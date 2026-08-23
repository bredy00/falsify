"""G1 (causality) and G7 (the leakage trap), in one file because neither is
meaningful alone.

G1 asserts that scrambling the future leaves the past bitwise identical. G7 is
the test of the test: deliberately leaky pipelines that G1 must reject. A harness
without its trap is unverified, which is why 03 Part I forbids splitting them
across two commits.

The traps are the four classes 02 Part A3 names as passing code review and
failing G1 -- a scaler fitted on the whole series, a rolling window with
`center=True`, a global-percentile clip, and a warm-up length that depends on a
whole-series statistic -- plus a genuine look-ahead oracle.

On A4's `LeakyOracle`: see `test_a4_oracle_does_not_leak_and_has_no_edge`. It is
kept here as a documented NON-violator, because the reasoning for that ruling is
worth more than the strategy was.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from falsify.core.causality import (
    CausalityViolation,
    Pipeline,
    Prices,
    causality_cut_test,
    sample_taus,
)
from falsify.core.types import Bars
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST
from falsify.features import rolling_mean, shift_one
from falsify.ledger import Ledger
from falsify.metrics import annualise_sharpe, sharpe
from falsify.strategies.base import Strategy
from falsify.strategies.null import RandomSign
from falsify.strategies.overlays import TurnoverBuffer, VolTarget
from falsify.strategies.simple import ZOO, BuyAndHold, CausalZScore
from falsify.synthetic import bars_from_close, gbm

# B3: the engines take a ledger, always. In-memory and non-persisting here --
# every invocation is still counted, which is what lets a test assert its own
# search size, but the gate suite does not write to the shipped ledger.
LEDGER = Ledger.memory()

T_BARS = 512
MASTER_SEED = 20140458

# 02 Part A4 test parameters. Held at spec strength for the honest strategies,
# since those are the ones a false pass would certify.
N_TAUS = 500
N_SEEDS = 20

# Everything that emits a weight has to be proven causal, including the overlays and
# the null. The overlays are the interesting case: `TurnoverBuffer` is a stateful
# left-to-right fold, which is exactly the shape that leaks if the state is seeded
# from a whole-series statistic instead of accumulated from the start. The null is
# the trivial case -- its path ignores prices entirely -- but a null that leaked
# would invalidate G6, so it is checked rather than assumed.
_BASE = CausalZScore(20)
OVERLAID: tuple[Strategy, ...] = (
    TurnoverBuffer(_BASE, 0.25),
    VolTarget(_BASE, 0.15, 60),
    TurnoverBuffer(VolTarget(_BASE, 0.15, 60), 0.25),
    RandomSign(T_BARS, np.random.default_rng(MASTER_SEED + 100), 0.2, 0.75),
)

HONEST = ZOO

# Reduced grid for the overlays. `TurnoverBuffer` folds over the whole series on every
# call, so spec strength across the composed overlays is ~36M loop iterations for no
# additional information: 640 independent cuts already covers every code path, and the
# base strategies underneath are separately checked at full strength above. Recorded
# rather than silently reduced, per F7's spirit -- a cap that is not stated reads as
# coverage that was never there.
N_TAUS_OVERLAY = 128
N_SEEDS_OVERLAY = 5


def make_pipeline(strategy: Strategy) -> Pipeline:
    """Compose the stages G1 runs end to end: prices -> Bars -> signals.

    When the data layer lands (Phase 5) this closure is where parse, validate,
    align and adjust get inserted, and G1 must be re-pointed at the widened
    pipeline. 02 Part A2 is explicit that leakage lives upstream more often than
    in the strategy, so testing `signals` alone would pass while a `bfill`
    quietly cheated.
    """

    def pipeline(prices: Prices) -> NDArray[np.float64]:
        return strategy.signals(bars_from_close(prices))

    return pipeline


# ------------------------------------------------------- the traps (G7)


class LookAheadOracle(Strategy):
    """Sets today's weight from tomorrow's move. A genuine A1 violation, and what
    Part A4 was reaching for when it named a strategy `LeakyOracle`."""

    lookback = 1

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        close = bars.close
        out = np.full(len(close), np.nan)
        out[self.lookback : -1] = np.sign(close[self.lookback + 1 :] - close[self.lookback : -1])
        out[-1] = 0.0
        return out


class GlobalScalerLeak(Strategy):
    """Standardises using the mean and std of the WHOLE series.

    The most common real leak, and invisible in review: `StandardScaler` fitted
    before the train/test split.
    """

    lookback = 30

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        z = (bars.close - bars.close.mean()) / bars.close.std(ddof=1)
        return shift_one(np.clip(-z, -1.0, 1.0))


class CenteredWindowLeak(Strategy):
    """A rolling mean centred on t, so half its window is the future."""

    lookback = 30

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        window, n = 21, len(bars)
        half = window // 2
        centred = np.full(n, np.nan)
        if window <= n:
            centred[half : n - half] = sliding_window_view(bars.close, window).mean(axis=1)
        return shift_one(np.sign(bars.close - centred))


class GlobalPercentileClipLeak(Strategy):
    """Clips at percentiles computed over the entire series."""

    lookback = 30

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        lo, hi = np.percentile(bars.close, [5.0, 95.0])
        clipped = np.clip(bars.close, lo, hi)
        return shift_one(np.sign(clipped - rolling_mean(clipped, self.lookback)))


class FutureDependentWarmupLeak(Strategy):
    """Values are causal, but the warm-up length depends on where the series
    peaks -- so the NaN pattern encodes the future.

    A pipeline can leak through its shape as surely as through its numbers, which
    is why G1 compares with `equal_nan=True` rather than dropping NaNs.
    """

    lookback = 30

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        raw = shift_one(np.sign(bars.close - rolling_mean(bars.close, self.lookback)))
        raw[: int(np.argmax(bars.close)) + 1] = np.nan
        return raw


class A4LeakyOracle(Strategy):
    """02 Part A4 verbatim: `sign(diff(close))`.

    Kept as a documented non-violator. See
    test_a4_oracle_does_not_leak_and_has_no_edge.
    """

    lookback = 1

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        return np.asarray(np.sign(np.diff(bars.close, prepend=bars.close[0])), dtype=np.float64)


TRAPS: tuple[Strategy, ...] = (
    LookAheadOracle(),
    GlobalScalerLeak(),
    CenteredWindowLeak(),
    GlobalPercentileClipLeak(),
    FutureDependentWarmupLeak(),
)


# ---------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def prices() -> Prices:
    """A driftless geometric random walk. No edge, which is irrelevant here --
    G1 is about information flow, not profitability."""
    rng = np.random.default_rng(MASTER_SEED)
    return np.asarray(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, T_BARS))), dtype=np.float64)


# --------------------------------------------------------------------- G1 tests


@pytest.mark.parametrize("strategy", HONEST, ids=lambda s: s.name)
def test_g1_honest_strategies_are_causal(strategy: Strategy, prices: Prices) -> None:
    """500 cuts, 20 seeds each, at spec strength (02 Part A4)."""
    rng = np.random.default_rng(MASTER_SEED + 1)
    taus = sample_taus(len(prices), strategy.lookback, rng, n_taus=N_TAUS)
    causality_cut_test(make_pipeline(strategy), prices, taus, rng, n_seeds=N_SEEDS)


@pytest.mark.parametrize("strategy", OVERLAID, ids=lambda s: s.name)
def test_g1_overlays_and_null_are_causal(strategy: Strategy, prices: Prices) -> None:
    """The overlays and the null, at 128 cuts x 5 seeds.

    `TurnoverBuffer` is the one that matters here. It is a stateful left-to-right fold,
    which is precisely the shape that leaks if the state is ever seeded from a
    whole-series statistic rather than accumulated from bar zero -- and the leak would
    be invisible in review because the fold itself looks obviously causal.

    The null is checked too. Its path ignores prices entirely so it cannot leak, but a
    leaking null would invalidate every number G6 produces, and that is not a thing to
    take on faith.
    """
    rng = np.random.default_rng(MASTER_SEED + 8)
    taus = sample_taus(len(prices), strategy.lookback, rng, n_taus=N_TAUS_OVERLAY)
    causality_cut_test(make_pipeline(strategy), prices, taus, rng, n_seeds=N_SEEDS_OVERLAY)


@pytest.mark.parametrize("strategy", OVERLAID, ids=lambda s: s.name)
def test_g1_overlays_survive_the_strict_cut(strategy: Strategy, prices: Prices) -> None:
    """The overlays must also pass the execution-alignment cut.

    This is what caught the real bug in `VolTarget`: the first version sized the
    position on same-bar volatility while the base signal was lagged a bar, so the
    weight was set from an information set the signal itself was not allowed to use.
    It passed the Part A1 causality contract and failed here -- which is the entire
    reason both cut modes exist.
    """
    rng = np.random.default_rng(MASTER_SEED + 9)
    taus = sample_taus(len(prices), max(strategy.lookback, 2), rng, n_taus=48)
    causality_cut_test(make_pipeline(strategy), prices, taus, rng, n_seeds=4, include_cut=True)


def test_g1_rejects_a_pipeline_whose_output_length_moves(prices: Prices) -> None:
    """Shape changes are leakage too, and would otherwise slip past a
    prefix-only comparison."""

    def pipeline(p: Prices) -> NDArray[np.float64]:
        keep = p > np.median(p)  # a global statistic, so the length is future-dependent
        return np.asarray(p[keep], dtype=np.float64)

    rng = np.random.default_rng(MASTER_SEED + 2)
    with pytest.raises(CausalityViolation):
        causality_cut_test(pipeline, prices, [100, 200], rng, n_seeds=2)


# ---------------------------------------------------------------- G7, the trap


@pytest.mark.parametrize("strategy", TRAPS, ids=lambda s: s.name)
def test_g7_leaks_are_caught(strategy: Strategy, prices: Prices) -> None:
    """Every trap must be rejected. If any of these passes, G1 certifies nothing
    and every downstream number in the project is unsupported."""
    rng = np.random.default_rng(MASTER_SEED + 3)
    taus = sample_taus(len(prices), strategy.lookback, rng, n_taus=32)
    with pytest.raises(CausalityViolation, match="causality violated"):
        causality_cut_test(make_pipeline(strategy), prices, taus, rng, n_seeds=4)


def test_a4_oracle_does_not_leak_and_has_no_edge(prices: Prices) -> None:
    """The A4 ruling, as executable evidence. Taken 2026-08-08: A1 stands.

    02 Part A4 asserts G1 must catch `sign(diff(close))` and that a harness which
    stays silent is broken. The harness is right and the trap was wrong, for two
    independent reasons this test pins down.

    Structurally: close[t] is inside bars[0:t+1], which A1 permits, and every
    Part D convention lags the weight at least one bar, so the weight earning
    return t was decided from strictly older bars. G1 is correctly silent.

    Empirically, which is the part that settles it: run through the engine the
    strategy earns nothing. If it genuinely saw one bar ahead its Sharpe would be
    enormous -- `LookAheadOracle`, which does, is in TRAPS and G1 rejects it. So
    tightening A1 to bars[0:t] strictly would have failed every legitimate
    close_to_close strategy while catching nothing real.
    """
    pipeline = make_pipeline(A4LeakyOracle())
    rng = np.random.default_rng(MASTER_SEED + 4)
    taus = sample_taus(len(prices), 2, rng, n_taus=64)

    # 1. Causality: silent, because nothing is violated.
    causality_cut_test(pipeline, prices, taus, rng, n_seeds=4)

    # 2. Execution alignment: caught, because the weight at t needs close[t].
    with pytest.raises(CausalityViolation, match="causality violated"):
        causality_cut_test(pipeline, prices, taus, rng, n_seeds=4, include_cut=True)

    # 3. No edge, which is what makes it not an oracle.
    oracle_sr: list[float] = []
    hold_sr: list[float] = []
    for i in range(12):
        bars = bars_from_close(gbm(0.08, 0.20, 1000, np.random.default_rng(7_000 + i)))
        for strat, bucket in ((A4LeakyOracle(), oracle_sr), (BuyAndHold(), hold_sr)):
            result = run_vectorized(
                bars, strat, ZERO_COST, 10_000.0, "close_to_close", ledger=LEDGER
            )
            bucket.append(annualise_sharpe(sharpe(result.net_ret[1:])))

    mean_oracle = float(np.mean(oracle_sr))
    se_oracle = float(np.std(oracle_sr, ddof=1) / np.sqrt(len(oracle_sr)))
    print(
        f"A4 sign(diff(close)) through the engine: SR = {mean_oracle:+.3f} +/- {se_oracle:.3f} "
        f"vs buy-and-hold {float(np.mean(hold_sr)):+.3f} -- not an oracle"
    )
    assert abs(mean_oracle) < 4.0, (
        f"A4's strategy earned SR={mean_oracle:+.3f}; if it really saw ahead this would "
        "be enormous, and the ruling that it does not leak would need revisiting"
    )


@pytest.mark.parametrize("strategy", HONEST, ids=lambda s: s.name)
def test_honest_strategies_also_survive_the_strict_cut(strategy: Strategy, prices: Prices) -> None:
    """The honest strategies lag their decisions, so they pass the stricter
    execution-alignment check too -- which is what makes `shift(1)` a proven claim
    here rather than an asserted one."""
    rng = np.random.default_rng(MASTER_SEED + 5)
    taus = sample_taus(len(prices), max(strategy.lookback, 2), rng, n_taus=64)
    causality_cut_test(make_pipeline(strategy), prices, taus, rng, n_seeds=4, include_cut=True)


# ------------------------------------------------- the harness cannot be vacuous


def test_scramble_actually_changes_the_future(prices: Prices) -> None:
    """A no-op scramble would make every G1 pass meaningless (F7)."""
    from falsify.core.causality import scramble_future

    rng = np.random.default_rng(MASTER_SEED + 6)
    tau = 200
    for include_cut in (False, True):
        s = scramble_future(prices, tau, rng, include_cut=include_cut)
        keep = tau if include_cut else tau + 1
        assert np.array_equal(s[:keep], prices[:keep]), "the preserved prefix was modified"
        assert not np.array_equal(s[keep:], prices[keep:]), "the future was left unchanged"
        assert np.all(s > 0.0), "scrambled prices must stay positive"


def test_g1_rejects_a_constant_pipeline_only_when_it_should() -> None:
    """A pipeline ignoring prices entirely is trivially causal, and G1 must not
    invent a failure for it. The complement of G7: the gate has to be silent when
    silence is correct, or a green suite means nothing."""
    rng = np.random.default_rng(MASTER_SEED + 7)
    p = np.asarray(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 256))), dtype=np.float64)
    causality_cut_test(lambda x: np.zeros(len(x)), p, [50, 100, 150], rng, n_seeds=4)
