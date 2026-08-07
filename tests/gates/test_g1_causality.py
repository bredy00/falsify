"""G1 (causality) and G7 (the leakage trap), in one file because neither is
meaningful alone.

G1 asserts that scrambling the future leaves the past bitwise identical. G7 is
the test of the test: deliberately leaky pipelines that G1 must reject. A harness
without its trap is unverified, which is why 03 Part I forbids splitting them
across two commits.

The traps are the four classes 02 Part A3 names as passing code review and
failing G1: a scaler fitted on the whole series, a rolling window with
`center=True`, a global-percentile clip, and a warm-up length that depends on a
whole-series statistic. Plus `LeakyOracle` from Part A4, whose behaviour is more
interesting than advertised -- see test_leaky_oracle_is_not_a_causality_violation.
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
from falsify.strategies.base import Strategy

T_BARS = 512
MASTER_SEED = 20140458

# 02 Part A4 test parameters. Held at spec strength for the honest strategies,
# since those are the ones a false pass would certify.
N_TAUS = 500
N_SEEDS = 20


# --------------------------------------------------------------- causal helpers


def rolling_mean(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Mean over the `window` bars ending at t. NaN before the first full window."""
    out = np.full(len(x), np.nan)
    if window <= len(x):
        out[window - 1 :] = sliding_window_view(x, window).mean(axis=1)
    return out


def rolling_std(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Sample std over the `window` bars ending at t."""
    out = np.full(len(x), np.nan)
    if window <= len(x):
        out[window - 1 :] = sliding_window_view(x, window).std(axis=1, ddof=1)
    return out


def shift_one(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Move each value one bar later: the weight held at t was decided at t-1."""
    out = np.full(len(x), np.nan)
    out[1:] = x[:-1]
    return out


def bars_from_close(close: Prices) -> Bars:
    """Build Bars from a close series, strictly per bar.

    Every field is a function of bar t and t-1 only, so this stage cannot itself
    leak. Deliberately so: if the fixture leaked, every G1 result in this file
    would be meaningless. The timestamps are a fixed calendar independent of
    price, which is what a real `align` stage produces (02 Part G).
    """
    n = len(close)
    ts = np.datetime64("2020-01-01", "ns") + np.arange(n) * np.timedelta64(1, "D")
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    return Bars(
        ts=ts,
        open=open_,
        high=np.maximum(open_, close),
        low=np.minimum(open_, close),
        close=close.copy(),
        volume=np.full(n, 1_000_000.0),
        adjustment="total_return",
    )


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


# ------------------------------------------------------------ honest strategies


class BuyAndHold(Strategy):
    lookback = 1

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        out = np.full(len(bars), 1.0)
        out[: self.lookback] = np.nan
        return out


class MACrossover(Strategy):
    """The reference repo's strategy, done correctly: long when the fast mean is
    above the slow one, with the decision lagged a bar."""

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast >= slow:
            raise ValueError(f"fast={fast} must be below slow={slow}")
        self.fast, self.slow = fast, slow
        self.lookback = slow

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        raw = np.sign(rolling_mean(bars.close, self.fast) - rolling_mean(bars.close, self.slow))
        return shift_one(raw)


class CausalZScore(Strategy):
    """Mean reversion on a z-score of price against its own trailing window."""

    def __init__(self, window: int = 30) -> None:
        self.window = window
        self.lookback = window

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        mu = rolling_mean(bars.close, self.window)
        sd = rolling_std(bars.close, self.window)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (bars.close - mu) / sd
        return shift_one(np.clip(-z, -1.0, 1.0))


HONEST: tuple[Strategy, ...] = (BuyAndHold(), MACrossover(), CausalZScore())


# ------------------------------------------------------- the traps (G7)


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


class LeakyOracle(Strategy):
    """02 Part A4 verbatim: trades on close[t] at close[t]."""

    lookback = 1

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        return np.asarray(np.sign(np.diff(bars.close, prepend=bars.close[0])), dtype=np.float64)


TRAPS: tuple[Strategy, ...] = (
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


def test_leaky_oracle_is_not_a_causality_violation(prices: Prices) -> None:
    """02 Part A4 says G1 must catch `LeakyOracle`. It provably cannot, and this
    records why rather than leaving it as a comment.

    `LeakyOracle` sets the weight at t from close[t] and close[t-1]. Both are
    inside bars[0:t+1], which the Part A1 contract explicitly permits, and
    neither is scrambled for t <= tau. So the signal prefix is invariant and G1
    is silent -- correctly. The one-bar lag is the engine's job (Part D), not the
    strategy's, and `close_to_close` is a permitted-but-optimistic convention
    rather than a leak.

    Scrambling the cut bar itself checks the stricter execution-alignment claim,
    and there the oracle is caught immediately. Both halves are asserted, so if
    either behaviour ever changes the build says so.
    """
    pipeline = make_pipeline(LeakyOracle())
    rng = np.random.default_rng(MASTER_SEED + 4)
    taus = sample_taus(len(prices), 2, rng, n_taus=64)

    # Part A1 contract: silent, because nothing is violated.
    causality_cut_test(pipeline, prices, taus, rng, n_seeds=4)

    # Execution alignment: caught, because the weight at t needs close[t].
    with pytest.raises(CausalityViolation, match="causality violated"):
        causality_cut_test(pipeline, prices, taus, rng, n_seeds=4, include_cut=True)


@pytest.mark.parametrize("strategy", HONEST, ids=lambda s: s.name)
def test_honest_strategies_also_survive_the_strict_cut(
    strategy: Strategy, prices: Prices
) -> None:
    """The honest strategies lag their decisions, so they pass the stricter check
    too -- which is what makes `shift(1)` a proven claim here rather than an
    asserted one."""
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
