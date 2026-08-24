"""Time-series momentum. PLAYBOOK Phase 6, after Moskowitz-Ooi-Pedersen (2012).

Every bound here was measured first. What was measured, 16 paths of 1,600 bars each,
`TimeSeriesMomentum(12, 1)` at zero cost, annualised Sharpe:

    GBM, mu = 0.08 (null)          -0.141 +/- 0.145     -1.0 SE
    GBM, mu = 0.00 (null)          -0.134 +/- 0.099     -1.4 SE
    persistent drift, psi = 0.98   +0.804 +/- 0.203     +4.0 SE
    persistent drift, psi = 0.99   +1.659 +/- 0.262     +6.3 SE
    mean-reverting AR(1)           -1.041 +/- 0.102    -10.2 SE

The last two rows are the pair worth having. A trend follower earns on a process whose
drift persists and loses, decisively, on the stationary process `CausalZScore` exploits --
same engine, same costs, same measurement. An edge is a property of a process, not of a
strategy, and this is the momentum-side statement of the same claim G3's power test makes
for mean reversion.

The `psi = 0.98` row landing on 0.804 against MOP's published 0.8 is a coincidence and is
not leaned on anywhere. See `test_momentum_is_bounded_well_below_the_bug_threshold` for
what the published figure is actually good for.
"""

from __future__ import annotations

import math
import statistics as st
from collections.abc import Callable
from functools import partial

import numpy as np
import pytest
from numpy.typing import NDArray

from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST, CostModel
from falsify.ledger import Ledger
from falsify.metrics import annualise_sharpe, sharpe
from falsify.strategies.base import Strategy
from falsify.strategies.momentum import BARS_PER_MONTH, TimeSeriesMomentum
from falsify.strategies.overlays import VolTarget
from falsify.synthetic import ar1, bars_from_close, gbm, persistent_drift

Prices = NDArray[np.float64]
Builder = Callable[[np.random.Generator, int], Prices]

# B3: the engines take a ledger, always. In-memory and non-persisting here.
LEDGER = Ledger.memory()

CAPITAL = 10_000.0
PATHS = 16
N_BARS = 1_600


def sharpe_over_paths(
    builder: Builder,
    strategy: Strategy | None = None,
    paths: int = PATHS,
    n: int = N_BARS,
) -> tuple[float, float]:
    """Annualised Sharpe of one strategy across independent paths. Mean and SE (B2)."""
    strat = strategy or TimeSeriesMomentum(12, 1)
    values = []
    for k in range(paths):
        bars = bars_from_close(builder(np.random.default_rng(400 + k), n))
        result = run_vectorized(bars, strat, ZERO_COST, CAPITAL, "next_open", ledger=LEDGER)
        values.append(annualise_sharpe(sharpe(result.net_ret[1:])))
    return st.mean(values), st.stdev(values) / math.sqrt(len(values))


def gbm_path(rng: np.random.Generator, n: int, mu: float = 0.08) -> Prices:
    return gbm(mu=mu, sigma=0.20, n_bars=n, rng=rng)


def trending_path(rng: np.random.Generator, n: int, psi: float = 0.99) -> Prices:
    return persistent_drift(psi, 0.0, 0.20, n, rng)


def reverting_path(rng: np.random.Generator, n: int) -> Prices:
    return ar1(phi=0.95, sigma=0.20, n_bars=n, rng=rng)


# --------------------------------------------------------------------------------------
# Structure. The holding period is the whole difference from a daily trend follower.
# --------------------------------------------------------------------------------------


def test_the_declared_lookback_accounts_for_the_decision_lag() -> None:
    """`close[t]/close[t-window]` is first computable at `t = window`, and `shift_one`
    moves the first usable weight to `window + 1`.

    The engine slices `signals[start - lag :]`, so it reads from exactly `lookback`.
    Declaring `window` instead of `window + 1` hands it one NaN and the run is refused --
    which is what happened on the first attempt, and is why this is asserted rather than
    left to the reader to rederive.
    """
    strategy = TimeSeriesMomentum(12, 1)
    assert strategy.window == 12 * BARS_PER_MONTH == 252
    assert strategy.lookback == 253

    bars = bars_from_close(gbm_path(np.random.default_rng(1), 600))
    signal = strategy.signals(bars)
    assert not np.isfinite(signal[strategy.lookback - 1]), "one bar earlier must still be NaN"
    assert np.isfinite(signal[strategy.lookback]), "the declared lookback must be usable"


def test_the_weight_is_held_between_rebalances() -> None:
    """The position changes only on rebalance bars. A daily-rebalanced version is a
    different strategy with several times the turnover, and quietly dropping the hold is
    how a published result becomes irreproducible."""
    strategy = TimeSeriesMomentum(12, 3)
    bars = bars_from_close(trending_path(np.random.default_rng(2), 2_000))
    signal = strategy.signals(bars)

    start = int(np.flatnonzero(np.isfinite(signal))[0])
    changed = np.flatnonzero(np.diff(signal[start:]) != 0.0) + 1
    assert changed.size > 0, "the signal never moved; this asserts nothing"
    off_schedule = [int(c) for c in changed if c % strategy.hold != 0]
    assert not off_schedule, f"the weight moved off the rebalance schedule at {off_schedule}"


def test_a_longer_hold_changes_when_the_flips_happen() -> None:
    """Measured: at a 12-month lookback the signal is so persistent that 1-month and
    3-month holds catch the *same number* of flips -- 5 each over 1,247 bars -- but at
    different bars (483 vs 504, 714 vs 756).

    So turnover per year can be identical while the strategies genuinely differ, which is
    worth knowing before reading equal turnover as evidence the hold does nothing.
    """
    bars = bars_from_close(gbm_path(np.random.default_rng(1), 1_500))
    monthly = TimeSeriesMomentum(12, 1).signals(bars)
    quarterly = TimeSeriesMomentum(12, 3).signals(bars)
    both = np.isfinite(monthly) & np.isfinite(quarterly)
    assert int(np.sum(monthly[both] != quarterly[both])) > 0, "the two holds are identical"


def test_the_rebalance_count_matches_the_schedule() -> None:
    strategy = TimeSeriesMomentum(12, 1)
    n = 1_500
    bars = bars_from_close(gbm_path(np.random.default_rng(3), n))
    signal = strategy.signals(bars)
    start = int(np.flatnonzero(np.isfinite(signal))[0])
    scheduled = len(range(start, n, strategy.hold))
    assert strategy.rebalance_count(n) == scheduled


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lookback_months": 0}, "lookback_months"),
        ({"hold_months": 0}, "hold_months"),
        ({"bars_per_month": 0}, "bars_per_month"),
    ],
)
def test_nonsensical_parameters_are_refused(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TimeSeriesMomentum(**kwargs)


# --------------------------------------------------------------------------------------
# Null and power. An edge is a property of a process.
# --------------------------------------------------------------------------------------


def test_there_is_no_edge_on_a_process_with_no_persistence() -> None:
    """GBM is momentum's null: constant drift, independent returns, nothing in the past
    that predicts the future. Measured -0.141 +/- 0.145 and -0.134 +/- 0.099."""
    for label, mu in (("drifting", 0.08), ("driftless", 0.00)):
        mean, se = sharpe_over_paths(partial(gbm_path, mu=mu))
        print(f"GBM {label}: {mean:+.3f} +/- {se:.3f}  ({mean / se:+.1f} SE)")
        assert abs(mean) < 4.0 * se, (
            f"TSMOM earned {mean:+.3f} +/- {se:.3f} on GBM, which has no persistence to "
            "find. Either the signal is reading something it should not, or the generator "
            "is not the null it claims to be."
        )


def test_there_is_a_real_edge_when_the_drift_persists() -> None:
    """Power. A trend follower must find a trend, or the null result above means nothing --
    a strategy that earns zero everywhere is not calibrated, it is broken.

    Measured +1.659 +/- 0.262, 6.3 SE above zero.
    """
    mean, se = sharpe_over_paths(trending_path)
    print(f"persistent drift psi=0.99: {mean:+.3f} +/- {se:.3f}  ({mean / se:+.1f} SE)")
    assert mean / se > 3.0, (
        f"TSMOM found only {mean:+.3f} +/- {se:.3f} on a process whose drift persists for "
        "months. The signal is not detecting the one structure it exists to detect."
    )


def test_the_edge_grows_with_persistence() -> None:
    """Monotone in the thing that causes it, which is the check that the power result is
    tracking the mechanism rather than an artefact. Measured 0.804 at psi=0.98 against
    1.659 at psi=0.99."""
    weak, _ = sharpe_over_paths(partial(trending_path, psi=0.98))
    strong, _ = sharpe_over_paths(partial(trending_path, psi=0.99))
    print(f"psi=0.98 {weak:+.3f}  ->  psi=0.99 {strong:+.3f}")
    assert strong > weak


def test_momentum_loses_on_the_process_mean_reversion_exploits() -> None:
    """The duality, on one engine. `CausalZScore` earns on a stationary AR(1) -- G3's power
    test asserts it -- and a trend follower loses on the same process at -10.2 SE.

    Both results are correct and the pair is the point. This is the momentum-side statement
    of the claim the A4 live test makes from the other direction on SPY.
    """
    mean, se = sharpe_over_paths(reverting_path)
    print(f"mean-reverting AR(1): {mean:+.3f} +/- {se:.3f}  ({mean / se:+.1f} SE)")
    assert mean < 0.0 and abs(mean / se) > 3.0, (
        f"TSMOM earned {mean:+.3f} on a mean-reverting process. It should lose there; if it "
        "does not, the signal is not doing what its name says."
    )


# --------------------------------------------------------------------------------------
# The literature check.
# --------------------------------------------------------------------------------------


def test_momentum_is_bounded_well_below_the_bug_threshold() -> None:
    """PLAYBOOK: "Published Sharpe ~ 0.8 on a diversified futures basket. If yours comes
    out at 3.0 on SPY, you have a bug."

    That is an upper sanity bound and not a target. The 0.8 is 58 futures across four
    asset classes, each vol-scaled and equally weighted, and most of it is diversification
    -- roughly uncorrelated bets stacked. A single instrument cannot reproduce it.

    So this asserts what the published figure can actually support: a single-asset TSMOM
    on a synthetic process must not reach the absurd. Even at `psi = 0.995`, a persistence
    no market exhibits, the measured Sharpe was 2.572 -- which is why 3.0 is a sound
    threshold rather than an arbitrary one.
    """
    mean, se = sharpe_over_paths(trending_path)
    print(f"most obliging process tested: {mean:+.3f} +/- {se:.3f} against a 3.0 bug bound")
    assert mean < 3.0, (
        f"TSMOM reached {mean:+.3f} on synthetic data. PLAYBOOK names 3.0 as the level at "
        "which the answer is a bug rather than an edge -- most likely a look-ahead in the "
        "signal, which G1 would also catch."
    )


# --------------------------------------------------------------------------------------
# Composition. MOP's full construction is this strategy plus the existing overlay.
# --------------------------------------------------------------------------------------


def test_vol_targeting_at_mops_forty_percent_is_a_no_op_without_leverage() -> None:
    """Worth asserting because it is surprising, and because it would otherwise look like
    the overlay silently failing.

    MOP scale each position to 40% annualised volatility. `VolTarget` defaults to
    `cap = 1.0`, which keeps weights inside the [-1, 1] contract in 02 Part B. On a series
    running at 20% vol the scale factor is 2.0 and the cap binds at every bar, so the
    overlay returns the base weights unchanged -- identical Sharpe, identical turnover.

    Reproducing MOP therefore requires asking for leverage explicitly, which is exactly
    what the cap is for: borrowing should be a decision, not a default.
    """
    bars = bars_from_close(persistent_drift(0.99, 0.10, 0.18, 2_000, np.random.default_rng(5)))
    base = TimeSeriesMomentum(12, 1)
    capped = VolTarget(base, 0.40, 60)

    plain = run_vectorized(bars, base, ZERO_COST, CAPITAL, "next_open", ledger=LEDGER)
    scaled = run_vectorized(bars, capped, ZERO_COST, CAPITAL, "next_open", ledger=LEDGER)
    assert np.array_equal(plain.weights, scaled.weights), (
        "VolTarget(0.40) changed the weights on a 18%-vol series. With cap=1.0 the scale "
        "factor is above 1 everywhere, so the cap should bind at every bar."
    )

    levered = VolTarget(base, 0.40, 60, cap=2.0)
    with_leverage = run_vectorized(bars, levered, ZERO_COST, CAPITAL, "next_open", ledger=LEDGER)
    assert not np.array_equal(plain.weights, with_leverage.weights), (
        "raising the cap must let the overlay actually scale, or MOP's construction is unreachable"
    )


def test_the_hold_keeps_turnover_far_below_a_daily_trend_follower() -> None:
    """The cost consequence of the holding period, which is why MOP specify one."""
    bars = bars_from_close(trending_path(np.random.default_rng(7), 2_000))
    costly = CostModel(commission_bps=10.0)

    held = run_vectorized(
        bars, TimeSeriesMomentum(12, 1), costly, CAPITAL, "next_open", ledger=LEDGER
    )
    turnover = float(np.sum(held.turnover) / len(held) * 252)
    print(f"TSMomentum(12m,1m) turnover: {turnover:.2f}/yr")
    assert turnover < 12.0, (
        f"turnover of {turnover:.1f}/yr for a strategy that rebalances monthly. It should "
        "be a handful of flips a year, not a trade a fortnight."
    )
