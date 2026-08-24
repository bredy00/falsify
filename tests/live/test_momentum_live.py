"""Time-series momentum on real SPY. The MOP calibration check, run against the market.

Marked `live`; CI deselects it. Requires the cache:

    uv run --group data python scripts/fetch_data.py

PLAYBOOK: "Published Sharpe ~ 0.8 on a diversified futures basket. If yours comes out at
3.0 on SPY, you have a bug -- this is a free calibration check against the literature."

Measured on SPY 2015-2024, zero cost, annualised:

    TSMomentum(12m,1m)                +0.606 +/- 0.340   turnover 1.56/yr   NW t +1.93
    MOP form: VolTarget(0.40, cap 2)  +0.673 +/- 0.340   turnover 3.27/yr   NW t +2.10
    TSMomentum(12m,3m)                +0.203 +/- 0.336   turnover 1.56/yr   NW t +0.65
    TSMomentum(6m,1m)                 +0.166 +/- 0.327   turnover 3.38/yr   NW t +0.55
    BuyAndHold                        +0.830 +/- 0.325   turnover 0.00/yr   NW t +2.80
    MACrossover(20,50)                +0.217 +/- 0.319   turnover 9.41/yr   NW t +0.72

0.606 against a published 0.8 looks like a hit and is not one. The 0.8 is 58 futures
across four asset classes, vol-scaled and equally weighted, and most of it is
diversification. One equity index reproducing it would be luck, and the standard error
here is 0.340 -- wide enough to contain both numbers and zero. What the check actually
licenses is the negative: nothing here is near 3.0, so nothing indicates a bug.

The honest headline is the row nobody wants: buy-and-hold beat every one of them, and it
is the only one whose Newey-West t-statistic clears 2. Same finding the A4 live test
reached from the other direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from falsify.core.types import Bars
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST, CostModel
from falsify.data.loaders import DEFAULT_CACHE, DEFAULT_MANIFEST, FetchSpec, load
from falsify.features import rolling_std
from falsify.ledger import Ledger
from falsify.metrics import annualise_sharpe, newey_west_t, sharpe, sharpe_se
from falsify.strategies.momentum import TimeSeriesMomentum
from falsify.strategies.overlays import VolTarget
from falsify.strategies.simple import BuyAndHold

pytestmark = pytest.mark.live

LEDGER = Ledger.memory()
SPEC = FetchSpec("SPY", "2015-01-01", "2025-01-01", "total_return")
CAPITAL = 10_000.0


@pytest.fixture(scope="module")
def spy() -> Bars:
    from falsify.data.loaders import DataUnavailable
    from falsify.data.manifest import ManifestMismatch

    try:
        return load(SPEC, cache_dir=DEFAULT_CACHE, manifest_path=DEFAULT_MANIFEST)
    except (DataUnavailable, ManifestMismatch) as exc:
        pytest.skip(f"SPY cache unavailable ({exc.__class__.__name__}); run scripts/fetch_data.py")


def annual_sharpe(bars: Bars, strategy: object, bps: float = 0.0) -> float:
    costs = ZERO_COST if bps == 0.0 else CostModel(commission_bps=bps)
    result = run_vectorized(bars, strategy, costs, CAPITAL, "next_open", ledger=LEDGER)  # type: ignore[arg-type]
    return annualise_sharpe(sharpe(result.net_ret[1:]))


def test_momentum_on_spy_is_nowhere_near_the_bug_threshold(spy: Bars) -> None:
    """The calibration check as PLAYBOOK actually states it: an upper bound.

    A single-asset trend follower reaching 3.0 on a decade of daily equity data would mean
    a look-ahead, not an edge. This asserts the bound and reports the number.
    """
    strategy = TimeSeriesMomentum(12, 1)
    result = run_vectorized(spy, strategy, ZERO_COST, CAPITAL, "next_open", ledger=LEDGER)
    returns = result.net_ret[1:]
    value = annualise_sharpe(sharpe(returns))
    turnover = float(np.sum(result.turnover) / len(result) * 252)

    print(
        f"TSMomentum(12m,1m) on SPY: SR {value:+.3f} +/- {sharpe_se(returns, 252):.3f}, "
        f"turnover {turnover:.2f}/yr, Newey-West t {newey_west_t(returns):+.2f}, "
        f"{strategy.rebalance_count(len(spy))} rebalances"
    )
    assert value < 3.0, (
        f"TSMOM earned {value:+.3f} on SPY. PLAYBOOK names 3.0 as the level at which the "
        "answer is a bug rather than an edge; G1 should also have caught it."
    )
    assert value > -1.0, f"{value:+.3f} is implausibly bad for a trend follower on an index"


def test_the_holding_period_keeps_turnover_low_on_real_data(spy: Bars) -> None:
    """1.56 turns a year against MACrossover's 9.41. That is the holding period, and it
    is why costs barely touch this strategy: 0.606 gross becomes 0.601 at 5 bps."""
    result = run_vectorized(
        spy, TimeSeriesMomentum(12, 1), ZERO_COST, CAPITAL, "next_open", ledger=LEDGER
    )
    turnover = float(np.sum(result.turnover) / len(result) * 252)
    gross = annual_sharpe(spy, TimeSeriesMomentum(12, 1))
    net = annual_sharpe(spy, TimeSeriesMomentum(12, 1), bps=5.0)

    print(f"turnover {turnover:.2f}/yr, SR {gross:+.3f} gross -> {net:+.3f} at 5 bps")
    assert turnover < 4.0, "a monthly-rebalanced strategy should not trade weekly"
    assert abs(gross - net) < 0.05, (
        "costs moved the Sharpe more than expected for 1.6 turns a year; either turnover "
        "is higher than measured or the cost model is being applied twice"
    )


def test_buy_and_hold_still_beats_it(spy: Bars) -> None:
    """Reported rather than buried, and it is the same answer the A4 live test gave.

    SPY over 2015-2024 rose for most of the decade. A trend follower that is long most of
    that time and occasionally steps out cannot beat simply staying in, and the honest
    version of this result says so instead of quoting 0.606 next to a published 0.8 and
    letting the reader assume a match.
    """
    hold = annual_sharpe(spy, BuyAndHold())
    momentum = annual_sharpe(spy, TimeSeriesMomentum(12, 1))
    print(f"buy-and-hold {hold:+.3f} vs TSMomentum {momentum:+.3f}")
    assert hold > momentum, (
        "TSMOM beat buy-and-hold on this window. That would be worth investigating rather "
        "than celebrating -- start with G1 and the turnover."
    )


def test_vol_targeting_bites_only_in_the_crash(spy: Bars) -> None:
    """MOP scale to 40% annualised volatility, and on real data that does something very
    specific: nothing at all, until it matters.

    `VolTarget` defaults to `cap = 1.0`, so it can only ever reduce exposure. SPY's 60-day
    realised volatility ranges from 0.050 to 0.614 across this decade, and sits below 0.40
    almost throughout -- so the scale factor exceeds 1, the cap binds, and the weights are
    untouched.

    Measured: the overlay changes the weight at exactly 65 of 2,261 bars (2.9%), which is
    exactly the number of bars whose trailing volatility exceeded 40%, and they run
    2020-03-18 to 2020-06-18. It is inert for 97% of the decade and de-risks through the
    COVID crash.

    On the constant-volatility synthetic series in `test_momentum.py` the same overlay is a
    strict no-op, because there is no volatility clustering for it to respond to. Both
    statements are true and the pair is the point: an overlay tested only on synthetic data
    would look like dead code.
    """
    base = TimeSeriesMomentum(12, 1)
    plain = run_vectorized(spy, base, ZERO_COST, CAPITAL, "next_open", ledger=LEDGER)
    capped = run_vectorized(
        spy, VolTarget(base, 0.40, 60), ZERO_COST, CAPITAL, "next_open", ledger=LEDGER
    )

    differing = np.flatnonzero(plain.weights != capped.weights)
    share = differing.size / plain.weights.size
    returns = np.diff(spy.close) / spy.close[:-1]
    high_vol = int(np.nansum(rolling_std(returns, 60) * np.sqrt(252) > 0.40))

    offset = len(spy) - plain.weights.size
    stamps = spy.ts[offset:][differing]
    print(
        f"vol target active at {differing.size}/{plain.weights.size} bars ({share:.1%}), "
        f"{high_vol} bars above 40% vol, {stamps[0]} .. {stamps[-1]}"
    )

    assert 0 < differing.size < 0.10 * plain.weights.size, (
        "the 40% target should be slack almost always on an index running near 18% vol"
    )
    assert differing.size == high_vol, (
        f"the overlay moved the weight at {differing.size} bars but only {high_vol} bars "
        "exceeded the target volatility. Those should be the same bars; if they are not, "
        "the overlay's volatility estimate is not the one measured here."
    )

    levered = run_vectorized(
        spy, VolTarget(base, 0.40, 60, cap=2.0), ZERO_COST, CAPITAL, "next_open", ledger=LEDGER
    )
    assert not np.array_equal(plain.weights, levered.weights), (
        "raising the cap must let the overlay scale up, or MOP's construction is unreachable"
    )


def test_no_configuration_of_it_is_significant_before_deflation(spy: Bars) -> None:
    """Four configurations, and the best Newey-West t-statistic among them is 2.10 -- and
    that is *before* any correction for having tried four.

    01 Part B1: use the HAC t-statistic, not the naive one, whenever a significance claim
    is made. This is what that discipline produces here, and the answer is that there is
    no claim to make.
    """
    configurations = [
        TimeSeriesMomentum(12, 1),
        TimeSeriesMomentum(12, 3),
        TimeSeriesMomentum(6, 1),
        VolTarget(TimeSeriesMomentum(12, 1), 0.40, 60, cap=2.0),
    ]
    stats = []
    for strategy in configurations:
        result = run_vectorized(spy, strategy, ZERO_COST, CAPITAL, "next_open", ledger=LEDGER)
        stats.append((strategy.name, newey_west_t(result.net_ret[1:])))
    for name, t_stat in stats:
        print(f"  {name:<34} Newey-West t {t_stat:+.2f}")

    best = max(abs(t) for _, t in stats)
    assert best < 3.0, (
        f"a HAC t-statistic of {best:.2f} on a single index would be a strong claim, and "
        "would need the trials ledger's N applied to it before anyone believed it."
    )
