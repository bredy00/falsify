"""G5 -- cost monotonicity and the break-even point.

Sweep cost from 0 to 100 bps and require net Sharpe to be non-increasing, then
report the break-even cost c* where it crosses zero. A sweep and a
`np.diff(...) <= 0` check, as 03 Part C says -- but the break-even number it
produces is the most useful figure the whole report will carry, because it
converts "this works" into "this works if you can trade for under c* bps a turn",
which a reader can check against their own execution.

Monotonicity is not a triviality worth skipping. It fails if cost is credited
anywhere, if turnover is computed with the wrong sign, if the charge is applied to
the wrong bar's equity, or if the net return is reconstructed rather than measured.
None of those raise an exception.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from falsify.analysis import CostSweep, sweep_costs
from falsify.core.types import Bars
from falsify.core.vectorized import run_vectorized
from falsify.costs import CostModel
from falsify.ledger import Ledger
from falsify.metrics import annualise_sharpe, sharpe
from falsify.strategies.base import Strategy
from falsify.strategies.simple import BuyAndHold, CausalZScore, MACrossover
from falsify.synthetic import ar1, bars_from_close, gbm

# B3: the engines take a ledger, always. In-memory and non-persisting here --
# every invocation is still counted, which is what lets a test assert its own
# search size, but the gate suite does not write to the shipped ledger.
LEDGER = Ledger.memory()

CAPITAL = 10_000.0
SEED = 5_050
GRID = np.linspace(0.0, 100.0, 41)  # 0 to 100 bps in 2.5 bps steps


@pytest.fixture(scope="module")
def trending_bars() -> Bars:
    """AR(1) so the mean-reversion strategy has a real edge to lose to costs.

    On a random walk the zero-cost Sharpe is already noise, so the sweep would
    start near zero and the break-even point would measure nothing.
    """
    return bars_from_close(ar1(0.95, 0.02, 1500, np.random.default_rng(SEED)))


@pytest.fixture(scope="module")
def sweep(trending_bars: Bars) -> CostSweep:
    return sweep_costs(trending_bars, CausalZScore(20), GRID, CAPITAL, "next_open", ledger=LEDGER)


# ------------------------------------------------------------------- the gate


def test_g5_net_sharpe_is_non_increasing_in_cost(sweep: CostSweep) -> None:
    """The gate itself."""
    steps = np.diff(sweep.sharpe_annual)
    worst = float(np.max(steps))
    print(
        f"net SR from {sweep.sharpe_annual[0]:+.4f} at 0 bps to "
        f"{sweep.sharpe_annual[-1]:+.4f} at {sweep.bps[-1]:g} bps; "
        f"largest step {worst:+.3e}; annualised turnover {sweep.turnover_annual:.2f}"
    )
    assert sweep.is_monotone(), (
        f"net Sharpe rose by {worst:.3e} somewhere in the sweep. Costs are being credited, "
        "or turnover has the wrong sign, or the charge hits the wrong bar's equity."
    )


def test_g5_break_even_cost_is_finite_and_reported(sweep: CostSweep, trending_bars: Bars) -> None:
    """c* must be a real number inside the swept range, and the sweep must be wide
    enough to have actually bracketed it rather than extrapolating."""
    c_star = sweep.break_even_bps()
    print(f"break-even cost c* = {c_star:.2f} bps per turn")
    assert math.isfinite(c_star), (
        "net Sharpe never crossed zero inside 0-100 bps, so c* is not determined here. "
        "Widen the grid rather than reporting a number the sweep did not see."
    )
    assert 0.0 < c_star < float(sweep.bps[-1])

    # The crossing is real: below c* the Sharpe is positive, above it is not.
    below = run_vectorized(
        trending_bars,
        CausalZScore(20),
        CostModel(commission_bps=c_star * 0.5),
        CAPITAL,
        "next_open",
        ledger=LEDGER,
    )
    above = run_vectorized(
        trending_bars,
        CausalZScore(20),
        CostModel(commission_bps=c_star * 1.5),
        CAPITAL,
        "next_open",
        ledger=LEDGER,
    )
    sr_below = annualise_sharpe(sharpe(below.net_ret[1:]))
    sr_above = annualise_sharpe(sharpe(above.net_ret[1:]))
    print(
        f"  at {c_star * 0.5:.2f} bps SR = {sr_below:+.4f};  at {c_star * 1.5:.2f} bps SR = {sr_above:+.4f}"
    )
    assert sr_below > 0.0 > sr_above, "c* does not actually separate profit from loss"


# ------------------------------------------------------------- properties


def test_g5_zero_turnover_is_indifferent_to_cost() -> None:
    """Buy-and-hold never trades after the anchor, so its Sharpe must be flat
    across the whole sweep. If it moves, cost is being charged on something other
    than traded notional -- which is the naive baseline's mistake."""
    bars = bars_from_close(gbm(0.08, 0.20, 800, np.random.default_rng(SEED + 1)))
    flat = sweep_costs(bars, BuyAndHold(), GRID, CAPITAL, "next_open", ledger=LEDGER)
    spread = float(np.ptp(flat.sharpe_annual))
    print(f"buy-and-hold SR spread across 0-100 bps: {spread:.3e}")
    assert spread == 0.0, (
        f"a strategy with no turnover changed by {spread:.3e} across the cost sweep; "
        "cost is not being charged on traded notional"
    )
    assert math.isnan(flat.break_even_bps()) or flat.break_even_bps() == 0.0


@pytest.mark.parametrize("strategy", [MACrossover(5, 15), CausalZScore(20)], ids=lambda s: s.name)
def test_g5_monotone_across_the_zoo(strategy: Strategy, trending_bars: Bars) -> None:
    """Monotonicity is a property of the accounting, not of one strategy."""
    result = sweep_costs(trending_bars, strategy, GRID, CAPITAL, "next_open", ledger=LEDGER)
    assert result.is_monotone(), f"{strategy.name} broke monotonicity"


def test_g5_higher_turnover_dies_sooner(trending_bars: Bars) -> None:
    """A faster strategy must have a lower break-even cost. This is the diagnostic
    the number exists for: it makes the cost of turnover legible instead of
    implicit."""
    fast = sweep_costs(trending_bars, CausalZScore(5), GRID, CAPITAL, "next_open", ledger=LEDGER)
    slow = sweep_costs(trending_bars, CausalZScore(60), GRID, CAPITAL, "next_open", ledger=LEDGER)
    fast_c, slow_c = fast.break_even_bps(), slow.break_even_bps()
    print(
        f"turnover {fast.turnover_annual:6.2f}/yr -> c* = {fast_c:6.2f} bps\n"
        f"turnover {slow.turnover_annual:6.2f}/yr -> c* = {slow_c:6.2f} bps"
    )
    assert fast.turnover_annual > slow.turnover_annual, "the fast rule should trade more"
    assert fast_c < slow_c, (
        f"the faster rule survives to {fast_c:.2f} bps while the slower one dies at "
        f"{slow_c:.2f}; turnover is not being paid for"
    )
    # The fast rule reports c* = 0 because it is already unprofitable at zero cost,
    # which is a real answer rather than a missing one: no execution price saves it.
    assert fast_c == 0.0 or fast.sharpe_annual[0] > 0.0


def test_g5_sweep_rejects_a_malformed_grid(trending_bars: Bars) -> None:
    for grid, match in (
        ([5.0], "at least 2"),
        ([10.0, 5.0], "strictly increasing"),
        ([5.0, 5.0], "strictly increasing"),
    ):
        with pytest.raises(ValueError, match=match):
            sweep_costs(trending_bars, BuyAndHold(), grid, CAPITAL, "next_open", ledger=LEDGER)


def test_g5_break_even_is_nan_when_the_grid_never_crosses(trending_bars: Bars) -> None:
    """Honest "not determined here" rather than a number invented by
    extrapolation (F7 for the break-even calculation itself)."""
    narrow = sweep_costs(
        trending_bars,
        CausalZScore(20),
        np.linspace(0.0, 0.5, 6),
        CAPITAL,
        "next_open",
        ledger=LEDGER,
    )
    print(f"c* over a 0-0.5 bps grid: {narrow.break_even_bps()}")
    assert math.isnan(narrow.break_even_bps()), (
        "a grid that never reaches the crossing must report NaN, not extrapolate"
    )
