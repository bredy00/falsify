"""Cost sweeps and the break-even point. Supports G5.

Break-even cost is the single most useful number in the whole report. A strategy
that dies at 3 bps is a plot; one that survives 40 bps is a business. It is also
the number that cannot be talked around: it converts "this works" into "this works
if you can trade for under c* basis points a turn", which is a claim a reader can
check against their own execution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from falsify.core.conventions import DEFAULT_CONVENTION, Convention
from falsify.core.types import BARS_PER_YEAR, Bars
from falsify.core.vectorized import run_vectorized
from falsify.costs import CostModel
from falsify.ledger import Ledger
from falsify.metrics import annualise_sharpe, sharpe
from falsify.strategies.base import Strategy


@dataclass(frozen=True, slots=True)
class CostSweep:
    """Annualised net Sharpe as a function of round-trip cost. Frozen (B7)."""

    bps: NDArray[np.float64]
    sharpe_annual: NDArray[np.float64]
    turnover_annual: float

    def is_monotone(self, tolerance: float = 0.0) -> bool:
        """True when net Sharpe never rises as cost rises.

        G5's condition. `tolerance` allows a floating-point epsilon, not a
        judgement call -- a genuine increase means costs are being credited
        somewhere, which is a bug and not a rounding artefact.
        """
        return bool(np.all(np.diff(self.sharpe_annual) <= tolerance))

    def break_even_bps(self) -> float:
        """Cost at which net Sharpe crosses zero, by linear interpolation.

        Returns NaN when the curve never crosses inside the swept range: an
        honest "not determined here" rather than a number invented by
        extrapolation.
        """
        sr = self.sharpe_annual
        if sr[0] <= 0.0:
            return 0.0
        crossings = np.flatnonzero(sr <= 0.0)
        if crossings.size == 0:
            return float("nan")
        i = int(crossings[0])
        lo_bps, hi_bps = float(self.bps[i - 1]), float(self.bps[i])
        lo_sr, hi_sr = float(sr[i - 1]), float(sr[i])
        if lo_sr == hi_sr:
            return hi_bps
        return lo_bps + (hi_bps - lo_bps) * lo_sr / (lo_sr - hi_sr)


def sweep_costs(
    bars: Bars,
    strategy: Strategy,
    bps_grid: Sequence[float] | NDArray[np.float64],
    initial_capital: float = 10_000.0,
    convention: Convention = DEFAULT_CONVENTION,
    base: CostModel | None = None,
    *,
    ledger: Ledger,
) -> CostSweep:
    """Run the strategy at each cost level and record annualised net Sharpe.

    `base` supplies the non-transaction terms (cash yield, borrow) so they stay
    fixed while the traded-notional cost varies -- otherwise the sweep would
    confound two different economics and the monotonicity claim would be about
    nothing in particular.
    """
    template = base or CostModel()
    grid = np.asarray(bps_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError(f"bps_grid must be a 1-D grid of at least 2 points, got {grid.shape}")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("bps_grid must be strictly increasing")

    sharpes = np.empty(grid.size)
    turnover_annual = float("nan")
    for i, bps in enumerate(grid):
        costs = CostModel(
            commission_bps=float(bps),
            borrow_bps_annual=template.borrow_bps_annual,
            cash_yield_annual=template.cash_yield_annual,
        )
        result = run_vectorized(bars, strategy, costs, initial_capital, convention, ledger=ledger)
        sharpes[i] = annualise_sharpe(sharpe(result.net_ret[1:]))
        if i == 0:
            turnover_annual = float(np.sum(result.turnover) / len(result) * BARS_PER_YEAR)

    return CostSweep(bps=grid, sharpe_annual=sharpes, turnover_annual=turnover_annual)
