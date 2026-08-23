"""The integration layer: strategy grid -> engine -> walk-forward -> selection.

Each gate so far certified one component. This is where they are joined into the
thing you would actually run: take a grid of configurations, push every one through
the certified engine, split the result with a purged walk-forward, choose among them
on each training block with a `SelectionRule`, and score only out of sample.

It is written now, at G8, rather than at G9 for a specific reason. CSCV's rank
bookkeeping is the fiddliest code in the project (`03` Part C), and it needs exactly
this: a `(T, N)` matrix of configuration returns, a block splitter, and a rule that
turns in-sample evidence into weights. Building those three here means G9 assembles
existing certified parts instead of inventing them alongside its own bookkeeping.

The honest thing this measures is not "does the strategy work" but "does the
procedure that picks the strategy work" -- which is the distinction `01` Part B6
insists on and the one people conflate.
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
from falsify.selection import SelectionRule
from falsify.strategies.base import Strategy
from falsify.walkforward import Split, WalkForwardSplitter

Returns = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StrategyGrid:
    """Net returns of every configuration on one aligned window. Frozen (B7).

    `returns` is `(T, N)`. Every column is the same length and covers the same bars,
    which is what makes cross-sectional selection meaningful -- configurations with
    different lookbacks have different warm-ups, so the grid is trimmed to the
    longest of them rather than padded. Padding would let a slow-warming
    configuration be judged on a period it was not actually trading.
    """

    returns: Returns
    names: tuple[str, ...]
    first_bar: int

    def __post_init__(self) -> None:
        if self.returns.ndim != 2:
            raise ValueError(f"expected a (T, N) matrix, got shape {self.returns.shape}")
        if self.returns.shape[1] != len(self.names):
            raise ValueError(f"{self.returns.shape[1]} columns but {len(self.names)} names")
        if not np.all(np.isfinite(self.returns)):
            raise ValueError("the grid contains non-finite returns; trim the warm-up first")

    @property
    def n_obs(self) -> int:
        return int(self.returns.shape[0])

    @property
    def n_configs(self) -> int:
        return int(self.returns.shape[1])


def build_grid(
    bars: Bars,
    strategies: Sequence[Strategy],
    costs: CostModel,
    initial_capital: float = 10_000.0,
    convention: Convention = DEFAULT_CONVENTION,
    *,
    ledger: Ledger,
) -> StrategyGrid:
    """Run every configuration through the certified engine and align the results.

    Alignment is the part worth care. Each strategy's reported window starts after its
    own warm-up, so the runs come back at different lengths and covering different
    bars. They are trimmed from the *end* -- keeping the most recent `T` bars common to
    all -- so every column describes the same calendar period. Aligning by position
    instead would silently compare a fast configuration's early bars against a slow
    one's later bars.
    """
    if not strategies:
        raise ValueError("a grid needs at least one configuration")

    runs = [
        run_vectorized(bars, s, costs, initial_capital, convention, ledger=ledger)
        for s in strategies
    ]
    # net_ret[0] is the anchor bar and earns nothing, so it is dropped everywhere.
    series = [r.net_ret[1:] for r in runs]
    common = min(len(s) for s in series)
    if common < 2:
        raise ValueError("configurations share fewer than 2 bars after warm-up alignment")

    matrix = np.column_stack([s[-common:] for s in series])
    return StrategyGrid(
        returns=np.asarray(matrix, dtype=np.float64),
        names=tuple(s.name for s in strategies),
        first_bar=len(bars) - common,
    )


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """What a selection rule achieved out of sample, fold by fold. Frozen (B7)."""

    rule: str
    splitter: str
    weights: Returns  # (n_splits, N) -- what the rule chose on each training block
    is_sharpe: Returns  # (n_splits,) annualised, the rule's portfolio in sample
    oos_sharpe: Returns  # (n_splits,) annualised, the same portfolio out of sample
    oos_returns: Returns  # stitched out-of-sample return series
    config_is_sharpe: Returns  # (n_splits, N) every configuration, in sample
    config_oos_sharpe: Returns  # (n_splits, N) every configuration, out of sample

    @property
    def n_splits(self) -> int:
        return int(self.is_sharpe.size)

    def stitched_sharpe(self) -> float:
        """Annualised Sharpe of the concatenated out-of-sample series.

        The number to report. Averaging per-fold Sharpes instead would weight a
        three-bar fold like a three-hundred-bar one and quietly flatter short folds.
        """
        return annualise_sharpe(sharpe(self.oos_returns))

    def degradation(self) -> float:
        """Mean in-sample Sharpe minus mean out-of-sample Sharpe.

        The price of selection, in Sharpe units. Positive is the ordinary case and
        the whole reason walk-forward exists; large and positive is overfitting
        measured rather than suspected.
        """
        return float(np.mean(self.is_sharpe) - np.mean(self.oos_sharpe))


def _block_sharpe(block: Returns, bars_per_year: int) -> Returns:
    """Annualised Sharpe of every column of a block."""
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_bar = np.where(sd > 0.0, mu / sd, np.nan)
    return np.asarray(per_bar * np.sqrt(bars_per_year), dtype=np.float64)


def walk_forward_select(
    grid: StrategyGrid,
    splitter: WalkForwardSplitter,
    rule: SelectionRule,
    bars_per_year: int = BARS_PER_YEAR,
) -> WalkForwardResult:
    """Choose on each training block, score on the block that follows.

    The rule sees only `grid.returns[split.train]`, which is enforced by the splitter
    rather than by convention -- `Split` cannot be constructed with overlapping
    indices, so a rule physically cannot be handed a bar it will later be scored on.

    Returns per-configuration in-sample and out-of-sample Sharpes alongside the
    rule's own portfolio, because the interesting quantity is usually not the level
    but the relationship: if the two are anticorrelated across the grid, the
    compensation effect from Gate 0.0 has been reproduced on engine output rather
    than on a toy.
    """
    splits: list[Split] = splitter.split(grid.n_obs)
    n_configs = grid.n_configs

    weights = np.empty((len(splits), n_configs))
    is_sharpe = np.empty(len(splits))
    oos_sharpe = np.empty(len(splits))
    config_is = np.empty((len(splits), n_configs))
    config_oos = np.empty((len(splits), n_configs))
    stitched: list[Returns] = []

    for k, split in enumerate(splits):
        train_block = grid.returns[split.train]
        test_block = grid.returns[split.test]

        w = rule.weights(train_block)
        weights[k] = w
        config_is[k] = _block_sharpe(train_block, bars_per_year)
        config_oos[k] = _block_sharpe(test_block, bars_per_year)

        is_sharpe[k] = annualise_sharpe(sharpe(train_block @ w), bars_per_year)
        oos_portfolio = test_block @ w
        oos_sharpe[k] = annualise_sharpe(sharpe(oos_portfolio), bars_per_year)
        stitched.append(oos_portfolio)

    return WalkForwardResult(
        rule=rule.name,
        splitter=repr(splitter),
        weights=weights,
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        oos_returns=np.concatenate(stitched),
        config_is_sharpe=config_is,
        config_oos_sharpe=config_oos,
    )


def selection_degradation_slope(result: WalkForwardResult) -> tuple[float, float]:
    """OLS slope of out-of-sample on in-sample Sharpe, pooled across folds and configs.

    Gate 0.0 showed this slope is zero on a memoryless process and negative under a
    compensation effect. Measuring it here, on real engine output rather than on
    synthetic Sharpes, is what connects the theory at the top of the project to the
    machinery at the bottom.

    Returns `(slope, standard_error)`. A slope near zero says in-sample ranking
    carries no out-of-sample information; a negative slope says it carries
    information with the wrong sign, which is worse than useless and is exactly the
    finding the source paper is about.
    """
    x = result.config_is_sharpe.ravel()
    y = result.config_oos_sharpe.ravel()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 3 or np.std(x) == 0.0:
        return (float("nan"), float("nan"))

    from scipy.stats import linregress

    fit = linregress(x, y)
    return (float(fit.slope), float(fit.stderr))


__all__ = [
    "StrategyGrid",
    "WalkForwardResult",
    "build_grid",
    "selection_degradation_slope",
    "walk_forward_select",
]
