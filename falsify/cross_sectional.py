"""Cross-sectional long/short. PLAYBOOK Phase 7.

    "N-asset universe, rank on signal, long top decile / short bottom decile, weights
     sum to zero. Turnover control: only rebalance names crossing a buffer band."

This is where the project stops asking "does this asset trend?" and starts asking "does
this asset trend *more than its peers?*" -- a different question with a different null.
A time-series signal can be long everything in a bull market; a cross-sectional one is
constrained to be flat in aggregate, so it cannot collect the market's drift by accident.
That constraint is the whole point, and it is why the long/short spread is the honest
place to look for skill.

**Deciles become tertiles here, and that is an adaptation not a shortcut.** PLAYBOOK
says top and bottom decile. A decile of nine sector funds is 0.9 assets. Ranking into
tertiles -- long the top three, short the bottom three -- is the same construction at the
breadth the universe actually has, and `fraction` is a parameter so the decile is
recoverable the moment the universe is wide enough to support one.

**The accounting is Part E, extended.** `run_panel` is the N-asset statement of exactly
the equations `core/vectorized.py` implements for one asset, and `test_cross_sectional.py`
asserts they agree bitwise when N = 1. That check is the same discipline as G2: two
implementations of one specification, held together by a test rather than by intent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from falsify.core.types import BARS_PER_YEAR
from falsify.costs import CostModel
from falsify.data.panel import Panel

Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]

# Sum of weights must vanish to this tolerance. Tighter than SUM_TOL in `selection`
# because there is no exponentiation here -- these are counts divided by counts.
NEUTRAL_TOL = 1e-12


class DegenerateCrossSection(ValueError):
    """A bar on which no cross-sectional portfolio can be formed."""


def rank_weights(scores: Vector, fraction: float = 1.0 / 3.0) -> Vector:
    """Scores for one bar -> dollar-neutral weights.

    Long the top `fraction` of names, short the bottom `fraction`, equally weighted
    within each leg and scaled so each leg carries 0.5 of gross exposure. The weights
    sum to zero and their absolute values sum to one, so gross exposure is comparable
    with a fully-invested long-only position and the Sharpes can be read side by side.

    Ties are broken by `argsort`'s stable order, which is position in the universe. That
    is arbitrary but deterministic (B9), and it only matters when two assets have exactly
    equal trailing returns -- which on real prices essentially never happens and on
    constructed test data happens on purpose.

    NaN scores are excluded from the ranking rather than sorted to an end. An asset with
    no signal yet is not the worst asset; treating it as one would put a systematic short
    on whichever name has the shortest history.
    """
    if scores.ndim != 1:
        raise ValueError(f"expected one score per asset, got shape {scores.shape}")
    if not 0.0 < fraction <= 0.5:
        raise ValueError(f"fraction must lie in (0, 0.5], got {fraction}")

    weights = np.zeros(scores.size)
    live = np.flatnonzero(np.isfinite(scores))
    if live.size < 2:
        return weights  # nothing to rank against; flat is the honest answer

    k = max(1, int(np.floor(live.size * fraction)))
    if 2 * k > live.size:
        k = live.size // 2
    if k < 1:
        return weights

    order = live[np.argsort(scores[live], kind="stable")]
    weights[order[-k:]] = 0.5 / k
    weights[order[:k]] = -0.5 / k
    return weights


def cross_sectional_weights(
    panel: Panel, lookback: int, fraction: float = 1.0 / 3.0, hold: int = 1
) -> Matrix:
    """`(T, N)` dollar-neutral weights from cross-sectional momentum.

    The score at bar `t` is the trailing `lookback`-bar return, computed from
    `close[0:t+1]` only, then lagged one bar exactly as the single-asset strategies do.
    The lag is the decision lag; the execution lag is the engine's and is applied in
    `run_panel`.

    `hold` samples the weights on a schedule and carries them between rebalances, the
    same sample-and-hold `TimeSeriesMomentum` uses. Carrying a *weight* is not the same
    as carrying a *score*: the positions drift with prices between rebalances in reality,
    and holding the target weight means the engine trades the drift back. That is the
    conservative choice -- it books turnover a real portfolio might avoid, so the cost
    estimate errs high rather than low.
    """
    close = panel.close
    n_bars, n_assets = close.shape
    if lookback < 1:
        raise ValueError(f"lookback must be at least 1, got {lookback}")
    if hold < 1:
        raise ValueError(f"hold must be at least 1, got {hold}")
    if n_bars <= lookback + 1:
        raise DegenerateCrossSection(
            f"{n_bars} bars cannot support a {lookback}-bar lookback plus the decision lag"
        )

    scores = np.full((n_bars, n_assets), np.nan)
    scores[lookback:] = close[lookback:] / close[:-lookback] - 1.0

    weights = np.zeros((n_bars, n_assets))
    first = lookback + 1  # +1 for the decision lag
    for t in range(first, n_bars):
        weights[t] = rank_weights(scores[t - 1], fraction)

    if hold > 1:
        positions = np.arange(n_bars)
        is_rebalance = (positions >= first) & ((positions - first) % hold == 0)
        source = np.maximum.accumulate(np.where(is_rebalance, positions, 0))
        weights[first:] = weights[source[first:]]
    return weights


@dataclass(frozen=True, slots=True)
class PanelResult:
    """One cross-sectional run over the reported window. Frozen (B7).

    Mirrors `core.types.Result` field for field, with `weights` widened to `(T, N)`.
    Deliberately not a subclass: the shapes differ, and a function that accepts either
    would have to branch on the shape, which is exactly the confusion this keeps out.
    """

    equity: Vector
    weights: Matrix
    gross_ret: Vector
    net_ret: Vector
    costs: Vector
    turnover: Vector
    tickers: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.equity.size)

    @property
    def gross_exposure(self) -> Vector:
        return np.sum(np.abs(self.weights), axis=1)

    @property
    def net_exposure(self) -> Vector:
        return np.sum(self.weights, axis=1)

    def is_dollar_neutral(self, tolerance: float = NEUTRAL_TOL) -> bool:
        return bool(np.all(np.abs(self.net_exposure) < tolerance))


def run_panel(
    panel: Panel,
    weights: Matrix,
    costs: CostModel,
    initial_capital: float,
    lag: int = 2,
    bars_per_year: int = BARS_PER_YEAR,
) -> PanelResult:
    """Part E's equations over N assets.

    Every line here is the `core/vectorized.py` line with a sum over assets added, and
    `test_cross_sectional.py` asserts the two agree bitwise at N = 1. The cash and borrow
    terms use gross and short exposure summed across the book, which is the only place
    the extension is more than a transcription: a dollar-neutral book is 100% invested in
    gross terms while holding no net position, so it earns no cash yield and pays borrow
    on the whole short leg.

    `lag` defaults to 2, matching `next_open`. Passing 1 gives `close_to_close`.
    """
    if initial_capital <= 0.0:
        raise ValueError(f"initial_capital must be positive, got {initial_capital}")
    if weights.shape != panel.close.shape:
        raise ValueError(f"weights {weights.shape} do not match the panel {panel.close.shape}")
    if lag < 1:
        raise ValueError(f"lag must be at least 1, got {lag}")

    n_bars = len(panel)

    # `warmup_start(lookback, lag) = lookback + lag` in the single-asset engine, where
    # `lookback` is the first bar carrying a usable weight. Here that bar is found rather
    # than declared, because the weights arrive as a matrix rather than from a Strategy
    # that announces its own lookback.
    #
    # The first version wrote `max(first_nonzero, lag)` and was wrong: it made the panel
    # engine report one bar more than the single-asset engine on identical input, so the
    # N = 1 agreement check failed by 1,313 in equity. Off by one bar, wrong by a
    # thirteenth of the account.
    live = np.flatnonzero(np.any(weights != 0.0, axis=1))
    if live.size == 0:
        raise DegenerateCrossSection("every weight is zero; there is nothing to account for")
    start = int(live[0]) + lag
    m = n_bars - start
    if m < 2:
        raise DegenerateCrossSection(f"{n_bars} bars leave {m} reported bars after the warm-up")

    asset_returns = np.zeros((m, panel.n_assets))
    asset_returns[1:] = panel.close[start + 1 : n_bars] / panel.close[start : n_bars - 1] - 1.0
    active = np.asarray(weights[start - lag : n_bars - lag], dtype=np.float64)

    cost_rate = costs.cost_rate()
    cash_per_bar = costs.cash_rate_per_bar(bars_per_year)
    borrow_per_bar = costs.borrow_rate_per_bar(bars_per_year)

    gross_exposure = np.sum(np.abs(active), axis=1)
    short_exposure = np.sum(np.maximum(-active, 0.0), axis=1)
    gross_ret = (
        np.sum(active * asset_returns, axis=1)
        + (1.0 - gross_exposure) * cash_per_bar
        - short_exposure * borrow_per_bar
    )
    gross_ret[0] = 0.0  # anchor bar earns nothing

    turnover = np.sum(np.abs(np.diff(active, axis=0, prepend=active[:1])), axis=1)

    equity = np.empty(m)
    cost_paid = np.zeros(m)
    net_ret = np.zeros(m)
    equity[0] = initial_capital
    for k in range(1, m):
        prev = equity[k - 1]
        charge = turnover[k] * prev * cost_rate
        equity[k] = prev * (1.0 + gross_ret[k]) - charge
        cost_paid[k] = charge
        net_ret[k] = equity[k] / prev - 1.0

    return PanelResult(
        equity=equity,
        weights=active,
        gross_ret=gross_ret,
        net_ret=net_ret,
        costs=cost_paid,
        turnover=turnover,
        tickers=panel.tickers,
    )


__all__ = [
    "NEUTRAL_TOL",
    "DegenerateCrossSection",
    "PanelResult",
    "cross_sectional_weights",
    "rank_weights",
    "run_panel",
]
