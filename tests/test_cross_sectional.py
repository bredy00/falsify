"""Cross-sectional long/short. PLAYBOOK Phase 7.

The load-bearing test in this file is the first one. `run_panel` is a second
implementation of Part E's accounting equations, and a second implementation of a
specification is exactly what G2 exists to police -- so it is held to the same standard:
at N = 1 it must reproduce `run_vectorized` bitwise, not approximately.

That check earned its place immediately. The first version computed the warm-up as
`max(first_nonzero, lag)` instead of `first_nonzero + lag`, reported one bar more than
the single-asset engine, and diverged by 1,313 in equity on a 10,000 account. Off by one
bar, wrong by a thirteenth of the account, and nothing else in the file would have
noticed.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from falsify.core.types import Bars
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST, CostModel
from falsify.cross_sectional import (
    NEUTRAL_TOL,
    DegenerateCrossSection,
    cross_sectional_weights,
    rank_weights,
    run_panel,
)
from falsify.data.panel import Panel
from falsify.ledger import Ledger
from falsify.strategies.base import Strategy
from falsify.synthetic import gbm

LEDGER = Ledger.memory()
CAPITAL = 10_000.0


def synthetic_panel(n_assets: int = 9, n_bars: int = 1_500, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    columns = [gbm(mu=0.02 + 0.02 * i, sigma=0.18, n_bars=n_bars, rng=rng) for i in range(n_assets)]
    ts = np.arange("2015-01-01", n_bars, dtype="datetime64[D]").astype("datetime64[ns]")
    return Panel(
        ts=ts,
        close=np.column_stack(columns),
        tickers=tuple(f"A{i}" for i in range(n_assets)),
        adjustment="total_return",
    )


class AlwaysLong(Strategy):
    """Weight 1.0 from the first bar after the lookback. The simplest input on which the
    two engines must agree."""

    lookback = 1

    def signals(self, bars: Bars) -> npt.NDArray[np.float64]:
        out = np.full(len(bars), np.nan)
        out[1:] = 1.0
        return out


# --------------------------------------------------------------------------------------
# The certification. Same discipline as G2.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bps", [0.0, 5.0, 25.0])
def test_the_panel_engine_reproduces_the_single_asset_engine_at_one_asset(bps: float) -> None:
    """Bitwise, not approximately. `run_panel` is Part E written a second time, and two
    implementations of one specification are held together by a test or they drift."""
    panel = synthetic_panel(n_assets=1)
    flat = panel.close.ravel()
    bars = Bars(
        ts=panel.ts,
        open=flat,
        high=flat,
        low=flat,
        close=flat,
        volume=np.full(len(panel), 1e6),
        adjustment="total_return",
    )
    costs = ZERO_COST if bps == 0.0 else CostModel(commission_bps=bps)

    reference = run_vectorized(bars, AlwaysLong(), costs, CAPITAL, "next_open", ledger=LEDGER)
    weights = np.zeros((len(panel), 1))
    weights[1:] = 1.0
    panel_run = run_panel(panel, weights, costs, CAPITAL, lag=2)

    assert len(panel_run) == len(reference), (
        f"the panel engine reported {len(panel_run)} bars against the single-asset "
        f"engine's {len(reference)}. The warm-up arithmetic differs."
    )
    for name in ("equity", "gross_ret", "net_ret", "costs", "turnover"):
        a = getattr(reference, name)
        b = getattr(panel_run, name)
        assert np.array_equal(a, b), f"{name} differs; max |diff| {np.max(np.abs(a - b)):.3e}"


# --------------------------------------------------------------------------------------
# What "cross-sectional" means, structurally.
# --------------------------------------------------------------------------------------


def test_the_book_is_dollar_neutral_on_every_bar() -> None:
    """Weights sum to zero, so the strategy cannot collect the market's drift by
    accident. That constraint is the whole reason the long/short spread is a cleaner
    place to look for skill than a long-only signal."""
    panel = synthetic_panel()
    weights = cross_sectional_weights(panel, lookback=252, fraction=1 / 3, hold=21)
    result = run_panel(panel, weights, ZERO_COST, CAPITAL)
    worst = float(np.max(np.abs(result.net_exposure)))
    print(f"max |sum of weights| = {worst:.3e}")
    assert result.is_dollar_neutral(), f"net exposure reached {worst:.3e}"
    assert worst < NEUTRAL_TOL


def test_gross_exposure_is_one_so_the_sharpe_is_comparable() -> None:
    """Each leg carries 0.5, so |w| sums to 1 and the book is as invested as a
    fully-invested long-only position. Without that the Sharpes could not be read next to
    each other -- a book at 30% gross would look less volatile for reasons that have
    nothing to do with the signal."""
    panel = synthetic_panel()
    weights = cross_sectional_weights(panel, lookback=126, fraction=1 / 3)
    result = run_panel(panel, weights, ZERO_COST, CAPITAL)
    assert np.allclose(result.gross_exposure, 1.0), (
        f"gross exposure ranged {result.gross_exposure.min():.3f}..{result.gross_exposure.max():.3f}"
    )


def test_the_legs_hold_the_number_of_names_the_fraction_asks_for() -> None:
    panel = synthetic_panel(n_assets=9)
    for fraction, expected in ((1 / 3, 3), (2 / 9, 2), (1 / 9, 1)):
        weights = cross_sectional_weights(panel, lookback=126, fraction=fraction)
        active = weights[300]
        assert int(np.sum(active > 0)) == expected
        assert int(np.sum(active < 0)) == expected


# --------------------------------------------------------------------------------------
# The ranking itself.
# --------------------------------------------------------------------------------------


def test_the_best_scores_are_long_and_the_worst_are_short() -> None:
    scores = np.array([0.5, -0.2, 0.9, 0.1, -0.7, 0.3])
    weights = rank_weights(scores, fraction=1 / 3)
    assert weights[2] > 0 and weights[0] > 0, "the two best scores must be long"
    assert weights[4] < 0 and weights[1] < 0, "the two worst must be short"
    assert abs(float(np.sum(weights))) < NEUTRAL_TOL
    assert np.isclose(float(np.sum(np.abs(weights))), 1.0)


def test_assets_with_no_signal_are_excluded_rather_than_ranked_last() -> None:
    """A NaN score is an asset with no history yet, not the worst asset. Sorting NaN to
    an end would put a systematic short on whichever name has the shortest history --
    a position taken for a data reason and reported as a signal."""
    scores = np.array([0.5, np.nan, 0.9, np.nan, -0.7, 0.3])
    weights = rank_weights(scores, fraction=1 / 3)
    assert weights[1] == 0.0 and weights[3] == 0.0, "NaN-scored assets must carry no weight"
    assert abs(float(np.sum(weights))) < NEUTRAL_TOL


def test_a_cross_section_too_thin_to_rank_goes_flat() -> None:
    """One live asset cannot be ranked against anything. Flat is the honest answer; a
    long position would be a time-series bet wearing a cross-sectional label."""
    assert np.array_equal(rank_weights(np.array([0.5, np.nan, np.nan])), np.zeros(3))
    assert np.array_equal(rank_weights(np.array([np.nan])), np.zeros(1))


def test_ranking_is_deterministic_under_ties() -> None:
    """B9. Equal scores must resolve the same way every time, or G10 cannot hold."""
    scores = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.9])
    first = rank_weights(scores, fraction=1 / 3)
    for _ in range(5):
        assert np.array_equal(rank_weights(scores, fraction=1 / 3), first)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 0.6, 1.0])
def test_a_nonsensical_fraction_is_refused(fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        rank_weights(np.array([1.0, 2.0, 3.0]), fraction=fraction)


# --------------------------------------------------------------------------------------
# Holding period and cost.
# --------------------------------------------------------------------------------------


def test_holding_the_weights_cuts_turnover() -> None:
    """The cost consequence of rebalancing monthly rather than daily, measured on the
    same panel and the same signal."""
    panel = synthetic_panel()
    daily = run_panel(panel, cross_sectional_weights(panel, 252, hold=1), ZERO_COST, CAPITAL)
    monthly = run_panel(panel, cross_sectional_weights(panel, 252, hold=21), ZERO_COST, CAPITAL)

    daily_turnover = float(np.sum(daily.turnover) / len(daily) * 252)
    monthly_turnover = float(np.sum(monthly.turnover) / len(monthly) * 252)
    print(f"daily {daily_turnover:.2f}/yr vs monthly {monthly_turnover:.2f}/yr")
    assert monthly_turnover < daily_turnover


def test_costs_reduce_the_return_and_are_charged_only_when_the_book_moves() -> None:
    panel = synthetic_panel()
    weights = cross_sectional_weights(panel, 252, hold=21)
    free = run_panel(panel, weights, ZERO_COST, CAPITAL)
    charged = run_panel(panel, weights, CostModel(commission_bps=10.0), CAPITAL)

    assert float(np.sum(charged.costs)) > 0.0
    assert charged.equity[-1] < free.equity[-1]
    traded = np.flatnonzero(charged.turnover > 0.0)
    paid = np.flatnonzero(charged.costs > 0.0)
    assert np.array_equal(paid, traded[traded > 0]), "cost was charged on a bar with no trade"


# --------------------------------------------------------------------------------------
# Refusals.
# --------------------------------------------------------------------------------------


def test_a_panel_too_short_for_the_lookback_is_refused() -> None:
    panel = synthetic_panel(n_bars=100)
    with pytest.raises(DegenerateCrossSection, match="lookback"):
        cross_sectional_weights(panel, lookback=252)


def test_weights_that_do_not_match_the_panel_are_refused() -> None:
    panel = synthetic_panel()
    with pytest.raises(ValueError, match="do not match the panel"):
        run_panel(panel, np.zeros((len(panel), 3)), ZERO_COST, CAPITAL)


def test_an_all_zero_book_is_refused_rather_than_reported_as_flat() -> None:
    """Gate 0.4's rule. A book that never took a position and a book that lost everything
    are different outcomes, and only one of them is an empty result."""
    panel = synthetic_panel()
    with pytest.raises(DegenerateCrossSection, match="every weight is zero"):
        run_panel(panel, np.zeros_like(panel.close), ZERO_COST, CAPITAL)
