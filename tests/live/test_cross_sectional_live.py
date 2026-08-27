"""Cross-sectional long/short on the real sector panel. PLAYBOOK Phase 7.

Marked `live`; CI deselects it. Requires the cache:

    uv run --group data python scripts/fetch_data.py

Nine SPDR sector funds, 2015-2024, dollar-neutral tertile long/short, zero cost:

    construction                    SR       ±SE     turns/yr   NW t
    XS momentum 12m, monthly     -0.065     0.334       4.79    -0.20
    XS momentum 12m, daily       -0.023     0.334      23.03    -0.07
    XS momentum  6m, monthly     +0.157     0.325       6.58    +0.51
    XS momentum  1m, monthly     -0.432     0.318      16.31    -1.37

**There is no edge here, and that is the result.** Not one construction produces a HAC
t-statistic even close to 2, and the best of the four is +0.157 against a standard error
of 0.325 -- an interval that contains zero comfortably in both directions. A project that
reports this plainly is doing what it was built to do; the failure mode would be running
a fifth, sixth and seventh construction until one of them cleared 2, which is precisely
the search the trials ledger exists to count.

The 1-month row is the one with a story. Short-horizon cross-sectional momentum is
negative here, at the largest absolute t in the set, which is the well-known short-term
reversal effect showing up unbidden -- last month's winners underperform next month.
It is still not significant, and it is reported as an observation rather than as a
strategy.

Note the difference from the time-series result. `TimeSeriesMomentum(12m,1m)` earned
+0.606 on SPY, but a dollar-neutral book of the same signal across sectors earns nothing.
That gap is the market: the time-series version is long a rising index most of the time
and collects its drift, while the cross-sectional version is constrained to zero net
exposure and cannot. Which is exactly why the long/short spread is the cleaner place to
look for skill, and why it so often has less to show.
"""

from __future__ import annotations

import numpy as np
import pytest

from falsify.costs import ZERO_COST, CostModel
from falsify.cross_sectional import cross_sectional_weights, run_panel
from falsify.data.panel import Panel, load_panel
from falsify.metrics import annualise_sharpe, newey_west_t, sharpe, sharpe_se

pytestmark = pytest.mark.live

CAPITAL = 10_000.0
YEAR, HALF, MONTH = 252, 126, 21


@pytest.fixture(scope="module")
def sectors() -> Panel:
    from falsify.data.loaders import DataUnavailable
    from falsify.data.manifest import ManifestMismatch

    try:
        return load_panel()
    except (DataUnavailable, ManifestMismatch) as exc:
        pytest.skip(
            f"sector cache unavailable ({exc.__class__.__name__}); run scripts/fetch_data.py"
        )


def spread(panel: Panel, lookback: int, hold: int, bps: float = 0.0) -> tuple[float, float, float]:
    """Annualised Sharpe, its SE, and the HAC t-statistic (B2, and 01 B1)."""
    costs = ZERO_COST if bps == 0.0 else CostModel(commission_bps=bps)
    result = run_panel(panel, cross_sectional_weights(panel, lookback, hold=hold), costs, CAPITAL)
    returns = result.net_ret[1:]
    return (
        annualise_sharpe(sharpe(returns)),
        sharpe_se(returns, 252),
        newey_west_t(returns),
    )


def test_the_panel_is_what_the_manifest_says(sectors: Panel) -> None:
    """The load path verified each sha256; this pins the shape the rest of the file
    reasons about, and that the calendars aligned without silently dropping a third of
    the sample."""
    print(sectors.describe())
    assert sectors.n_assets == 9, "nine sector funds"
    assert len(sectors) > 2_400, "a decade of daily bars is about 2,516"
    assert np.all(sectors.close > 0.0)
    assert sectors.adjustment == "total_return"


def test_no_cross_sectional_construction_is_significant(sectors: Panel) -> None:
    """Four constructions, best HAC t = +0.51. That is the Phase 7 result.

    Asserted as a bound rather than a target: if some construction here started clearing
    2, the first question would be what changed in the ranking, not whether an edge had
    appeared.
    """
    rows = [
        ("12m monthly", YEAR, MONTH),
        ("12m daily", YEAR, 1),
        ("6m monthly", HALF, MONTH),
        ("1m monthly", MONTH, MONTH),
    ]
    stats = []
    for label, lookback, hold in rows:
        sr, se, t_stat = spread(sectors, lookback, hold)
        stats.append((label, sr, se, t_stat))
        print(f"  {label:<14} SR {sr:+.3f} +/- {se:.3f}   HAC t {t_stat:+.2f}")

    best = max(abs(t) for _, _, _, t in stats)
    assert best < 2.0, (
        f"a cross-sectional construction reached HAC t = {best:.2f}. Before believing it, "
        "count the constructions tried and deflate by that N -- four were tried here."
    )


def test_the_book_stays_dollar_neutral_on_real_prices(sectors: Panel) -> None:
    """Synthetic data cannot break neutrality; real calendars and corporate actions might.
    Same assertion, run where it could actually fail."""
    weights = cross_sectional_weights(sectors, YEAR, hold=MONTH)
    result = run_panel(sectors, weights, ZERO_COST, CAPITAL)
    worst = float(np.max(np.abs(result.net_exposure)))
    print(f"max |net exposure| on real prices: {worst:.3e}")
    assert result.is_dollar_neutral(), f"net exposure reached {worst:.3e}"
    assert np.allclose(result.gross_exposure, 1.0)


def test_short_horizon_ranking_is_the_worst_of_the_four(sectors: Panel) -> None:
    """Short-term reversal, showing up unbidden: last month's winning sectors underperform
    over the next month, so ranking on a one-month lookback is the worst construction of
    the four. Measured -0.432, the largest absolute t in the set at -1.37.

    Still not significant, and recorded as an observation rather than promoted into a
    reversal strategy -- flipping the sign because a backtest came out negative is how a
    search becomes a story.
    """
    short_sr, _, short_t = spread(sectors, MONTH, MONTH)
    long_sr, _, _ = spread(sectors, YEAR, MONTH)
    print(f"1m {short_sr:+.3f} (t {short_t:+.2f}) vs 12m {long_sr:+.3f}")
    assert short_sr < long_sr, "the one-month ranking should be the weaker of the two"
    assert abs(short_t) < 2.0, "and it is still not significant in either direction"


def test_costs_bite_harder_on_the_daily_rebalance(sectors: Panel) -> None:
    """23 turns a year against 4.8. The holding period is worth more here than in the
    single-asset case, because a nine-name book rebalances more names per decision."""
    daily = run_panel(sectors, cross_sectional_weights(sectors, YEAR, hold=1), ZERO_COST, CAPITAL)
    monthly = run_panel(
        sectors, cross_sectional_weights(sectors, YEAR, hold=MONTH), ZERO_COST, CAPITAL
    )
    daily_turnover = float(np.sum(daily.turnover) / len(daily) * 252)
    monthly_turnover = float(np.sum(monthly.turnover) / len(monthly) * 252)

    daily_net = spread(sectors, YEAR, 1, bps=10.0)[0]
    daily_gross = spread(sectors, YEAR, 1)[0]
    monthly_net = spread(sectors, YEAR, MONTH, bps=10.0)[0]
    monthly_gross = spread(sectors, YEAR, MONTH)[0]

    print(f"daily {daily_turnover:.2f}/yr, 10bps costs {daily_gross - daily_net:+.3f} Sharpe")
    print(f"monthly {monthly_turnover:.2f}/yr, 10bps costs {monthly_gross - monthly_net:+.3f}")
    assert daily_turnover > 4 * monthly_turnover
    assert (daily_gross - daily_net) > (monthly_gross - monthly_net)
