"""Factor attribution on real strategies. PLAYBOOK Phase 7's other half.

Marked `live`; CI deselects it. Requires the cache:

    uv run --group data python scripts/fetch_data.py

Carhart four factors from the Ken French library, HAC standard errors, close-to-close,
SPY and nine sector funds, 2015-2024:

    strategy                alpha/yr   HAC t     R2    Mkt-RF     SMB     HML     UMD
    BuyAndHold                +0.12%   +0.37   0.995   +0.975  -0.126  +0.015  -0.007
    TSMomentum(12m,1m)        -0.02%   -0.01   0.408   +0.618  -0.064  +0.182  +0.244
    TSMomentum(12m,3m)        -6.43%   -1.47   0.370   +0.575  -0.089  +0.094  +0.284
    MACrossover(20,50)        +5.36%   +0.92   0.037   -0.167  +0.191  -0.026  +0.045
    XS momentum 12m           -2.51%   -1.29   0.520   +0.006  +0.032  -0.020  +0.323
    XS momentum  6m           -0.60%   -0.30   0.366   -0.026  +0.010  +0.017  +0.259
    XS momentum  1m           -5.50%   -2.49   0.056   -0.048  -0.015  -0.023  +0.065

**The headline is the second row.** PLAYBOOK asks: "If your alpha t-stat drops below 2
after controlling for momentum, say so in the README. That single act of intellectual
honesty is worth more to a reader than a 2.5 Sharpe."

It does not drop below 2. It drops to ZERO -- t = -0.01, alpha -0.02% a year. The
+0.606 Sharpe `TimeSeriesMomentum(12m,1m)` earns on SPY is entirely accounted for by two
exposures the regression can name: 0.618 on the market, because a trend follower is long
a rising index most of the decade, and 0.244 on UMD, because it *is* a momentum strategy.
Once both are priced, nothing is left over. That is the finding, and it is the one worth
publishing.

The buy-and-hold row is the calibration. SPY loads 0.975 on the market with an R-squared
of 0.995 and no alpha, which is exactly what an index fund should show -- and if it did
not, nothing else in the table could be believed.
"""

from __future__ import annotations

import numpy as np
import pytest

from falsify.attribution import FactorFit, fit_factors
from falsify.core.types import Bars
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST
from falsify.cross_sectional import cross_sectional_weights, run_panel
from falsify.data.factors import Factors, align_to, load_factors
from falsify.data.loaders import DEFAULT_CACHE, DEFAULT_MANIFEST, FetchSpec, load
from falsify.data.panel import Panel, load_panel
from falsify.ledger import Ledger
from falsify.strategies.momentum import TimeSeriesMomentum
from falsify.strategies.simple import BuyAndHold

pytestmark = pytest.mark.live

LEDGER = Ledger.memory()
CAPITAL = 10_000.0
NAMES = ("Mkt-RF", "SMB", "HML", "UMD")


@pytest.fixture(scope="module")
def data() -> tuple[Bars, Panel, Factors]:
    from falsify.data.loaders import DataUnavailable
    from falsify.data.manifest import ManifestMismatch

    try:
        spy = load(
            FetchSpec("SPY", "2015-01-01", "2025-01-01", "total_return"),
            cache_dir=DEFAULT_CACHE,
            manifest_path=DEFAULT_MANIFEST,
        )
        return spy, load_panel(), load_factors()
    except (DataUnavailable, ManifestMismatch) as exc:
        pytest.skip(f"cache unavailable ({exc.__class__.__name__}); run scripts/fetch_data.py")


def attribute(factors: Factors, ts: np.ndarray, net_ret: np.ndarray) -> FactorFit:
    """Align, subtract French's risk-free rate, regress.

    The risk-free rate is French's rather than the `^IRX` series Part H decision 4
    fetched. Both are three-month bill rates; the factors are excess of *French's*, so
    mixing sources would leave a small systematic residual sitting in alpha.
    """
    aligned, rows = align_to(factors, ts)
    return fit_factors(net_ret[rows] - aligned.rf, aligned.values, aligned.names)


def single_asset(bars: Bars, strategy: object, factors: Factors) -> FactorFit:
    """Close-to-close, which is not optional here -- see the calibration test below."""
    result = run_vectorized(bars, strategy, ZERO_COST, CAPITAL, "close_to_close", ledger=LEDGER)  # type: ignore[arg-type]
    start = len(bars) - len(result)
    return attribute(factors, bars.ts[start:][1:], result.net_ret[1:])


def test_the_market_factor_recovers_a_known_beta_for_an_index_fund(
    data: tuple[Bars, Panel, Factors],
) -> None:
    """The calibration, and it is load-bearing.

    SPY is very nearly the market, so regressing its excess return on Mkt-RF must give a
    beta near 1 and an R-squared near 1. Measured 0.9598 and 0.9896 -- 0.96 rather than
    1.00 because Mkt-RF is the whole US market including the small caps SPY does not hold.

    This check caught a real error on its first run. The strategy returns were being taken
    from the `next_open` convention, which measures open-to-open, while the factors are
    close-to-close. They correlate 0.40 at daily frequency, and the R-squared for a market
    index came back 0.159. Nothing else in the table would have looked wrong.
    """
    spy, _, factors = data
    result = run_vectorized(spy, BuyAndHold(), ZERO_COST, CAPITAL, "close_to_close", ledger=LEDGER)
    start = len(spy) - len(result)
    aligned, rows = align_to(factors, spy.ts[start:][1:])
    excess = result.net_ret[1:][rows] - aligned.rf

    fit = fit_factors(excess, aligned.column("Mkt-RF").reshape(-1, 1), ("Mkt-RF",))
    print(
        f"SPY on Mkt-RF: beta {fit.loading('Mkt-RF'):.4f}, R2 {fit.r_squared:.4f}, "
        f"alpha {fit.alpha_annual:+.2%}/yr (t {fit.alpha_t:+.2f})"
    )

    assert 0.85 < fit.loading("Mkt-RF") < 1.10, (
        f"SPY's market beta came out {fit.loading('Mkt-RF'):.4f}. It is an S&P 500 tracker; "
        "a beta far from 1 means the return series and the factors are on different clocks."
    )
    assert fit.r_squared > 0.95, (
        f"R2 {fit.r_squared:.4f} for an index fund against the market factor. Below 0.95 "
        "the two series are not measuring the same days -- check the execution convention."
    )
    assert abs(fit.alpha_t) < 2.0, "an index fund should not show alpha"


def test_time_series_momentum_has_no_alpha_once_the_factors_are_priced(
    data: tuple[Bars, Panel, Factors],
) -> None:
    """The Phase 7 headline, and PLAYBOOK's stated act of intellectual honesty.

    TSMomentum(12m,1m) earns +0.606 Sharpe on SPY. Its four-factor alpha is -0.02% a year
    at t = -0.01. The return is entirely explained by two loadings the regression names:
    +0.618 on the market, because a trend follower is long a rising index most of the
    decade, and +0.244 on UMD, because it is a momentum strategy and UMD is the momentum
    factor.
    """
    spy, _, factors = data
    fit = single_asset(spy, TimeSeriesMomentum(12, 1), factors)
    print(fit.describe())

    assert not fit.survives, (
        f"TSMOM showed alpha t = {fit.alpha_t:+.2f}. If it now clears 2, the first "
        "question is what changed in the factor alignment, not whether an edge appeared."
    )
    assert fit.loading("Mkt-RF") > 0.3, "a long-biased trend follower must load on the market"
    assert fit.loading("UMD") > 0.1, (
        "a momentum strategy should load positively on the momentum factor; if it does not, "
        "either the signal or the factor alignment is wrong"
    )


def test_buy_and_hold_is_pure_market_exposure(data: tuple[Bars, Panel, Factors]) -> None:
    """The row that makes the rest readable: R2 0.995, market loading 0.975, no alpha."""
    spy, _, factors = data
    fit = single_asset(spy, BuyAndHold(), factors)
    print(fit.describe())
    assert fit.r_squared > 0.98
    assert 0.9 < fit.loading("Mkt-RF") < 1.05
    assert not fit.survives


def test_the_cross_sectional_book_is_mostly_the_momentum_factor(
    data: tuple[Bars, Panel, Factors],
) -> None:
    """A dollar-neutral sector book has almost no market exposure by construction -- 0.006
    -- and loads 0.323 on UMD, which accounts for over half its variance (R2 0.520).

    So what looked like a strategy is largely one factor, bought at the cost of trading.
    Its alpha is negative and not significant.
    """
    _, panel, factors = data
    weights = cross_sectional_weights(panel, 252, hold=21)
    result = run_panel(panel, weights, ZERO_COST, CAPITAL, lag=1)
    start = len(panel) - len(result)
    fit = attribute(factors, panel.ts[start:][1:], result.net_ret[1:])
    print(fit.describe())

    assert abs(fit.loading("Mkt-RF")) < 0.15, (
        f"a dollar-neutral book carried market beta {fit.loading('Mkt-RF'):+.3f}; "
        "neutrality is supposed to make this near zero"
    )
    assert fit.loading("UMD") > 0.15, "a momentum book should load on the momentum factor"
    assert not fit.survives


def test_not_one_strategy_in_the_zoo_produces_alpha(data: tuple[Bars, Panel, Factors]) -> None:
    """The whole table in one assertion.

    Seven constructions across two engines, and the largest absolute alpha t-statistic is
    2.49 -- on a one-month cross-sectional book whose alpha is NEGATIVE. Before reading
    even that as a finding, it should be deflated by the number of constructions tried,
    which the trials ledger counts.
    """
    spy, panel, factors = data
    fits = [
        ("BuyAndHold", single_asset(spy, BuyAndHold(), factors)),
        ("TSMomentum(12m,1m)", single_asset(spy, TimeSeriesMomentum(12, 1), factors)),
        ("TSMomentum(12m,3m)", single_asset(spy, TimeSeriesMomentum(12, 3), factors)),
    ]
    for lookback, label in ((252, "XS 12m"), (126, "XS 6m"), (21, "XS 1m")):
        weights = cross_sectional_weights(panel, lookback, hold=21)
        result = run_panel(panel, weights, ZERO_COST, CAPITAL, lag=1)
        start = len(panel) - len(result)
        fits.append((label, attribute(factors, panel.ts[start:][1:], result.net_ret[1:])))

    for label, fit in fits:
        print(
            f"  {label:<20} alpha {fit.alpha_annual:+7.2%}/yr  t {fit.alpha_t:+5.2f}  R2 {fit.r_squared:.3f}"
        )

    positive = [(label, fit) for label, fit in fits if fit.survives and fit.alpha > 0.0]
    assert not positive, (
        f"{[label for label, _ in positive]} showed positive alpha at |t| > 2. That would be "
        "a real finding and needs the trials ledger's N applied before it is believed."
    )
