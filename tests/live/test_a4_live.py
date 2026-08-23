"""The A4 ruling, verified on real SPY prices. Marked `live`; CI deselects it.

The ruling was taken on 2026-08-08 and rested on two arguments: a structural one --
`close[t]` lies inside `bars[0:t+1]`, which the Part A1 contract permits, and every Part
D convention lags the weight at least one bar -- and an empirical one measured on
synthetic GBM, where `sign(diff(close))` earned +0.054 annualised Sharpe against a true
one-bar-ahead oracle's +21.10.

Synthetic evidence is the weaker half of that. A generator with no microstructure, no
fat tails and no autocorrelation in returns is exactly the setting where a spurious
same-bar edge would fail to appear even if the reasoning were wrong. So this repeats the
measurement on ten years of real SPY, which is the verification that was outstanding.

Requires the cache. Populate it once with:

    uv run --group data python scripts/fetch_data.py

and every run after that is offline, reading the parquet and checking it against the
manifest before trusting it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from falsify.core.types import Bars
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST, CostModel
from falsify.data.loaders import DEFAULT_CACHE, DEFAULT_MANIFEST, FetchSpec, load
from falsify.metrics import annualise_sharpe, sharpe
from falsify.strategies.base import Strategy
from falsify.strategies.simple import BuyAndHold, CausalZScore, MACrossover

pytestmark = pytest.mark.live

SPEC = FetchSpec("SPY", "2015-01-01", "2025-01-01", "total_return")
CAPITAL = 10_000.0


class A4LeakyOracle(Strategy):
    """02 Part A4 verbatim: `sign(diff(close))`, said there to be a leaking oracle."""

    lookback = 1

    def signals(self, bars: Bars) -> npt.NDArray[np.float64]:
        return np.asarray(np.sign(np.diff(bars.close, prepend=bars.close[0])), dtype=np.float64)


class LookAheadOracle(Strategy):
    """Genuinely sees one bar ahead. What A4 was reaching for."""

    lookback = 1

    def signals(self, bars: Bars) -> npt.NDArray[np.float64]:
        close = bars.close
        out = np.full(len(close), np.nan)
        out[1:-1] = np.sign(close[2:] - close[1:-1])
        out[-1] = 0.0
        return out


@pytest.fixture(scope="module")
def spy() -> Bars:
    from falsify.data.loaders import DataUnavailable
    from falsify.data.manifest import ManifestMismatch

    try:
        return load(SPEC, cache_dir=DEFAULT_CACHE, manifest_path=DEFAULT_MANIFEST)
    except (DataUnavailable, ManifestMismatch) as exc:
        pytest.skip(f"SPY cache unavailable ({exc.__class__.__name__}); run scripts/fetch_data.py")


def annual_sharpe(bars: Bars, strategy: Strategy, costs: CostModel) -> float:
    result = run_vectorized(bars, strategy, costs, CAPITAL, "close_to_close")
    return annualise_sharpe(sharpe(result.net_ret[1:]))


def test_the_cached_series_is_what_the_manifest_says(spy: Bars) -> None:
    """The load path already verified the sha256; this pins the shape of the window
    the rest of the file reasons about."""
    print(f"SPY {len(spy)} bars, {spy.ts[0]} .. {spy.ts[-1]}, adjustment={spy.adjustment}")
    assert len(spy) > 2_000, "ten years of daily bars should be about 2,500"
    assert spy.adjustment == "total_return"
    assert np.all(spy.close > 0.0)


def test_a4_oracle_has_no_edge_on_real_spy(spy: Bars) -> None:
    """The live half of the ruling, and it is starker than the synthetic half.

    On real SPY the A4 strategy does not merely fail to earn -- it loses, and loses
    more once it pays to trade at 257 turns a year. A strategy that genuinely saw one
    bar ahead on this same series earns an unmistakable double-digit Sharpe, which is
    what the comparison is for.
    """
    gross = annual_sharpe(spy, A4LeakyOracle(), ZERO_COST)
    net = annual_sharpe(spy, A4LeakyOracle(), CostModel(commission_bps=5.0))
    oracle = annual_sharpe(spy, LookAheadOracle(), ZERO_COST)
    print(f"A4 sign(diff(close)): SR {gross:+.3f} gross, {net:+.3f} at 5 bps")
    print(f"true one-bar-ahead oracle: SR {oracle:+.3f}")

    assert gross < 1.0, (
        f"A4's strategy earned {gross:+.3f} on real SPY. The 2026-08-08 ruling that it "
        "does not leak rests on it having no edge; a large positive Sharpe here would "
        "reopen it."
    )
    assert oracle > 5.0, (
        f"the genuine look-ahead oracle earned only {oracle:+.3f}, so this comparison "
        "is not discriminating and proves nothing about A4"
    )
    assert oracle - gross > 5.0, "the two must be far apart or the contrast is empty"


def test_costs_punish_the_a4_strategy_for_its_turnover(spy: Bars) -> None:
    """It flips on almost every bar -- 257 turns a year -- so cost, not signal,
    dominates its result. Exactly the failure G6's turnover matching exists to stop
    a null from inheriting."""
    result = run_vectorized(spy, A4LeakyOracle(), ZERO_COST, CAPITAL, "close_to_close")
    turnover = float(np.sum(result.turnover) / len(result) * 252)
    print(f"A4 turnover: {turnover:.1f}/yr")
    assert turnover > 200.0, "the premise is that this strategy trades constantly"
    assert annual_sharpe(spy, A4LeakyOracle(), CostModel(commission_bps=5.0)) < annual_sharpe(
        spy, A4LeakyOracle(), ZERO_COST
    )


def test_buy_and_hold_is_the_bar_everything_else_has_to_clear(spy: Bars) -> None:
    """The honest benchmark, and on this window nothing beats it.

    Reported rather than hidden: SPY's own total return over 2015-2024 gives an
    annualised Sharpe near 0.8, and neither the moving-average crossover nor the
    z-score mean reversion comes close. A framework that cannot say this plainly about
    its own strategy zoo is not doing its job.
    """
    hold = annual_sharpe(spy, BuyAndHold(), ZERO_COST)
    costs = CostModel(commission_bps=5.0)
    others = {
        "MACrossover(20,50)": annual_sharpe(spy, MACrossover(20, 50), costs),
        "CausalZScore(20)": annual_sharpe(spy, CausalZScore(20), costs),
    }
    print(f"buy-and-hold SR {hold:+.3f}")
    for name, value in others.items():
        print(f"  {name:<20} SR {value:+.3f} at 5 bps")

    assert hold > 0.4, (
        f"SPY buy-and-hold over 2015-2024 should be clearly positive, got {hold:+.3f}"
    )
    assert all(value < hold for value in others.values()), (
        "a strategy beat buy-and-hold on this window; that would be worth investigating "
        "rather than celebrating"
    )


def test_mean_reversion_does_not_work_on_a_trending_index(spy: Bars) -> None:
    """The counterpart to the AR(1) result, and it points the other way.

    `CausalZScore` earns a real edge on a stationary mean-reverting series, which G3's
    power test asserts. SPY is not that: it trends, so the same rule loses. Both
    results are correct and the pair is the point -- an edge is a property of a
    process, not of a strategy.
    """
    value = annual_sharpe(spy, CausalZScore(20), CostModel(commission_bps=5.0))
    print(f"CausalZScore(20) on SPY at 5 bps: SR {value:+.3f}")
    assert value < 0.5, (
        f"mean reversion earned {value:+.3f} on a trending index. Not impossible, but it "
        "contradicts the process argument and would need explaining."
    )
