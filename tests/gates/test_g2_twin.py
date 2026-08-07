"""G2 -- twin-engine agreement. Specified by 02-ENGINE-SPEC.md Part F3.

The claim: a vectorised engine is fast and subtly wrong in ways that raise no
exception, an explicit event loop is slow and obviously right, and agreement to
1e-12 certifies the fast one. Run across the strategy zoo, all three conventions,
and a cost sweep, with hypothesis generating the price series so the property is
quantified over arbitrary inputs rather than the three anyone would think of.

Read Part F2's failure mode before relaxing anything here. "Agrees to 1e-6 but
not 1e-12" is an accumulation-order difference in the equity recursion, not a
rounding artefact to wave through.

The two engines share exactly one thing -- `warmup_start`, because an off-by-one
in the warm-up is a specification question rather than an implementation one, and
duplicating it would let both be wrong in the same direction while agreeing
perfectly. Everything else, including all of the Part E arithmetic, is written
twice on purpose. Two engines that share their accounting agree by construction
and certify nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from falsify.core.conventions import CONVENTIONS, Convention
from falsify.core.event import benchmark_equity, run_event
from falsify.core.types import Bars, InsufficientHistory
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST, CostModel
from falsify.strategies.base import Strategy
from tests.gates.test_g1_causality import (
    HONEST,
    BuyAndHold,
    CausalZScore,
    MACrossover,
    bars_from_close,
)

TOLERANCE = 1e-12
INITIAL_CAPITAL = 10_000.0
MASTER_SEED = 20140458

# Kept small because the event engine is O(T^2) by design: it recomputes the
# strategy on a growing prefix at every bar. 220 bars across the zoo, three
# conventions and a cost sweep is the honest trade between coverage and minutes.
T_BARS = 220

COST_SWEEP: tuple[CostModel, ...] = (
    ZERO_COST,
    CostModel(commission_bps=1.0),
    CostModel(commission_bps=2.0, half_spread_bps=1.5, slippage_bps=0.5),
    # The cash-yield and borrow terms exercise the two parts of the gross-return
    # equation that a long-only zero-rate test never touches.
    CostModel(commission_bps=5.0, cash_yield_annual=0.05, borrow_bps_annual=75.0),
    CostModel(commission_bps=100.0, half_spread_bps=25.0, cash_yield_annual=0.02),
)


def gbm_close(seed: int, n: int = T_BARS, sigma: float = 0.012) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return np.asarray(100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma, n))), dtype=np.float64)


def max_relative_deviation(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Part F3's metric: max |a - b| / |a|, NaN-tolerant."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(a - b) / np.abs(a)
    return float(np.nanmax(rel))


@pytest.fixture(scope="module")
def bars() -> Bars:
    return bars_from_close(gbm_close(MASTER_SEED))


# ------------------------------------------------------------------ the gate


@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize("strategy", HONEST, ids=lambda s: s.name)
def test_g2_engines_agree_across_zoo_and_conventions(
    strategy: Strategy, convention: Convention, bars: Bars
) -> None:
    """The gate itself, at Part F3's tolerance."""
    a = run_event(bars, strategy, ZERO_COST, INITIAL_CAPITAL, convention)
    b = run_vectorized(bars, strategy, ZERO_COST, INITIAL_CAPITAL, convention)

    deviation = max_relative_deviation(a.equity, b.equity)
    print(f"{strategy.name:14s} {convention:15s} max relative equity deviation {deviation:.3e}")
    assert deviation < TOLERANCE, (
        f"{strategy.name}/{convention}: max relative deviation {deviation:.3e} "
        f"exceeds {TOLERANCE:.0e}. Per Part F2 this is an accumulation-order "
        "difference in the equity recursion, not a rounding artefact."
    )


@pytest.mark.parametrize("costs", COST_SWEEP, ids=lambda c: f"{c.total_bps:g}bps")
def test_g2_holds_across_the_cost_sweep(costs: CostModel, bars: Bars) -> None:
    """Costs enter the recursion through `equity[k-1]`, so they are exactly where
    an accumulation-order difference would surface."""
    strategy = MACrossover(5, 15)
    a = run_event(bars, strategy, costs, INITIAL_CAPITAL, "next_open")
    b = run_vectorized(bars, strategy, costs, INITIAL_CAPITAL, "next_open")

    for name, left, right in (
        ("equity", a.equity, b.equity),
        ("costs", a.costs, b.costs),
        ("turnover", a.turnover, b.turnover),
        ("gross_ret", a.gross_ret, b.gross_ret),
        ("net_ret", a.net_ret, b.net_ret),
    ):
        assert np.allclose(left, right, rtol=TOLERANCE, atol=0.0, equal_nan=True), (
            f"{name} diverges at total_bps={costs.total_bps:g}: "
            f"max relative deviation {max_relative_deviation(left, right):.3e}"
        )
    print(
        f"total_bps={costs.total_bps:<7g} equity deviation "
        f"{max_relative_deviation(a.equity, b.equity):.3e}  "
        f"final equity {a.equity[-1]:.4f}"
    )


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    seed=st.integers(0, 2**31 - 1),
    sigma=st.floats(0.002, 0.05),
    convention=st.sampled_from(CONVENTIONS),
)
def test_g2_holds_for_arbitrary_price_series(
    seed: int, sigma: float, convention: Convention
) -> None:
    """Part F3 asks for hypothesis here, so the claim is "for all price series"
    rather than "for the one I generated"."""
    bars = bars_from_close(gbm_close(seed, sigma=sigma))
    strategy = CausalZScore(12)
    costs = CostModel(commission_bps=3.0, cash_yield_annual=0.04, borrow_bps_annual=50.0)

    a = run_event(bars, strategy, costs, INITIAL_CAPITAL, convention)
    b = run_vectorized(bars, strategy, costs, INITIAL_CAPITAL, convention)
    deviation = max_relative_deviation(a.equity, b.equity)
    assert deviation < TOLERANCE, (
        f"seed={seed} sigma={sigma:.4f} convention={convention}: deviation {deviation:.3e}"
    )


# --------------------------------------------- the gate can fail (F7)


class OffByOneEngineBug(Strategy):
    """A strategy whose vectorised and per-prefix signals differ by one bar.

    Not a leak -- G1 passes it, because every value depends only on the past. It
    is the *other* failure: two implementations of one idea disagreeing. The
    prefix path sees a shorter array than the whole-series path and indexes from
    the end, so the two engines assign different weights to the same bar.

    Exactly the class of bug Part F exists to catch, and proof that G2 can fail.
    """

    lookback = 4

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        close = bars.close
        out = np.full(len(close), np.nan)
        for t in range(self.lookback, len(close)):
            # Depends on how far the array happens to extend, not only on the past.
            offset = 1 if len(close) % 2 == 0 else 0
            out[t] = 1.0 if close[t - offset] > close[t - self.lookback] else -1.0
        return out


def test_g2_fires_on_an_engine_disagreement(bars: Bars) -> None:
    """A gate that has never failed is not a gate (03 Part F, F7)."""
    strategy = OffByOneEngineBug()
    a = run_event(bars, strategy, ZERO_COST, INITIAL_CAPITAL, "next_open")
    b = run_vectorized(bars, strategy, ZERO_COST, INITIAL_CAPITAL, "next_open")
    deviation = max_relative_deviation(a.equity, b.equity)
    print(f"planted disagreement: max relative equity deviation {deviation:.3e}")
    assert deviation > TOLERANCE, (
        "the planted engine disagreement was not detected; G2 certifies nothing"
    )


# --------------------------------------------------- accounting identities


@pytest.mark.parametrize("convention", CONVENTIONS)
def test_equity_and_benchmark_start_at_initial_capital(convention: Convention, bars: Bars) -> None:
    """Part E: assert `equity[0] == bench[0] == initial_capital`, exactly.

    This is the reference repo's benchmark bug expressed as a test. Slice first,
    compound second, and the two curves share a base.
    """
    for engine in (run_event, run_vectorized):
        result = engine(bars, BuyAndHold(), ZERO_COST, INITIAL_CAPITAL, convention)
        bench = benchmark_equity(bars, BuyAndHold().lookback, INITIAL_CAPITAL, convention)
        assert result.equity[0] == INITIAL_CAPITAL
        assert bench[0] == INITIAL_CAPITAL
        assert len(bench) == len(result)


@pytest.mark.parametrize("convention", CONVENTIONS)
def test_buy_and_hold_at_zero_cost_is_the_benchmark(convention: Convention, bars: Bars) -> None:
    """A preview of G4's zero-cost identity: weight 1, no costs, so the strategy
    curve *is* buy-and-hold. Exact float equality, not approximate."""
    bench = benchmark_equity(bars, BuyAndHold().lookback, INITIAL_CAPITAL, convention)
    for engine in (run_event, run_vectorized):
        result = engine(bars, BuyAndHold(), ZERO_COST, INITIAL_CAPITAL, convention)
        assert np.array_equal(result.equity, bench), (
            f"{engine.__name__}/{convention}: zero-cost buy-and-hold deviates from "
            f"buy-and-hold by up to {max_relative_deviation(bench, result.equity):.3e}"
        )


def test_net_return_reconstructs_the_equity_path(bars: Bars) -> None:
    """`net_ret` must be the actual realised per-bar return, so compounding it
    returns the equity curve. Guards against reporting a net return that is
    `gross - cost_rate`, the additive approximation Part E rejects."""
    costs = CostModel(commission_bps=8.0, half_spread_bps=2.0, cash_yield_annual=0.03)
    for engine in (run_event, run_vectorized):
        result = engine(bars, MACrossover(5, 15), costs, INITIAL_CAPITAL, "next_open")
        rebuilt = INITIAL_CAPITAL * np.cumprod(1.0 + result.net_ret)
        assert np.allclose(rebuilt, result.equity, rtol=1e-13, atol=0.0), engine.__name__


def test_cost_is_charged_on_traded_notional(bars: Bars) -> None:
    """Part E charges `turnover * equity[t-1] * rate`, not a deduction from the
    portfolio return. The two coincide only while positions are 0/1 and fully
    allocated, and diverge the moment sizing is added."""
    costs = CostModel(commission_bps=10.0)
    result = run_vectorized(bars, MACrossover(5, 15), costs, INITIAL_CAPITAL, "next_open")
    expected = result.turnover[1:] * result.equity[:-1] * costs.cost_rate()
    assert np.allclose(result.costs[1:], expected, rtol=0.0, atol=0.0)
    assert result.costs[0] == 0.0
    assert result.turnover[0] == 0.0


def test_cash_yield_is_earned_on_the_unallocated_fraction() -> None:
    """The term everybody drops. Flat position, zero costs, positive cash yield:
    equity must grow at exactly the cash rate."""

    class Flat(Strategy):
        lookback = 1

        def signals(self, bars: Bars) -> NDArray[np.float64]:
            out = np.zeros(len(bars))
            out[: self.lookback] = np.nan
            return out

    bars = bars_from_close(gbm_close(MASTER_SEED + 9))
    costs = CostModel(cash_yield_annual=0.05)
    for engine in (run_event, run_vectorized):
        result = engine(bars, Flat(), costs, INITIAL_CAPITAL, "next_open")
        per_bar = 0.05 / 252.0
        assert np.allclose(result.gross_ret[1:], per_bar, rtol=1e-15, atol=0.0), engine.__name__
        assert result.equity[-1] > INITIAL_CAPITAL, "idle cash must accrue"


# -------------------------------------------------------------- degenerate


def test_too_few_bars_raises_rather_than_returning_an_empty_result() -> None:
    """Gate 0.4: a run with no history and a run that found nothing are different
    outcomes, and only one of them is an empty frame."""
    short = bars_from_close(gbm_close(1, n=20))
    for engine in (run_event, run_vectorized):
        with pytest.raises(InsufficientHistory, match="too few"):
            engine(short, MACrossover(20, 50), ZERO_COST, INITIAL_CAPITAL, "next_open")


@pytest.mark.parametrize("capital", [0.0, -1.0])
def test_non_positive_capital_is_rejected(capital: float, bars: Bars) -> None:
    for engine in (run_event, run_vectorized):
        with pytest.raises(ValueError, match="initial_capital"):
            engine(bars, BuyAndHold(), ZERO_COST, capital, "next_open")


def test_unknown_convention_is_rejected(bars: Bars) -> None:
    for engine in (run_event, run_vectorized):
        with pytest.raises(ValueError, match="unknown convention"):
            engine(bars, BuyAndHold(), ZERO_COST, INITIAL_CAPITAL, "at_the_vwap")  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["commission_bps", "cash_yield_annual", "borrow_bps_annual"])
def test_negative_costs_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CostModel(**{field: -1.0})


def test_conventions_are_actually_distinct(bars: Bars) -> None:
    """If two conventions produced identical curves the comparison figure Part D
    asks for would be meaningless, and a lag bug would hide."""
    strategy = MACrossover(5, 15)
    curves = {
        c: run_vectorized(bars, strategy, ZERO_COST, INITIAL_CAPITAL, c).equity[-1]
        for c in CONVENTIONS
    }
    print("final equity by convention: " + "  ".join(f"{k}={v:.4f}" for k, v in curves.items()))
    assert len(set(curves.values())) == len(CONVENTIONS), (
        f"conventions collapsed onto the same result: {curves}"
    )
