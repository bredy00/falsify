"""G4 -- the zero-cost identity.

Costs zero, position identically one: the strategy equity curve *is* buy-and-hold.
Exact float equality, not approximate. One assertion, and 03 Part C budgets a
quarter of a session for it, which is right -- but it is the assertion that catches
a whole family of accounting mistakes at once, because almost anything wrong in the
equity recursion breaks exact equality while leaving `allclose` happy.

What it pins down:
  - the anchor bar starts at exactly `initial_capital`, so the strategy and the
    benchmark share a base (Part E: slice first, compound second)
  - no phantom cost is charged when the cost model is zero
  - the multiplicative recursion is exact rather than the additive approximation
  - the weight alignment introduces no drift relative to the price series
"""

from __future__ import annotations

import numpy as np
import pytest

from falsify.core.conventions import CONVENTIONS, Convention
from falsify.core.event import benchmark_equity, run_event
from falsify.core.types import Bars
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST, CostModel
from falsify.strategies.simple import BuyAndHold, Flat
from falsify.synthetic import bars_from_close, gbm

CAPITAL = 10_000.0
SEED = 4_040

ENGINES = (run_event, run_vectorized)


@pytest.fixture(scope="module")
def bars() -> Bars:
    return bars_from_close(gbm(0.08, 0.20, 300, np.random.default_rng(SEED)))


@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize("engine", ENGINES, ids=lambda e: e.__name__)
def test_g4_zero_cost_buy_and_hold_is_exactly_buy_and_hold(
    engine: object, convention: Convention, bars: Bars
) -> None:
    """The gate. Exact float equality, both engines, all three conventions."""
    result = engine(bars, BuyAndHold(), ZERO_COST, CAPITAL, convention)  # type: ignore[operator]
    bench = benchmark_equity(bars, BuyAndHold().lookback, CAPITAL, convention)

    assert len(bench) == len(result)
    assert result.equity[0] == CAPITAL, "the anchor bar must start at initial capital exactly"
    assert bench[0] == CAPITAL, "the benchmark must share that base"
    assert np.array_equal(result.equity, bench), (
        "zero-cost buy-and-hold is not bitwise identical to buy-and-hold; max relative "
        f"deviation {np.max(np.abs(result.equity - bench) / bench):.3e}"
    )


@pytest.mark.parametrize("engine", ENGINES, ids=lambda e: e.__name__)
def test_g4_no_cost_is_charged_when_the_model_is_zero(engine: object, bars: Bars) -> None:
    """Turnover still happens; the charge must not."""
    from falsify.strategies.simple import MACrossover

    result = engine(bars, MACrossover(5, 15), ZERO_COST, CAPITAL, "next_open")  # type: ignore[operator]
    assert np.all(result.costs == 0.0), "a zero cost model charged something"
    assert np.sum(result.turnover) > 0.0, "the test is vacuous if nothing traded"

    # Mathematically net == gross here, but not bitwise, and demanding exactness
    # would be wrong rather than strict: net_ret is measured back off the equity
    # path as equity[k]/equity[k-1] - 1, so it takes a multiply, a divide and a
    # subtract to recover a number gross_ret reached directly. Those round
    # differently in the last bits. The exact comparisons in this file are between
    # two curves built by the *same* recursion shape, where exactness is available.
    assert np.allclose(result.net_ret, result.gross_ret, rtol=1e-12, atol=1e-18), (
        "with no costs the net and gross return series must agree to rounding"
    )


@pytest.mark.parametrize("engine", ENGINES, ids=lambda e: e.__name__)
def test_g4_flat_position_at_zero_cost_and_zero_yield_is_flat(
    engine: object, bars: Bars
) -> None:
    """Weight zero, no cash yield: equity must not move by a single bit. Catches a
    cash term that leaks in when it should be switched off."""
    result = engine(bars, Flat(), ZERO_COST, CAPITAL, "next_open")  # type: ignore[operator]
    assert np.all(result.equity == CAPITAL), "flat equity moved"
    assert np.all(result.gross_ret == 0.0)
    assert np.all(result.net_ret == 0.0)


def test_g4_fires_when_a_phantom_cost_is_introduced(bars: Bars) -> None:
    """A gate that has never failed is not a gate (F7). One basis point is enough
    to break exact equality, which is the point of demanding exactness."""
    bench = benchmark_equity(bars, BuyAndHold().lookback, CAPITAL, "next_open")
    nearly_free = run_vectorized(
        bars, BuyAndHold(), CostModel(commission_bps=1.0), CAPITAL, "next_open"
    )
    # Buy-and-hold never trades after the anchor, so even a cost model changes
    # nothing -- which is itself worth asserting.
    assert np.array_equal(nearly_free.equity, bench), (
        "buy-and-hold has no turnover after the anchor bar, so cost must not bite"
    )

    from falsify.strategies.simple import MACrossover

    traded_free = run_vectorized(bars, MACrossover(5, 15), ZERO_COST, CAPITAL, "next_open")
    traded_paid = run_vectorized(
        bars, MACrossover(5, 15), CostModel(commission_bps=1.0), CAPITAL, "next_open"
    )
    assert not np.array_equal(traded_free.equity, traded_paid.equity), (
        "a strategy that trades must be affected by a 1 bps cost, or the cost model "
        "is not wired into the equity path at all"
    )
    assert traded_paid.equity[-1] < traded_free.equity[-1]
