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
import numpy.typing as npt
import pytest

from falsify.core.conventions import CONVENTIONS, Convention, fill_prices, signal_lag
from falsify.core.event import benchmark_equity, run_event, warmup_start
from falsify.core.types import Bars
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST, CostModel
from falsify.strategies.base import Strategy
from falsify.strategies.simple import BuyAndHold, Flat
from falsify.synthetic import bars_from_close, gbm

CAPITAL = 10_000.0
SEED = 4_040

ENGINES = (run_event, run_vectorized)


class ConstantWeight(Strategy):
    """A fixed exposure held forever -- never trades after the anchor bar.

    Exists because the gate as originally written only ever exercised w = 1, so a
    weight that was propagated, scaled or signed incorrectly at any other exposure
    would have slipped past G4 entirely. G2 and G3 would eventually have caught it,
    but G4 is the gate whose job is the accounting identity, and an identity that
    holds only at w = 1 is not an identity.
    """

    def __init__(self, weight: float) -> None:
        self.weight = weight
        self.lookback = 1

    @property
    def name(self) -> str:
        return f"ConstantWeight({self.weight:+g})"

    def signals(self, bars: Bars) -> npt.NDArray[np.float64]:
        out = np.full(len(bars), self.weight)
        out[: self.lookback] = np.nan
        return out


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


@pytest.mark.parametrize("weight", [1.0, 0.5, 0.25, -0.5, -1.0])
@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize("engine", ENGINES, ids=lambda e: e.__name__)
def test_g4_constant_weight_compounds_exactly_one_plus_w_times_r(
    engine: object, convention: Convention, weight: float, bars: Bars
) -> None:
    """The identity generalised off w = 1, which is the coverage G4 was missing.

    A constant exposure never trades after the anchor, so at zero cost the equity
    path must be `capital * prod(1 + w*r)`. That is checked with `np.cumprod`,
    which is a genuinely independent formulation rather than a copy of the engine's
    loop.

    The tolerance is ulp-scale rather than exact, and the reason is worth stating
    because it is easy to mistake for a defect. `cumprod` forms the product of the
    growth factors and scales by capital once at the end; the engine multiplies the
    running equity at every step. Those associate the multiplications differently,
    and floating-point multiplication is not associative -- so the two agree to
    about 1e-15 and cannot agree bitwise. Requiring exactness here would be
    requiring that multiplication be associative. The comparison against
    `benchmark_equity` elsewhere in this file *is* exact, because there both curves
    are sequential recursions from the same base and the association matches.

    This is what catches a weight applied with the wrong sign, scaled by exposure
    twice, or aligned a bar off at fractional exposure -- none of which w = 1 can
    reveal, because 1 is a fixed point of most of those mistakes.
    """
    strategy = ConstantWeight(weight)
    result = engine(bars, strategy, ZERO_COST, CAPITAL, convention)  # type: ignore[operator]

    price = fill_prices(bars, convention)
    start = warmup_start(strategy.lookback, signal_lag(convention))
    r = price[start + 1 : len(bars)] / price[start : len(bars) - 1] - 1.0
    expected = CAPITAL * np.cumprod(np.concatenate(([1.0], 1.0 + weight * r)))

    assert np.all(result.weights == weight), "a constant strategy produced varying weights"
    assert np.all(result.turnover == 0.0), "a constant weight must never trade after the anchor"
    assert np.all(result.costs == 0.0)

    deviation = float(np.max(np.abs(result.equity - expected) / expected))
    assert deviation < 1e-13, (
        f"w={weight:+g}/{convention}: equity departs from capital*prod(1 + w*r) by "
        f"{deviation:.3e}, far above the ~1e-15 that reassociating the multiplications "
        "explains. At this size it is an accounting error, not rounding."
    )


@pytest.mark.parametrize("weight", [0.5, -0.75])
def test_g4_both_engines_agree_on_the_weight_array_itself(weight: float, bars: Bars) -> None:
    """G2 compares equity; this compares the weights that produced it.

    The event engine derives each weight from a hard-sliced prefix and the
    vectorised one slices a whole-series array, so agreement here is a statement
    about the alignment arithmetic rather than about the accounting.
    """
    strategy = ConstantWeight(weight)
    for convention in CONVENTIONS:
        a = run_event(bars, strategy, ZERO_COST, CAPITAL, convention)
        b = run_vectorized(bars, strategy, ZERO_COST, CAPITAL, convention)
        assert np.array_equal(a.weights, b.weights), f"weights differ under {convention}"
        assert len(a.weights) == len(b.weights) == len(a.equity)


def test_g4_short_position_pays_borrow_and_earns_no_cash() -> None:
    """The signed terms of the Part E gross return, isolated.

    At w = -1: exposure is 1 so the cash term is zero, and the short leg pays
    borrow. Getting `max(-w, 0)` backwards would credit borrow to a long position
    and charge it to a short one -- and the equity curve would still look
    plausible, which is why it is asserted rather than eyeballed.
    """
    bars = bars_from_close(gbm(0.08, 0.20, 300, np.random.default_rng(SEED + 3)))
    borrow_bps = 500.0
    per_bar = borrow_bps / 10_000.0 / 252.0

    short = run_vectorized(
        bars, ConstantWeight(-1.0), CostModel(borrow_bps_annual=borrow_bps), CAPITAL, "next_open"
    )
    short_free = run_vectorized(bars, ConstantWeight(-1.0), ZERO_COST, CAPITAL, "next_open")
    long_paid = run_vectorized(
        bars, ConstantWeight(1.0), CostModel(borrow_bps_annual=borrow_bps), CAPITAL, "next_open"
    )
    long_free = run_vectorized(bars, ConstantWeight(1.0), ZERO_COST, CAPITAL, "next_open")

    drag = short_free.gross_ret[1:] - short.gross_ret[1:]
    print(f"borrow drag per bar: {np.mean(drag):.10f}  expected {per_bar:.10f}")
    assert np.allclose(drag, per_bar, rtol=1e-12, atol=0.0), "short leg did not pay borrow"
    assert np.array_equal(long_paid.equity, long_free.equity), (
        "a long-only position was charged borrow; the max(-w, 0) term has the wrong sign"
    )


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
