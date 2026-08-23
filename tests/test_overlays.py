"""Unit tests for the strategy overlays.

They exist because of what G5 measured: `CausalZScore(5)` trades ~134 times a year
and is unprofitable *even at zero cost*, while `CausalZScore(60)` trades ~52 times and
survives to 90 bps. Turnover was doing more damage than the signal was doing good and
there was no dial. These are the dial.

The causality of both overlays is verified by G1 in `test_g1_causality.py`, which is
the right place for it -- these tests cover behaviour.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import numpy.typing as npt
import pytest

from falsify.core.types import Bars
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST, CostModel
from falsify.metrics import annualise_sharpe, sharpe
from falsify.strategies.null import realised_exposure, realised_turnover_annual
from falsify.strategies.overlays import TurnoverBuffer, VolTarget, simple_returns
from falsify.strategies.simple import BuyAndHold, CausalZScore
from falsify.synthetic import ar1, bars_from_close, gbm

CAPITAL = 10_000.0
SEED = 7_070


@pytest.fixture(scope="module")
def bars() -> Bars:
    return bars_from_close(ar1(0.95, 0.02, 1000, np.random.default_rng(SEED)))


# ------------------------------------------------------------- turnover buffer


@pytest.mark.parametrize("band", [0.1, 0.25, 0.5, 1.0])
def test_buffer_never_increases_turnover(band: float, bars: Bars) -> None:
    """A band can only suppress trades, never create them."""
    base = CausalZScore(20)
    plain = run_vectorized(bars, base, ZERO_COST, CAPITAL, "next_open")
    buffered = run_vectorized(bars, TurnoverBuffer(base, band), ZERO_COST, CAPITAL, "next_open")
    plain_to = realised_turnover_annual(plain.turnover)
    buffered_to = realised_turnover_annual(buffered.turnover)
    print(f"band={band:.2f}: turnover {plain_to:.2f} -> {buffered_to:.2f}/yr")
    assert buffered_to <= plain_to + 1e-12


def test_buffer_turnover_decreases_monotonically_with_the_band(bars: Bars) -> None:
    """Wider band, less trading. If this inverted, the comparison the overlay exists
    to enable would be meaningless."""
    base = CausalZScore(20)
    turnovers = [
        realised_turnover_annual(
            run_vectorized(
                bars, TurnoverBuffer(base, band), ZERO_COST, CAPITAL, "next_open"
            ).turnover
        )
        for band in (0.0, 0.1, 0.25, 0.5, 0.75)
    ]
    print("turnover by band: " + "  ".join(f"{t:.2f}" for t in turnovers))
    assert all(b <= a + 1e-12 for a, b in pairwise(turnovers)), f"not monotone: {turnovers}"


def test_buffer_with_zero_band_is_a_no_op(bars: Bars) -> None:
    """band = 0 means "trade on any change", which is the base strategy exactly."""
    base = CausalZScore(20)
    plain = run_vectorized(bars, base, ZERO_COST, CAPITAL, "next_open")
    zero = run_vectorized(bars, TurnoverBuffer(base, 0.0), ZERO_COST, CAPITAL, "next_open")
    assert np.array_equal(plain.weights, zero.weights)
    assert np.array_equal(plain.equity, zero.equity)


def test_buffer_holds_its_position_between_trades() -> None:
    """The weight series must be piecewise constant, changing only when the base
    target moves outside the band."""

    class Ramp(CausalZScore):
        """A target that drifts by 0.05 a bar -- five bars to cross a 0.25 band."""

        def signals(self, bars: Bars) -> npt.NDArray[np.float64]:
            out = np.full(len(bars), np.nan)
            out[self.lookback :] = np.linspace(0.0, 1.0, len(bars) - self.lookback)
            return out

    bars = bars_from_close(gbm(0.0, 0.2, 60, np.random.default_rng(1)))
    weights = TurnoverBuffer(Ramp(5), 0.25).signals(bars)
    finite = weights[np.isfinite(weights)]
    distinct = np.unique(finite)
    print(f"{len(finite)} bars collapsed onto {len(distinct)} distinct weights")
    assert len(distinct) < len(finite) / 3, "the buffer is not holding positions"


def test_buffer_rejects_a_negative_band() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TurnoverBuffer(BuyAndHold(), -0.1)


# ---------------------------------------------------------------- vol target


def test_vol_target_scales_exposure_toward_the_target(bars: Bars) -> None:
    """A lower target must produce lower exposure, monotonically."""
    base = CausalZScore(20)
    exposures = [
        realised_exposure(
            run_vectorized(
                bars, VolTarget(base, target, 60), ZERO_COST, CAPITAL, "next_open"
            ).weights
        )
        for target in (0.05, 0.10, 0.20, 0.40)
    ]
    print("exposure by target vol: " + "  ".join(f"{e:.4f}" for e in exposures))
    assert all(a <= b + 1e-12 for a, b in pairwise(exposures)), f"not monotone: {exposures}"


def test_vol_target_respects_the_cap(bars: Bars) -> None:
    """Even at an absurd target, weights must stay inside the [-1, 1] contract."""
    overlay = VolTarget(CausalZScore(20), target_annual_vol=50.0, window=60, cap=1.0)
    weights = overlay.signals(bars)
    finite = weights[np.isfinite(weights)]
    assert np.all(np.abs(finite) <= 1.0 + 1e-12)
    assert np.max(np.abs(finite)) == pytest.approx(1.0), "the cap should actually bind here"


def test_vol_target_keeps_the_warmup_visible(bars: Bars) -> None:
    """NaN must survive the overlay. G1 compares NaN patterns, so a warm-up silently
    filled with a number would be a causality failure rather than a cosmetic one."""
    overlay = VolTarget(CausalZScore(20), 0.15, 60)
    weights = overlay.signals(bars)
    assert np.all(np.isnan(weights[: overlay.lookback]))
    assert overlay.lookback >= 61, "the vol estimate needs window+1 bars of returns"
    assert np.isfinite(weights[overlay.lookback :]).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"target_annual_vol": 0.0}, "target_annual_vol"),
        ({"window": 1}, "window"),
        ({"cap": 0.0}, "cap"),
    ],
)
def test_vol_target_rejects_bad_parameters(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        VolTarget(BuyAndHold(), **kwargs)  # type: ignore[arg-type]


def test_simple_returns_is_causal_and_starts_with_nan() -> None:
    close = np.asarray([100.0, 110.0, 99.0])
    got = simple_returns(close)
    assert np.isnan(got[0])
    assert got[1] == pytest.approx(0.10)
    assert got[2] == pytest.approx(99.0 / 110.0 - 1.0)


# ------------------------------------------------- the point of the overlays


def test_overlays_improve_net_sharpe_at_realistic_cost(bars: Bars) -> None:
    """The claim that justifies the overlays, asserted rather than described.

    Composed, they must raise the net Sharpe at a realistic cost while barely moving
    the gross Sharpe -- which is what shows the gain came from paying less tax and not
    from a different signal. If the gross Sharpe moved a lot too, the overlay would be
    a new strategy wearing a risk-management label.
    """
    base = CausalZScore(20)
    composed = TurnoverBuffer(VolTarget(base, 0.15, 60), 0.25)
    costs = CostModel(commission_bps=20.0)

    base_gross = annualise_sharpe(
        sharpe(run_vectorized(bars, base, ZERO_COST, CAPITAL, "next_open").net_ret[1:])
    )
    base_net = annualise_sharpe(
        sharpe(run_vectorized(bars, base, costs, CAPITAL, "next_open").net_ret[1:])
    )
    over_run = run_vectorized(bars, composed, costs, CAPITAL, "next_open")
    over_gross = annualise_sharpe(
        sharpe(run_vectorized(bars, composed, ZERO_COST, CAPITAL, "next_open").net_ret[1:])
    )
    over_net = annualise_sharpe(sharpe(over_run.net_ret[1:]))
    base_to = realised_turnover_annual(
        run_vectorized(bars, base, ZERO_COST, CAPITAL, "next_open").turnover
    )
    over_to = realised_turnover_annual(over_run.turnover)

    print(
        f"base:     turnover {base_to:6.2f}/yr  gross {base_gross:+.4f}  net@20bps {base_net:+.4f}\n"
        f"overlaid: turnover {over_to:6.2f}/yr  gross {over_gross:+.4f}  net@20bps {over_net:+.4f}"
    )
    assert over_to < base_to, "the overlays must cut turnover"
    assert over_net > base_net, "cutting turnover must improve the net Sharpe at 20 bps"
    assert abs(over_gross - base_gross) < 0.35, (
        "the gross Sharpe moved materially, so the overlay changed the signal rather "
        "than only its cost -- that would be a new strategy, not risk management"
    )
