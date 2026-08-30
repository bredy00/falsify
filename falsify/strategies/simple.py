"""The baseline strategies. PLAYBOOK Phase 6, minus the ones that need G6.

All three emit target weights and never orders (B4), declare their lookback, and
lag their decisions. They are the zoo G2 runs both engines over, and the honest
half of G1's evidence -- the traps live in the test suite, because deliberately
leaky code has no business shipping in the package.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from falsify.core.types import Bars
from falsify.features import rolling_mean, rolling_std, shift_one
from falsify.strategies.base import Strategy


class BuyAndHold(Strategy):
    """Fully invested, always. The reference against which everything else has to
    justify its turnover."""

    lookback = 1

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        out = np.full(len(bars), 1.0)
        out[: self.lookback] = np.nan
        return out


class MACrossover(Strategy):
    """The naive baseline's strategy, done correctly: long when the fast mean is
    above the slow one, short when below, decision lagged a bar."""

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast < 1:
            raise ValueError(f"fast must be at least 1, got {fast}")
        if fast >= slow:
            raise ValueError(f"fast={fast} must be below slow={slow}")
        self.fast, self.slow = fast, slow
        self.lookback = slow

    @property
    def name(self) -> str:
        return f"MACrossover({self.fast},{self.slow})"

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        fast = rolling_mean(bars.close, self.fast)
        slow = rolling_mean(bars.close, self.slow)
        return shift_one(np.sign(fast - slow))


class CausalZScore(Strategy):
    """Mean reversion on a z-score of price against its own trailing window.

    Weight is `-clip(z, -1, 1)`: long when price sits below its recent mean, short
    when above, capped at full exposure. On a random walk this has no edge by
    construction; on a stationary AR(1) it has a real one, which is what makes it
    the natural instrument for Gate 0.3's power test.
    """

    def __init__(self, window: int = 30) -> None:
        if window < 2:
            raise ValueError(f"window must be at least 2, got {window}")
        self.window = window
        self.lookback = window

    @property
    def name(self) -> str:
        return f"CausalZScore({self.window})"

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        mu = rolling_mean(bars.close, self.window)
        sd = rolling_std(bars.close, self.window)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (bars.close - mu) / sd
        return shift_one(np.clip(-z, -1.0, 1.0))


class Flat(Strategy):
    """Never invested. Exists so the cash-yield term can be tested in isolation."""

    lookback = 1

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        out = np.zeros(len(bars))
        out[: self.lookback] = np.nan
        return out


# TimeSeriesMomentum joins the zoo here rather than being certified separately, so G1
# and G2 pick it up automatically -- a strategy that is not in the zoo is a strategy
# whose causality and twin-engine agreement nobody checked. Imported at the bottom
# because `momentum` imports the `Strategy` base this module also uses.
from falsify.strategies.momentum import TimeSeriesMomentum  # noqa: E402

ZOO: tuple[Strategy, ...] = (
    BuyAndHold(),
    MACrossover(),
    CausalZScore(),
    TimeSeriesMomentum(),
)
