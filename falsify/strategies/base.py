"""The Strategy contract. Specified by 02-ENGINE-SPEC.md Part C."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from falsify.core.types import Bars


class Strategy(ABC):
    """Emits a target weight per bar. Never an order (B4).

    Sizing, rebalancing and execution belong to the engine. A strategy that
    emits orders cannot be run through both engines, so it cannot be certified
    by G2, and cannot be vol-targeted without a rewrite.
    """

    lookback: int
    """Bars of history required before the first valid signal.

    Declared, not inferred, and checked: the event engine slices exactly this
    many bars. A strategy that silently needs more produces NaN and fails
    loudly rather than quietly reading further back.
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def signals(self, bars: Bars) -> NDArray[np.float64]:
        """Target weight per bar, in [-1, 1].

        CONTRACT: signals[t] may depend only on bars[0:t+1].
        Enforced by G1, not by convention -- `shift(1)` is a claim about this
        property, never a proof of it.

        Returns NaN for t < lookback.
        """
