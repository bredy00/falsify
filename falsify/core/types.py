"""Immutable value types. Specified by 02-ENGINE-SPEC.md Part B.

`frozen=True` is load-bearing rather than stylistic (B7). The twin-engine
comparison at G2 is only meaningful if an object has exactly one state to
compare; a type that mutates across a dozen assignments has as many meanings as
it has assignment sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Adjustment = Literal["raw", "split", "total_return"]

_PRICE_FIELDS = ("open", "high", "low", "close", "volume")

# Trading bars per year. Annualisation happens at the reporting boundary only
# (B8); this constant exists so the cost model and the engines agree on the
# per-bar conversion of annual rates.
BARS_PER_YEAR = 252


class InsufficientHistory(ValueError):
    """Fewer bars than the strategy and convention require.

    Raised rather than returning an empty result, per Gate 0.4: a run that had no
    history to work with and a run that found nothing are different outcomes.
    """


@dataclass(frozen=True, slots=True)
class Bars:
    """Immutable OHLCV. `ts` is strictly increasing and tz-aware (UTC).

    Validation happens here and nowhere else, and it *rejects* rather than
    repairs. A NaN in `close` is a data-layer defect; filling it at this
    boundary would be exactly the silent backward information flow that B6
    forbids, so this type refuses to hide it.
    """

    ts: NDArray[np.datetime64]
    open: NDArray[np.float64]
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    close: NDArray[np.float64]
    volume: NDArray[np.float64]
    adjustment: Adjustment

    def __post_init__(self) -> None:
        n = len(self.ts)
        for field in _PRICE_FIELDS:
            got = len(getattr(self, field))
            if got != n:
                raise ValueError(f"{field} length {got} != ts length {n}")
        if n > 1 and not np.all(np.diff(self.ts) > np.timedelta64(0, "ns")):
            raise ValueError("ts is not strictly increasing")
        if np.isnan(self.close).any():
            raise ValueError("NaN in close; fix upstream, do not fill here")

    def __len__(self) -> int:
        return len(self.ts)

    def slice(self, start: int, stop: int) -> Bars:
        """A hard slice, `[start, stop)`.

        Load-bearing for the event engine: it passes the strategy a prefix, so
        the function is *structurally* incapable of seeing bar `stop` or later.
        That is a stronger guarantee than a shift convention, which merely
        promises not to look.
        """
        if not 0 <= start < stop <= len(self):
            raise ValueError(f"slice [{start}, {stop}) out of range for {len(self)} bars")
        return Bars(
            ts=self.ts[start:stop],
            open=self.open[start:stop],
            high=self.high[start:stop],
            low=self.low[start:stop],
            close=self.close[start:stop],
            volume=self.volume[start:stop],
            adjustment=self.adjustment,
        )


@dataclass(frozen=True, slots=True)
class Result:
    """One engine run over the *reported* window. Frozen (B7).

    Every array is the same length and covers the window after warm-up, sliced
    before anything is compounded. That ordering is the fix for the reference
    repo's benchmark bug: it compounds the market through the warm-up while the
    strategy curve restarts, then plots the two against different bases. Slice
    first, compound second -- so `equity[0] == initial_capital` exactly, and a
    benchmark built on the same window starts at the same number.

    Index 0 is the anchor bar: the position is established there and no return,
    cost or turnover is attributed to it. `equity[0]` is therefore exactly the
    initial capital, which is what makes G4's zero-cost identity an exact float
    comparison rather than an approximate one. The entry cost of that first
    position is not charged -- a stated simplification, revisited with the cost
    model at G5, and worth remembering as one bar of optimism.
    """

    equity: NDArray[np.float64]
    weights: NDArray[np.float64]
    gross_ret: NDArray[np.float64]
    net_ret: NDArray[np.float64]
    costs: NDArray[np.float64]
    turnover: NDArray[np.float64]

    def __post_init__(self) -> None:
        n = len(self.equity)
        for field in ("weights", "gross_ret", "net_ret", "costs", "turnover"):
            got = len(getattr(self, field))
            if got != n:
                raise ValueError(f"{field} length {got} != equity length {n}")
        if n < 2:
            raise ValueError(f"a result needs at least 2 bars, got {n}")

    def __len__(self) -> int:
        return len(self.equity)
