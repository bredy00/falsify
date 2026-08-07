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
