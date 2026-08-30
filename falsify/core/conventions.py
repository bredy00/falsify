"""Execution conventions. Specified by 02-ENGINE-SPEC.md Part D.

A convention answers two questions, and both must be answered the same way by
both engines (B5): *at what price does a trade execute*, and *how many bars after
the signal*. Everything else about execution follows from that pair.

    convention        fill price   lag   honesty
    close_to_close    close        1     optimistic -- trades the price just observed
    next_open         open         2     realistic, and the default
    next_close        close        2     conservative, full overnight gap risk

The lag is the number of bars between the signal index and the first return index
that signal earns. Reading `close_to_close`: a signal decided at the close of bar
s is filled at that same close, so it earns close[s] -> close[s+1], i.e. return
index s+1, so lag = 1. Reading `next_open`: a signal decided at the close of bar
s is filled at open[s+1] and held to open[s+2], earning open-indexed return s+2,
so lag = 2.

`close_to_close` with lag 1 is what the naive baseline implements via
`position = raw_signal.shift(1)`. It is defensible on daily data and it flatters
the result, which is why the default here is `next_open`.

Part D also asks for the three-way comparison figure. That belongs to the
reporting phase; the spread between these curves is the execution-assumption risk
and quantifying it is worth more than any single number.
"""

from __future__ import annotations

from typing import Final, Literal, get_args

import numpy as np
from numpy.typing import NDArray

from falsify.core.types import Bars

Convention = Literal["close_to_close", "next_open", "next_close"]

DEFAULT_CONVENTION: Final[Convention] = "next_open"

# convention -> (Bars field used as the fill price, signal-to-return lag in bars)
_SPEC: Final[dict[Convention, tuple[str, int]]] = {
    "close_to_close": ("close", 1),
    "next_open": ("open", 2),
    "next_close": ("close", 2),
}

CONVENTIONS: Final[tuple[Convention, ...]] = get_args(Convention)


def _resolve(convention: Convention) -> tuple[str, int]:
    try:
        return _SPEC[convention]
    except KeyError:
        raise ValueError(
            f"unknown convention {convention!r}; expected one of {list(_SPEC)}"
        ) from None


def signal_lag(convention: Convention) -> int:
    """Bars between the signal index and the first return index it earns."""
    return _resolve(convention)[1]


def fill_prices(bars: Bars, convention: Convention) -> NDArray[np.float64]:
    """The price series trades execute at under this convention."""
    field = _resolve(convention)[0]
    prices: NDArray[np.float64] = getattr(bars, field)
    return prices
