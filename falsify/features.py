"""Causal feature primitives.

Every function here maps a series to a series where element `t` is a function of
elements `[0, t]` only, and returns NaN before its window is full. That is the
contract G1 verifies; nothing in this module is allowed to look forward, and the
NaN prefix must depend only on the declared window length rather than on anything
the data happens to contain.

Windows are computed with `sliding_window_view`, not with a cumulative sum. That
choice is load-bearing for G2: a cumsum-based rolling mean gives bitwise different
answers when evaluated on a prefix versus on the whole series, because the running
total accumulates differently, and the twin engines would then disagree in the last
bits for no reason anyone could find. Windowed means over identical elements are
bitwise identical.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

Series = NDArray[np.float64]


def _empty_like(x: Series) -> Series:
    return np.full(len(x), np.nan)


def rolling_mean(x: Series, window: int) -> Series:
    """Mean of the `window` values ending at t. NaN before the first full window."""
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    out = _empty_like(x)
    if window <= len(x):
        out[window - 1 :] = sliding_window_view(x, window).mean(axis=1)
    return out


def rolling_std(x: Series, window: int) -> Series:
    """Sample standard deviation (ddof=1) of the `window` values ending at t."""
    if window < 2:
        raise ValueError(f"window must be at least 2 for a sample std, got {window}")
    out = _empty_like(x)
    if window <= len(x):
        out[window - 1 :] = sliding_window_view(x, window).std(axis=1, ddof=1)
    return out


def shift_one(x: Series) -> Series:
    """Move every value one bar later, so `out[t] == x[t-1]`.

    A claim, not a proof: this expresses the intent that a decision is acted on no
    earlier than the following bar. G1 is what proves the claim, and the engine's
    convention lag is what actually enforces the delay.
    """
    out = _empty_like(x)
    out[1:] = x[:-1]
    return out
