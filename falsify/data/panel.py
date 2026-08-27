"""An N-asset panel. Phase 7's data layer, and the revisit of 03 Part H decision 1.

    Part H decision 1: "Universe: SPY only for v1. [...] Revisit when: the engine is
    certified and you want factor attribution."

Both conditions now hold -- G1 through G10 are green and Phase 7 is cross-sectional -- so
the universe opens. This is the first deliberate departure from a Part H decision, taken
under the condition Part H itself wrote for it rather than by re-litigating it.

**The universe is nine sector ETFs, and the choice is about bias, not convenience.**
`describe_biases()` already records that yfinance returns currently-listed tickers only,
so a universe of individual equities selected today is survivorship-biased upward over a
2015 start: the companies that failed are invisible. The nine original SPDR sector funds
have existed continuously since 1998, none was delisted, and the set was not chosen by
looking at returns. The bias does not vanish -- the sectors themselves are a chosen
partition -- but it is far weaker than picking stocks that are still around.

XLC and XLRE are deliberately excluded. XLC launched in June 2018 and XLRE in October
2015, so including either would leave a ragged panel whose early cross-section is
narrower than its later one. A cross-sectional rank over a universe that grows mid-sample
is not the same statistic before and after, and aligning on the intersection is the
honest fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from falsify.core.types import Bars
from falsify.data.loaders import DEFAULT_CACHE, DEFAULT_MANIFEST, FetchSpec, load

Matrix = NDArray[np.float64]

# The nine original SPDR select sector funds, continuously listed since 1998.
SECTORS: tuple[str, ...] = (
    "XLB",  # materials
    "XLE",  # energy
    "XLF",  # financials
    "XLI",  # industrials
    "XLK",  # technology
    "XLP",  # consumer staples
    "XLU",  # utilities
    "XLV",  # health care
    "XLY",  # consumer discretionary
)

DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-01-01"


class PanelMismatch(ValueError):
    """Series that cannot be aligned into one panel."""


@dataclass(frozen=True, slots=True)
class Panel:
    """`(T, N)` aligned closes for a universe. Frozen (B7).

    Every column covers the same timestamps. Alignment is on the intersection of the
    trading calendars rather than a union with fills, because a filled bar is a price
    nobody traded at and B6 forbids inventing one. In practice the sector funds share the
    NYSE calendar exactly, so the intersection loses nothing -- which is asserted rather
    than assumed, since a silent drop of a third of the sample would look identical to a
    clean alignment from the outside.
    """

    ts: NDArray[np.datetime64]
    close: Matrix
    tickers: tuple[str, ...]
    adjustment: str

    def __post_init__(self) -> None:
        if self.close.ndim != 2:
            raise PanelMismatch(f"expected a (T, N) matrix, got shape {self.close.shape}")
        if self.close.shape[0] != self.ts.size:
            raise PanelMismatch(f"{self.close.shape[0]} rows against {self.ts.size} timestamps")
        if self.close.shape[1] != len(self.tickers):
            raise PanelMismatch(
                f"{self.close.shape[1]} columns against {len(self.tickers)} tickers"
            )
        if self.ts.size < 2:
            raise PanelMismatch("a panel needs at least two bars")
        if not np.all(np.diff(self.ts.astype("int64")) > 0):
            raise PanelMismatch("timestamps are not strictly increasing")
        if not np.all(np.isfinite(self.close)):
            raise PanelMismatch("panel contains non-finite prices; alignment should have cut them")
        if np.any(self.close <= 0.0):
            raise PanelMismatch("panel contains non-positive prices")

    def __len__(self) -> int:
        return int(self.ts.size)

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    def returns(self) -> Matrix:
        """Simple returns, `(T-1, N)`. Row `t` is the return earned from `t` to `t+1`."""
        return np.asarray(self.close[1:] / self.close[:-1] - 1.0, dtype=np.float64)

    def column(self, ticker: str) -> NDArray[np.float64]:
        return self.close[:, self.tickers.index(ticker)]

    def describe(self) -> str:
        return (
            f"{self.n_assets} assets, {len(self)} bars, {self.ts[0]} .. {self.ts[-1]}, "
            f"adjustment={self.adjustment}"
        )


def align(series: dict[str, Bars]) -> Panel:
    """Intersect the calendars and stack the closes.

    Intersection, not union-with-forward-fill. A forward-filled price is a bar on which
    the asset did not trade, and a cross-sectional rank computed against one is ranking a
    stale number against live ones -- which flatters whichever asset was stale in a
    falling market. B6 forbids the fill; this is what forbidding it costs.
    """
    if not series:
        raise PanelMismatch("no series to align")
    tickers = tuple(sorted(series))
    adjustments = {b.adjustment for b in series.values()}
    if len(adjustments) != 1:
        raise PanelMismatch(f"mixed adjustment policies in one panel: {sorted(adjustments)}")

    common = series[tickers[0]].ts
    for ticker in tickers[1:]:
        common = np.intersect1d(common, series[ticker].ts)
    if common.size < 2:
        raise PanelMismatch(f"only {common.size} shared timestamps across {tickers}")

    columns = []
    for ticker in tickers:
        bars = series[ticker]
        index = np.searchsorted(bars.ts, common)
        if not np.array_equal(bars.ts[index], common):
            raise PanelMismatch(f"{ticker} is missing timestamps present in the intersection")
        columns.append(bars.close[index])

    return Panel(
        ts=np.asarray(common),
        close=np.column_stack(columns),
        tickers=tickers,
        adjustment=adjustments.pop(),
    )


def load_panel(
    tickers: tuple[str, ...] = SECTORS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    adjustment: str = "total_return",
    *,
    allow_network: bool = False,
    cache_dir: Path = DEFAULT_CACHE,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> Panel:
    """Cached-first, verified against the manifest, network only when asked (B1)."""
    series = {
        ticker: load(
            FetchSpec(ticker, start, end, adjustment),  # type: ignore[arg-type]
            allow_network=allow_network,
            cache_dir=cache_dir,
            manifest_path=manifest_path,
        )
        for ticker in tickers
    }
    return align(series)


__all__ = [
    "DEFAULT_END",
    "DEFAULT_START",
    "SECTORS",
    "Panel",
    "PanelMismatch",
    "align",
    "load_panel",
]
