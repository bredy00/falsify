"""Fetch, cache and validate real market data. Specified by 02 Part G, PLAYBOOK Phase 1.

The pipeline stages, and the rule each one obeys:

    parse     no fills, no drops, NaN preserved as NaN
    validate  reject non-monotonic timestamps, duplicates, non-positive prices, high < low
    align     a missing session is NaN, never interpolated
    adjust    explicit policy, never a library default, recorded in Bars.adjustment
    cache     parquet, keyed on (ticker, start, end, adjustment)
    manifest  sha256 written on fetch, verified on load

**No `bfill`, anywhere.** `02` Part A2 identifies `data.ffill().bfill()` in the reference
repository as the actual leak in the whole project -- backward fill on an interior gap
carries a future price into the past, and it lives in the file nobody reads. This module
does not fill at all: a gap stays NaN and `Bars` refuses to construct with NaN in close,
so bad data fails loudly at the boundary instead of quietly downstream.

**Network access is opt-in per call.** `allow_network` defaults to False, so a test that
accidentally reaches this code cannot reach Yahoo. B1's prohibition is lifted now that
Gate 0 is green, but the property it bought -- a suite that runs with no network, no API
key and no rate limit -- is worth keeping deliberately rather than losing by drift.

**`auto_adjust` is set explicitly and recorded.** yfinance changed that default between
versions, which silently converts a price series into a total-return series. Both are
legitimate, neither is the other, and they do not give the same Sharpe.

Known biases, to state rather than fix (`02` Part G): yfinance returns currently-listed
tickers only, so every delisted company is invisible and any universe study is
survivorship-biased upward. Prices are back-adjusted, so the series you see is not the
series that traded. Neither is fixable on free data; both belong in a limitations
section, because a reader who knows the field will check whether you know.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from falsify.core.types import Adjustment, Bars
from falsify.data.manifest import ManifestEntry, record, sha256_of, verify

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO_ROOT / "data" / "cache"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "MANIFEST.json"

# The explicit policy map. `raw` and `total_return` differ by exactly this flag, and
# writing it down here is the point -- the alternative is inheriting whatever the
# installed yfinance happens to default to this month.
AUTO_ADJUST: dict[str, bool] = {"raw": False, "total_return": True}

OHLCV = ("Open", "High", "Low", "Close", "Volume")


class DataUnavailable(RuntimeError):
    """The requested series could not be obtained."""


class ValidationFailed(ValueError):
    """The fetched series violates the Part G data contract."""


@dataclass(frozen=True, slots=True)
class FetchSpec:
    """What to fetch, and under which policy. Frozen (B7)."""

    ticker: str
    start: str
    end: str
    adjustment: Adjustment = "total_return"

    def __post_init__(self) -> None:
        if self.adjustment not in AUTO_ADJUST:
            raise ValueError(
                f"adjustment must be one of {sorted(AUTO_ADJUST)}, got {self.adjustment!r}. "
                "'split' is a valid Bars adjustment but has no yfinance mapping, so it "
                "cannot be fetched -- it would have to be constructed."
            )
        if self.start >= self.end:
            raise ValueError(f"start {self.start} must precede end {self.end}")

    @property
    def auto_adjust(self) -> bool:
        return AUTO_ADJUST[self.adjustment]

    @property
    def cache_key(self) -> str:
        """The filename. Carries every input that changes the bytes, so two different
        policies can never collide on one cache entry."""
        return f"{self.ticker}_{self.start}_{self.end}_{self.adjustment}.parquet"


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns a MultiIndex even for a single ticker; take the field level.

    Handled explicitly rather than with a bare `droplevel` so a genuinely
    multi-ticker frame raises instead of silently keeping one arbitrary column.
    """
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    tickers = frame.columns.get_level_values(-1).unique()
    if len(tickers) != 1:
        raise ValidationFailed(f"expected one ticker, frame carries {list(tickers)}")
    out = frame.copy()
    out.columns = frame.columns.get_level_values(0)
    return out


def validate_frame(frame: pd.DataFrame, spec: FetchSpec) -> pd.DataFrame:
    """The Part G validate stage. Rejects; never repairs."""
    if frame.empty:
        raise DataUnavailable(f"{spec.ticker} returned no rows for {spec.start}..{spec.end}")

    frame = _flatten_columns(frame)
    missing = [c for c in OHLCV if c not in frame.columns]
    if missing:
        raise ValidationFailed(f"missing columns {missing}; got {list(frame.columns)}")

    index = pd.DatetimeIndex(frame.index)
    if index.has_duplicates:
        dupes = index[index.duplicated()][:5]
        raise ValidationFailed(f"duplicate timestamps: {list(dupes)}")
    if not index.is_monotonic_increasing:
        raise ValidationFailed("timestamps are not monotonically increasing")

    prices = frame[["Open", "High", "Low", "Close"]]
    if (prices <= 0).to_numpy().any():
        raise ValidationFailed("non-positive price present; the series is not usable")
    if (frame["High"] < frame["Low"]).any():
        bad = int((frame["High"] < frame["Low"]).sum())
        raise ValidationFailed(f"high < low on {bad} bar(s)")
    if frame["Close"].isna().any():
        gaps = int(frame["Close"].isna().sum())
        raise ValidationFailed(
            f"{gaps} NaN close price(s). Not filled here on purpose: forward fill is a "
            "declared feature-stage policy and backward fill is a look-ahead (B6). Fix "
            "the range or the source."
        )
    return frame


def frame_to_bars(frame: pd.DataFrame, adjustment: Adjustment) -> Bars:
    """Parse stage: a validated frame becomes the immutable engine type."""
    index = pd.DatetimeIndex(frame.index)
    ts = index.tz_localize(None) if index.tz is not None else index
    return Bars(
        ts=ts.to_numpy(dtype="datetime64[ns]"),
        open=frame["Open"].to_numpy(dtype=np.float64),
        high=frame["High"].to_numpy(dtype=np.float64),
        low=frame["Low"].to_numpy(dtype=np.float64),
        close=frame["Close"].to_numpy(dtype=np.float64),
        volume=frame["Volume"].to_numpy(dtype=np.float64),
        adjustment=adjustment,
    )


def _download(spec: FetchSpec) -> tuple[pd.DataFrame, str]:
    """The only function in the project that touches the network."""
    try:
        import yfinance
    except ImportError as exc:  # pragma: no cover - depends on the install group
        raise DataUnavailable("yfinance is not installed; run `uv sync --group data`") from exc

    frame: pd.DataFrame = yfinance.download(
        spec.ticker,
        start=spec.start,
        end=spec.end,
        auto_adjust=spec.auto_adjust,  # explicit, never the library default
        progress=False,
        actions=False,
    )
    return frame, f"yfinance=={getattr(yfinance, '__version__', 'unknown')}"


def load(
    spec: FetchSpec,
    *,
    allow_network: bool = False,
    cache_dir: Path = DEFAULT_CACHE,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> Bars:
    """Return `Bars` for `spec`, from cache when possible.

    Cache hits are verified against the manifest before they are trusted, so a file
    that changed under a recorded identity raises instead of quietly producing
    different numbers. Misses require `allow_network=True`; the default refuses, which
    is what keeps an accidental call in the gate suite from reaching Yahoo.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / spec.cache_key

    if target.exists():
        entry = verify(manifest_path, cache_dir, spec.cache_key)
        frame = pd.read_parquet(target)
        if len(frame) != entry.rows:
            raise ValidationFailed(
                f"{spec.cache_key} has {len(frame)} rows, manifest says {entry.rows}"
            )
        return frame_to_bars(validate_frame(frame, spec), spec.adjustment)

    if not allow_network:
        raise DataUnavailable(
            f"{spec.cache_key} is not cached and allow_network is False. Fetch it "
            "deliberately with `uv run python scripts/fetch_data.py`, then the cached "
            "copy serves every later run offline."
        )

    frame, source = _download(spec)
    frame = validate_frame(frame, spec)
    frame.to_parquet(target)

    index = pd.DatetimeIndex(frame.index)
    record(
        manifest_path,
        spec.cache_key,
        ManifestEntry(
            sha256=sha256_of(target),
            rows=len(frame),
            fetched_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            source=source,
            adjustment=spec.adjustment,
            auto_adjust=spec.auto_adjust,
            first_ts=str(index[0]),
            last_ts=str(index[-1]),
        ),
    )
    return frame_to_bars(frame, spec.adjustment)


def cached_specs(manifest_path: Path = DEFAULT_MANIFEST) -> list[str]:
    """Keys the manifest knows about, for reporting what is available offline."""
    from falsify.data.manifest import load_manifest

    return sorted(load_manifest(manifest_path))


AdjustmentPolicy = Literal["raw", "total_return"]


def describe_biases() -> dict[str, str]:
    """The biases that are stated rather than fixed, as data rather than prose.

    Kept in code so the README and the research note quote one source instead of
    drifting apart, and so a reader can see they were acknowledged deliberately.
    """
    return {
        "survivorship": (
            "yfinance returns currently-listed tickers only, so delisted companies are "
            "invisible and any universe study is biased upward. Not fixable on free data."
        ),
        "back_adjustment": (
            "Prices are back-adjusted for splits and, under total_return, dividends. The "
            "series you see is not the series that traded."
        ),
        "single_source": (
            "One vendor, unaudited. A second source would let disagreements be detected; "
            "with one, they are invisible."
        ),
    }


__all__ = [
    "AUTO_ADJUST",
    "DEFAULT_CACHE",
    "DEFAULT_MANIFEST",
    "AdjustmentPolicy",
    "DataUnavailable",
    "FetchSpec",
    "ValidationFailed",
    "cached_specs",
    "describe_biases",
    "frame_to_bars",
    "load",
    "validate_frame",
]
