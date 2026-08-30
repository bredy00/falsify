"""Risk-free rate series. 03 Part H decision 4.

    "Risk-free: constant, set to the period mean of 3-month T-bills. The cash-yield
     term must exist (that's the naive baseline's omission), but a full T-bill path is
     Phase 8 polish. Record the constant used in the manifest."

So this fetches the path, stores it under the same SHA256 discipline as the price cache,
and reduces it to the one constant the decision calls for. The path is kept rather than
discarded because the constant is only defensible if the thing it averages is inspectable,
and because Phase 8 will want the path itself.

Separate from `loaders.py` on purpose. That module is about OHLCV bars and everything in
it -- `validate_frame`, `frame_to_bars`, the high/low checks -- assumes a price series.
A yield is not a price: it has no high or low worth validating, it is quoted in percent
rather than currency, and near-zero values that would be a red flag for a price are
ordinary for a 2020-2021 T-bill. Forcing it through the bars path would mean loosening
checks that exist for good reasons on the data they were written for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from falsify.data.loaders import DEFAULT_CACHE, DEFAULT_MANIFEST, DataUnavailable, ValidationFailed
from falsify.data.manifest import ManifestEntry, record, sha256_of, verify

# ^IRX is the CBOE 13-week Treasury bill yield -- the 3-month T-bill Part H names.
DEFAULT_TICKER = "^IRX"

# ^IRX is quoted in percent: 5.25 means 5.25% per annum.
PERCENT = 100.0


@dataclass(frozen=True, slots=True)
class RateSpec:
    """What rate series to fetch. Frozen (B7)."""

    ticker: str
    start: str
    end: str

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError(f"start {self.start} must precede end {self.end}")

    @property
    def cache_key(self) -> str:
        return f"{self.ticker}_{self.start}_{self.end}_rate.parquet"


@dataclass(frozen=True, slots=True)
class RiskFreeRate:
    """A risk-free path and the constant Part H decision 4 asks for. Frozen (B7)."""

    ts: NDArray[np.datetime64]
    annual_rate: NDArray[np.float64]  # decimal, so 0.0525 is 5.25%
    ticker: str

    def __len__(self) -> int:
        return int(self.annual_rate.size)

    @property
    def period_mean(self) -> float:
        """The constant. Decimal annual rate, averaged over the window."""
        return float(np.mean(self.annual_rate))

    def per_bar(self, bars_per_year: int = 252) -> float:
        """The constant as a per-observation rate (B8), for `sharpe(risk_free_per_bar=)`.

        Compounded, not divided: `(1 + r)^(1/252) - 1`. At 2% the difference from naive
        division is about 1e-7 per bar, which is immaterial for a Sharpe -- but the
        naive form is wrong in a way that grows with the rate, and this costs nothing.
        """
        return float((1.0 + self.period_mean) ** (1.0 / bars_per_year) - 1.0)

    def describe(self) -> str:
        return (
            f"{self.ticker}: {len(self)} observations, "
            f"{self.ts[0]} .. {self.ts[-1]}, "
            f"period mean {self.period_mean:.4%} annual "
            f"({self.per_bar():.3e} per bar), "
            f"range {self.annual_rate.min():.4%}..{self.annual_rate.max():.4%}"
        )


def validate_rate_frame(frame: pd.DataFrame, spec: RateSpec) -> pd.DataFrame:
    """Reject a rate series that cannot be an annual yield path.

    Deliberately not the price checks. Non-positive is allowed -- the 3-month bill
    yielded essentially zero through 2020-2021 and printing 0.0 is correct, not a
    corruption -- but NaN, non-monotonic timestamps and duplicates are still failures,
    and a yield above 100% or below -10% is a units error rather than a market event.
    """
    if frame.empty:
        raise ValidationFailed(f"{spec.ticker}: no rows returned for {spec.start}..{spec.end}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.droplevel(1, axis=1)
    if "Close" not in frame.columns:
        raise ValidationFailed(f"{spec.ticker}: no Close column, got {list(frame.columns)}")

    out = frame[["Close"]].dropna()
    if out.empty:
        raise ValidationFailed(f"{spec.ticker}: every Close was NaN")

    index = pd.DatetimeIndex(out.index)
    if not index.is_monotonic_increasing:
        raise ValidationFailed(f"{spec.ticker}: timestamps are not monotonic")
    if index.has_duplicates:
        raise ValidationFailed(f"{spec.ticker}: duplicate timestamps")

    values = out["Close"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValidationFailed(f"{spec.ticker}: non-finite yields survived the dropna")
    if np.any(values > PERCENT) or np.any(values < -10.0):
        raise ValidationFailed(
            f"{spec.ticker}: yields outside [-10, 100] percent "
            f"({values.min():.3f}..{values.max():.3f}); this is a units error, not a market"
        )
    return out


def _download(spec: RateSpec) -> tuple[pd.DataFrame, str]:
    """Network access. `auto_adjust` is meaningless for a yield and passed as False
    explicitly rather than left to the library default (02 Part G)."""
    try:
        import yfinance
    except ImportError as exc:  # pragma: no cover - depends on the install group
        raise DataUnavailable("yfinance is not installed; run `uv sync --group data`") from exc

    frame: pd.DataFrame = yfinance.download(
        spec.ticker,
        start=spec.start,
        end=spec.end,
        auto_adjust=False,
        progress=False,
        actions=False,
    )
    return frame, f"yfinance=={getattr(yfinance, '__version__', 'unknown')}"


def load_rate(
    spec: RateSpec,
    *,
    allow_network: bool = False,
    cache_dir: Path = DEFAULT_CACHE,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> RiskFreeRate:
    """Cached-first, verified against the manifest, network only when asked.

    Same contract as `loaders.load`: offline by default (B1), and a cached file whose
    digest does not match its manifest row is an error rather than a warning.
    """
    path = cache_dir / spec.cache_key
    if path.exists():
        verify(manifest_path, cache_dir, spec.cache_key)
        return _to_rate(pd.read_parquet(path), spec)

    if not allow_network:
        raise DataUnavailable(
            f"{spec.cache_key} is not cached and allow_network=False. "
            "Populate it with `uv run --group data python scripts/fetch_data.py`."
        )

    frame, source = _download(spec)
    validated = validate_rate_frame(frame, spec)
    cache_dir.mkdir(parents=True, exist_ok=True)
    validated.to_parquet(path)

    index = pd.DatetimeIndex(validated.index)
    record(
        manifest_path,
        spec.cache_key,
        ManifestEntry(
            sha256=sha256_of(path),
            rows=len(validated),
            fetched_utc=datetime.now(UTC).isoformat(),
            source=source,
            adjustment="rate",
            auto_adjust=False,
            first_ts=str(index[0]),
            last_ts=str(index[-1]),
        ),
    )
    return _to_rate(validated, spec)


def _to_rate(frame: pd.DataFrame, spec: RateSpec) -> RiskFreeRate:
    validated = validate_rate_frame(frame, spec)
    index = pd.DatetimeIndex(validated.index)
    return RiskFreeRate(
        ts=index.to_numpy(dtype="datetime64[ns]"),
        annual_rate=validated["Close"].to_numpy(dtype=np.float64) / PERCENT,
        ticker=spec.ticker,
    )


__all__ = [
    "DEFAULT_TICKER",
    "PERCENT",
    "RateSpec",
    "RiskFreeRate",
    "load_rate",
    "validate_rate_frame",
]
