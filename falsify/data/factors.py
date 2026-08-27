"""Fama-French factors from the Ken French data library. PLAYBOOK Phase 7.

    "Factor attribution: regress strategy excess returns on Mkt-RF, SMB, HML, UMD from
     Ken French."

A second data source, and it behaves nothing like the first. yfinance returns a tidy
frame; this is a ZIP containing a CSV with a prose preamble, a table, a blank line, then
a *second* table of annual figures under the same header. Parsing it by `read_csv` and
hoping is how a daily series quietly acquires a hundred annual rows.

**Units.** French publishes factors in PERCENT: a market return of 1% is `1.0`, not
`0.01`. Everything in this project is in decimal, so the conversion happens here, once,
at the boundary -- and `validate_factors` refuses a frame that looks like it has already
been converted, because a silent factor-of-100 in a regression produces betas that are
off by 100 and an alpha that looks enormous.

**The risk-free rate.** French ships `RF` alongside the factors, and it is used rather
than the `^IRX` series 03 Part H decision 4 fetched. Both are three-month bill rates and
they differ only in construction, but the factors are excess of *French's* RF, so mixing
sources would leave a small, systematic residual in alpha. The `^IRX` series remains what
`CostModel.cash_yield_annual` is set from; the two are for different jobs.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from falsify.data.loaders import DEFAULT_CACHE, DEFAULT_MANIFEST, DataUnavailable, ValidationFailed
from falsify.data.manifest import ManifestEntry, record, sha256_of, verify

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

# The three-factor set and momentum arrive in separate archives.
THREE_FACTOR = "F-F_Research_Data_Factors_daily_CSV.zip"
MOMENTUM = "F-F_Momentum_Factor_daily_CSV.zip"

FACTOR_NAMES: tuple[str, ...] = ("Mkt-RF", "SMB", "HML", "UMD")
PERCENT = 100.0

# French's own missing-data sentinels.
SENTINELS = (-99.99, -999.0)


@dataclass(frozen=True, slots=True)
class Factors:
    """Daily factor returns in DECIMAL, plus the risk-free rate. Frozen (B7)."""

    ts: NDArray[np.datetime64]
    values: NDArray[np.float64]  # (T, 4) -- Mkt-RF, SMB, HML, UMD
    rf: NDArray[np.float64]
    names: tuple[str, ...] = FACTOR_NAMES

    def __len__(self) -> int:
        return int(self.ts.size)

    def column(self, name: str) -> NDArray[np.float64]:
        return self.values[:, self.names.index(name)]

    def describe(self) -> str:
        annual = self.values.mean(axis=0) * 252
        parts = "  ".join(f"{n} {v:+.2%}/yr" for n, v in zip(self.names, annual, strict=True))
        return (
            f"{len(self)} days, {self.ts[0]} .. {self.ts[-1]}, "
            f"RF {self.rf.mean() * 252:.2%}/yr  |  {parts}"
        )


def parse_french_csv(text: str, value_columns: int) -> pd.DataFrame:
    """Pull the daily table out of a Ken French CSV.

    The file is a preamble, a daily table, a blank line, and often a second table of
    monthly or annual figures under the same column names. Rows are selected by shape --
    an eight-digit date followed by exactly `value_columns` numbers -- rather than by
    slicing at a fixed offset, because the preamble length changes between releases and a
    fixed offset would silently start mid-table one day.

    Stopping at the first blank line after the table has begun is what keeps the annual
    block out. Without it the frame gains rows dated `1927` and the regression quietly
    mixes two frequencies.
    """
    rows: list[list[float]] = []
    started = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if started:
                break  # the daily table has ended; what follows is a different frequency
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != value_columns + 1 or len(parts[0]) != 8 or not parts[0].isdigit():
            if started:
                break
            continue
        try:
            values = [float(p) for p in parts[1:]]
        except ValueError:
            if started:
                break
            continue
        started = True
        rows.append([float(parts[0]), *values])

    if not rows:
        raise ValidationFailed(
            f"no daily rows found; expected an 8-digit date plus {value_columns} numbers"
        )
    frame = pd.DataFrame(rows)
    frame[0] = pd.to_datetime(frame[0].astype(int).astype(str), format="%Y%m%d")
    return frame.set_index(0)


def validate_factors(values: NDArray[np.float64], label: str) -> None:
    """Refuse a factor block that is not in percent, or that carries French's sentinels.

    The percent check is the one that matters. A daily factor in percent has a standard
    deviation near 1.0; in decimal it is near 0.01. If this file were ever shipped already
    converted, dividing by 100 again would scale every beta by 100 and inflate alpha to
    something that looks like a discovery.
    """
    if not np.all(np.isfinite(values)):
        raise ValidationFailed(f"{label}: non-finite values survived parsing")
    for sentinel in SENTINELS:
        if np.any(np.isclose(values, sentinel)):
            raise ValidationFailed(f"{label}: contains French's missing-data sentinel {sentinel}")
    spread = float(np.std(values))
    if not 0.05 < spread < 20.0:
        raise ValidationFailed(
            f"{label}: standard deviation {spread:.4f} is not consistent with percent units. "
            "Daily factors in percent sit near 1.0; near 0.01 means the file is already "
            "decimal and dividing again would scale every beta by 100."
        )


def validate_rate(rf: NDArray[np.float64]) -> None:
    """The risk-free column, checked on its LEVEL rather than its spread.

    `validate_factors` refuses anything whose standard deviation is far from 1, which is
    right for a daily factor return in percent and wrong for a rate. RF is a near-constant
    small number -- 1.76% a year is 0.007% a day -- so its measured spread is 0.021, and
    running the factor check against it failed on real data the first time this fetched.

    The level is what discriminates here. A daily rate in percent sits in roughly
    [0, 0.05]; the same series already converted to decimal would sit near 7e-5, three
    orders of magnitude down, and dividing again would put a rounding error where the
    risk-free rate belongs.
    """
    if not np.all(np.isfinite(rf)):
        raise ValidationFailed("RF: non-finite values survived parsing")
    for sentinel in SENTINELS:
        if np.any(np.isclose(rf, sentinel)):
            raise ValidationFailed(f"RF: contains French's missing-data sentinel {sentinel}")
    if np.any(rf < -0.01) or np.any(rf > 0.10):
        raise ValidationFailed(
            f"RF: daily values range {rf.min():.5f}..{rf.max():.5f}, outside the "
            "[-0.01, 0.10] percent-per-day a three-month bill can plausibly pay"
        )
    annual = float(np.mean(rf)) * 252.0
    if not -0.5 < annual < 15.0:
        raise ValidationFailed(
            f"RF: implies {annual:.2f}% a year, which is not a plausible bill rate. "
            "A value near 0.02 would mean the file is already decimal."
        )


def _download(name: str) -> bytes:
    """Network access. The only function here that leaves the machine."""
    url = f"{BASE_URL}/{name}"
    try:
        request = Request(url, headers={"User-Agent": "falsify/research (offline-first cache)"})
        with urlopen(request, timeout=60) as response:
            return bytes(response.read())
    except Exception as exc:  # pragma: no cover - network shape varies
        raise DataUnavailable(f"could not fetch {url}: {type(exc).__name__}: {exc}") from exc


def _extract(payload: bytes, value_columns: int) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValidationFailed(f"expected one CSV in the archive, found {members}")
        return parse_french_csv(archive.read(members[0]).decode("latin-1"), value_columns)


def load_factors(
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    *,
    allow_network: bool = False,
    cache_dir: Path = DEFAULT_CACHE,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> Factors:
    """Cached-first, verified against the manifest, network only when asked (B1)."""
    key = f"FF4_{start}_{end}_daily.parquet"
    path = cache_dir / key

    if path.exists():
        verify(manifest_path, cache_dir, key)
        return _to_factors(pd.read_parquet(path))

    if not allow_network:
        raise DataUnavailable(
            f"{key} is not cached and allow_network=False. "
            "Populate it with `uv run --group data python scripts/fetch_data.py`."
        )

    three = _extract(_download(THREE_FACTOR), 4)  # Mkt-RF, SMB, HML, RF
    momentum = _extract(_download(MOMENTUM), 1)  # Mom
    three.columns = ["Mkt-RF", "SMB", "HML", "RF"]
    momentum.columns = ["UMD"]

    merged = three.join(momentum, how="inner")
    merged = merged.loc[(merged.index >= start) & (merged.index < end)]
    if merged.empty:
        raise ValidationFailed(f"no factor rows between {start} and {end}")

    validate_factors(merged[["Mkt-RF", "SMB", "HML", "UMD"]].to_numpy(dtype=np.float64), "factors")
    validate_rate(merged["RF"].to_numpy(dtype=np.float64))

    cache_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path)
    record(
        manifest_path,
        key,
        ManifestEntry(
            sha256=sha256_of(path),
            rows=len(merged),
            fetched_utc=datetime.now(UTC).isoformat(),
            source="Ken French data library (Dartmouth)",
            adjustment="factor_percent",
            auto_adjust=False,
            first_ts=str(merged.index[0]),
            last_ts=str(merged.index[-1]),
        ),
    )
    return _to_factors(merged)


def _to_factors(frame: pd.DataFrame) -> Factors:
    index = pd.DatetimeIndex(frame.index)
    return Factors(
        ts=index.to_numpy(dtype="datetime64[ns]"),
        values=frame[list(FACTOR_NAMES)].to_numpy(dtype=np.float64) / PERCENT,
        rf=frame["RF"].to_numpy(dtype=np.float64) / PERCENT,
    )


def align_to(factors: Factors, ts: NDArray[np.datetime64]) -> tuple[Factors, NDArray[np.int64]]:
    """Restrict the factors to timestamps present in both, and say which rows survived.

    Returns the index into `ts` as well as the factors, so a caller can cut its own return
    series to the same bars. Intersecting rather than reindexing keeps B6: a factor value
    forward-filled onto a day French did not publish is a number nobody computed.
    """
    common = np.intersect1d(factors.ts, ts)
    if common.size < 3:
        raise ValidationFailed(f"only {common.size} shared dates between factors and returns")
    factor_rows = np.searchsorted(factors.ts, common)
    return (
        Factors(ts=common, values=factors.values[factor_rows], rf=factors.rf[factor_rows]),
        np.searchsorted(ts, common).astype(np.int64),
    )


__all__ = [
    "BASE_URL",
    "FACTOR_NAMES",
    "PERCENT",
    "Factors",
    "align_to",
    "load_factors",
    "parse_french_csv",
    "validate_factors",
    "validate_rate",
]
