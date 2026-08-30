"""Phase 5 -- the data layer, tested entirely offline.

Every test here builds its own frames and its own temporary cache. Nothing reaches the
network, which is deliberate: B1's prohibition is lifted now that Gate 0 is green, but
the property it bought -- a suite that runs with no network, no API key and no rate
limit -- is worth keeping on purpose rather than losing by drift.

The live verification against real SPY prices lives in `tests/live/`, marked so CI
deselects it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from falsify.data.loaders import (
    AUTO_ADJUST,
    DataUnavailable,
    FetchSpec,
    ValidationFailed,
    describe_biases,
    frame_to_bars,
    load,
    validate_frame,
)
from falsify.data.manifest import (
    ManifestEntry,
    ManifestMismatch,
    load_manifest,
    record,
    sha256_of,
    verify,
    verify_all,
)

SPEC = FetchSpec("TEST", "2020-01-01", "2020-03-01", "total_return")


def good_frame(n: int = 40) -> pd.DataFrame:
    """A clean OHLCV frame that satisfies the Part G contract."""
    idx = pd.bdate_range("2020-01-02", periods=n)
    close = 100.0 * np.exp(np.cumsum(np.random.default_rng(3).normal(0.0, 0.01, n)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * 1.001,
            "Low": np.minimum(open_, close) * 0.999,
            "Close": close,
            "Volume": np.full(n, 1e6),
        },
        index=idx,
    )


# ------------------------------------------------------------------ manifest


def test_manifest_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "MANIFEST.json"
    entry = ManifestEntry(
        sha256="a" * 64,
        rows=10,
        fetched_utc="2026-08-20T00:00:00+00:00",
        source="yfinance==1.6.0",
        adjustment="total_return",
        auto_adjust=True,
        first_ts="2020-01-02",
        last_ts="2020-02-28",
    )
    record(path, "k.parquet", entry)
    assert load_manifest(path)["k.parquet"] == entry


def test_manifest_is_byte_stable_and_sorted(tmp_path: Path) -> None:
    """G10 compares bytes, so the manifest must not reorder itself between writes."""
    path = tmp_path / "MANIFEST.json"
    entry = ManifestEntry("b" * 64, 1, "t", "s", "raw", False, "a", "b")
    for key in ("z.parquet", "a.parquet", "m.parquet"):
        record(path, key, entry)
    first = path.read_bytes()
    record(path, "a.parquet", entry)  # rewrite an existing key with the same value
    assert path.read_bytes() == first, "rewriting an unchanged entry changed the bytes"
    assert list(json.loads(path.read_text())) == ["a.parquet", "m.parquet", "z.parquet"]


def test_verify_detects_a_tampered_file(tmp_path: Path) -> None:
    """The failure the manifest exists to catch: the file changed under a recorded
    identity, so every number computed from it is unverifiable."""
    cache, path = tmp_path / "cache", tmp_path / "MANIFEST.json"
    cache.mkdir()
    target = cache / "k.parquet"
    target.write_bytes(b"original bytes")
    record(path, "k.parquet", ManifestEntry(sha256_of(target), 1, "t", "s", "raw", False, "a", "b"))
    verify(path, cache, "k.parquet")  # clean

    target.write_bytes(b"tampered bytes")
    with pytest.raises(ManifestMismatch, match="does not match its manifest row"):
        verify(path, cache, "k.parquet")
    assert verify_all(path, cache) == ["k.parquet"]


def test_verify_rejects_a_file_with_no_provenance(tmp_path: Path) -> None:
    """A cache entry written by something other than the loader has unknown origin."""
    cache, path = tmp_path / "cache", tmp_path / "MANIFEST.json"
    cache.mkdir()
    (cache / "stray.parquet").write_bytes(b"x")
    with pytest.raises(ManifestMismatch, match="no manifest row"):
        verify(path, cache, "stray.parquet")


def test_verify_reports_a_stale_row(tmp_path: Path) -> None:
    cache, path = tmp_path / "cache", tmp_path / "MANIFEST.json"
    cache.mkdir()
    record(path, "gone.parquet", ManifestEntry("c" * 64, 1, "t", "s", "raw", False, "a", "b"))
    with pytest.raises(ManifestMismatch, match="file is missing"):
        verify(path, cache, "gone.parquet")


# ------------------------------------------------------------------- the spec


def test_adjustment_policy_is_explicit_and_recorded() -> None:
    """yfinance flipped this default between versions, silently turning a price series
    into a total-return series. Both are legitimate; they are not the same data."""
    assert AUTO_ADJUST == {"raw": False, "total_return": True}
    assert FetchSpec("SPY", "2020-01-01", "2021-01-01", "raw").auto_adjust is False
    assert FetchSpec("SPY", "2020-01-01", "2021-01-01", "total_return").auto_adjust is True


def test_cache_key_separates_the_policies() -> None:
    """Two policies must never collide on one cache entry, or the manifest would
    record one file's provenance for another file's numbers."""
    raw = FetchSpec("SPY", "2020-01-01", "2021-01-01", "raw").cache_key
    total = FetchSpec("SPY", "2020-01-01", "2021-01-01", "total_return").cache_key
    assert raw != total
    assert raw.endswith(".parquet") and "raw" in raw


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"adjustment": "split"}, "adjustment must be one of"),
        ({"start": "2021-01-01", "end": "2020-01-01"}, "must precede"),
        ({"start": "2020-01-01", "end": "2020-01-01"}, "must precede"),
    ],
)
def test_malformed_spec_is_rejected(kwargs: dict[str, str], match: str) -> None:
    base = {"ticker": "SPY", "start": "2020-01-01", "end": "2021-01-01"}
    with pytest.raises(ValueError, match=match):
        FetchSpec(**{**base, **kwargs})  # type: ignore[arg-type]


# -------------------------------------------------------------- validation


def test_a_clean_frame_passes_and_becomes_bars() -> None:
    frame = validate_frame(good_frame(), SPEC)
    bars = frame_to_bars(frame, "total_return")
    assert len(bars) == 40
    assert bars.adjustment == "total_return"
    assert np.all(np.diff(bars.ts) > np.timedelta64(0, "ns"))


def test_multiindex_columns_are_flattened() -> None:
    """yfinance returns a MultiIndex even for one ticker."""
    frame = good_frame()
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["SPY"]])
    assert len(validate_frame(frame, SPEC)) == 40


def test_a_multi_ticker_frame_is_rejected_rather_than_silently_narrowed() -> None:
    """Dropping a level would keep one arbitrary ticker's column and look fine."""
    frame = good_frame()
    frame.columns = pd.MultiIndex.from_product([["Open"], ["SPY", "QQQ", "IWM", "DIA", "VTI"]])
    with pytest.raises(ValidationFailed, match="expected one ticker"):
        validate_frame(frame, SPEC)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda f: f.assign(Close=np.nan), "NaN close"),
        (lambda f: f.assign(Close=-1.0), "non-positive price"),
        (lambda f: f.assign(High=f["Low"] - 1.0), "high < low"),
        (lambda f: f.iloc[::-1], "not monotonically increasing"),
        (lambda f: pd.concat([f, f.iloc[[0]]]), "duplicate timestamps"),
        (lambda f: f.drop(columns=["Volume"]), "missing columns"),
    ],
)
def test_the_contract_rejects_rather_than_repairs(mutate: object, match: str) -> None:
    """Part G's validate stage rejects; it never fixes.

    A loader that quietly repairs bad data is how a silent leak enters -- the naive
    baseline's `ffill().bfill()` being the canonical example, and it lives in the file
    nobody reads.
    """
    with pytest.raises((ValidationFailed, DataUnavailable), match=match):
        validate_frame(mutate(good_frame()), SPEC)  # type: ignore[operator]


def test_an_empty_frame_is_unavailable_not_invalid() -> None:
    with pytest.raises(DataUnavailable, match="no rows"):
        validate_frame(pd.DataFrame(), SPEC)


def test_nan_close_is_not_filled() -> None:
    """B6: no bfill anywhere, and no silent ffill in the loader either.

    Forward fill is a declared feature-stage policy with a stated bar limit. Doing it
    here, invisibly, is how a gap becomes a number nobody chose.
    """
    frame = good_frame()
    frame.iloc[10, frame.columns.get_loc("Close")] = np.nan
    with pytest.raises(ValidationFailed, match="Not filled here on purpose"):
        validate_frame(frame, SPEC)


# ------------------------------------------------------------------- loading


def test_load_refuses_the_network_by_default(tmp_path: Path) -> None:
    """The guard that keeps an accidental call in the gate suite from reaching Yahoo."""
    with pytest.raises(DataUnavailable, match="allow_network is False"):
        load(SPEC, cache_dir=tmp_path / "cache", manifest_path=tmp_path / "M.json")


def test_a_cached_file_without_a_manifest_row_is_refused(tmp_path: Path) -> None:
    """Provenance is required, not optional: an unrecorded cache file could be anything."""
    cache = tmp_path / "cache"
    cache.mkdir()
    good_frame().to_parquet(cache / SPEC.cache_key)
    with pytest.raises(ManifestMismatch, match="no manifest row"):
        load(SPEC, cache_dir=cache, manifest_path=tmp_path / "M.json")


def test_a_cache_hit_is_verified_before_it_is_trusted(tmp_path: Path) -> None:
    """The round trip: write, record, load, then tamper and watch it refuse."""
    cache, manifest = tmp_path / "cache", tmp_path / "M.json"
    cache.mkdir()
    target = cache / SPEC.cache_key
    frame = good_frame()
    frame.to_parquet(target)
    record(
        manifest,
        SPEC.cache_key,
        ManifestEntry(sha256_of(target), len(frame), "t", "test", "total_return", True, "a", "b"),
    )

    bars = load(SPEC, cache_dir=cache, manifest_path=manifest)
    assert len(bars) == len(frame)

    good_frame(39).to_parquet(target)  # same shape of file, different bytes
    with pytest.raises(ManifestMismatch):
        load(SPEC, cache_dir=cache, manifest_path=manifest)


def test_row_count_disagreement_is_caught(tmp_path: Path) -> None:
    """A digest match with a row-count mismatch cannot happen, but the check is cheap
    and the two are recorded independently."""
    cache, manifest = tmp_path / "cache", tmp_path / "M.json"
    cache.mkdir()
    target = cache / SPEC.cache_key
    frame = good_frame()
    frame.to_parquet(target)
    record(
        manifest,
        SPEC.cache_key,
        ManifestEntry(
            sha256_of(target), 999, len(frame) * "t", "test", "total_return", True, "a", "b"
        ),
    )
    with pytest.raises(ValidationFailed, match="manifest says 999"):
        load(SPEC, cache_dir=cache, manifest_path=manifest)


def test_the_known_biases_are_stated() -> None:
    """`02` Part G says to state these rather than fix them, because a reader who knows
    the field will check whether the author knows."""
    biases = describe_biases()
    assert set(biases) == {"survivorship", "back_adjustment", "single_source"}
    assert "delisted" in biases["survivorship"]
    assert "back-adjusted" in biases["back_adjustment"]
