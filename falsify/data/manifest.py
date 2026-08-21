"""The data manifest. Specified by 02-ENGINE-SPEC.md Part G.

Every cached file gets a row recording its sha256, its shape, when it was fetched and
under exactly which adjustment policy. Written on fetch, verified on load, and shipped
in the repository so a reader can check that the numbers came from the data claimed.

This is what G10 reads. A mismatch fails the build rather than producing quietly
different numbers, which is the failure mode that matters: a re-fetch six months later
returns a *slightly* different history -- a restated dividend, a corrected split, a
vendor backfill -- and every downstream Sharpe shifts with no error and no notice.

`auto_adjust` gets its own field for the same reason. yfinance changed that default
between versions, and a silent flip converts a price series into a total-return series.
Both are legitimate; they are not the same data and they do not give the same Sharpe.
Recording the resolved value makes the difference detectable instead of invisible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One cached file's provenance. Frozen (B7)."""

    sha256: str
    rows: int
    fetched_utc: str
    source: str
    adjustment: str
    auto_adjust: bool
    first_ts: str
    last_ts: str

    def to_json(self) -> dict[str, Any]:
        return dict(asdict(self))

    @staticmethod
    def from_json(raw: dict[str, Any]) -> ManifestEntry:
        return ManifestEntry(
            sha256=str(raw["sha256"]),
            rows=int(raw["rows"]),
            fetched_utc=str(raw["fetched_utc"]),
            source=str(raw["source"]),
            adjustment=str(raw["adjustment"]),
            auto_adjust=bool(raw["auto_adjust"]),
            first_ts=str(raw["first_ts"]),
            last_ts=str(raw["last_ts"]),
        )


class ManifestMismatch(RuntimeError):
    """A cached file does not match what the manifest recorded.

    Raised rather than warned. The whole point of the manifest is that silently
    different data is the failure being prevented, so a soft warning would defeat it.
    """


def sha256_of(path: Path) -> str:
    """Streaming sha256, so a large cache file does not have to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    """Read the manifest, or an empty mapping if it does not exist yet."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {key: ManifestEntry.from_json(value) for key, value in raw.items()}


def save_manifest(path: Path, entries: dict[str, ManifestEntry]) -> None:
    """Write the manifest with sorted keys and a trailing newline.

    Sorted and indented so a diff shows what actually changed rather than a
    reordering, and so the file is byte-stable across runs -- G10 compares bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: entries[key].to_json() for key in sorted(entries)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record(path: Path, key: str, entry: ManifestEntry) -> None:
    """Add or replace one entry, leaving the rest untouched."""
    entries = load_manifest(path)
    entries[key] = entry
    save_manifest(path, entries)


def verify(path: Path, cache_dir: Path, key: str) -> ManifestEntry:
    """Check a cached file against its manifest row, or raise.

    Three distinct failures, each with its own message, because the fix differs: no
    manifest row at all (the cache was populated outside the loader), a missing file
    (the row is stale), and a digest mismatch (the file changed under a recorded
    identity, which is the dangerous one).
    """
    entries = load_manifest(path)
    if key not in entries:
        raise ManifestMismatch(
            f"{key} is cached but has no manifest row. It was written by something "
            "other than the loader, so its provenance is unknown -- delete it and re-fetch."
        )
    entry = entries[key]
    target = cache_dir / key
    if not target.exists():
        raise ManifestMismatch(f"{key} has a manifest row but the file is missing")

    actual = sha256_of(target)
    if actual != entry.sha256:
        raise ManifestMismatch(
            f"{key} does not match its manifest row.\n"
            f"  recorded {entry.sha256}\n"
            f"  actual   {actual}\n"
            "The cached data has changed since it was recorded. Every number computed "
            "from it is now unverifiable; re-fetch and re-run rather than trusting it."
        )
    return entry


def verify_all(path: Path, cache_dir: Path) -> list[str]:
    """Verify every manifest row, returning the keys that failed rather than raising.

    The batch form is for a health check that wants to report all the damage at once;
    the loader uses `verify` and raises on the file it actually needs.
    """
    failures = []
    for key in load_manifest(path):
        try:
            verify(path, cache_dir, key)
        except ManifestMismatch:
            failures.append(key)
    return failures


__all__ = [
    "ManifestEntry",
    "ManifestMismatch",
    "load_manifest",
    "record",
    "save_manifest",
    "sha256_of",
    "verify",
    "verify_all",
]
