"""The trials ledger. Invariant B3, specified by 01 Part C.

    B3: "The trials ledger is append-only and machine-written. Never hand-set `N`.
         Never delete a row. A bug fix marks `superseded_by`."

    01 Part C rule 3: "`N` is read from the ledger by counting non-superseded rows
         matching the reporting scope. It is never a hand-typed constant."

01 Part C on what this file is for: "the difference between a project that discusses
backtest overfitting and a project that measures its own."

Design and the two conflicts it resolves are in
`docs/superpowers/specs/2026-08-21-trials-ledger-design.md`. In short:

  - Rule 1 says EVERY engine invocation writes a row, and the gate suite invokes the
    engines hundreds of thousands of times. Content-addressing resolves that with no
    exemption: G1's tau-test recomputing one configuration 10,000 times is one distinct
    trial, which is what it actually is. `N` counts distinct ids.

  - A `uuid4` trial id, as Part C writes it, would have broken G10 -- `n_trials_raw`
    is read from a file every run appends to, so `N` would grow and `metrics.json`
    would differ between runs. The id is a content hash instead, so reruns are
    idempotent.

Departures from Part C, each deliberate:

  1. `trial_id` is `sha256(canonical)[:16]`, not `uuid4`. See above.
  2. `sharpe_se` is added. B2: no performance number without an error bar, and Part C's
     record carries a bare `sharpe`, which is the same defect as a bare Sharpe in a
     README.
  3. `recording` names the policy that produced the row, so a later policy change is
     interpretable rather than silently mixing bases.
  4. `series_digest` is added, and it is load-bearing. Part C identifies data by
     `universe`, but `Bars` carries no ticker, and `data_manifest_hash` covers the whole
     manifest rather than the one series used -- so two different instruments over the
     same dates would otherwise hash identically. A digest of the close prices makes
     identity exact without threading a symbol through the engine.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from falsify.data.manifest import sha256_of

ID_LENGTH = 16  # hex characters; 64 bits, ample at the N this project reaches
UNKNOWN = "unknown"

# Where the shipped ledger lives. Spec: "the file ships in the repository where a reader
# can verify `N` for themselves" (Part C rule 4). Relative to the repository root.
LEDGER_PATH = Path("data/trials.jsonl")

ParamValue = str | int | float | bool | None
Params = dict[str, ParamValue]


class LedgerError(RuntimeError):
    """The ledger cannot record a trial faithfully."""


class Recording(Enum):
    """What the ledger does with an observation.

    The `ledger` argument to the engines is not optional -- that is what makes bypass
    impossible -- but what the ledger *does* with an observation is configurable, is
    recorded on every row, and is revisable later through supersession rather than by
    rewriting history.
    """

    TRIALS = "trials"  # one row per distinct content address (the default)
    ALL = "all"  # every observation, duplicates included
    NONE = "none"  # observe and count, persist nothing


# --------------------------------------------------------------------------------------
# Provenance primitives. `reporting` imports these rather than keeping its own copies.
# --------------------------------------------------------------------------------------


@cache
def git_sha(repo: Path | None = None) -> str:
    """`<sha>` or `<sha>-dirty`.

    **Cached, and it has to be.** Each call spawns two subprocesses, and B3 rule 1 puts
    this on the path of every engine invocation -- G1's tau-test alone runs about ten
    thousand. Uncached, the gate suite went from 33s to over ten minutes and was still
    running when it was killed. Two process spawns are nothing; twenty thousand are the
    whole build.

    Caching is also the more correct semantics: one process run is one code state, so a
    commit landing mid-run should not split that run's trials across two identities.

    The suffix is part of the identity, not a warning printed elsewhere: a trial run on
    uncommitted code hashes differently from one run on shipped code, which is exactly
    what should happen. Confusing the two is how a result outlives the code that made it.

    **The trials ledger is excluded from the dirty check**, and it has to be. The ledger
    is tracked, so appending a trial to it would mark the tree dirty, which would change
    this SHA, which would give the *next* run a different `trial_id` for the identical
    configuration -- so `N` would grow on every run and two runs would disagree, taking
    G10 with them. The exclusion is not a convenience: `git_sha` identifies the state of
    the *code*, and the ledger is output the code produced. A run recording what it did
    is not a change to what it is.
    """
    root = repo or Path(__file__).resolve().parent.parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", f":(exclude){LEDGER_PATH.as_posix()}"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return UNKNOWN
    return f"{sha}-dirty" if dirty else sha


@cache
def manifest_hash(manifest_path: Path) -> str:
    """SHA256 of the data manifest -- one value standing for every cached input.

    Cached for the same reason as `git_sha`: it is a file read plus a digest on the path
    of every engine invocation. The manifest is written by `fetch_data.py`, which is a
    separate process, so it cannot change under a running gate suite.
    """
    return sha256_of(manifest_path) if manifest_path.exists() else UNKNOWN


def series_digest(values: NDArray[np.float64]) -> str:
    """Content address of a price series. Byte-exact and cheap.

    `np.ascontiguousarray` first: a sliced or transposed view has the same values in a
    different memory layout, and hashing raw bytes would call those two different series.
    """
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:ID_LENGTH]


# --------------------------------------------------------------------------------------
# Identity.
# --------------------------------------------------------------------------------------


def canonical_value(value: object) -> ParamValue:
    """Normalise one parameter so equal configurations hash equally.

    `bool` is checked before `int` and that ordering is load-bearing: Python makes
    `True` an instance of `int`, so the obvious ordering silently records `True` as `1`,
    and `1` is not the same trial as `True`.

    Integers and floats both normalise through `float`, so `MACrossover(20, 50)` and
    `MACrossover(20.0, 50.0)` are one configuration rather than two. Without that a
    caller who writes one lookback as a float inflates `N`, and inflating `N` makes the
    deflated Sharpe look more rigorous while describing a search that never happened.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):  # BEFORE int -- see above
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise LedgerError(
        f"parameter of type {type(value).__name__} cannot be canonicalised. A trial that "
        "cannot be identified must not be recorded as one; give the strategy a scalar "
        "parameter or extend `canonical_value`."
    )


def canonical_params(params: Mapping[str, object]) -> Params:
    return {str(k): canonical_value(v) for k, v in sorted(params.items())}


def compute_trial_id(
    *,
    git_sha: str,
    data_manifest_hash: str,
    series_digest: str,
    strategy: str,
    params: Params,
    universe: tuple[str, ...],
    date_range: tuple[str, str],
    cost_bps: float,
) -> str:
    """`sha256(canonical)[:16]` over everything that defines a trial.

    `timestamp` is deliberately absent. It is provenance, not identity; including it
    would make every rerun a new trial and reopen the G10 conflict this design exists to
    avoid.

    16 hex characters is 64 bits. At the `N` this project reaches -- thousands, not
    billions -- collision probability is negligible, and the full digest is recoverable
    by recomputation if it is ever wanted.
    """
    defining = {
        "git_sha": git_sha,
        "data_manifest_hash": data_manifest_hash,
        "series_digest": series_digest,
        "strategy": strategy,
        "params": canonical_params(dict(params)),
        "universe": list(universe),
        "date_range": list(date_range),
        "cost_bps": float(cost_bps),
    }
    canonical = json.dumps(defining, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:ID_LENGTH]


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One configuration evaluated. 01 Part C's record. Frozen (B7)."""

    trial_id: str
    timestamp: str  # ISO 8601 UTC -- provenance, NOT part of the id
    git_sha: str
    data_manifest_hash: str
    series_digest: str
    strategy: str
    params: Params
    universe: tuple[str, ...]
    date_range: tuple[str, str]
    cost_bps: float
    sharpe: float  # annualised, at the reporting boundary (B8)
    sharpe_se: float
    n_obs: int
    recording: str
    superseded_by: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload = dict(asdict(self))
        payload["universe"] = list(self.universe)
        payload["date_range"] = list(self.date_range)
        return payload

    @staticmethod
    def from_json(raw: dict[str, Any]) -> TrialRecord:
        return TrialRecord(
            trial_id=str(raw["trial_id"]),
            timestamp=str(raw["timestamp"]),
            git_sha=str(raw["git_sha"]),
            data_manifest_hash=str(raw["data_manifest_hash"]),
            series_digest=str(raw["series_digest"]),
            strategy=str(raw["strategy"]),
            params=dict(raw["params"]),
            universe=tuple(raw["universe"]),
            date_range=(str(raw["date_range"][0]), str(raw["date_range"][1])),
            cost_bps=float(raw["cost_bps"]),
            sharpe=float(raw["sharpe"]),
            sharpe_se=float(raw["sharpe_se"]),
            n_obs=int(raw["n_obs"]),
            recording=str(raw["recording"]),
            superseded_by=(None if raw.get("superseded_by") is None else str(raw["superseded_by"])),
        )


def make_record(
    *,
    strategy: str,
    params: Mapping[str, object],
    sharpe: float,
    sharpe_se: float,
    n_obs: int,
    cost_bps: float,
    series_digest: str,
    date_range: tuple[str, str],
    universe: tuple[str, ...] = (),
    git_sha_value: str | None = None,
    data_manifest_hash_value: str | None = None,
    recording: Recording = Recording.TRIALS,
    timestamp: str | None = None,
) -> TrialRecord:
    """Build a record, computing its content address."""
    resolved_params = canonical_params(params)
    sha = git_sha() if git_sha_value is None else git_sha_value
    manifest = UNKNOWN if data_manifest_hash_value is None else data_manifest_hash_value
    return TrialRecord(
        trial_id=compute_trial_id(
            git_sha=sha,
            data_manifest_hash=manifest,
            series_digest=series_digest,
            strategy=strategy,
            params=resolved_params,
            universe=universe,
            date_range=date_range,
            cost_bps=cost_bps,
        ),
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        git_sha=sha,
        data_manifest_hash=manifest,
        series_digest=series_digest,
        strategy=strategy,
        params=resolved_params,
        universe=universe,
        date_range=date_range,
        cost_bps=float(cost_bps),
        sharpe=float(sharpe),
        sharpe_se=float(sharpe_se),
        n_obs=int(n_obs),
        recording=recording.value,
    )


# --------------------------------------------------------------------------------------
# Scope. Part C rule 3's "matching the reporting scope".
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scope:
    """Which trials a reported `N` should count.

    This is what keeps G6's 1,000 nulls out of a real strategy's `N` while both live in
    one file. Those 1,000 are genuinely 1,000 distinct trials and belong in the ledger;
    they are simply not part of the search that produced the strategy being reported.
    """

    strategies: tuple[str, ...] = ()
    universe: tuple[str, ...] = ()
    date_range: tuple[str, str] | None = None
    series_digest: str | None = None
    cost_bps: float | None = None
    """Count only trials run at this cost.

    A cost sweep evaluates one configuration at eight cost levels and records eight
    trials, correctly -- they are eight distinct evaluations. But they are not eight
    candidates a strategy was *selected* from, and a deflated Sharpe wants the width of
    the choice, not the count of the runs. A report of the zero-cost Sharpe scopes to
    the zero-cost trials, which keeps `N` like-for-like with the number being deflated.
    """

    def matches(self, record: TrialRecord) -> bool:
        if self.strategies and not any(record.strategy.startswith(s) for s in self.strategies):
            return False
        if self.universe and record.universe != self.universe:
            return False
        if self.date_range is not None and record.date_range != self.date_range:
            return False
        if self.series_digest is not None and record.series_digest != self.series_digest:
            return False
        return self.cost_bps is None or record.cost_bps == self.cost_bps


# --------------------------------------------------------------------------------------
# The ledger.
# --------------------------------------------------------------------------------------


@dataclass
class Ledger:
    """Append-only trials ledger, backed by JSONL.

    JSONL because append-only is native to the format, because it is text so git shows
    real diffs, and because Part C rule 4 requires the file to ship in the repository
    where a reader can verify `N` for themselves.

    `seen` is maintained regardless of `recording`, so a `NONE` ledger still counts what
    passed through it. "Observe and count, persist nothing" is a storage policy, not a
    licence to stop counting.
    """

    path: Path | None
    recording: Recording = Recording.TRIALS
    seen: set[str] = field(default_factory=set)
    _cache: list[TrialRecord] = field(default_factory=list)
    _primed: bool = False

    @staticmethod
    def memory(recording: Recording = Recording.NONE) -> Ledger:
        """An in-memory ledger. What every gate call site uses: counted, not persisted."""
        return Ledger(path=None, recording=recording)

    def observe(self, record: TrialRecord) -> None:
        """Record one engine invocation. Never conditional on a debug flag (B3 rule 1).

        Two things here are performance-critical, because B3 puts this on the path of
        every engine invocation and the gate suite makes hundreds of thousands of them.

        `_cache` is only filled for an in-memory ledger that is actually retaining rows.
        Appending unconditionally grew it to one record per invocation -- tens of
        megabytes of `TrialRecord` the gate suite never reads -- and the resulting
        allocation and GC pressure took the suite from 33s to 113s. Counting needs the
        `seen` set, and nothing else.

        `seen` is primed from disk once rather than re-read per observation. The previous
        version scanned and parsed the whole file on every new trial, which is quadratic
        in the size of the ledger and would have degraded exactly as the project's own
        history of trials grew.
        """
        self._prime()
        first_time = record.trial_id not in self.seen
        self.seen.add(record.trial_id)

        if self.recording is Recording.NONE:
            return
        if self.path is None:
            self._cache.append(record)
            return
        if self.recording is Recording.TRIALS and not first_time:
            return  # idempotent, still append-only: nothing is rewritten
        self._append(record)

    def _prime(self) -> None:
        """Load the ids already on disk, once, so re-running a search is idempotent
        across processes and not merely within one."""
        if self._primed or self.path is None:
            return
        self._primed = True
        for existing in self.records():
            self.seen.add(existing.trial_id)

    def _append(self, record: TrialRecord) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_json(), sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def records(self) -> tuple[TrialRecord, ...]:
        """Every row in file order, superseded ones included. The audit view."""
        if self.path is None:
            return tuple(self._cache)
        if not self.path.exists():
            return ()
        out = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append(TrialRecord.from_json(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise LedgerError(
                    f"{self.path}:{number} is not a readable trial record ({exc}). The "
                    "ledger is append-only, so a corrupt line means something wrote to it "
                    "that was not this module."
                ) from exc
        return tuple(out)

    def live(self) -> tuple[TrialRecord, ...]:
        """Collapsed by `trial_id`, last row winning, superseded rows dropped.

        Supersession appends rather than edits, so a trial can appear several times. The
        last row is what the project currently believes; the earlier ones stay readable,
        which is the whole point of append-only.
        """
        latest: dict[str, TrialRecord] = {}
        for record in self.records():
            latest[record.trial_id] = record
        return tuple(r for r in latest.values() if r.superseded_by is None)

    def n_trials(self, scope: Scope | None = None) -> int:
        """`N`. Part C rule 3: read from the ledger, never hand-typed.

        Distinct non-superseded trials, optionally within a reporting scope. For a
        `NONE` ledger this counts what was observed in this process, which is what makes
        an in-memory ledger useful to a gate that wants to assert its own search size.
        """
        if self.path is None:
            if scope is None:
                return len(self.seen)
            if self.recording is Recording.NONE:
                raise LedgerError(
                    "a NONE ledger retains no rows, so it cannot count a scoped subset "
                    "of them -- it would answer 0 regardless of what passed through. "
                    "Under-reporting N is the direction that flatters a deflated "
                    "Sharpe, so this raises instead. Use Recording.TRIALS to keep the "
                    "rows, or scope=None to count everything this process observed."
                )
            return sum(1 for r in self._cache if scope.matches(r))
        self._prime()
        records = self.live()
        if scope is None:
            return len(records)
        return sum(1 for r in records if scope.matches(r))

    def supersede(self, trial_id: str, by: str) -> None:
        """Mark a trial superseded by appending, never by editing (B3 rule 2).

        A bug fix does not delete the rows it invalidates. What the project used to
        believe stays in the file, which is the difference between a corrected record
        and a rewritten one.
        """
        if self.path is None:
            raise LedgerError("an in-memory ledger has nothing to supersede")
        matches = [r for r in self.live() if r.trial_id == trial_id]
        if not matches:
            raise LedgerError(f"no live trial {trial_id!r} to supersede")
        self._append(
            replace(matches[-1], superseded_by=by, timestamp=datetime.now(UTC).isoformat())
        )

    def extend(self, records: Iterable[TrialRecord]) -> None:
        for record in records:
            self.observe(record)


def scope_for(strategies: Iterable[str]) -> Scope:
    """Convenience: a scope naming the strategy families a report covers."""
    return Scope(strategies=tuple(strategies))


ObserveHook = Callable[[TrialRecord], None]

__all__ = [
    "ID_LENGTH",
    "LEDGER_PATH",
    "UNKNOWN",
    "Ledger",
    "LedgerError",
    "Params",
    "Recording",
    "Scope",
    "TrialRecord",
    "canonical_params",
    "canonical_value",
    "compute_trial_id",
    "git_sha",
    "make_record",
    "manifest_hash",
    "scope_for",
    "series_digest",
]
