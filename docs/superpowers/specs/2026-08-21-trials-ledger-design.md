# Trials ledger — design

**Date:** 2026-08-21
**Implements:** `03` invariant B3, `01` Part C
**Status:** approved, not yet implemented

---

## Why

`03` Part B, B3: *"The trials ledger is append-only and machine-written. Never hand-set `N`. Never delete a row. A bug fix marks `superseded_by`."*

`01` Part C rule 3: *"`N` is read from the ledger by counting non-superseded rows matching the reporting scope. It is never a hand-typed constant."*

**This invariant is currently violated.** `falsify/deflated.py` exposes `expected_max_sharpe(n_trials: int, ...)` and `min_backtest_length_years(n_trials: int, ...)`, both taking `N` as a caller-supplied integer. There is no ledger. Every DSR the project has computed rests on a hand-typed `N`.

`01` Part C on what this file is for: *"the difference between a project that discusses backtest overfitting and a project that measures its own."*

---

## Two conflicts found while reading the spec, and how they resolve

### Conflict 1 — "every engine invocation" against the gate suite

Part C rule 1 says **every** engine invocation writes a row, *"not optional, not conditional, no `if not debug`."* Taken naively that is millions of rows: G1 runs 500 τ × 20 seeds per strategy, G6 pushes 1,000 nulls through the full pipeline.

**Resolved by content-addressing, with no exemption needed.** G1's τ-test recomputes the *same* `(strategy, params, universe, date_range, cost_bps)` ten thousand times. Same tuple, same hash, **one** distinct trial — which is correct, because that is one configuration evaluated. Rule 1 holds literally; `N` counts *distinct* ids.

G6's 1,000 nulls are 1,000 *different* strategies and land as 1,000 distinct rows. That is not pollution: it is exactly the `N` a deflated Sharpe should be deflating by. Scope filtering (below) keeps them out of a real strategy's count while leaving them in one file.

### Conflict 2 — the ledger would break G10

G10's pass condition is *"two runs produce byte-identical figures and metrics JSON."* `n_trials_raw` (`01` Part D) is read from the ledger; B3 says every run appends. Run twice, `N` grows, `metrics.json` differs — **G10 fails by construction.**

**Resolved by replacing `uuid4` with a content-addressed `trial_id`.** Identical trials produce identical ids, so re-running is idempotent, `N` is stable, and G10 stays achievable.

---

## Departures from `01` Part C, stated explicitly

| Part C | This design | Reason |
|---|---|---|
| `trial_id: str  # uuid4` | `sha256(canonical)[:16]` | uuid4 is non-deterministic; it breaks G10 (Conflict 2) and B9. Content-addressing also makes "`N` = distinct configurations evaluated" literally true. |
| `sharpe: float` | adds `sharpe_se: float` | **B2**: no performance number without an error bar. The ledger is where `N`-many of them live; a bare Sharpe here is the same defect as a bare Sharpe in a README. |
| — | adds `recording: str` | Records which policy produced the row, so a later policy change is interpretable rather than silently mixing bases. |

Nothing is removed from the Part C record.

---

## Data model

```python
class Recording(Enum):
    TRIALS = "trials"   # persist one row per distinct content address (default)
    ALL    = "all"      # persist every observation, duplicates included
    NONE   = "none"     # observe and count, persist nothing


@dataclass(frozen=True, slots=True)          # B7
class TrialRecord:
    trial_id: str                # sha256(canonical)[:16]
    timestamp: str               # ISO 8601 UTC; provenance, NOT part of the id
    git_sha: str                 # "<sha>" or "<sha>-dirty"
    data_manifest_hash: str
    strategy: str
    params: dict[str, object]    # JSON scalars only
    universe: tuple[str, ...]
    date_range: tuple[str, str]
    cost_bps: float
    sharpe: float                # annualised, per the reporting boundary (B8)
    sharpe_se: float
    n_obs: int
    recording: str
    superseded_by: str | None = None
```

`Recording` is the adjustable dial. The `ledger` parameter on the engines is **not** optional — that is what makes bypass impossible — but what the ledger *does* with an observation is configurable, is recorded on every row, and is revisable later through supersession without destroying the old basis.

---

## Identity

`trial_id = sha256(canonical.encode("utf-8")).hexdigest()[:16]`, where `canonical` is
`json.dumps(defining, sort_keys=True, separators=(",", ":"))` over
`{git_sha, data_manifest_hash, strategy, params, universe, date_range, cost_bps}`.

Three rules that decide whether this is correct or silently wrong:

1. **Numbers normalise through `float` then `repr`.** `MACrossover(20, 50)` and `MACrossover(20.0, 50.0)` are one configuration and must not produce two rows. `bool` is tested *before* `int`, because Python makes `True` an integer and `1.0` is not the same trial as `True`.
2. **`timestamp` is excluded from the hash.** It is provenance. Including it makes every rerun a new trial and reopens Conflict 2.
3. **A dirty working tree records `<sha>-dirty`,** which hashes differently from the clean commit. A trial run on uncommitted code can then never be confused with one run on shipped code. That is intended: it *should* look different, because it is.

`params` values are restricted to `str | int | float | bool | None`. The constructor raises on anything else — a param that cannot be canonicalised is a trial that cannot be identified, and an unidentifiable trial must not be recorded as one.

**Truncation.** 16 hex characters is 64 bits. At the scale this project will ever reach (`N` in the thousands) collision probability is negligible; the full digest is recoverable by recomputation if it is ever wanted.

---

## Storage

JSONL at `data/trials.jsonl`, one record per line, appended.

- Append-only is native to the format (B3 rule 2).
- Text, so git shows real diffs and Part C rule 4 holds: *"The ledger ships in the repo. A reader can verify your `N`."*
- Nested `params` needs no encoding gymnastics.

Under `Recording.TRIALS` a write is skipped when the `trial_id` is already present — idempotent, still append-only, never a rewrite.

---

## API

```python
Ledger(path: Path, recording: Recording)
Ledger.memory(recording: Recording = Recording.NONE)   # tests

.observe(record: TrialRecord) -> None      # unconditional, called by the engines
.records() -> tuple[TrialRecord, ...]      # file order, everything
.live() -> tuple[TrialRecord, ...]         # collapsed, superseded dropped
.n_trials(scope: Scope | None = None) -> int
.supersede(trial_id: str, by: str) -> None
```

**Supersession is append-only.** `supersede` appends a new row carrying the same `trial_id` with `superseded_by` set; `live()` collapses by `trial_id` on a last-row-wins rule and drops superseded entries. History is never rewritten, and what the project used to believe stays readable.

**Scope** is a predicate over `strategy`, `universe`, and `date_range`, satisfying Part C rule 3's *"matching the reporting scope"*. It is what keeps G6's 1,000 nulls out of a real strategy's `N` while both live in one file.

---

## Engine integration

`ledger` becomes a **required** parameter on `run_vectorized` and `run_event`.

**B5 applies**: both engines change in one commit, and G2 re-runs in that same commit. Every existing gate call site takes `Ledger.memory(Recording.NONE)` — observed and counted, nothing persisted.

`deflated.py` keeps its `n_trials: int` signatures — the functions stay pure and testable. What changes is that no caller in the repo may pass a literal; `N` comes from `Ledger.n_trials(scope)`.

---

## Testing

Ordinary coverage: round-trip, canonicalisation (`20` vs `20.0`, `True` vs `1`, key order, unicode), append-only under supersession, scope filtering, `NONE`/`ALL`/`TRIALS` behaviour, dirty-tree suffix.

Two that carry the weight, in `tests/gates/test_b3_ledger.py`:

**The trap (F7).** A "researcher" evaluates 50 configurations and reports the best. Assert the ledger reports **50**, not 1. This is the `N`-side equivalent of G7's leakage trap — the failure mode is publishing the winner and forgetting the search, which is the entire thesis of the project. Per F7 it must be demonstrated failing: remove the ledger write and the count reports 1.

**Idempotency.** Run the same sweep twice; assert `n_trials` is unchanged. This is the guard on Conflict 2 and the precondition for G10 going green later.

The trap lives under B3 rather than as a new gate row. `03` Part G is explicit about scope discipline, and B3 is already a named hard invariant — making a stated constraint enforceable does not require minting G11.

---

## Out of scope

Deliberately not in this spec, each getting its own:

- Effective number of trials (`01` B4)
- Stationary bootstrap (`01` B5, PLAYBOOK 5a)
- `metrics.json` and the Part D reporting contract
- Completing G10

Also not doing: **retrofitting historical trials.** The ledger starts empty. A reconstructed `N` would be a hand-typed constant wearing a costume, which is the thing B3 exists to forbid.

---

## Risks

| Risk | Mitigation |
|---|---|
| Canonicalisation drift silently splits one config into two ids, inflating `N` | Dedicated tests on numeric/bool/key-order normalisation; inflation is the *conservative* direction for a DSR, but still wrong |
| Required parameter is a large mechanical diff across every gate | Contained; B5 forces the two engines together and G2 verifies in the same commit |
| Someone reintroduces a literal `N` at a call site | `n_trials` is the only supported source; a follow-up spec can add a lint check when `metrics.json` lands |
| Ledger file grows unboundedly under `Recording.ALL` | `TRIALS` is the default; `ALL` is opt-in for debugging |
