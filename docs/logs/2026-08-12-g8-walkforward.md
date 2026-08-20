# Session log — 2026-08-12 — G8, the integration layer, and a parallel suite

## 1. Starting state

Verified before touching anything: clean tree, 0 unpushed, 22 commits, local and
`origin/main` both at `60274ee`. No PRs — the project pushes directly to `main`.

## 2. G8 — purged, embargoed walk-forward

PLAYBOOK's condition is blunt: **zero index overlap train/test, asserted in code, not
in docs.** `Split` refuses to construct when the two intersect, so a leaking partition
is *unconstructable* rather than caught later inside a Sharpe.

Three geometries, because "walk-forward" is not one thing:

| Splitter | Geometry | Why it exists |
|---|---|---|
| `ExpandingWindow` | anchored at bar 0, grows | closest to how a strategy is actually run |
| `RollingWindow` | fixed training length, slides | folds comparable; forgets the distant past |
| `PurgedKFold` | trains both sides of the block | **not** a walk-forward — the geometry CSCV needs at G9, and the only one where embargo does work |

Purge and embargo are tested separately, because transposing them produces folds that
are non-overlapping, plausible, and wrong.

### The integration layer

`build_grid` pushes a configuration grid through the certified engine and aligns
columns **from the end**, so every column covers the same calendar period — aligning
by position would compare a fast configuration's early bars against a slow one's later
ones. `walk_forward_select` then splits, selects on each training block with a
`SelectionRule`, and scores only out of sample.

Built here rather than at G9 deliberately: CSCV needs exactly a `(T, N)` grid, a block
splitter and a rule turning in-sample evidence into weights. Building them now means G9
assembles certified parts instead of inventing them beside its own rank bookkeeping.

## 3. Three assertions written, measured, then rewritten

This was the substance of the session, and all three failed for the same underlying
reason: **an assertion written from expectation rather than from measurement, on a
quantity whose sampling error was never worked out.**

### 3.1 "ArgMax degrades out of sample"

Failed with **OOS above IS** (+1.98 vs +1.19) — failure mode F6 in `03`, which reads as
a leak. Investigated per F6's own instruction ("check the purge and embargo indices
directly rather than trusting the splitter"):

```
fold 0: train [0,626] n=627  test [637,716]  gap=10
fold 1: train [0,706] n=707  test [717,796]  gap=10
...
```

Every fold clean: gap exactly 10, train strictly before test, zero overlap. **The
splitter is correct.** Over 20 seeds, degradation is **−0.041 ± 0.125**, negative in 10
of 20. An 80-bar out-of-sample Sharpe carries SE ≈ `sqrt(252/80)` = 1.77, so a six-fold
mean carries ≈ 0.72 — the observed gap is about one SE of nothing.

Replaced by the **exact** structural claim: ArgMax's in-sample Sharpe *equals* the grid
maximum (20/20, exactly). Degradation is measured and reported, not gated.

### 3.2 "IS→OOS slope is negative (the compensation effect)"

Measured **+0.979 ± 0.085, positive in 20/20** on AR(1). This is correct behaviour, not
a defect, and understanding why matters:

| Process | slope over 20 seeds | reading |
|---|---|---|
| AR(1), configs genuinely differ | **+0.979 ± 0.085**, >0 in 20/20 | ranking carries real information |
| GBM, no config has an edge | +0.215 ± 0.265, <0 in 8/20 | indistinguishable from zero |

A negative slope is the signature of selecting among configurations with **no true
differential merit** — ranking noise. This grid runs on a stationary mean-reverting
series where slow z-scores earn a real edge and trend-followers genuinely lose on it
(their full-window Sharpes are −1.08 and −0.68). In-sample ranking *should* predict
out-of-sample there.

The compensation effect belongs to the memoryless case. So the slope is reported, not
gated — one GBM path puts it anywhere from −1.78 to +2.26 — and what is asserted is
what survives: a real edge survives the walk-forward, and none is manufactured on GBM.

### 3.3 "All splitters raise on 20 observations"

`PurgedKFold` does not, and should not: 20 observations is a perfectly legal 5-fold
split. Parametrised per splitter with a size appropriate to each.

## 4. Suite performance

| | before | after |
|---|---|---|
| tests | 225 | **263** |
| skipped | 4 | **0** |
| local runtime | 49.8 s | **28 s** |
| CI total | 55 s | **47 s** |

CI step breakdown after: gate suite **27 s**, typecheck 8 s, ~12 s fixed overhead.

- **`pytest-xdist` at `-n auto`**: 72.5 s → 35.2 s serial-to-parallel on 8 cores, with
  every test still at full strength. G1 keeps its spec-mandated 500 cuts × 20 seeds;
  nothing was trimmed.
- **The 4 skips are gone.** They were `TopK(3)` receiving a grid narrower than 3
  columns. N is now drawn `>= k`, so every example exercises the rule. A skip
  conditioned on a random draw is worse than a skip: how much of the property actually
  got checked varied run to run.
- **mypy cached in CI**, pinned to `actions/cache@v6.1.0` like every other action. Key
  tracks the lockfile and the sources, so a real change still gets a cold, honest run.

### Parallelising broke the collection floor

Worth recording because it is exactly the failure the floor exists to prevent. Under
xdist it raised `UsageError` inside a **worker**, where pytest surfaces it as an
`INTERNALERROR` with a traceback rather than the clean message it exists to print — a
countermeasure evaporating at the moment the way tests run changed.

Fixed by enforcing controller-side via `pytest_xdist_node_collection_finished`, and
verified firing in **both** serial and parallel, with a normal run still passing.

## 5. Final state

```
263 collected · 263 pass · 0 skip · 28 s local · 47 s CI · offline
ruff clean · mypy --strict clean (40 files) · action refs resolve
health check: 5/5 PASS
```

Green: Gate 0.0, G1–G8. Partial: G10. Not started: G9.

## 6. Decisions taken this session

- **SPY fetch authorised**, and deliberately **not spent** — scope was set to G8 depth,
  so Phase 5 is queued for next session with the authorisation banked.
- **G7 was already green** (leakage trap, built alongside G1 in session 2). The A4
  ruling lives in `test_a4_oracle_does_not_leak_and_has_no_edge`. The requested *live*
  verification on real SPY prices needs Phase 5 and is queued with it.
- **On Gate 0 and `auto_adjust` / caches / sha256 manifests**: those are the Part G data
  contract, gated by **G10**, not Gate 0. Gate 0 is the pre-flight on synthetic data and
  it is complete. The congruence being asked for is real work — it is Phase 5 plus G10.

## 7. Next — G9

PBO via CSCV, and the hardest code in the project: 12,870 splits, per-split argmax
across N columns, rank computation, logit transform. Every one is a place to be off by
one and none of them raises.

It is also the best-prepared gate in the build. `SelectionRule` landed in session 3, and
G8 has just supplied the `(T, N)` grid, the block splitter and the walk-forward harness
it needs — so CSCV assembles certified parts rather than inventing them alongside its
own bookkeeping.
