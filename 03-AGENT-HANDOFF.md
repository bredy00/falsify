# 03 — Agent Handoff

> **Purpose:** the operating manual for any LLM working on this repo — a coding agent, a fresh session, or a future you with no memory of these decisions. Load this first.

---

## Part A — Read order

Load in this sequence. Each depends on the ones above it.

| # | Document | What it is |
|---|---|---|
| 1 | `03-AGENT-HANDOFF.md` | this file — invariants and rules |
| 2 | `02-ENGINE-SPEC.md` | interfaces, equations, conventions |
| 3 | `00-VALIDATION-FIRST.md` | phase ordering, Gate 0 |
| 4 | `01-STATS-FOUNDATIONS.md` | the statistical machinery |
| 5 | `PLAYBOOK.md` | the original roadmap, superseded on ordering |
| 6 | `thesis.pdf` | the mathematics: expected maximum Sharpe, MinBTL, compensation effects |
| 7 | `companion.pdf` | why any of this matters, and where to read further |

Items 6 and 7 are background. No code depends on reading them, and Gate 0.0 in `00` already
restates everything from `thesis.pdf` that the build needs. Read them last, or in parallel.

Where documents conflict, precedence runs highest-numbered-rule-first: this file beats `02`, which beats `PLAYBOOK.md`. `PLAYBOOK.md` phase ordering is **superseded** by `00`.

---

## Part B — Invariants

Violating any of these is a build failure, not a style preference. Do not "improve" past them.

**B1. No network call before Gate 0 passes.** Phases 0 through 4 run entirely offline on synthetic data. If a task requires market data before Gate 0 is green, the task is out of order — stop and say so.

**B2. Every performance number carries an error bar.** A bare Sharpe in a log line, a docstring, a README or a commit message is a defect. The reporting contract in `01`, Part D, has six required fields.

**B3. The trials ledger is append-only and machine-written.** Never hand-set `N`. Never delete a row. A bug fix marks `superseded_by`.

**B4. Strategies emit target weights, never orders.** Sizing, rebalancing and execution belong to the engine.

**B5. Both engines implement the Part E equations identically.** Changing an equation means changing it in two places and re-running G2 in the same commit. Never one without the other.

**B6. No fills in the feature layer beyond declared forward-fill.** `bfill` anywhere in the pipeline is an automatic revert. It is the leak in the naive baseline.

**B7. Frozen dataclasses.** No in-place mutation of `Bars` or `Result`. If a function needs a modified copy, it constructs one.

**B8. Per-observation units internally, annualised only at the reporting boundary.** Any function that takes a Sharpe names the unit in its signature or docstring.

**B9. Seeds are threaded explicitly.** `rng = np.random.default_rng(seed)` passed as an argument. No `np.random.seed`, no module-level global RNG, ever. G10 depends on this.

**B10. Non-excess kurtosis in every PSR/DSR call.** `scipy.stats.kurtosis(..., fisher=False)`. The default is wrong for these formulas.

---

## Part C — Gate cost, honestly re-estimated

You said: *"from G1 to G5 I could code those in a single session or day, but for G6 we have to think like the exact same amount of time that we've quoted five other modules."*

That's the right instinct and it's directionally correct, but the cost is not where you think it is. Revised estimates, in focused sessions of roughly three hours:

| Gate | Sessions | Where the time actually goes |
|---|---|---|
| G1 Causality | 1.5 | Harness is 30 lines. The time goes into making the *full pipeline* callable as one function, and then into fixing the leaks it finds. Budget most of it for fixes, not for the test. |
| G2 Twin agreement | 2.0 | Writing two engines is one session. Chasing the last 1e-9 of disagreement is the other, and it will be a float accumulation-order difference. |
| G3 Analytic recovery | 0.5 | Genuinely easy once the synthetic generators exist. |
| G4 Zero-cost identity | 0.25 | One assertion. |
| G5 Cost monotonicity | 0.5 | A sweep and a `np.diff(...) <= 0` check. |
| **G1–G5 subtotal** | **4.75** | Not one day. Call it two solid days. |
| G6 Null calibration | 1.5 | **Cheaper than you think.** See below. |
| G7 Leakage trap | 0.25 | 10 lines, given G1 exists. Do it in the same commit as G1. |
| G8 Walk-forward | 1.5 | The splitter is easy; the index-overlap assertions and the off-by-one at block edges are not. |
| G9 PBO / CSCV | 2.0 | 12,870 splits needs vectorising. The rank bookkeeping is fiddly and easy to get subtly wrong. |
| G10 Reproducibility | 1.0 | Mostly plumbing: manifest verification, git SHA capture, deterministic figure output. |

### Why G6 is cheaper than it looks

Running 1,000 strategies sounds like 1,000× the work. It isn't, for two reasons:

1. **The pipeline already exists.** By the time you reach G6 you have a vectorised engine that runs one strategy in milliseconds. A thousand of them is a loop and a coffee. The compute is not the problem.
2. **The random strategy is trivial.** `rng.choice([-1, 0, 1], size=T)` is the whole implementation.

The one genuinely hard part, and the only part worth budgeting for: **turnover matching.** A random strategy that flips every bar has enormous turnover and will be destroyed by costs, making your real strategy look good for the wrong reason. The null must trade at the same rate as the thing it's testing. Implement it as a Markov chain whose transition probability is tuned so realised turnover matches the target within 5%, then assert that match in the test. That calibration is where your session goes.

### Where the real cost is

**G9, not G6.** CSCV has more moving parts than anything else in the build: block partitioning, combination enumeration, per-split argmax across N columns, rank computation, logit transform, and a final probability. Every one of those is a place to be off by one, and none of them will throw an exception when they are. Budget two full sessions and write the test first.

### Suggested ordering

```
Session 1–2    G1 + G7 together        (the harness and its trap in one commit)
Session 3–4    G2
Session 5      G3 + G4 + G5            (all three, they're small)
─── checkpoint: engine certified, still offline ───
Session 6      data layer + G10 plumbing
Session 7      G6
Session 8–9    G8
Session 10–11  G9
Session 12     reporting, README, research note
```

Twelve sessions. If a session slips, G8 is the one to defer — walk-forward is standard and expected, whereas G1, G2, G6 and G9 are what make the repo unusual. Protect those four.

---

## Part D — Task template

Every task handed to a subagent uses this shape. Anything vaguer produces plausible code that fails a gate three tasks later.

```markdown
### Task N: <component>

**Files**
- Create: exact/path.py
- Test:   tests/gates/test_gN.py

**Invariants touched:** B5, B8

**Reference:** 02-ENGINE-SPEC.md Part E

- [ ] Write the failing test (full code, not a description)
- [ ] Run it, confirm it fails with the expected message
- [ ] Implement the minimum that passes
- [ ] Run it, confirm pass
- [ ] Run the full gate suite, confirm no regression
- [ ] Commit
```

Test first, always. In this project the test *is* the specification — a gate is not a check on the code, it is the definition of what correct means.

---

## Part E — Review checklist

Run before any merge to `main`.

**Correctness**
- [ ] Full gate suite green, including gates unrelated to this change
- [ ] G2 re-run if any Part E equation changed
- [ ] No new bare Sharpe without its five companion fields
- [ ] Kurtosis calls pass `fisher=False`
- [ ] Per-observation vs annualised units consistent at every boundary

**Causality**
- [ ] Any new feature declares its `lookback`
- [ ] G1 covers the new code path (verify by adding a deliberate leak and watching it fail)
- [ ] No `bfill`, no `center=True`, no global-percentile clipping, no scaler fitted outside the window

**Reproducibility**
- [ ] Seeds passed explicitly, no global RNG
- [ ] Manifest updated if data changed
- [ ] `make reproduce` produces byte-identical output across two runs

**Honesty**
- [ ] Trials ledger has a row for every configuration evaluated in this change
- [ ] Any result that improved — check whether it improved because of a relaxed assumption

That last item is the one that matters. When a Sharpe jumps after a refactor, the default hypothesis is a new leak, not a better strategy. Investigate before celebrating.

---

## Part F — Failure modes to watch for

Ranked by how often they appear and how quietly.

**F1 — Sharpe improves after a "cleanup" commit.** Near-certainly a leak. Bisect and find it. Real improvements come from strategy changes, not from tidying.

**F2 — G2 agrees to 1e-6 but not 1e-12.** Accumulation-order difference in the equity recursion. The vectorised engine is probably doing the cost deduction in a different sequence. Not a rounding artefact to wave through; find it.

**F3 — PBO comes out at exactly 0.0 or 1.0.** The rank computation is broken, or every column is identical. Check that your N configurations actually differ.

**F4 — DSR near 1.0 for everything.** `N` is being read as 1. Check the ledger query.

**F5 — Bootstrap CI narrower than the analytic SE.** The resampler is destroying autocorrelation, meaning `p` is too large and blocks are too short. Validate against the i.i.d. case first.

**F6 — Walk-forward OOS Sharpe higher than in-sample.** Possible but rare. Usually indicates train and test are overlapping — check the purge and embargo indices directly rather than trusting the splitter.

**F7 — A gate that has never failed.** A test that cannot fail is not a test. Every gate needs a known-bad input in the suite proving it fires. G7 does this for G1; do the equivalent for the rest.

---

## Part G — Scope discipline

Out of scope for v1. Each is a different project and each will consume the time this one needs.

| Excluded | Why |
|---|---|
| Live trading, broker APIs | Different project. Invites a security review of key handling and adds nothing to the argument. |
| Intraday data | Microstructure, session handling and tape quirks will eat a month. Daily first. |
| Options, futures roll logic | Contract specification and continuous-series construction are their own build. |
| ML strategies | The point is validating *any* strategy honestly. Adding a neural net multiplies your N and weakens the story. |
| Web dashboard | Static figures in a README are read; dashboards are not. |
| Multi-currency | FX conversion, non-overlapping calendars, two sets of holidays. |

**The finished project can honestly conclude that the strategy does not work.** A repo reporting `PBO = 0.61`, `DSR = 0.22`, break-even cost 6 bps, plus a rigorous engine and a calibrated null, is a stronger artefact than one reporting a 2.4 Sharpe with no error bars. The first proves you can evaluate a strategy. The second proves you can overfit one.

Any agent proposing to relax a gate to make a number look better is proposing to defeat the project.

---

## Part H — Resolved decisions

These were open. They are now decided, so an agent does **not** need to ask before proceeding.
Each records the rationale and the condition under which it should be revisited.

| # | Decision | Rationale | Revisit when |
|---|---|---|---|
| 1 | **Universe: SPY only for v1.** | Single ticker removes survivorship bias entirely as a confound, which keeps the statistical argument clean. The cross-sectional work is Phase 8+. | The engine is certified and you want factor attribution. |
| 2 | **`N_eff` via participation ratio.** `N_eff = (Σλ)²/Σλ²` on the trial correlation matrix. | No threshold to justify, and the source paper itself recommends dimension reduction for non-independent trials. Report clustering as a secondary estimate only. | The two estimates diverge by more than 2×; then report the smaller and say why. |
| 3 | **Bootstrap `p = 1/√T`.** | Mean block length √T. Sensitivity across `{T^(-1/3), T^(-1/2), T^(-2/3)}` still gets reported. | CI width moves >20% across that range. |
| 4 | **Risk-free: constant, set to the period mean of 3-month T-bills.** | The cash-yield term must exist (that's the naive baseline's omission), but a full T-bill path is Phase 8 polish. Record the constant used in the manifest. | Comparing across regimes with very different rate levels. |
| 5 | **Repo name: `falsify`.** | Matches the thesis — the engine's job is to try to kill the strategy. | Never. Pick it and move. |

If a genuinely new ambiguity appears, flag it. Do not re-litigate the five above.

---

## Part I — First three sessions

Start here. Do not begin with scaffolding.

### Session 1 — Reproduce the theory before building anything

Implement `tests/gates/test_prop.py` containing Experiments A, B and C from
`00-VALIDATION-FIRST.md` Part 0.0. numpy and scipy only. No engine, no data layer, no yfinance.

Deliverable: a passing test file and one saved figure — the in-sample vs out-of-sample Sharpe
scatter under a compensation effect, showing a negative slope. That figure is the project's
thesis in one image and it goes in the README.

### Session 2 — Scaffold, then G1 + G7 in a single commit

`uv` env, ruff, mypy strict, GitHub Actions running the gate suite. Then the causality-cut
harness (`02-ENGINE-SPEC.md` Part A) and the `LeakyOracle` trap together. The trap must fail
the build if the harness stops detecting it.

Do not split these across two commits. A harness without its trap is unverified.

### Session 3 — `SelectionRule` interface, before any engine work

Specified in `02-ENGINE-SPEC.md` Part H. Implement `ArgMax`, `Softmax`, `EqualWeight`, `TopK`
against the ABC with unit tests only — no integration yet.

Reason for the ordering: G9 (CSCV/PBO) is the hardest code in the project and hardcoding it to
argmax means rewriting the rank bookkeeping later. The interface must exist first.

After session 3, resume the ordering in Part C from "Session 3–4: G2".
