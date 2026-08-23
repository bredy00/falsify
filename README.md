# falsify — workspace

A backtesting framework that reports an estimate **and its error bar**, plus the probability that
the number survived the search which produced it.

**Start with `03-AGENT-HANDOFF.md`.** It carries the invariants, the precedence order, and the
first three sessions. Everything else is downstream of it.

---

## Contents

| File | Role | Read |
|---|---|---|
| `03-AGENT-HANDOFF.md` | operating manual — invariants B1–B10, resolved decisions, session plan | **first** |
| `02-ENGINE-SPEC.md` | interfaces, equations, causality contract, selection rules | second |
| `00-VALIDATION-FIRST.md` | phase ordering, Gate 0.0 and Gate 0 | third |
| `01-STATS-FOUNDATIONS.md` | the statistical machinery and terminology | fourth |
| `PLAYBOOK.md` | original roadmap — **superseded on ordering** by `00` | fifth |
| `thesis.pdf` | the mathematics, 4 pages | background |
| `companion.pdf` | context and further reading, 7 pages | background |

**Precedence when documents conflict:** `03` > `02` > `00` > `01` > `PLAYBOOK`.

Part H of `03` records five decisions that are already made. Do not re-open them.

---

## The thesis in three lines

1. **Causality is a structural constraint, not a comment.** Scramble the future, assert the past
   is bit-identical. A `.shift(1)` is a claim; the τ-test is the proof.
2. **Two independent engines must agree.** Vectorised code is fast and silently wrong; an explicit
   event loop is slow and obviously right. Agreement to 1e-12 certifies the fast one.
3. **The null must be calibrated.** A thousand coin-flip strategies through the whole pipeline.
   If the machinery doesn't reject ~95% of them, nothing downstream means anything.

---

## Gate 0.0 — the thesis in one image

![In-sample versus out-of-sample Sharpe under three processes](docs/figures/compensation_effect.png)

Left: selecting the best of 1,000 backtests on a memoryless process buys you nothing. Middle and
right: on a process with memory it costs you, and the cost grows with how hard you searched. That
is Propositions 3 and 5 of the source paper, reproduced from a seed on this machine rather than
quoted.

`tests/gates/test_prop.py` — 14 tests, ~12 s, numpy and scipy only, no network, no engine.
Regenerate with `pytest tests/gates/test_prop.py -v -s`.

## G9 — the price of selectivity, measured

![PBO against softmax temperature for three kinds of grid](docs/figures/pbo_vs_temperature.png)

`01` Part E3 predicts this curve falls monotonically as selection is diluted, with
`EqualWeight` as the asymptote. Measured over the full C(16,8) = 12,870 splits, it does not.
On a grid built so that the in-sample winner is mechanically the out-of-sample loser, PBO
*rises* to a peak of 0.733 at τ = 1 before falling — a mild softmax still concentrates on the
top few in-sample performers, which are exactly the configurations that reverse. On a grid
with real, persistent merit the curve runs the other way, sitting near zero at τ = 1 and
climbing to 0.324 by τ = 32, because diluting selection when the differences are real throws
away the information that made selection worth doing.

The safe temperature is therefore neither 0 nor infinity; it depends on whether the grid has
genuine merit, which is the one thing you cannot know in advance. That is the argument for
measuring PBO rather than assuming a selection rule is safe.

`EqualWeight` cannot be the asymptote in any case: its weights are identical on every split,
so all 12,870 are near perfectly dependent and its PBO is effectively a single draw (sd 0.26
to 0.34 across grids). It is plotted with that spread and excluded from every assertion.

`tests/gates/test_g9_pbo.py` — 16 tests, ~8 s at C(8,4). The null calibrates to 0.5 at every
block count tested (0.4930, 0.4850, 0.4798, 0.4809 at S = 8, 10, 12, 16 over 80 grids each),
which is what makes the 0.5 ship threshold mean anything. Regenerate the figure with
`make g9-figure` — ~25 minutes, or instant from the cached measurements beside it.

| Experiment | Measured | Reference | Verdict |
|---|---|---|---|
| **A** E[max z], N = 1,000, 10,000 reps brute force | 3.2394 ± 0.0035 | 3.241436 exact | 0.58 SE |
| **A** Monte Carlo vs exact, N = 2 … 10⁶ (13 values) | worst gap 1.98 SE | exact quadrature | holds at every N |
| **A** growth vs `√(2 ln N)` | ratio 0.479 → 0.925 | — | approaches the bound from below |
| **B** memoryless slope | −0.019 ± 0.030 (p = 0.53) | 0 | no information in the in-sample number |
| **B** winner's OOS Sharpe, 200 reps | +0.053 ± 0.048 | 0 | selection is free |
| **B** winner's IS Sharpe, 200 reps | 2.321 ± 0.017 | 2.311 (Prop 1) | A predicts B to 0.45% |
| **C** common-mean slope | −0.9989 ± 0.0014 (p < 1e-300) | −1 | the exact reversal of Prop 3 |
| **C** AR(1) φ = 0.995 slope | −0.481 ± 0.028 (p = 2.1e-57) | < 0 | the winner reverts (Prop 5) |

Intercepts are insignificant in both C panels (−0.0000 ± 0.0007 and −0.019 ± 0.012), as the
propositions require: the effect is a rotation, not a level shift.

### The error budget of SR₀

Experiment A separates two things that a single tolerance would conflate. The empirical mean
estimates the **true** `E[max z]`, computed here by quadrature and checked against the closed forms
`1/√π` and `3/(2√π)` to machine precision. The two-term Gumbel expression is only an
*approximation* to that truth — and it is the approximation `01` Part B3 feeds into the Deflated
Sharpe as the benchmark SR₀, so its error is a bias in the DSR itself:

| N | 2 | 3 | 5 | 10 | 50 | 100 | 1,000 | 10⁶ |
|---|---|---|---|---|---|---|---|---|
| error in SR₀ | −7.88% | +0.77% | +2.55% | +2.33% | +1.21% | +0.92% | +0.42% | +0.10% |

The sign matters: the formula **overstates for every N ≥ 3**, so SR₀ is too strict rather than too
lax and the resulting DSR is conservative. Below N = 100 the error exceeds 1% and should be quoted
alongside any DSR computed there. At the N this project reports — hundreds of configurations — it is
about half a percent. All of this is asserted, not assumed.

Four negative controls ship alongside, because a gate that cannot fail is not a gate (`03` F7). One
holds the seed and the estimator fixed and shows the slope moving from +0.029 ± 0.031 to
−0.997 ± 0.001 under recentring alone, so the effect is the treatment and not the machinery. Another
shows that at 200,000 repetitions the exact reference sits 0.54 SE from the empirical mean while the
Gumbel approximation sits 16.88 SE away — which is why Experiment A is judged against exact
quadrature and not against the formula.

Numeric output and the figure's bytes are identical across two runs at the same seed.

---

## Gate status

**297 tests, 0 skipped**, entirely offline. `make ci` runs exactly what CI runs.

**Timings carry error bars, including this project's own.** Over 14 successful runs CI
averages **49.3 s ± 1.9 (SE)**, with a standard deviation of 7.2 s — 14.7% of the mean,
range 35–59 s. So a difference smaller than about 14 s between two single runs is not
evidence of anything, and earlier single-run figures quoted here (40 s, 47 s) were lucky
draws rather than measurements. Regenerate with
`uv run python scripts/ci_timing_study.py`.

| Gate | Statement | Status |
|---|---|---|
| **0.0** | Reproduce the propositions before building anything | **green** — see above |
| **G1** | Causality: scramble the future, the past stays bit-identical | **green** — 500 cuts × 20 seeds per strategy |
| **G2** | Twin engines agree to 1e-12 | **green** — `0.000e+00`, exact |
| **G3** | Analytic recovery on synthetic GBM | **green** — two Sharpe conventions, vol, drift |
| **G4** | Zero-cost identity | **green** — bitwise, both engines, all conventions |
| **G5** | Cost monotonicity, break-even cost | **green** — c\* = 27.03 bps per turn |
| **G6** | Null calibration, 1,000 coin flips | **green** — turnover matched to 0.35%, verified in 15 worlds |
| **G7** | Leakage trap: deliberately leaky pipelines must be caught | **green** — 5 traps rejected |
| **G8** | Purged, embargoed walk-forward | **green** — 3 splitters, purge + embargo asserted |
| G9 | PBO via CSCV over 12,870 splits, on `SelectionRule` | green; null calibrates to 0.5, fires on a compensation trap at 0.79 |
| G10 | Reproducibility from pinned hashes | green — two runs byte-identical across three figures and `metrics.json` |

Everything through G8 runs with no network, no API key and no rate limit. That is the whole point of
the ordering in `00`: the certified core is testable in CI without a single flaky test that fails
because Yahoo timed out.

### G2 came out exact, not merely within tolerance

The two engines are written independently — the event engine loops bar by bar over hard-sliced
prefixes, the vectorised one uses whole-series rolling operations and array arithmetic — and they
share nothing but the warm-up index. Agreement is bitwise across the strategy zoo, all three
execution conventions, and a cost sweep to 125 bps. A planted one-bar engine disagreement is caught
at `7.4e-02`, so the gate discriminates rather than passing vacuously.

The three conventions are genuinely distinct on the same series: `close_to_close` 12924.37,
`next_open` 12915.83, `next_close` 11688.41. That spread is execution-assumption risk.

### G3 recovers known truth, in both directions

| Quantity | Measured (200 paths) | True value | Gap |
|---|---|---|---|
| Simple-return Sharpe | +0.37513 ± 0.02358 | `μ/σ` = 0.40 | 1.05 SE |
| Log-return Sharpe | +0.27461 ± 0.02364 | `(μ−σ²/2)/σ` = 0.30 | 1.07 SE |
| Annualised vol | 0.20018 ± 0.00020 | `σ` = 0.20 | 0.09% rel |
| Annualised log drift | +0.05502 ± 0.00474 | `μ−σ²/2` = 0.06 | 1.05 SE |

**Both Sharpe conventions are asserted, and that is deliberate.** `00` Gate 0.1 states the true
Sharpe is 0.30 and warns that 0.40 means you used `μ` instead of `μ−σ²/2`. Both numbers are right,
for different estimators: the Sharpe of *log* returns is 0.30, the Sharpe of *simple* returns is
`μ/σ` = 0.40, because `E[exp(g)−1] = μ/252` exactly. Part E's equity recursion compounds simple
returns, so the engine measures 0.40 and is correct to — asserting 0.30 against it would reject a
working engine. Pinning both means confusing them fails one direction or the other, which is what
the gate was reaching for.

Two companions ship with it. The Monte Carlo standard error falls with a log-log slope of
**−0.5173** (r² = 0.997), confirming `1/√M` — that is the *estimator* error, reducible by
simulation, and not the estimate error, which shrinks only with history. And the power test:
mean reversion earns **+0.0051 ± 0.0738 on GBM** but **+1.1534 ± 0.0703 on a stationary AR(1)**.
The first says the engine invents no edge; the second says it does not destroy edge that exists,
which no real-data test can establish because there "found nothing" is always a plausible answer.

### G5 — the number that matters

Break-even cost **c\* = 27.03 bps per turn**, verified to separate profit from loss: +0.35 Sharpe at
half of c\*, −0.35 at 1.5×. Net Sharpe is monotone non-increasing across 0–100 bps, and
buy-and-hold's Sharpe moves by exactly `0.000e+00` across the whole sweep, which confirms cost is
charged on traded notional rather than on portfolio return. Faster rules die sooner: at 135 turns a
year the strategy is unprofitable even for free, at 49 turns it survives to 78 bps.

### G6 — the null, and the result that matters

A thousand coin flips through the entire pipeline, matched to the strategy's realised
**turnover and exposure**, both read off its own engine run. Turnover matching is the whole job:
a naive coin flip trades 249.7 times a year and earns −1.081 at 20 bps, while the matched null
trades 72.5 times and earns −0.477. Compare a strategy against the naive null and it clears the
bar trivially, for entirely the wrong reason.

**The headline, and it is the honest kind:**

| `CausalZScore(20)` | Sharpe | empirical p vs its own null |
|---|---|---|
| gross (0 bps) | +1.0566 | **0.0110** |
| net (20 bps) | +0.4853 | **0.1628** |

The edge is real against noise and **does not survive realistic costs** at the 5% level. Across
15 independent worlds the pattern holds: mean p = 0.029 gross, **0.172 net**.

The machinery is calibrated too. Under a true null PSR(0) is approximately uniform, and the
rejection rate at α must be about α — measured 0.085 / 0.040 against nominal 0.10 / 0.05. DSR
rejects all 1,000; the luckiest coin flip of a thousand reaches 0.18.

**Bounds set by a 15-world replication study** (`scripts/g6_replication.py`), not by one run —
and the study earned its keep. It showed the gate as first written **would have failed**: the
`< 3 SE` bounds hit 4.08 SE and 3.27 SE in some worlds. The cause was a statistical error in the
bound rather than in the null — the 1,000 nulls all trade one price path, so they are not
independent draws and the naive/binomial SE understates the true uncertainty. The bounds are now
scale-free or evidence-based, and the study verifies them: **15/15 worlds pass**. The 1% level is
reported but not asserted, because 1,000 draws cannot resolve a 1% tail — 10 expected exceedances
carries Poisson noise of ±3, and the worlds duly spread 0.003 to 0.020.

### The propagation layer

G5 showed turnover was doing more damage than the signal was doing good, with no dial to turn.
Two composable overlays now provide one. On `CausalZScore(20)`:

| | turnover/yr | gross SR | net SR @ 20 bps |
|---|---|---|---|
| base | 74.32 | +0.9464 | +0.3864 |
| `VolTarget` + `TurnoverBuffer` | **24.74** | +1.1176 | **+0.7076** |

Turnover −67%, net Sharpe **+83%**, while the gross Sharpe barely moves — which is what shows the
gain came from paying less transaction tax rather than from a different signal. G1's strict cut
caught a real bug on the way: `VolTarget` first sized on same-bar volatility while the base signal
was lagged a bar, setting the position from an information set the signal itself was not allowed
to use. It passed the Part A1 causality contract and failed execution alignment — precisely the
distinction the two cut modes exist to separate.

### G8 — walk-forward, and the integration layer

The gate's condition is blunt: **zero index overlap, asserted in code, not in docs.**
`Split` refuses to construct when train and test intersect, so a leaking partition is
*unconstructable* rather than caught later inside a Sharpe.

Three geometries, because "walk-forward" is not one thing. `ExpandingWindow` anchors at
bar 0 and grows — how a strategy is actually run. `RollingWindow` holds the training
length fixed, making folds comparable and deliberately forgetting the distant past.
`PurgedKFold` trains on both sides of the block: explicitly **not** a walk-forward, kept
because it is the geometry CSCV needs at G9 and the only one where the embargo does any
work. Purge and embargo are tested separately, because transposing them yields folds
that are non-overlapping, plausible and wrong.

The integration layer is the other half: `build_grid` pushes a configuration grid
through the certified engine, `walk_forward_select` splits it, selects on each training
block with a `SelectionRule`, and scores only out of sample. Built here rather than at
G9 on purpose — CSCV needs exactly a `(T, N)` grid, a block splitter and a rule, so G9
assembles certified parts instead of inventing them beside its own rank bookkeeping.

**Three assertions were written, measured, and then rewritten.** That is the substance:

*Draft one* asserted ArgMax degrades out of sample. It failed with OOS **above** IS
(+1.98 vs +1.19) — failure mode F6, which reads as a leak. It was not one: every fold
showed a purge gap of exactly 10 with train strictly before test. Over 20 seeds,
degradation is **−0.041 ± 0.125**, negative in 10 of 20. An 80-bar Sharpe carries an SE
near 1.77, so a six-fold mean carries 0.72 — the assertion was on noise. Replaced by the
exact structural claim (ArgMax's IS Sharpe *equals* the grid maximum, 20/20), with
degradation reported rather than gated.

*Draft two* expected the compensation effect — a negative IS→OOS slope. It measured
**+0.979 ± 0.085, positive in 20/20**, and that is correct rather than broken:

| Process | IS→OOS slope over 20 seeds | reading |
|---|---|---|
| AR(1), configs genuinely differ | **+0.979 ± 0.085**, >0 in 20/20 | ranking carries real information |
| GBM, no config has an edge | +0.215 ± 0.265, <0 in 8/20 | indistinguishable from zero |

A negative slope is the signature of selecting among configurations with **no true
differential merit**. On a stationary mean-reverting series where slow z-scores earn a
real edge and trend-followers genuinely lose, in-sample ranking *should* predict
out-of-sample. The compensation effect belongs to the memoryless case. So the slope is
reported, not gated — one GBM path puts it anywhere from −1.78 to +2.26 — and what is
asserted is what survives: a real edge survives the walk-forward, and none is
manufactured on GBM.

*Draft three* asserted all three splitters raise on 20 observations. `PurgedKFold` does
not, and should not — 20 observations is a perfectly legal 5-fold split.

### Suite performance

| | before | after |
|---|---|---|
| tests | 225 | **263** |
| skipped | 4 | **0** |
| local runtime | 49.8 s | **28 s** |
| CI total | 55 s | **47 s** (gate step 27 s) |

`pytest-xdist` at `-n auto` does the work, with **every test still at full strength** —
G1 keeps its spec-mandated 500 cuts × 20 seeds. The 4 skips were `TopK(3)` receiving a
grid narrower than 3 columns; N is now drawn `>= k` so every example exercises the rule.
A skip conditioned on a random draw is worse than a skip, because how much of the
property got checked varied run to run.

One thing parallelising broke and had to be fixed: **the collection floor stopped
working under xdist**, raising `UsageError` inside a worker where pytest surfaces it as
an `INTERNALERROR`. A countermeasure that evaporates the moment you change how tests run
is worse than none, so it now enforces controller-side and is verified firing in both
serial and parallel.

### The A4 ruling

`02` Part A4 asserts G1 must catch `LeakyOracle` — `sign(diff(close))` — and that a silent harness
is broken. **Ruled 2026-08-08: A1 stands, and the trap was mis-specified.** `close[t]` lies inside
`bars[0:t+1]`, which A1 permits, and every Part D convention lags the weight at least one bar, so
that strategy never trades on information it could not have had. Measured through the engine it
earns **+0.054** annualised Sharpe against buy-and-hold's +0.372; a strategy that genuinely sees one
bar ahead earns **+21.10**. Tightening A1 would have failed every legitimate `close_to_close`
strategy while catching nothing real. G7 now traps a true look-ahead oracle, and the A4 strategy is
kept as a documented non-violator so the reasoning is not lost.

---

## The number to hold onto

800 configurations, four years of daily data, pure noise: expected best annualised Sharpe ≈ **1.6**.
Report 1.4 on that grid and you underperformed randomness.

---

## Source

Bailey, Borwein, López de Prado & Zhu (2014), *Notices of the AMS* 61(5), 458–471.
Open access: https://www.ams.org/notices/201405/rnoti-p458.pdf

The two PDFs here are original exposition written to accompany that paper. They reproduce none
of it. Read the original.

## Prior art

`github.com/restiverumble/algorithmic-trading-backtester` — read as reference, not copied. No
LICENSE file, so all rights reserved. Build from scratch.
