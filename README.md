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
