# Session log — 2026-08-08 — A4 ruling, G3–G5, and the yellow-gate investigation

Two things happened in this session: the offline core was completed (G3, G4, G5), and
three gates were then reported as failing and had to be investigated. The second part
is the more useful record, so it is written up first.

---

## 1. The report: "G2, G3, G4 are yellow, not green"

The gates were reported as broken on the basis of experimental data, with three
specific claims. Each was investigated by stressing the property far past what its
own gate covers, rather than by re-running the gate that had already passed.

The stress harness is committed as `scripts/health_check.py` so any of this is
repeatable.

### Claim 1 — "the propagation layer isn't passing on both engines" (G4)

**Not reproduced.** 288 combinations of (engine × convention × process × length ×
seed): **288/288 bitwise identical**. Zero-cost buy-and-hold equals the benchmark
curve exactly, in both engines, under all three conventions, across GBM and AR(1)
at T ∈ {50, 137, 400, 1500} and six seeds each.

Three edge probes were added on top, all correct:

| Cost model on buy-and-hold | Effect on equity | Why that is right |
|---|---|---|
| `cash_yield_annual=0.05` | none | exposure is 1, so `(1 − \|w\|) = 0` |
| `borrow_bps_annual=300` | none | `max(−w, 0) = 0` for a long position |
| `commission_bps=50` | none | turnover is 0 after the anchor bar |

### Claim 2 — "implied value larger than actual" / "volatility and mean confused" (G3)

**Half right, and the half that was right mattered.**

There is no bias in the expectation value. Convergence to truth is textbook:

| Paths | gap | absolute error |
|---|---|---|
| 200 | 1.07 SE | 0.025407 |
| 800 | 0.07 SE | 0.000790 |
| 3000 | 0.09 SE | 0.000543 |

The absolute error falls 47× as paths grow, which is `1/√M` convergence, not bias.
The 1.05 SE gap visible at 200 paths was that seed set and nothing else.

**But the targets were first-order approximations, and that was a real defect in a
gate whose entire job is known-truth recovery:**

| Quantity | was asserted | exact lognormal | error |
|---|---|---|---|
| Simple-return Sharpe | `μ/σ` = 0.400000 | **0.399921** | 0.0198% |
| Simple-return vol | `σ` = 0.200000 | **0.200071** | 0.0357% |

A simple return is `exp(g) − 1` for normal `g`, i.e. lognormal, so
`E[r] = exp(m + s²/2) − 1` and `sd[r] = exp(m + s²/2)·√(exp(s²) − 1)`. `σ` is the
volatility of the **log** returns; the simple returns are right-skewed and slightly
more volatile. Both targets are now computed exactly in
`falsify.metrics.gbm_simple_return_sharpe` / `gbm_simple_return_vol`.

On "pick one before working on the other": the opposite is what protects us, and it
is now enforced by a much sharper test. The two conventions differ by exactly `σ/2`:

```
μ/σ − (μ − σ²/2)/σ = σ/2 = 0.10
```

Measured as a **paired** difference on the same paths, so path-to-path noise cancels:

```
paired gap = +0.100001 ± 0.000330   exact = +0.100000   gap = 0.00 SE
SE is 2.6% of the level SE — a ~38× tighter constraint on the pair
```

That is why asserting both conventions prevents a mix-up rather than inviting one:
swapping them, or applying one estimator's annualisation to the other's returns,
moves this difference off `σ/2` and fails **even when both level tests still pass**.

### Claim 3 — "G2 is fucked up as well"

**Not reproduced.** 1,890 combinations (7 strategies × 3 conventions × 5 cost models
× 3 lengths × 3 seeds × 2 processes), checking **all six `Result` fields** rather
than equity alone:

```
worst relative deviation: 0.000e+00
```

Not "within 1e-12" — identically zero, on equity, weights, gross_ret, net_ret, costs
and turnover, including a cost model at 340 bps with cash yield and borrow active.

### Claim 4 — G5

Agreed, left alone. 12/12 sweeps monotone across three seeds and four strategies.

One correction for the record: the `0.00e+00` figure belongs to **G2**'s
twin-engine deviation and to Gate 0.0's quadrature identity. G5's headline number is
the break-even cost, `c* = 27.03 bps` per turn on the gate's fixture, ranging 41.89
to 93.49 bps across the health-check seeds.

### Verdict

G2 and G5 were green and remain green. G3 carried a real defect — approximate
targets — now fixed. G4 was correct but **under-covered**, which is a fair reading of
"needs functionality improvements"; see below.

---

## 2. What was actually improved

### G3 — exact targets and a paired invariant

- targets computed exactly from the lognormal, not first-order
- volatility asserted against the exact simple-return vol **and** within 3 SE, which
  is ~5× tighter than the spec's 1% band
- new `test_g3_the_two_sharpe_conventions_differ_by_sigma_over_two`

### G4 — the coverage gap that was genuinely there

G4 only ever exercised `w ≡ 1`. **1 is a fixed point of most weight bugs**: a weight
applied with the wrong sign, scaled by exposure twice, or aligned a bar off at
fractional exposure all leave `w = 1` untouched. An identity that holds only at
`w = 1` is not an identity.

Added, 33 tests:

- constant exposure at `w ∈ {1, 0.5, 0.25, −0.5, −1}` × 3 conventions × 2 engines,
  each checked against `capital · Π(1 + w·r)` built by `cumprod`
- both engines agree on the **weights array itself**, not only on equity
- the short leg pays borrow at exactly `bps/10⁴/252` per bar, and a long position is
  **not** charged it — which catches `max(−w, 0)` having the wrong sign, a mistake
  that leaves a perfectly plausible-looking equity curve

---

## 3. Two places where exactness is not available, and why

Both surfaced as failing assertions I had written too strictly. Recording them
because the distinction is easy to mistake for a defect, and the reasoning is now in
the test files.

**`net_ret` vs `gross_ret` at zero cost.** Mathematically equal. Not bitwise: `net_ret`
is measured back off the equity path as `equity[k]/equity[k−1] − 1`, so it takes a
multiply, a divide and a subtract to recover a number `gross_ret` reached directly.
Tolerance `1e-12`.

**`capital · cumprod(factors)` vs the engine recursion.** `cumprod` forms the product
of growth factors and scales by capital once at the end; the engine multiplies the
running equity every step. Different association, and **floating-point multiplication
is not associative** — so they agree to ~1e-15 and cannot agree bitwise. Requiring
exactness there would be requiring associativity. Tolerance `1e-13`.

The comparisons that **are** exact — engine equity vs `benchmark_equity`, and the twin
engines against each other — are exact because both sides are sequential recursions
from the same base, so the association matches.

---

## 4. The runtime scare

Reported as 51 s → 3:38 after the G3/G4 additions. Investigated by isolation rather
than accepted:

| Isolated measurement | Time |
|---|---|
| All 33 **new** G4 tests | 1.39 s |
| G4 whole file | 2.28 s |
| G3 whole file | 16.22 s |
| G1 alone — **code unchanged** since the 51 s run | 34.01 s |

G1 took ~23 s inside the 51 s run and 34 s in isolation later, on byte-identical
code. The new tests account for ~13 s. **The swing was machine load and cache
warmth, not a regression.** Confirmed by re-running the full suite on the same
commit: **46.76 s**, against 218.91 s an hour earlier.

Conclusion: steady state is ~45–55 s locally; CI is the authoritative figure because
its environment is consistent. No code change made, and none warranted — inventing a
fix for a non-regression would have been worse than leaving it.

Earlier in the session a **genuine** 33 s regression was found and fixed:
`bars_from_close` called `np.busday_offset` on every invocation and G1 rebuilds the
pipeline 10,000 times per strategy. The index is a pure function of `(start, n)`, so
it is now cached and returned read-only. That took the suite from 2:11 to 51 s.

---

## 5. Final state

```
183 collected · 179 pass · 4 skip · 46.76 s · offline
ruff clean · mypy --strict clean (31 files) · action refs resolve
figure bytes identical across two runs
```

Health check, all five: **PASS**

```
G2 twin engines        worst 0.000e+00 across 1890 combinations
G3 recovery            worst level gap 0.90 SE, paired gap 0.00 SE
G4 zero-cost identity  288/288 bitwise identical
G5 cost monotonicity   12/12 sweeps monotone
Invariants B7-B10      0 violations
```

Green: Gate 0.0, G1, G2, G3, G4, G5, G7. Partial: G10. Not started: G6, G8, G9.
No market data has entered the system — **B1 still holds**.

---

## 6. The A4 ruling (first half of the session)

`02` Part A4 asserts G1 must catch `LeakyOracle` = `sign(diff(close))`, and that a
silent harness is broken. **Ruled: A1 stands, the trap was mis-specified.**

I had framed this as a one-line either/or. It was a false dichotomy and either
choice would have been wrong.

- **Structurally:** `close[t]` is inside `bars[0:t+1]`, which A1 permits, and every
  Part D convention lags the weight at least one bar — so the weight earning return
  `t` was decided from strictly older bars, under every convention.
- **Empirically**, which is what settled it, over 40 GBM paths through the engine:

| Strategy | mean annualised Sharpe |
|---|---|
| A4's `sign(diff(close))` | **+0.054** |
| a true look-ahead oracle | **+21.10** |
| buy-and-hold | +0.372 |

It earns nothing. Tightening A1 to `bars[0:t]` strictly would have failed every
legitimate `close_to_close` strategy — which Part D explicitly permits — while
catching nothing real. G7 now traps a genuine `LookAheadOracle`; the A4 strategy is
kept as a documented non-violator with the reasoning attached.

---

## 7. Next session — G6

First gate that needs the data layer. Compute is trivial; `03` Part C is explicit
that the hard part is **turnover matching**. A null flipping every bar has enormous
turnover and gets destroyed by costs, which would make the real strategy look good
for entirely the wrong reason.

G5 already supplies the numbers to match against: `c*` between 41.89 and 93.49 bps
depending on window, at 40–138 turns a year — and at 134 turns a year `CausalZScore(5)`
is unprofitable **even for free**. So turnover matching is not a formality; it is the
difference between a calibrated null and a flattering one.
