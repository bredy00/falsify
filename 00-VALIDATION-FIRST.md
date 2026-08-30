# 00 — Validation First

> **Companion to:** `PLAYBOOK.md`. This document supersedes the phase ordering in that file.
> **Status:** binding. Nothing in `core/` gets written until Gate 0 is green.

---

## The correction you made

You said: *"the playbook should have a run this test before everything, that's what I think the playbook is missing, because we're running off assumptions right now."*

You're right, and it's a real gap. The original playbook put the data layer at Phase 1 and the engine at Phase 2, which means the first thing that happens is real yfinance data flowing through unverified code. Every number after that is an assumption wearing a result's clothing.

The fix is an ordering rule:

> **Gate 0 — no real market data enters the system until the engine has recovered known parameters from synthetic data with known truth.**

Synthetic data is the only data where you know the right answer in advance. That makes it the only data that can tell you whether your code is correct. Real data can tell you a strategy is profitable; it can never tell you your Sharpe function has a `ddof` bug, because there's nothing to compare against.

Physics analogue you'll recognise: you don't point a new detector at the sky first. You point it at a calibration source with a known spectrum, confirm you recover the known lines, and only then look at something you don't know the answer to.

---

## Gate 0.0 — reproduce the theory first

Before Gate 0, before the scaffold, before anything. numpy and scipy only, no engine, no data
layer. Three experiments, roughly sixty lines, under a minute to run. They convert the
overfitting propositions from things you have read into things you have watched happen.

These become `tests/gates/test_prop.py` and they are session one.

### Experiment A — the expected maximum

Draw `N = 1000` standard normals, take the max, repeat 10,000 times, average. Compare against

```
E[max z] = (1-γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))
```

**Expect** the empirical mean near 3.24 against a predicted 3.255 — agreement to about two
decimals. Then sweep `N` across four orders of magnitude and confirm growth tracks `√(2 ln N)`
from below. **Assert:** relative error < 1% at every `N`.

### Experiment B — the memoryless case

Generate `N = 1000` driftless Gaussian random walks of length `T = 1000`. Split each at the
midpoint into in-sample and out-of-sample halves. Scatter in-sample Sharpe against out-of-sample
Sharpe.

**Expect** a structureless circular cloud centred on the origin, and an OLS slope
indistinguishable from zero. Then select the single path with the highest in-sample Sharpe.
**Expect** its in-sample Sharpe in roughly 1.2 to 2.6 and its out-of-sample Sharpe scattered
around 0. **Assert:** `|slope| < 2·SE(slope)`, and the selected path's mean out-of-sample Sharpe
across repetitions is within 2 SE of zero.

### Experiment C — compensation effects

Rerun B twice more. First recentre every path to a common mean before splitting. Second, replace
the random walk with a stationary AR(1) at `φ = 0.995` (half-life `−ln2/lnφ ≈ 138` bars).

**Expect** in both cases a clearly negative slope in the in-sample vs out-of-sample scatter, with
the slope significant at any conventional level and the intercept not. **Assert:** `slope < 0`
with `p < 0.01`.

That negative slope is Propositions 3 and 5 rendered as a picture. Save the figure — it is the
single most important plot in this subject and it belongs at the top of the README.

**Why this ordering.** Experiment B says a lucky backtest costs you nothing. Experiment C says
that in a market with memory — which is every market — the same lucky backtest costs you money in
expectation, in proportion to how hard you searched. Build the framework only after you have seen
that with your own seed.

---

## Gate 0 — the pre-flight

Five checks. All five must pass before `data/loaders.py` is allowed to make a network call.

### 0.1 — Known-truth recovery

Generate geometric Brownian motion with parameters you chose:

```
S_t = S_0 · exp[ (μ − σ²/2)·t + σ·W_t ]
```

Discretised at daily frequency, log-returns are exactly `N((μ − σ²/2)/252, σ²/252)`. So the *true* annualised Sharpe of buy-and-hold, with zero risk-free rate, is known in closed form.

**Test:** generate 200 independent paths at `μ = 0.08`, `σ = 0.20`, `T = 2520` (ten years). Run buy-and-hold through the full engine. Assert:

| Quantity | True value | Pass condition |
|---|---|---|
| Mean estimated annualised Sharpe across 200 paths | `(μ − σ²/2)/σ = 0.30` | within 2 standard errors of the ensemble mean |
| Mean estimated annualised vol | `0.20` | within 1% relative |
| Mean estimated CAGR | `exp(μ − σ²/2) − 1 ≈ 6.18%` | within 2 SE |

If your Sharpe comes out at 0.40 because you used `μ` instead of `μ − σ²/2`, this catches it. If your annualisation is `√365` instead of `√252`, this catches it. Neither would ever be visible on AAPL data.

### 0.2 — The Monte Carlo convergence study

This is the thing you described: *"what if we simulated like a hundred times and we see that the error reduces."* Your instinct is right, but there is a distinction that will bite you if it isn't nailed down now, so it gets its own section below (**"Two kinds of error"**). Read that before implementing.

**Test:** for `M ∈ {10, 50, 100, 500, 1000, 5000}` paths, compute the standard error of the ensemble-mean Sharpe estimate. Fit `log(SE)` against `log(M)`.

**Pass condition:** slope within `[−0.55, −0.45]`. You are confirming the `1/√M` scaling law. If your slope is flat, your paths are correlated — most likely you seeded the RNG once outside the loop and every path is identical, or you used `np.random.seed` in a way that reuses state.

Save the log-log plot. It goes in the README. It is the single cheapest piece of evidence that you know what a Monte Carlo estimator is.

### 0.3 — Signal-recovery on a process with a *known* edge

GBM has no exploitable structure, so a strategy that beats buy-and-hold on GBM is a bug. Build the opposite case: an AR(1) mean-reverting log-price,

```
x_t = φ·x_{t−1} + ε_t,    ε_t ~ N(0, σ²),    0 < φ < 1
```

Here a z-score mean-reversion rule *does* have a true edge, and you can compute roughly what it should be. Two assertions:

- On GBM, the mean-reversion strategy's annualised Sharpe across 200 paths has an ensemble mean statistically indistinguishable from 0.
- On AR(1) with `φ = 0.95`, the same strategy has an ensemble mean Sharpe reliably above 0.

**This is your power test.** 0.1 shows the engine doesn't invent edge. 0.3 shows the engine doesn't destroy edge that exists. A framework that fails to find signal in a series that provably contains signal is broken in a way that no real-data test will reveal, because on real data "found nothing" is a plausible answer.

### 0.4 — Degenerate inputs don't silently produce numbers

Run the engine on each of these and assert the specified behaviour:

| Input | Required behaviour |
|---|---|
| Constant price series | Sharpe returns `NaN` or raises, never `0.0` |
| Series shorter than the strategy lookback | raises `InsufficientHistory`, never returns an empty frame |
| Single `NaN` in the middle of the price series | raises, never forward-fills silently |
| All-zero returns | Sharpe `NaN`, MDD `0.0`, CAGR `0.0` |
| Position array containing `NaN` | raises before the equity curve is computed |

The naive baseline returns `0.0` when the standard deviation is zero. A zero Sharpe and an undefined Sharpe are different claims and only one of them is true.

### 0.5 — Determinism

Run the entire Gate 0 suite twice with the same seed. Assert byte-identical output. Then run with a different seed and assert the results differ. Both directions matter: the first catches hidden global state, the second catches a seed that isn't actually being threaded through.

---

## Two kinds of error

This is the correction that changes your design, so it gets its own section.

You wrote about backtests having *"some irreducible error term"* and about simulating a hundred times to *"reduce the error."* Both ideas are correct, but they apply to two different quantities that must never be confused.

### Error of the estimator (reducible by simulation)

**Question:** does my Sharpe function compute the right thing?

**Ensemble:** 200 synthetic paths, all from the same known generator. Every path is an independent draw.

**Behaviour:** the standard error of the ensemble mean falls as `1/√M`. Simulating more paths genuinely does reduce this error. Gate 0.2 measures exactly this.

**What it buys you:** confidence that the code is right.

### Error of the estimate (not reducible by simulation)

**Question:** what is SPY's true Sharpe over 2020–2024?

**Ensemble:** there is one. One path. `n = 1`. The universe ran the experiment once and didn't keep a control group.

**Behaviour:** the standard error is approximately

```
SE(SR) ≈ √( (1 + SR²/2) / T )
```

and it shrinks only with `T` — with more *history*, not with more *runs*. Backtesting SPY a hundred times gives you the same number a hundred times.

**What it buys you:** nothing, if you run it repeatedly. The error is a property of how much data exists.

### The consequence

| | Estimator error | Estimate error |
|---|---|---|
| Reduce by | more synthetic paths | more history, or nothing |
| Where it lives | Gate 0 | the actual result |
| Tool | Monte Carlo | bootstrap, DSR, walk-forward |

The bootstrap in Phase 5 does **not** reduce the estimate error. It *measures* it. Resampling your one SPY path 10,000 times manufactures a pseudo-ensemble so you can put an interval around the number. The interval doesn't get narrower because you bootstrapped more; it converges to the true width faster. That distinction is worth internalising because half the people who use bootstraps in finance believe they've added information, and they haven't. They've only quantified how little they had.

### Where the actual irreducible error is

Your instinct about an irreducible term is sound but points at the wrong dominant source. Decompose the expected squared error of your out-of-sample performance prediction:

```
E[(realised − predicted)²]  =  bias²  +  variance  +  σ²_irreducible
                                 ↑         ↑            ↑
                          selection    estimation    market
                             bias      noise (1/T)   randomness
```

The market's randomness `σ²` is genuinely irreducible; nothing you build touches it. But it is **not the biggest term**. The biggest term in a typical retail backtest is the selection bias — the `bias²` component — which arises entirely because you chose the strategy after looking at the data. That term is *reducible*. It's reducible by not doing that, or by correcting for it, which is what DSR and PBO exist for.

So the thesis sharpens: **your project attacks the reducible error that everyone else calls irreducible.**

---

## On "ML wearing a statistics T-shirt"

You had it right on the second guess. Bias–variance decomposition and the irreducible error term `σ²` come from statistics — they're standard in regression theory long before machine learning existed as a field. ML adopted the vocabulary and made it famous. So: **ML wearing a statistics T-shirt.**

Worth knowing because it tells you where to look things up. When you want the rigorous treatment, you search statistics literature, not ML tutorials.

---

## Revised phase ordering

Replaces Part 4 of `PLAYBOOK.md`.

```
Phase 0    Scaffold + CI                          (unchanged)
Phase 0.5  Synthetic generators + Gate 0    ← NEW, blocking
Phase 1    Engine on synthetic data only     ← moved up
Phase 2    Twin-engine agreement (G2)             on synthetic
Phase 3    Cost model (G4, G5)                    on synthetic
Phase 4    Metrics (G3)                           on synthetic
──────────  first network call happens here  ──────────
Phase 5    Data layer, cache, manifest
Phase 6    Real-data smoke test
Phase 7    Statistical validation (G6, G8, G9)
Phase 8+   Strategy zoo, attribution, reporting
```

Everything through Phase 4 runs offline. That's a bonus: the whole certified core is testable in CI with no network, no API keys, no rate limits, and no flaky test that fails because Yahoo timed out.

---

## Yes, run it locally first

You said you want to try this on a local machine before trusting the reading. Correct instinct, and Gate 0 is designed for it. Minimum viable local loop:

```bash
uv venv && source .venv/bin/activate
uv pip install numpy scipy pandas pytest hypothesis matplotlib
pytest tests/gates/test_gate0.py -v
```

No network, no yfinance, no data directory. If Gate 0 passes on your machine in under thirty seconds, the foundation is real. If it takes five minutes, your synthetic generator is looping in Python where it should be vectorised, and that will matter enormously at Gate 6 when you run a thousand strategies.

**Target:** full Gate 0 suite under 30 seconds. Enforce it with `pytest --durations=10` and treat a regression as a bug.
