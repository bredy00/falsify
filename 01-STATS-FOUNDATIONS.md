# 01 — Statistical Foundations

> **Purpose:** the vocabulary and the mathematics, stated precisely once so the rest of the build can reference it instead of re-deriving it. Also the corrections to the terms you used, because two of them would have led you somewhere wrong.

---

## Part A — Terms, corrected

You asked to be corrected on your terms. Six of them, in the order you raised them.

### A1. "Backtests should be estimations / simulations"

**Correct, with one sharpening.** A backtest is a *simulation of a decision process over a realised path*, which makes it an estimator of an unknown parameter. But it is not a Monte Carlo simulation — a distinction that matters because "simulation" invites you to think there's an ensemble.

There isn't. A backtest on SPY 2020–2024 is one draw. You cannot average it with anything. Everything in Phase 5 of the main playbook exists to manufacture a pseudo-ensemble out of that single draw:

| Method | How it fabricates an ensemble |
|---|---|
| Bootstrap | resample blocks of the observed path |
| Walk-forward | slice the path into quasi-independent windows |
| Cross-validation (CSCV) | recombine blocks into many train/test partitions |
| Synthetic paths | generate new paths from a fitted or assumed process |

Each is a different bet about what "another draw from the same world" would look like. None of them is free.

### A2. "Irreducible error is ML wearing a statistics T-shirt"

**Right on the second attempt.** The bias–variance decomposition and the irreducible term are statistics; ML borrowed them. See `00-VALIDATION-FIRST.md` for why the irreducible term is *not* your dominant error source — the selection bias is, and unlike the irreducible term it can be attacked.

### A3. "Every integer that gets tried by the algorithm"

**Term:** *configuration*, or *trial*. Not integer — the parameters need not be integers (a volatility target of 0.15 is a configuration), and a trial can vary the strategy family, not just its numbers.

Define it precisely, because `N` in the Deflated Sharpe formula is exactly this count:

> A **trial** is one complete evaluation of one fully specified strategy on one dataset, producing one return series.

Changing the lookback from 20 to 21 is a new trial. Changing the ticker is a new trial. Changing the cost assumption to see if it "still works" is a new trial. Re-running after fixing a bug is *not* a new trial, provided you discard the pre-fix results and never look at them again — which is why the ledger records both and marks superseded rows.

### A4. "They have a p-value and a bootstrap CI, which is basically just stats"

**Roughly right, wrong in one place that matters.** Each configuration does produce a point estimate to which you can attach both. But:

**Correction 1 — a p-value and a CI are the same object seen twice.** A 95% CI is the set of null hypotheses you would not reject at α = 0.05. If the 95% CI for Sharpe excludes zero, then `p < 0.05` for the null "Sharpe = 0." They are not independent pieces of evidence and quoting both as if they were is double-counting.

**Correction 2 — this is the important one — your N trials are not independent.** SMA(20,50) and SMA(21,50) trade almost identically; their return series might correlate at 0.98. Treating them as two independent tests badly overstates how much searching you did in a *statistical* sense.

Consequence: the honest `N` in the DSR is not the raw grid size but an **effective** number of trials, and it is much smaller. Estimating it is a real task, specified in Part C below. Skipping it makes your DSR wildly conservative — you'll reject strategies that are actually fine. Assuming it away makes your DSR wildly optimistic. Both failure modes are live.

### A5. "You graph out how many times the CI / p-values intersect and write an estimated value"

**Your intuition maps onto two real, named methods.** What you're reaching for is: *given a whole family of results, how do I make one honest statement about the family?* That's the multiple-comparisons problem, and it has established answers:

| What you described | The established method | What it controls |
|---|---|---|
| Counting how many CIs exclude zero | Benjamini–Hochberg FDR | expected proportion of false discoveries among your rejections |
| Judging the *best* result against the family | White's Reality Check (2000) | probability the best of N beats a benchmark by luck |
| Same, more powerful | Hansen's SPA (2005) | as above, less sensitive to bad strategies padding N |
| Correcting the winner's statistic | Deflated Sharpe Ratio | as above, expressed as a probability the true Sharpe is positive |

For this project, use DSR as the headline and White's Reality Check as the cross-check. Two independent routes to the same conclusion is worth more than either alone, and when they disagree you've learned something about your effective-N estimate.

**Do not** invent a procedure by eyeballing where confidence intervals overlap on a plot. Visual CI-overlap reasoning is a known trap: two estimates with overlapping 95% CIs can still differ significantly, and the error rate of the eyeball method is not something you can quote.

### A6. "Apply a regression model, find the pattern, then optimize it"

**Half right, and the wrong half is dangerous.**

**The dangerous version:** fit a surface `Sharpe = f(params)`, find its maximum, adopt those parameters. You have now overfit at a second level. Your grid search overfit the price data; your surface fit overfit the grid search. Nothing in your validation machinery will catch it, because the machinery measures the first level only. Worse, the fitted surface will look smooth and convincing, which is exactly the pseudo-mathematics the AMS paper is about.

**The legitimate version:** fit the surface and *never take its argmax*. Use it to ask a robustness question:

- Is the high-Sharpe region a **broad plateau** or an **isolated spike**? A plateau means neighbouring parameters work too, which is weak evidence of real structure. A spike means one lucky cell.
- Compute a **plateau score**: mean Sharpe of the 8 grid neighbours divided by the centre cell's Sharpe. Near 1.0 is a plateau. Near 0 is a spike, and you discard it regardless of how good the centre looks.
- Plot the in-sample surface next to the out-of-sample surface for the same grid. If the second is structureless noise, you have a picture of overfitting that needs no caption.

Report the plateau score alongside the Sharpe. It's cheap, it's honest, and almost nobody does it.

---

## Part B — The estimators, stated precisely

All formulas in **per-observation** (daily) units unless marked. Annualise only at the reporting boundary. Mixing units is the single most common bug in this area.

### B1. Sharpe ratio and its standard error

```
SR = (mean(r) − r_f) / std(r, ddof=1)
SR_annual = SR · √252
```

Under i.i.d. normal returns, Lo (2002):

```
SE(SR) ≈ √( (1 + SR²/2) / T )
```

With non-normal returns, using skewness `g3` and **non-excess** kurtosis `g4`:

```
SE(SR) ≈ √( (1 − g3·SR + ((g4 − 1)/4)·SR²) / (T − 1) )
```

**Convention check.** Normal returns give `g3 = 0`, `g4 = 3`, so the numerator collapses to `1 + SR²/2` and the two expressions agree. If yours doesn't, you passed excess kurtosis where non-excess was expected. `scipy.stats.kurtosis` defaults to `fisher=True`, which returns *excess*. Pass `fisher=False`.

**Autocorrelation.** Strategy returns are serially correlated, which inflates the true standard error above both formulas. Report the Newey–West corrected t-statistic on the mean excess return with lag

```
L = floor(4·(T/100)^(2/9))
```

For `T = 1008`, `L = 7`. Use this t-statistic, not the naive one, whenever you make a significance claim.

### B2. Probabilistic Sharpe Ratio

Probability the true Sharpe exceeds a benchmark `SR*`:

```
PSR(SR*) = Φ[ (SR − SR*)·√(T − 1) / √(1 − g3·SR + ((g4 − 1)/4)·SR²) ]
```

`PSR(0)` is the probability the strategy has any edge at all, ignoring selection. That's the number to report for a strategy you did not search for.

### B3. Deflated Sharpe Ratio

Set the benchmark to the expected best-of-N result under the null. With `γ ≈ 0.5772` (Euler–Mascheroni), `e` Euler's number, and `V = Var(SR_n)` across your N trials:

```
SR_0 = √V · [ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]
DSR  = PSR(SR_0)
```

Reference implementation:

```python
import numpy as np
from scipy.stats import norm, skew, kurtosis

EULER = 0.5772156649015329


def expected_max_sharpe(n_trials: int, var_across_trials: float) -> float:
    """SR_0: expected maximum per-observation Sharpe under the null."""
    if n_trials < 2:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return np.sqrt(var_across_trials) * ((1 - EULER) * a + EULER * b)


def psr(returns: np.ndarray, sr_benchmark: float) -> float:
    """Probabilistic Sharpe Ratio. All inputs per-observation."""
    T = len(returns)
    sr = returns.mean() / returns.std(ddof=1)
    g3 = skew(returns, bias=False)
    g4 = kurtosis(returns, fisher=False, bias=False)  # NON-excess
    denom = np.sqrt(1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr**2)
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / denom))


def deflated_sharpe(returns: np.ndarray, all_trial_sharpes: np.ndarray) -> float:
    sr0 = expected_max_sharpe(len(all_trial_sharpes), all_trial_sharpes.var(ddof=1))
    return psr(returns, sr0)
```

**Unit test that must pass:** with `g3 = 0`, `g4 = 3`, `psr` must equal `norm.cdf((sr − sr_b) * sqrt(T−1) / sqrt(1 + sr**2/2))`. Assert to 1e-12.

### B4. Effective number of trials

Raw `N` overstates the search when configurations are correlated. Two acceptable estimators — implement both and report the pair.

**Method 1 — clustering.** Build the `N×N` correlation matrix of trial return series. Convert to a distance `d_ij = √(0.5·(1 − ρ_ij))`. Hierarchical-cluster with average linkage and cut at a fixed height (0.5 is a reasonable default, stated in the README). `N_eff` is the number of clusters.

**Method 2 — participation ratio.** With `λ_i` the eigenvalues of the correlation matrix:

```
N_eff = (Σ λ_i)² / Σ λ_i²
```

This is the standard participation ratio and it needs no threshold choice, which makes it the more defensible of the two. Report both; if they differ by more than a factor of two, say so and use the smaller (conservative).

Then compute DSR with `N_eff`, and also with raw `N` as a lower bound. Report the interval.

### B5. Stationary bootstrap

Politis–Romano (1994). Geometric block lengths with mean `1/p`, wrapping at the end, which preserves the autocorrelation an i.i.d. bootstrap destroys.

```python
def stationary_bootstrap(x: np.ndarray, p: float, n_boot: int, rng) -> np.ndarray:
    T = len(x)
    out = np.empty((n_boot, T), dtype=x.dtype)
    for b in range(n_boot):
        idx = np.empty(T, dtype=np.int64)
        idx[0] = rng.integers(T)
        new_block = rng.random(T) < p
        jumps = rng.integers(0, T, size=T)
        for t in range(1, T):
            idx[t] = jumps[t] if new_block[t] else (idx[t - 1] + 1) % T
        out[b] = x[idx]
    return out
```

**Choosing `p`.** Default `p = 1/√T`, giving mean block length `√T` (≈32 bars at T=1008). Report sensitivity across `p ∈ {1/T^(1/3), 1/√T, 1/T^(2/3)}`. If your confidence interval width moves by more than 20% across that range, the autocorrelation structure is doing real work and you should say so rather than pick the flattering one.

**Validation test:** bootstrap an i.i.d. normal sample. The bootstrap SE of the mean must match `σ/√T` within Monte Carlo error. If it doesn't, the resampler is broken.

### B6. PBO via CSCV

```
1.  M  ← T×N matrix of trial return series
2.  split rows into S contiguous blocks, S even (S = 16)
3.  for each of C(S, S/2) choices of half the blocks as in-sample:
        n*  ← argmax over columns of in-sample Sharpe
        ω   ← rank of column n* among the N out-of-sample Sharpes
        r   ← ω / (N + 1)
        λ   ← ln(r / (1 − r))
4.  PBO ← fraction of splits with λ ≤ 0
```

`C(16,8) = 12,870`. Vectorise the Sharpe computation across all N columns at once; the loop is over splits only.

**Reading it:** `PBO ≈ 0.5` means your selection rule is a coin flip. `PBO > 0.5` means it is worse than a coin flip, which is the empirical signature of the mean-reversion result — your in-sample winner is systematically the wrong pick. `PBO < 0.2` means the procedure carries information.

**What PBO is not:** it does not say the strategy works. It says whether the *method you used to choose it* works. Different claims.

---

## Part C — The trials ledger

`N` is the input to everything in Part B, and human memory is not an acceptable source for it. Make the harness record it.

```python
@dataclass(frozen=True)
class TrialRecord:
    trial_id: str  # uuid4
    timestamp: str  # ISO 8601, UTC
    git_sha: str
    data_manifest_hash: str
    strategy: str
    params: dict
    universe: tuple[str, ...]
    date_range: tuple[str, str]
    cost_bps: float
    sharpe: float
    n_obs: int
    superseded_by: str | None = None
```

Rules:

1. **Every** engine invocation writes a row. Not optional, not conditional, no `if not debug`.
2. Rows are append-only. A bug fix marks old rows `superseded_by`; it does not delete them.
3. `N` is read from the ledger by counting non-superseded rows matching the reporting scope. It is never a hand-typed constant.
4. The ledger ships in the repo. A reader can verify your `N`.

This one file is the difference between a project that discusses backtest overfitting and a project that measures its own.

---

## Part D — Reporting contract

Every performance claim in the README carries all six fields. No exceptions, including for results you like.

```json
{
  "sharpe_annual":        1.42,
  "sharpe_ci95":          [0.31, 2.48],
  "n_trials_raw":         800,
  "n_trials_effective":   47,
  "deflated_sharpe":      0.61,
  "pbo":                  0.34,
  "min_backtest_length_years": 10.2,
  "actual_history_years": 4.0,
  "break_even_cost_bps":  11.3,
  "newey_west_t":         1.87
}
```

Note the two fields in tension: `min_backtest_length_years: 10.2` against `actual_history_years: 4.0`. When those don't reconcile, the honest headline is that the result is uninterpretable at this sample length — and printing that is the point of the project.

---

## Part E — Selection rules

Added after the first build briefing. Two questions came up that change the architecture, so they get specified here rather than left in conversation.

### E1. The CI-intersection idea, made legitimate

You described graphing where confidence intervals and p-values intersect and reading an estimate off the crossing. Two things about that.

**Why the naive version fails.** Comparing two estimates by checking whether their 95% CIs overlap is a known trap. For two independent estimates with comparable standard errors, non-overlapping 95% intervals imply roughly `p < 0.006` — far stricter than 0.05, so you miss real differences. Meanwhile `p = 0.05` corresponds to intervals that overlap by about half the average margin of error, so you can have visible overlap and a significant difference at the same time. The error rate of the eyeball test is not constant and not quotable. When you want to compare two things, build the CI on their **difference**, never compare two separate CIs.

**Why the intuition is right anyway.** What you're reaching for — one honest statement about a whole family of results, read off a crossing point on a graph — is exactly the Benjamini–Hochberg procedure, and it is literally a graph with an intersection.

```
1.  compute p_n for each of the N configurations (null: Sharpe ≤ 0)
2.  sort ascending:  p_(1) ≤ p_(2) ≤ … ≤ p_(N)
3.  plot p_(i) against i, and overlay the line  y = i·α/N
4.  find the largest i where the sorted curve is still BELOW the line
5.  reject the null for configurations 1 … i
```

The crossing point of those two curves is the decision boundary, and the procedure controls the **false discovery rate** — the expected proportion of your rejections that are false — at level `α`. That is a real, named, defensible answer to the question you were asking, and the plot is worth putting in the README.

Use it as the family-level view. Use DSR and White's Reality Check for the best-of-N question. They answer different questions: BH asks "how many of these are real," DSR asks "is the winner real."

### E2. Softmax instead of argmax

You asked what happens if the selection rule takes a softmax over configurations rather than the argmax. It's a good question and the answer is better than you'd guess.

**Definition.** Standardise the in-sample Sharpes across the grid to z-scores `z_n`, then

```
w_n = exp(z_n / τ) / Σ_m exp(z_m / τ)
r_blend[t] = Σ_n w_n · r_n[t]
```

Standardising first is not cosmetic: it makes the temperature `τ` interpretable as "one cross-sectional standard deviation" rather than an arbitrary scale. Subtract `max(z)` before exponentiating or you will overflow.

`τ → 0` recovers argmax. `τ → ∞` recovers the equal-weighted grid average. So temperature is a continuous dial between maximum selection bias and none, which makes softmax a **shrinkage estimator** — the same family as James–Stein, ridge, and Bayesian model averaging.

**What improves.**

*The extreme-value penalty largely evaporates.* Under the null, argmax gives you `E[max of N draws] ≈ √(2 ln N)`, which is the whole problem. Softmax gives you a weighted mean, whose expectation under the null is zero and whose variance falls roughly as `1/N_eff`. The deflation required collapses. You have not corrected the selection bias — you have declined to incur most of it.

*Under compensation effects, it converts a loss into a wash.* This is the sharp result. Propositions 3 and 5 say that with memory in the process, the in-sample winner is the out-of-sample loser, exactly. So argmax doesn't merely fail to help — it reliably selects the configuration most likely to reverse. Softmax dilutes that: as `τ` rises, the blend approaches the grid mean, whose expected out-of-sample Sharpe is zero rather than negative. **Softmax turns the negative expectation into a zero one.** For a memoryless process it's a wash either way. There is no regime in which argmax is safer.

*Turnover usually falls.* When configurations disagree, their weighted average position is smaller and smoother than any individual position. Lower turnover, lower cost. Watch the confound though — see below.

**What it costs.**

*Genuine edge gets diluted.* If one region of the grid holds real signal, blending it with 799 configurations that don't will wash it out. The trade-off is governed by the shape of the surface, which is why the plateau score from Part A6 matters: a broad plateau survives softmax nearly intact, an isolated spike does not. Since spikes are almost always noise, that asymmetry works in your favour.

*Gross exposure drops, and that confounds the comparison.* A blend of disagreeing configurations may run at 30% gross exposure while argmax runs at 100%. Comparing their Sharpes is fine (Sharpe is scale-invariant), but comparing CAGR or drawdown is not. **Renormalise the blend to matched gross exposure before any comparison**, and assert the match in a test.

*Choosing `τ` is itself a trial.* Sweeping temperature and keeping the best is the original sin at one level up. Either fix `τ` a priori, or select it by nested cross-validation on the training blocks only, and either way add the temperature sweep to the trials ledger.

**The connection worth knowing.** Exponential weighting over N experts is the Hedge algorithm from online learning, and its regret against the best expert in hindsight is bounded by `O(√(T ln N))`. The same `ln N` that appears in `√(2 ln N)` from extreme-value theory. Two entirely different fields, one measuring the danger of selection and the other the price of learning, arrive at the same logarithmic dependence on the number of options. That is not a coincidence and it is worth a paragraph in the research note.

### E3. Architectural consequence

Make the selection rule a first-class object, and validate **the rule** rather than the strategy.

```python
class SelectionRule(ABC):
    @abstractmethod
    def weights(self, is_returns: np.ndarray) -> np.ndarray:
        """T_is × N in-sample returns → length-N weights summing to 1."""


class ArgMax(SelectionRule): ...


class Softmax(SelectionRule):
    def __init__(self, temperature: float): ...


class EqualWeight(SelectionRule): ...


class TopK(SelectionRule):
    def __init__(self, k: int): ...
```

Then generalise PBO. The standard definition assumes argmax; replace "rank of the argmax column" with "rank of the rule's portfolio among the N individual out-of-sample Sharpes." That gives `PBO(rule)`, and the headline figure of the whole project becomes:

> **PBO as a function of softmax temperature**, with `argmax` at `τ = 0` and `EqualWeight` as the asymptote.

Expect it to decrease monotonically. If it does, you have measured the price of selectivity on your own data, in one plot, with a named quantity on each axis. Nobody's student repo has that figure.

**Build order note.** Specify `SelectionRule` *before* implementing G9. Retrofitting CSCV from a hardcoded argmax to a rule interface means rewriting the rank bookkeeping, which is the fiddliest code in the project.
