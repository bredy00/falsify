# 02 — Engine Specification

> **Purpose:** the exact interfaces, equations and conventions. Precise enough that an agent with no other context can implement it and two implementations will agree bit-for-bit.

---

## Part A — The causality contract

### A1. You named it; here is the definition

You said: *"we need to randomize data for more precise error, but we cannot do that from the start then we wouldn't have an algorithm, so just introduce the index."*

Exactly right. The index is called the **causality cut**, written `τ`.

> **Definition.** For a price series of length `T` and a cut `τ ∈ [L, T)` where `L` is the strategy's declared lookback, the *scrambled series* `P^τ` is identical to `P` on `[0, τ]` and replaced by independent noise on `(τ, T)`.
>
> **Contract.** For any strategy `S` and any `τ`, the signals satisfy
>
> ```
> S(P)[0:τ+1]  ==  S(P^τ)[0:τ+1]     exactly, bitwise
> ```

Everything at index `s ≤ τ` must be invariant to arbitrary mutilation of everything at `s > τ`. In physics terms: the signal at `τ` depends only on the past light cone. In code terms: if you can change the future and the past moves, you have leakage.

### A2. One correction to what you said about the reference repo

You wrote: *"there is clear leakage from what this guy wrote."* Half true, and the half that's wrong matters for where you spend effort.

**The engine alignment is correct.** In `run_moving_average_crossover`, `position = raw_signal.shift(1)` means the position held during bar `t` was decided using data through bar `t−1`, and `daily_return[t]` spans `close[t−1] → close[t]`. No look-ahead. That part is fine.

**The leakage is in the data loader.** `data.ffill().bfill()` — the `bfill` fills leading `NaN`s using values *from the future*. On a single leading gap it's cosmetically harmless; on any interior gap that survives the `ffill`, it is a genuine backward information flow. That is the real leak, and it's in the file nobody looks at.

**The general lesson:** leakage hides in the data layer far more often than in the strategy layer, because the strategy layer is where people concentrate. Your `τ` harness must run on the **full pipeline from raw bytes to signal**, not on the strategy function in isolation. If you only test the strategy, you will pass while `bfill` quietly cheats upstream.

### A3. `shift(1)` versus the τ-test — they are not the same layer

You asked whether this is "built on top of shift1 or an upgrade to shift1." Neither. They sit at different levels:

| | `shift(1)` | τ-perturbation test |
|---|---|---|
| What it is | the implementation | the verification |
| What it does | enforces one alignment convention | proves that *whatever* you implemented is causal |
| Fails when | you forget it | anything anywhere leaks, including things `shift` cannot express |
| Analogy | writing the proof | checking the proof |

`shift(1)` is a claim. The τ-test is the proof of the claim. Keep `shift(1)`, and never trust it without G1.

The τ-test also catches classes of leakage that no amount of shifting fixes: a scaler fitted on the whole series, a `dropna()` whose row count depends on future rows, an outlier clip using a global percentile, a rolling window with `center=True`. Every one of those passes a code review and fails G1.

### A4. Reference implementation

```python
def causality_cut_test(pipeline, prices: np.ndarray, taus: list[int],
                       rng, n_seeds: int = 20) -> None:
    """G1. Raises AssertionError on any leakage."""
    baseline = pipeline(prices)                       # full-pipeline signals
    for tau in taus:
        for _ in range(n_seeds):
            scrambled = prices.copy()
            # replace the future with noise of matching scale
            tail = len(prices) - tau - 1
            shocks = rng.normal(0.0, np.diff(np.log(prices)).std(), size=tail)
            scrambled[tau + 1:] = prices[tau] * np.exp(np.cumsum(shocks))
            out = pipeline(scrambled)
            assert np.array_equal(out[:tau + 1], baseline[:tau + 1], equal_nan=True), \
                f"causality violated at tau={tau}"
```

**Test parameters:** 500 values of `τ` sampled uniformly over `[L, T)`, 20 seeds each. `equal_nan=True` because warm-up `NaN`s are legitimate and must also be stable.

**G7 — the trap.** Register a deliberately leaky strategy in the test suite:

```python
class LeakyOracle(Strategy):
    """Trades on close[t] at close[t]. MUST be caught by G1."""
    lookback = 1
    def signals(self, close):
        return np.sign(np.diff(close, prepend=close[0]))   # uses close[t]
```

If `causality_cut_test` does not raise on `LeakyOracle`, the harness itself is broken. This is the test of the test, and CI fails if it doesn't fire.

---

## Part B — Types

```python
from dataclasses import dataclass
from typing import Literal
import numpy as np

@dataclass(frozen=True, slots=True)
class Bars:
    """Immutable OHLCV. Index is a monotonic, unique, tz-aware DatetimeIndex."""
    ts:     np.ndarray   # datetime64[ns, UTC], strictly increasing
    open:   np.ndarray
    high:   np.ndarray
    low:    np.ndarray
    close:  np.ndarray
    volume: np.ndarray
    adjustment: Literal["raw", "split", "total_return"]

    def __post_init__(self):
        n = len(self.ts)
        for f in ("open", "high", "low", "close", "volume"):
            assert len(getattr(self, f)) == n, f"{f} length mismatch"
        assert np.all(np.diff(self.ts) > np.timedelta64(0)), "ts not strictly increasing"
        assert not np.isnan(self.close).any(), "NaN in close; fix upstream, do not fill here"

@dataclass(frozen=True, slots=True)
class Result:
    equity:     np.ndarray   # portfolio value, equity[0] == initial_capital
    weights:    np.ndarray   # target weight per bar, in [-1, 1] for single asset
    gross_ret:  np.ndarray
    net_ret:    np.ndarray
    costs:      np.ndarray   # currency units per bar
    turnover:   np.ndarray   # |Δw| per bar
```

`frozen=True` is load-bearing. The reference repo mutates `self.df` in place across a dozen assignments, which means the object's meaning depends on how far through the method you are. Frozen dataclasses make the twin-engine comparison meaningful, because there is exactly one state to compare.

---

## Part C — Strategy interface

```python
class Strategy(ABC):
    lookback: int          # bars of history required before the first valid signal

    @abstractmethod
    def signals(self, bars: Bars) -> np.ndarray:
        """Target weight per bar, in [-1, 1].

        CONTRACT: signals[t] may depend only on bars[0:t+1].
        Enforced by G1, not by convention.
        Return NaN for t < lookback.
        """
```

Two rules that are not negotiable:

1. **The strategy returns a target weight, not a trade.** Position sizing, rebalancing and execution belong to the engine. A strategy that emits orders cannot be run through both engines and cannot be vol-targeted without rewriting it.
2. **The declared `lookback` is checked.** The event engine slices exactly `lookback` bars and passes them in. A strategy that silently needs more will produce `NaN` and fail loudly rather than quietly reading further back.

---

## Part D — Execution conventions

Decide once, write it in the README, expose it as a flag.

| Convention | Signal from | Fill at | Honesty |
|---|---|---|---|
| `close_to_close` | close of `t` | close of `t` | optimistic — assumes you trade the price you just observed |
| `next_open` | close of `t` | open of `t+1` | realistic, the default |
| `next_close` | close of `t` | close of `t+1` | conservative, full overnight gap risk |

**Default: `next_open`.** The reference repo implicitly uses `close_to_close` with a one-bar shift, which is defensible for daily data but flatters the result.

**Produce the comparison figure.** Run the identical strategy under all three and plot the three equity curves. The spread between them is your execution-assumption risk, and quantifying it is worth more than any single number. On daily equities expect the gap between `close_to_close` and `next_open` to be material for high-turnover strategies and negligible for low-turnover ones — which is itself a useful diagnostic.

---

## Part E — Cost and accounting equations

Canonical. Both engines must implement these identically.

**Turnover** at bar `t`, with `w` the target weight:

```
turnover[t] = |w[t] − w[t−1]|
```

**Cost in currency**, charged on traded notional, not on portfolio return:

```
cost[t] = turnover[t] · equity[t−1] · (commission_bps + half_spread_bps + slippage_bps) / 10_000
```

The reference repo charges `turnover × cost_rate` as a *return* deduction. Equivalent only while positions are 0/1 and fully allocated. It breaks the moment you add vol targeting, which you will.

**Gross return** with cash yield on the unallocated fraction and borrow on shorts:

```
gross_ret[t] = w[t]·r[t]
             + (1 − |w[t]|)·(cash_yield_annual / 252)
             − max(−w[t], 0)·(borrow_bps_annual / 10_000 / 252)
```

The middle term is the one everybody drops. A long/cash strategy sitting in cash half the time during a 5% rate environment forgoes about 2.5% a year, and omitting it understates the strategy.

**Equity recursion**, multiplicative:

```
equity[t] = equity[t−1] · (1 + gross_ret[t]) − cost[t]
net_ret[t] = equity[t] / equity[t−1] − 1
```

Not `net = gross − cost_rate`. The additive form is a first-order approximation and there is no reason to accept its error when the exact form is one line.

**Benchmark, computed on the identical index after all slicing:**

```
bench[t] = bench[t−1] · (1 + r[t]),    bench[t_0] = initial_capital
```

`t_0` is the first bar of the *reported* window, after warm-up removal. This is the reference repo's benchmark bug: it compounds through the warm-up while the strategy curve restarts. Slice first, compound second. Assert `equity[0] == bench[0] == initial_capital` as a unit test.

---

## Part F — Twin engines

### F1. Event engine — the reference

```python
def run_event(bars: Bars, strategy: Strategy, costs: CostModel,
              initial_capital: float, convention: str) -> Result:
    n = len(bars.close)
    equity = np.full(n, np.nan); equity[strategy.lookback] = initial_capital
    w_prev = 0.0
    for t in range(strategy.lookback + 1, n):
        window = bars.slice(t - strategy.lookback, t)   # hard slice: no future access
        w = strategy.signal_at(window)
        # ... apply Part E equations, one bar at a time
    return Result(...)
```

Slow and obviously correct. The hard slice is the point: the function is structurally incapable of seeing bar `t+1`.

### F2. Vectorised engine — the product

Same equations, numpy rolling operations, no Python loop over bars.

One genuine subtlety: the equity recursion is not a `cumprod`, because `cost[t]` depends on `equity[t−1]`. Options:

- **Exact:** a single loop over bars for the equity path only (signals stay vectorised). Still fast, ~1000× faster than the event engine.
- **Approximate:** express cost as a return deduction and use `cumprod`. Introduces error of order `cost²`.

**Take the exact route.** G2 demands agreement to 1e-12 and the approximation will not deliver it. The performance win from vectorisation lives in the signal computation, not in the accounting.

### F3. G2

```python
def test_g2_twin_agreement(bars, strategy, costs):
    a = run_event(bars, strategy, costs, 10_000.0, "next_open")
    b = run_vectorized(bars, strategy, costs, 10_000.0, "next_open")
    rel = np.abs(a.equity - b.equity) / np.abs(a.equity)
    assert np.nanmax(rel) < 1e-12, f"max relative deviation {np.nanmax(rel):.3e}"
```

Run it across the full strategy zoo, both engines, three conventions, and a cost sweep. Use `hypothesis` to generate the price series so the property is quantified over arbitrary inputs rather than the three you thought of.

---

## Part G — Data contract

The pipeline that G1 tests runs end to end:

```
raw bytes → parse → validate → align → adjust → feature → signal
```

| Stage | Rule |
|---|---|
| parse | no fills, no drops, `NaN` preserved as `NaN` |
| validate | reject non-monotonic timestamps, duplicates, non-positive prices, `high < low` |
| align | reindex to the exchange calendar; a missing session is `NaN`, never interpolated |
| adjust | explicit policy, never a library default. Record which in `Bars.adjustment`. |
| feature | forward-fill only, maximum `k` bars, `k` declared and logged. Never `bfill`. |
| signal | causal by contract, verified by G1 |

**On `auto_adjust`.** yfinance changed this default between versions. Set it explicitly, record the resolved value in the manifest, and pin the yfinance version. A silent change here shifts every Sharpe in the repo and produces no error.

**Manifest, written on fetch and verified on load:**

```json
{
  "AAPL_2020-01-01_2024-01-01_total_return.parquet": {
    "sha256": "…",
    "rows": 1006,
    "fetched_utc": "2026-08-05T10:14:22Z",
    "source": "yfinance==0.2.x",
    "auto_adjust": true,
    "first_ts": "2020-01-02T00:00:00Z",
    "last_ts": "2023-12-29T00:00:00Z"
  }
}
```

G10 reads this. A mismatch fails the build rather than producing quietly different numbers.

**Known biases, to state in the README rather than fix.** yfinance returns currently-listed tickers only, so every delisted company is invisible and any universe test is survivorship-biased upward. Prices are back-adjusted, so the series you see is not the series that traded. Neither is fixable on free data. Both belong in a limitations section, because a reader who knows the field will check whether you know.


---

## Part H — Selection rules

Added so that G9 is built against an interface rather than a hardcoded `argmax`. Statistical
justification is in `01-STATS-FOUNDATIONS.md` Part E2; this section is the contract.

```python
class SelectionRule(ABC):
    name: str

    @abstractmethod
    def weights(self, is_returns: np.ndarray) -> np.ndarray:
        """(T_is, N) in-sample returns -> (N,) weights.

        CONTRACT:
          - returns a non-negative vector summing to 1.0 (within 1e-12)
          - depends ONLY on the in-sample block passed in
          - deterministic: same input, same output, no RNG
        """
```

Four implementations:

| Rule | Weights | Notes |
|---|---|---|
| `ArgMax()` | 1 on the best in-sample Sharpe, 0 elsewhere | the baseline everyone uses |
| `Softmax(tau)` | `exp(z_n/tau)` normalised, `z` = cross-sectional z-scores of in-sample Sharpe | subtract `max(z)` before `exp`; `tau -> 0` recovers `ArgMax` |
| `EqualWeight()` | `1/N` | the `tau -> inf` asymptote |
| `TopK(k)` | `1/k` on the best k, 0 elsewhere | discrete middle ground |

**Standardise before the exponential.** Softmax on raw Sharpe values is scale-dependent and
`tau` becomes uninterpretable. On z-scores, `tau = 1` means one cross-sectional standard
deviation.

**Exposure normalisation is mandatory.** A blend of disagreeing configurations runs at lower
gross exposure than `ArgMax`. Sharpe is scale-invariant so it survives, but CAGR, drawdown and
turnover do not. Rescale the blend to matched gross exposure before any comparison, and assert
the match:

```python
assert abs(blend.gross_exposure.mean() - reference.gross_exposure.mean()) < 1e-9
```

**Generalised PBO.** The standard CSCV definition ranks the argmax column. Replace that with the
rank of the rule's *portfolio* among the N individual out-of-sample Sharpes. This yields
`PBO(rule)` and makes the headline figure possible:

> PBO as a function of softmax temperature, with `ArgMax` at `tau = 0` and `EqualWeight` as the
> asymptote. Expect monotone decrease.

**`tau` is a trial.** Every temperature evaluated gets a ledger row. Fix it a priori, or select
it by nested CV on training blocks only. Sweeping `tau` and keeping the best is the original sin
one level up.
