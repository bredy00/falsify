# Backtester Build Playbook

**Target:** a backtester that reports an estimate *and its error bar*, plus the probability the number survived by luck.

---

## Part 0 — What a demo-grade backtester gets wrong

The starting point is the shape almost every first backtester takes: three modules, a couple
of hundred lines. A loader that pulls daily bars from yfinance, fills gaps and computes
`pct_change`. An engine holding one class with one method — SMA(20) against SMA(50),
long-or-cash, signal shifted one bar, friction charged on position changes. A metrics module
computing Sharpe, maximum drawdown and CAGR.

None of that is stupid. Shifting the signal by one bar is correct and it is the single thing
most people get wrong. Modelling friction at all puts it ahead of the median. But it is a
demo, and the distance between a demo and an instrument is the whole of this project.

### Defects worth naming (these become the feature list)

**1. The benchmark curve starts from the wrong base.** Strategy return is NaN through the
lookback warm-up, so `cumprod` skips it and the strategy curve starts at ~1.0 after a
`dropna`. Market return has no NaNs and compounds through all of those warm-up bars. The two
curves are then plotted against each other from different bases. Fix: slice first, compound
second.

**2. CAGR assumes 252 bars per year.** `years = len(values) / 252` counts bars, not time. Use
elapsed calendar time, `(idx[-1] - idx[0]).days / 365.25`, or every holiday-heavy year
silently inflates the annualisation.

**3. Cash earns nothing.** A long-or-cash strategy sits in cash roughly half the time and is
credited 0%. At a 5% policy rate that is a large fraction of the return being discarded.
`risk_free_rate` typically defaults to 0.0 and is used only in the Sharpe numerator, never in
the equity path — so the equity curve and the Sharpe are misstated in opposite directions.

**4. Sharpe over a series containing exact zeros.** When the position is flat the daily return
is exactly 0.0. Those zeros shrink the sample standard deviation without shrinking the mean
proportionally, so a long/cash strategy earns a flattering Sharpe against a fully-invested
one. Report exposure and a deployed-capital Sharpe alongside it, or the number is not
comparable to anything.

**5. Costs are additive, not multiplicative.** `net = gross - friction`. The truth is
`(1 + r)(1 - c) - 1`. Second-order, but it is free to get right.

**6. Costs are charged on portfolio return, not traded notional.** Fine while positions are
0/1 and fully allocated. Breaks the instant fractional sizing or vol targeting arrives, which
it will.

**7. No statistical validation of any kind.** One asset, one strategy, one parameter pair,
entirely in-sample. The reported Sharpe carries no confidence interval, no correction for the
parameter pairs tried before this one was chosen, and no out-of-sample split. This is the
whole ballgame, and it is the gap this project exists to fill.

**8. Survivorship bias.** yfinance returns currently-listed tickers with back-adjusted prices.
Every company that went to zero is invisible.

**9. `pytest` pinned in the requirements file, and no tests.**

Every one of these is a general property of naive backtesting rather than a fact about any
particular repository, which is why they generalise into invariants B1–B10 rather than into a
list of patches.

---

## Part 1 — The thesis

Any backtest is an estimator of an unknown parameter: the strategy's true expected Sharpe. Everyone reports the point estimate. Almost nobody reports the sampling distribution, and nobody at all reports the selection process that produced the estimate.

**Your project reports all three.** Every number ships with a confidence interval and a p-value that accounts for how many strategies you tried before you found this one.

Three things follow, and they're the spine of the build:

1. **Causality is a structural constraint, not a comment.** A signal at time `t` must be a functional of information in the past light cone only. Enforce it with a test that *perturbs the future and asserts the past is bit-identical*, not with a `.shift(1)` and a prayer.
2. **Two independent engines must agree.** A vectorised engine is fast and subtly wrong in ways that don't raise exceptions. An explicit event loop is slow and obviously correct. Build both, assert agreement to 1e-12, and you've certified the fast one.
3. **The null must be calibrated.** Run 1000 coin-flip strategies through the whole pipeline. If your significance machinery doesn't reject ~95% of them at the 5% level, your machinery is broken and every result downstream is noise.

That third one is what a quant desk actually cares about and it's almost never in a student repo.

**Name candidates:** `falsify` · `null-first` · `causal-backtester`. Pick one and commit to the framing in the README.

---

## Part 2 — Gate set

Ship nothing until the relevant gate is green. Same discipline as the Field Engine.

| Gate | Statement | Pass condition |
|---|---|---|
| **G1** | **Causality.** Randomise all data after index `t`; recompute signals. | `signals[:t]` bit-identical, for 500 random `t` across 20 seeds |
| **G2** | **Twin-engine agreement.** Vectorised vs event-driven on identical inputs. | max relative equity deviation ≤ 1e-12 |
| **G3** | **Analytic recovery.** Synthetic GBM, known μ and σ, buy-and-hold. | estimated Sharpe within 2 SE of √252·μ/σ over 200 paths |
| **G4** | **Zero-cost identity.** Costs = 0, position ≡ 1. | strategy equity == buy-and-hold equity, exact float equality |
| **G5** | **Cost monotonicity.** Sweep cost 0→100 bps. | net Sharpe non-increasing; break-even cost `c*` finite and reported |
| **G6** | **Null calibration.** 1000 random-sign strategies through full pipeline. | mean Sharpe ≈ 0 (within MC error); DSR rejects ≥95% at α=0.05 |
| **G7** | **Leakage trap.** Deliberately inject a strategy trading on `close[t]` at `close[t]`. | G1 harness flags it; CI fails |
| **G8** | **Walk-forward integrity.** Purged + embargoed splits. | zero index overlap train/test; assertion in code, not in docs |
| **G9** | **Overfit probability.** CSCV over the full parameter grid. | PBO computed and reported; ship only if < 0.5 |
| **G10** | **Reproducibility.** `make reproduce` from pinned data hashes. | two runs produce byte-identical figures and metrics JSON |

G6 and G7 are the ones that will make a reader stop scrolling.

---

## Part 3 — Stack

```
Python 3.12
uv                  env + lockfile (fast, deterministic)
numpy               hot path
pandas              I/O and analytics edges
scipy               stats (norm.ppf, OLS, distributions)
pyarrow             parquet cache
matplotlib          figures, no seaborn
pytest + hypothesis property-based tests — these ARE the gates
ruff + mypy --strict
GitHub Actions      runs the full gate suite on every push
```

`hypothesis` is not optional. The causality gate, the cost-monotonicity gate, and the zero-cost identity are all naturally expressed as properties quantified over arbitrary price series. Property tests are how you say "for all inputs" instead of "for the three inputs I happened to try."

### Layout

```
falsify/
├── data/
│   ├── loaders.py          fetch + cache + hash manifest
│   ├── synthetic.py        GBM, AR(1), regime-switch generators
│   └── calendar.py         trading-day alignment, session handling
├── core/
│   ├── types.py            Bar, Position, Fill, Trade (frozen dataclasses)
│   ├── features.py         Feature ABC with declared lookback
│   ├── vectorized.py       fast engine
│   └── event.py            reference engine (explicit loop)
├── costs/
│   └── model.py            commission, spread, slippage, borrow, cash yield
├── strategies/
│   ├── base.py             Strategy ABC
│   ├── ma_crossover.py
│   ├── ts_momentum.py
│   ├── mean_reversion.py
│   └── null.py             random-sign generator for G6
├── stats/
│   ├── bootstrap.py        stationary bootstrap (Politis–Romano)
│   ├── deflated.py         PSR, DSR
│   ├── cscv.py             PBO via combinatorially symmetric CV
│   ├── walkforward.py      purge + embargo splitters
│   └── attribution.py      Fama–French regression, Newey–West SEs
├── report/
│   ├── metrics.py
│   ├── tearsheet.py
│   └── figures.py
├── tests/
│   └── gates/              one file per gate, G1..G10
├── docs/
│   └── research-note.md    the writeup
└── Makefile
```

Files that change together live together. `costs/` is separate from `core/` because you will iterate on the cost model far more than the engine.

---

## Part 4 — Phased build

Assumption: evenings, alongside INVERSE-0. Phases 0–5 land in about two weeks and are already a top-decile portfolio piece. 6–9 are the extension.

### Phase 0 — Scaffold (half a day)

- [ ] `uv init`, pin Python 3.12, add deps, commit the lockfile
- [ ] `ruff` + `mypy --strict` configured in `pyproject.toml`, both passing on an empty repo
- [ ] GitHub Actions workflow: lint → typecheck → `pytest tests/gates -v`
- [ ] `Makefile` with `make test`, `make gates`, `make reproduce`
- [ ] `.gitignore` covering `.venv/`, `data/cache/`, `outputs/`

Set up CI before you write code. It costs 30 minutes now and saves the "it worked on my machine" week later.

### Phase 1 — Data layer

- [ ] `loaders.fetch(ticker, start, end) -> pd.DataFrame` with a parquet cache keyed on `(ticker, start, end, adjustment_policy)`
- [ ] Explicit `adjustment: Literal["raw", "split", "total_return"]` — never rely on a library default. yfinance's `auto_adjust` flipped defaults between versions and silently changed everyone's results.
- [ ] `data/MANIFEST.json`: sha256 per cached file, written on fetch, verified on load. G10 depends on this.
- [ ] `synthetic.gbm(mu, sigma, T, seed)` returning a price path with *known* parameters. This is your ground truth for G3 and G4.
- [ ] `synthetic.ar1(phi, sigma, T, seed)` — mean-reverting series, for testing that a mean-reversion strategy finds signal that genuinely exists.
- [ ] Explicit missing-data policy. `ffill().bfill()` as in the naive baseline backfills the *first* row from the future. On a single leading NaN it's harmless; on a gap it is a look-ahead. Forward-fill only, then drop the leading NaN block.

**Free data beyond yfinance:** Ken French's data library (factor returns, the academic standard, free), Stooq via `pandas-datareader`, Nasdaq Data Link. Using French's factors for attribution signals that you read the literature.

### Phase 2 — Twin engines (the core)

The `Feature` ABC declares its own lookback, and the engine uses that declaration to enforce causality:

```python
@dataclass(frozen=True)
class Feature(ABC):
    lookback: int  # bars of history required; engine asserts on this

    @abstractmethod
    def compute(self, window: np.ndarray) -> float:
        """Receives exactly `lookback` bars ending at t. Returns signal at t."""
```

The event engine calls `compute` with a hard-sliced window. It is structurally incapable of seeing the future. The vectorised engine computes the same quantity with rolling ops. G2 asserts they agree.

- [ ] `core/event.py` — explicit `for` loop over bars, maintains cash + shares, applies fills at the next bar's open (or close, but *state the convention in the README*), records a `Trade` per fill
- [ ] `core/vectorized.py` — numpy-first, same semantics
- [ ] **G2 test**, and don't move on until it's green to 1e-12

The event engine will be 100× slower. That's fine — it exists to certify the fast one, and you run it on a subsample.

Also decide and document: **when do you trade?** Signal computed on close of `t`, filled at close of `t` is the naive baseline's implicit assumption and it's optimistic. Filled at open of `t+1` is honest and costs you real return. Pick one, write it in the README, and offer both as a config flag so you can quantify the difference. That comparison is itself a good figure.

### Phase 3 — Cost model

```python
@dataclass(frozen=True)
class CostModel:
    commission_bps: float  # per side
    half_spread_bps: float  # per side
    slippage_bps: float  # per side, or a function of participation rate
    borrow_bps_annual: float  # short financing
    cash_yield_annual: float  # what idle cash earns — the naive baseline drops this
```

- [ ] Costs charged on **traded notional**: `cost_t = |Δw_t| · V_t · total_bps`, not on portfolio return
- [ ] Multiplicative application: `equity_t = equity_{t-1} · (1 + r_t) · (1 - c_t)`
- [ ] Idle cash accrues `cash_yield_annual / 252` per bar on the unallocated fraction
- [ ] **G4 and G5 tests**
- [ ] `break_even_cost()` — sweep `total_bps` upward, find `c*` where net Sharpe crosses zero, and separately where it crosses buy-and-hold. Report in bps per turn.

Break-even cost is the single most useful number in the whole report. A strategy that dies at 3 bps is a plot; one that survives 40 bps is a business.

### Phase 4 — Metrics

Beyond CAGR / MDD / Sharpe:

- [ ] Sortino (downside deviation only)
- [ ] Calmar (CAGR / |MDD|)
- [ ] Exposure % (fraction of bars with non-zero position)
- [ ] Turnover (annualised sum of `|Δw|`)
- [ ] Hit rate, average win / average loss, profit factor
- [ ] Longest drawdown *duration*, not just depth
- [ ] Rolling 1-year Sharpe (the figure that shows whether the edge is stable or comes from one lucky quarter)
- [ ] t-statistic on mean excess return, with **Newey–West** standard errors at lag `L = floor(4(T/100)^(2/9))` — daily strategy returns are autocorrelated and OLS SEs will lie to you
- [ ] **G3 test**: metrics recover known GBM parameters

### Phase 5 — Statistical validation (this is the project)

**5a. Stationary bootstrap** (Politis–Romano 1994). Resample blocks of geometric length with mean `1/p`, wrapping around, to preserve autocorrelation that an i.i.d. bootstrap would destroy.

```python
def stationary_bootstrap(x: np.ndarray, p: float, n_boot: int, rng) -> np.ndarray:
    T = len(x)
    out = np.empty((n_boot, T))
    for b in range(n_boot):
        idx = np.empty(T, dtype=np.int64)
        idx[0] = rng.integers(T)
        for t in range(1, T):
            idx[t] = rng.integers(T) if rng.random() < p else (idx[t - 1] + 1) % T
        out[b] = x[idx]
    return out
```

Tune `p ≈ 1/√T` as a default; report sensitivity to it. Produces CIs on Sharpe, CAGR, MDD.

**5b. Deflated Sharpe Ratio** (Bailey & López de Prado 2014). Corrects for non-normal returns *and* for the number of trials.

Probabilistic Sharpe, with all SRs in per-observation (daily) units, `γ₃` skew, `γ₄` non-excess kurtosis:

```
PSR(SR*) = Φ[ (SR − SR*)·√(T−1) / √(1 − γ₃·SR + ((γ₄−1)/4)·SR²) ]
```

Expected maximum Sharpe under the null across `N` independent trials, with `γ ≈ 0.5772` (Euler–Mascheroni) and `V = Var(SR_n)` across trials:

```
SR₀ = √V · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]
DSR = PSR(SR₀)
```

Sanity check the implementation: normal returns give `γ₃=0, γ₄=3`, so the denominator collapses to `√(1 + SR²/2)`, matching Lo (2002). If yours doesn't, you have a kurtosis convention bug.

Report `N` honestly — it's the total number of configurations you *evaluated*, not the number you kept.

**5c. Probability of Backtest Overfitting** via CSCV (Bailey, Borwein, López de Prado, Zhu 2014).

- Build `M`, a `T × N` matrix of returns across all `N` parameter configs
- Split `T` into `S = 16` contiguous blocks
- For each of the C(16,8) = 12,870 ways to choose 8 blocks as in-sample:
  - `n* = argmax` in-sample Sharpe
  - find `n*`'s rank `ω` among the `N` out-of-sample Sharpes
  - relative rank `r = ω / (N+1)`, logit `λ = ln(r / (1−r))`
- `PBO = P(λ ≤ 0)` — the fraction of splits where the in-sample winner lands in the bottom half out-of-sample

PBO near 0.5 means your selection procedure is a coin flip. Publishing this number about your own strategy is the credibility move.

**5d. Purged walk-forward** with embargo (López de Prado, AFML ch. 7).

- [ ] Purge: drop training observations whose label horizon overlaps the test window
- [ ] Embargo: additionally drop `e·T` observations immediately after the test window
- [ ] Assert zero index overlap in code — G8

For a 1-day-horizon MA crossover, purging is nearly trivial. It becomes essential the moment you add multi-day holds or triple-barrier labels, so build it correctly now.

**5e. Null calibration — G6.** Generate 1000 random-sign strategies with turnover matched to your real strategy. Push each through the entire pipeline. Their Sharpe distribution is your empirical null. Your real strategy's Sharpe must sit in the tail of *that*, not of a textbook normal.

### Phase 6 — Strategy zoo

- [ ] MA crossover (the demo strategy, done correctly — your baseline)
- [ ] Time-series momentum (Moskowitz–Ooi–Pedersen 2012: 12-month lookback, 1-month hold). Published Sharpe ≈ 0.8 on a diversified futures basket. If yours comes out at 3.0 on SPY, you have a bug — this is a free calibration check against the literature.
- [ ] Mean reversion on a z-score of price vs rolling mean
- [ ] Vol-targeted overlay: scale position by `σ_target / σ̂_t`. Almost always improves Sharpe and adds turnover — a clean demonstration of the cost/benefit tension.
- [ ] `null.RandomSign` for G6

### Phase 7 — Cross-sectional and attribution

- [ ] N-asset universe, rank on signal, long top decile / short bottom decile, weights sum to zero
- [ ] Turnover control: only rebalance names crossing a buffer band
- [ ] **Factor attribution**: regress strategy excess returns on Mkt-RF, SMB, HML, UMD from Ken French. Report alpha with Newey–West SEs.

If your alpha t-stat drops below 2 after controlling for momentum, say so in the README. That single act of intellectual honesty is worth more to a reader than a 2.5 Sharpe.

### Phase 8 — Reporting

- [ ] Tearsheet: equity curve (log scale) vs benchmark, drawdown underwater plot, rolling Sharpe, monthly return heatmap, turnover
- [ ] Break-even cost curve: net Sharpe vs cost bps
- [ ] Parameter surface: in-sample Sharpe heatmap next to the out-of-sample heatmap for the same grid. When the second one is flat noise, you've just shown overfitting visually. Best figure in the repo.
- [ ] `metrics.json` written on every run with the git SHA and data manifest hash embedded

### Phase 9 — The writeup

`docs/research-note.md`, structured like a paper: motivation, data and its biases, engine design and the causality argument, cost model, results with CIs, DSR and PBO, factor attribution, limitations, what would need to be true for this to be tradeable.

The README leads with the thesis, shows the two heatmaps, and states the gate table with pass/fail. Anyone screening for a quant role reads the README and nothing else.

---

## Part 5 — Reading list

Short, real, in order of value to you.

1. **Bailey, Borwein, López de Prado, Zhu — "Pseudo-Mathematics and Financial Charlatanism"** (*Notices of the AMS*, 2014). Written for mathematicians, ten pages, and it is the entire thesis of your project. Start here.
2. **Bailey & López de Prado — "The Deflated Sharpe Ratio"** (2014). Formulas for 5b.
3. **López de Prado — *Advances in Financial Machine Learning***, chapters 7 (cross-validation, purging, embargo), 11 (backtest dangers), 12 (backtesting through CV), 14 (metrics).
4. **Harvey, Liu, Zhu — "…and the Cross-Section of Expected Returns"** (*RFS*, 2016). Why t > 2 is not enough and t > 3 is the honest bar.
5. **Politis & Romano — "The Stationary Bootstrap"** (*JASA*, 1994). Method for 5a.
6. **White — "A Reality Check for Data Snooping"** (*Econometrica*, 2000). The multiple-testing framework DSR sits inside.
7. **Moskowitz, Ooi, Pedersen — "Time Series Momentum"** (*JFE*, 2012). A real published strategy with real published numbers to calibrate against.
8. **Lo — "The Statistics of Sharpe Ratios"** (*FAJ*, 2002). Where the Sharpe standard error comes from.

Optional if you go deeper on execution: Hasbrouck, *Empirical Market Microstructure*. Optional on portfolio theory: Grinold & Kahn, *Active Portfolio Management* (the fundamental law, IC × √breadth).

---

## Part 6 — Two-week sprint

| Day | Work | Gate |
|---|---|---|
| 1 | Scaffold, CI, Makefile, lockfile | — |
| 2 | Data layer, cache, manifest, synthetic generators | — |
| 3 | `Feature` ABC, causality harness | **G1, G7** |
| 4 | Event engine | — |
| 5 | Vectorised engine | **G2** |
| 6 | Cost model, cash yield, break-even sweep | **G4, G5** |
| 7 | Metrics suite, Newey–West t-stats | **G3** |
| 8 | Stationary bootstrap, CIs on everything | — |
| 9 | PSR / DSR | — |
| 10 | Null calibration, 1000 random strategies | **G6** |
| 11 | Purged walk-forward splitter | **G8** |
| 12 | CSCV / PBO over the parameter grid | **G9** |
| 13 | Tearsheet, heatmaps, break-even curve | — |
| 14 | `make reproduce`, README, research note | **G10** |

Slip days 7–9 if INVERSE-0 needs the evening. The gates that matter for the story are G1, G2, G6, G9 — protect those four.

---

## Part 7 — Traps

**Don't chase Sharpe.** The finished repo's headline result can honestly be "the MA crossover has a PBO of 0.61 and its edge dies at 8 bps." That's a *better* portfolio piece than a 2.4 Sharpe with no error bars, because the first one proves you can evaluate a strategy and the second proves you can overfit one.

**Don't add a live-trading module.** It's a different project, it invites a security review of your API key handling, and it adds nothing to the story.

**Don't use intraday data yet.** Microstructure effects, session handling, and consolidated-tape quirks will eat a month. Daily bars, then extend.

**Don't skip the event engine** because the vectorised one "obviously works." The whole point is that vectorised bugs don't raise exceptions.

**Watch the yfinance adjustment default.** Set `auto_adjust` explicitly. Total-return series and price series give different Sharpes and the difference is entirely dividends.

**Report N honestly in the DSR.** If you swept a 20×20 parameter grid across 4 strategies, N = 1600, not 1.
