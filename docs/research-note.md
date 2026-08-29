# A backtest that declines to ship its own strategy

**falsify** — a backtesting engine whose output is an estimate, its error bar, and the
probability the number survived the search that produced it.

Time-series momentum on SPY, 2015–2024. Annualised Sharpe **+0.606 ± 0.340**. Four-factor
alpha **−0.02% a year at t = −0.01**. The engine does not recommend trading it.

---

## 1. Motivation

A backtest is an estimate. Estimates have error bars, and estimates chosen from among many
candidates have a second problem on top of the first: the act of choosing inflates the
number you kept. Most backtesting code reports neither. It reports a Sharpe ratio, to three
decimal places, with no interval and no record of how many configurations were tried before
that one was written down.

This project was built to report all three: the estimate, the error bar, and the
probability the estimate survived its own selection process. The engineering is in service
of that, and the test suite is the specification rather than a check on it — 475 offline
tests and 20 against real market data, ten gates that each name a specific way the result
could be wrong.

The strategy is the excuse. The interesting output is the machinery's verdict on it, which
is negative, and the fact that seven independent parts of the machinery reach that verdict
without being tuned to agree.

## 2. Data, and the biases that come with it

| dataset | rows | window | source |
|---|---|---|---|
| SPY (`total_return`, `raw`) | 2,516 | 2015-01-02 → 2024-12-31 | yfinance 1.6.0 |
| Nine SPDR sector ETFs | 2,516 each | same | yfinance 1.6.0 |
| `^IRX` 13-week T-bill | 2,515 | same | yfinance 1.6.0 |
| Fama–French 4-factor daily | 2,516 | same | Ken French library, Dartmouth |

Thirteen files, each pinned. A `FetchSpec` names ticker, start date, end date and
adjustment policy explicitly; `auto_adjust` is set by hand rather than left to the library
default, because yfinance has changed that default between versions and a silent change
there moves every Sharpe in the repository without raising an error. Each file is SHA256'd
on fetch, the digest is committed, and every subsequent load re-verifies it. The cache is
gitignored — a repository is not a data store — but the manifest ships so a reader can
regenerate and check.

Deliberately **not** used: `yf.download(..., period="1y")`. `period` is relative to today,
so the same code would describe a different backtest every time it ran.

Two biases are stated rather than fixed, because neither is fixable on free data:

- **Survivorship.** yfinance returns currently-listed tickers only. Companies that failed
  are invisible. This is why the cross-sectional universe is nine sector funds that have
  traded continuously since 1998 rather than nine stocks selected today — it weakens the
  bias substantially without pretending to remove it.
- **Back-adjustment.** Prices are adjusted for splits and, under `total_return`, dividends.
  The series you see is not the series that traded.

The risk-free rate exists in two versions and they are not interchangeable: `^IRX` averages
**1.757%** a year over the window, French's `RF` averages **1.700%**. Both are three-month
bill rates differing only in construction. The factor regression uses French's, because the
factors are excess of *that* series and mixing the two would leave a small systematic
residual sitting in alpha.

## 3. The engine, and the causality argument

The contract is that `signals[t]` may depend only on `bars[0:t+1]`. `shift(1)` is a *claim*
about that property, never a proof of it, so the property is tested directly: scramble
every price after an index τ, recompute, and assert the signals before τ are **bitwise**
identical. 500 values of τ across 20 seeds, for every strategy in the zoo. Two modes —
Part A1 causality, and the stricter execution-alignment cut.

The gate has a deliberate counterexample. A strategy that trades on `close[t]` at `close[t]`
is injected, and the harness must flag it. A gate that has never failed is not a gate.

Two implementations of the accounting equations exist and must agree exactly: a bar-by-bar
event engine and a vectorised one. Measured agreement across 1,890 combinations of
strategy, convention and cost level is **0.000e+00** — not "within tolerance", identical.
A third implementation was added for the N-asset panel and held to the same standard: at
one asset it reproduces the single-asset engine bitwise.

**A finding worth recording.** A version of the momentum strategy with `shift_one` removed
*passes both causality cuts*. That is not a broken gate. `sign(close[t]/close[t−252])` reads
only `bars[0:t+1]` and satisfies the contract exactly; omitting the lag is an *execution*
assumption — that you can trade at a close you have only just observed — not a causality
violation. What prevents that here is the engine's convention lag, which applies the weight
from `t−2` structurally for every strategy. The decision lag is a second bar of prudence on
top, and it is exactly one bar: `lagged[t] == unlagged[t−1]` for every finite bar, verified
bitwise.

Anyone who assumed the τ-test covered the standard "use `shift(1)`" warning would have been
wrong in a way no existing test would have told them.

## 4. Cost model

Costs are charged on turnover: `turnover × capital × cost_rate`, where turnover is
`|Δw|` summed across the book. For a strategy that flips between +1 and −1 this charges
exactly 2.0 of turnover on the flip bar and exactly 0.0 elsewhere — verified as an identity,
that the set of bars charged equals the set of bars on which the weight moved.

The cost model carries commission, half-spread, slippage, borrow on short exposure and cash
yield on the uninvested balance. The cash-yield term exists because omitting it is the
reference implementation's error: a strategy that is flat half the time earns nothing on its
cash, which understates it.

Net Sharpe is monotone decreasing in cost across every sweep tested (12/12), and the
**break-even cost is 658 bps**. That number looks implausible until you notice the strategy
turns over 1.56 times a year. A monthly-held position is very nearly cost-insensitive, and
this is the one dimension on which it clearly beats a daily-rebalanced trend follower.

## 5. Results, with intervals

`TimeSeriesMomentum(12m, 1m)` on SPY, `next_open` execution, zero cost, 2,261 reported bars
beginning 2016-01-07 — a full 12-month lookback after the data starts.

| | strategy | buy-and-hold |
|---|---|---|
| annualised Sharpe | **+0.6058 ± 0.3401** | **+0.9232** |
| Newey–West *t* | **+1.926** | **+2.924** |
| max drawdown | −32.05% | −32.05% |
| turnover | 1.56 /yr | 0 |
| decisions | 108 rebalances, 7 flips | 1 |

The benchmark beats the strategy, and it is the only one of the two whose HAC *t*-statistic
clears 2. Stating that is the point of putting it in the table.

**The drawdowns are identical, to the same trough day.** The equity curves differ elsewhere;
through the COVID crash the strategy held weight exactly 1.0, because a 12-month signal
cannot turn inside a four-week decline. When protection would have mattered most, the trend
follower simply *was* the benchmark.

**Stationary bootstrap**, 2,000 replicates, geometric blocks with mean length √T ≈ 47 bars:

> 95% interval **[+0.042, +1.288]**

That interval excludes zero, and the exclusion is stable — across 20 bootstrap seeds the
lower bound averages +0.0422 with a standard deviation of 0.0175, and is positive in 20 of
20. So on this one measurement the result is marginally positive.

It is the *only* measurement that says so. It is also flagged by its own diagnostic: the
interval width moves **21.5%** across the block-length sensitivity grid, above the 20% level
at which the specification says the autocorrelation structure is doing real work and should
be discussed rather than parameterised away.

## 6. Deflation and the probability of overfitting

The strategy was not conceived in isolation. A grid of 24 configurations was evaluated —
six lookbacks by four holding periods — and every one is recorded. `N` is read from an
append-only trials ledger, never typed by hand; re-running the same search leaves `N`
unchanged, because a trial's identity is a content hash of its inputs rather than a fresh
identifier per run.

| quantity | value |
|---|---|
| trials evaluated (`N`) | 24 |
| effective trials (`N_eff`, participation ratio) | **2.21** |
| compression | 0.09 |
| PBO over C(10,5) = 252 splits | **0.8214** |
| median logit λ | −0.575 |
| PSR against a zero benchmark | **0.9626** |
| **Deflated Sharpe** | **0.000000** |
| minimum backtest length required | **10.7 years** |
| history actually available | **10.0 years** |

Two rows carry the argument.

**PSR 0.963 against DSR 0.000.** The probabilistic Sharpe ratio asks whether the Sharpe
beats zero given the sample's skew, kurtosis and length, and answers 96%. The deflated
Sharpe asks whether it beats *the best of 24 tries*, and answers zero. Those are the same
data. The entire difference is the correction for having looked 24 times, and it is the
difference between a result and a coincidence.

**PBO 0.82.** Across 252 symmetric half-splits, the configuration that looked best in
sample landed in the *bottom* half out of sample 82% of the time. The threshold for
shipping is 0.5. Selection on this grid is worse than a coin flip.

And the last two rows do not reconcile: 10.7 years would be needed to distinguish this
Sharpe from selection luck, and 10.0 are available. The result is not weak — it is
**uninterpretable at this sample length**, which is a different and more honest statement.

![In-sample against out-of-sample parameter surfaces](figures/parameter_surface.png)

The same claim without any statistics. Left: Sharpe over the parameter grid in the first
half. Right: the same grid in the second. Both on one shared colour scale, because
per-panel scales would make noise look as structured as signal.

**Spearman ρ between the surfaces = −0.015.** In-sample rank carries no information about
out-of-sample rank. The in-sample winner, (15m, 1m) at +0.874, earns +0.432 out of sample.

The bright 15-month band on the left has a mundane explanation, and it is worth giving
because it is the mechanism: at a 15-month lookback the signal has **zero flips in the first
half**. SPY rose continuously from 2016 to 2020, the trailing return never turned negative,
and every holding period produced an identical constant-long path. The ridge a reader's eye
reads as "the good parameters" is *being long a market that only went up*.

## 7. Factor attribution

Carhart four factors, Newey–West standard errors at the automatic lag (7 for n = 2,261),
close-to-close, excess of French's risk-free rate.

| | coefficient | SE | *t* |
|---|---|---|---|
| **alpha** | **−0.0226 %/yr** | 4.4866 %/yr | **−0.005** |
| Mkt-RF | +0.6177 | 0.0760 | +8.12 |
| SMB | −0.0640 | 0.0401 | −1.60 |
| HML | +0.1822 | 0.0449 | +4.06 |
| UMD | +0.2445 | 0.0410 | +5.96 |

R² = 0.408.

The specification asked for one thing above all others: *"If your alpha t-stat drops below 2
after controlling for momentum, say so in the README."*

It does not drop below 2. It drops to **zero**. The +0.606 Sharpe is fully accounted for by
two exposures the regression names — **+0.618 on the market**, because a trend follower is
long a rising index most of the decade, and **+0.244 on the momentum factor**, because it is
a momentum strategy. Both are available cheaply and neither is skill. Price them and nothing
remains.

**The calibration that makes this readable.** Regressing SPY's own excess return on the
market factor alone gives β = **0.9598**, R² = **0.9896**, alpha not significant — exactly
what an index fund must show. Under the four-factor model, β = 0.975 at R² = 0.995. If those
rows were wrong, nothing else in the table could be believed.

That check earned its place. Run against the `next_open` convention it gave **β = 0.367,
R² = 0.159** for a market index — because `next_open` measures open-to-open while the
factors are close-to-close, and the two correlate 0.40 at daily frequency. Every loading in
the first version of this table was wrong, and nothing else would have flagged it.
Attribution requires the close-to-close convention; there is no way to detect the mismatch
from a return series alone.

**On what the betas are.** The loadings are covariance over variance. At one factor,
`b = (X'X)⁻¹X'r` reduces to exactly `cov(x,y)/var(x)`, and the multivariate solver agrees
with the project's bivariate routine to **2.2e-16** — machine epsilon, asserted rather than
described. The multivariate form matters because the factors are correlated: four separate
`cov/var` fits would attribute the same return to several factors at once.

## 8. The cross-sectional test

A dollar-neutral book removes the market exposure by construction, so it is the cleaner
place to look for skill. Nine sector ETFs, tertile long/short, weights summing to zero.

| construction | Sharpe | ±SE | turns/yr | HAC *t* |
|---|---|---|---|---|
| 12-month, monthly | −0.065 | 0.334 | 4.79 | −0.20 |
| 12-month, daily | −0.023 | 0.334 | 23.03 | −0.07 |
| 6-month, monthly | +0.157 | 0.325 | 6.58 | +0.51 |
| 1-month, monthly | −0.432 | 0.318 | 16.31 | −1.37 |

Nothing approaches significance. The one-month row is short-term reversal appearing
unbidden — last month's winning sectors underperform — and it is recorded as an observation
rather than promoted into a strategy by flipping the sign.

The gap from the single-asset result is the finding. The *same signal* earns +0.606 on SPY
and nothing at all when held dollar-neutral. That difference is the market's drift: the
long-biased version collects it, the neutral book cannot. Which is another way of saying
what the factor regression said, arrived at from a different direction.

## 9. Every layer, independently

| layer | measured | verdict |
|---|---|---|
| B3 trials ledger | 24 trials for 24 configurations, idempotent | counted |
| B4 effective N | 24 → 2.21 | the search was narrow |
| B5 bootstrap | [+0.042, +1.288], width dispersion 21.5% | marginal, and flagged |
| B1 HAC *t* | +1.926 | below 2 |
| G9 PBO | 0.8214 | above the 0.5 line |
| DSR | 0.000000 | nothing survives deflation |
| Phase 7 alpha | −0.02%/yr at *t* = −0.005 | no alpha |
| Part D contract | `ships = False` | does not ship |

Six of these say no; one, the bootstrap interval, says *marginally yes* and flags itself.
None was tuned to agree with the others. The bootstrap knows nothing about the factor model.
PBO knows nothing about sample length. The ledger counts configurations without looking at
their returns.

That they converge is the argument. A single negative statistic is a result; seven
independent ones pointing the same way is a conclusion.

## 10. Limitations

Stated plainly, because a limitations section that lists only comfortable caveats is
marketing.

- **One instrument, one decade, one regime.** SPY 2015–2024 contains one crash and a long
  bull market. A trend follower's case rests on regimes this window does not contain — 2000
  and 2008 are exactly the periods where trend following earned its reputation, and they are
  absent.
- **Daily bars, close-to-close or open-to-open.** No intraday execution, no market impact
  model, no partial fills. The 658 bps break-even is generous precisely because impact is
  not modelled.
- **The universe is nine sector funds.** A tertile of nine is three names. PLAYBOOK asks for
  deciles; a decile of nine assets is 0.9 of one, and the adaptation is stated rather than
  hidden.
- **Survivorship and back-adjustment remain**, per section 2.
- **The trials ledger counts this project's search, not the field's.** Time-series momentum
  has been studied for decades. The `N` that ought to deflate it is not 24; it is every
  configuration every researcher has tried since 1993, and nobody has that number.
- **The HAC estimator under-corrects for strongly persistent series.** At AR(1) φ = 0.8 it
  reaches 2.29 against a theoretical 3.00, because the automatic lag truncates at 8 while
  the autocorrelation is still 0.17 there.
- **`N_eff` at 2.21 is a property of a narrow grid.** Six lookbacks by four holds on one
  asset is not a wide search, and the compression says so.

## 10.1 What would have to be true for this to be tradeable

Not "what would make the number better" — what would have to be *true*.

1. **The market and momentum exposures would have to be unavailable or expensive.** They are
   neither. A reader can buy both through liquid funds at a few basis points, and the
   regression says that is all this strategy delivers. The alpha would need to be positive
   and significant *after* those exposures are priced. It is −0.02% at t = −0.005.
2. **The sample would have to be long enough to distinguish the result from selection luck.**
   10.7 years are required; 10.0 exist. That gap closes with time, not with more
   configurations — and evaluating more configurations widens it.
3. **PBO would have to fall below 0.5.** It is 0.82. Selection on this grid is worse than a
   coin flip, so any "best" parameters chosen from it are more likely to be the worst out of
   sample than the best.
4. **The out-of-sample surface would have to show the in-sample ridge.** ρ = −0.015. It does
   not.
5. **It would have to beat holding the index.** It does not: +0.606 against +0.923 on the
   same window, with the same maximum drawdown on the same day.

None of the five holds. The honest conclusion is that this strategy, on this data, is a
repackaging of two cheap exposures, and the correct action is not to trade it.

## 11. What the project is actually for

The strategy failing is not a failure of the project — it is the project working. The
machinery was built to be capable of returning this answer, and most backtesting code is
not: it can report a Sharpe of 0.606 and stop, and the number is not even wrong, only
unaccompanied.

What accompanies it here is an interval, a deflation by a machine-counted search, a
probability that the selection was a coin flip, and a regression naming where the return
came from. Those took considerably more work than the strategy did, and they are the
deliverable.

---

## Reproducing this

```bash
uv sync --all-groups
uv run --group data python scripts/fetch_data.py   # the only network access
make ci                                            # 475 tests, offline
make live                                          # 20 tests against the cache
make reproduce                                     # byte-identical figures + metrics.json
make tearsheet surface                             # the Phase 8 figures
```

Every statistic in this note is asserted by a test in the tree and carries the standard
error it was measured with. None was rounded toward a nicer number.
