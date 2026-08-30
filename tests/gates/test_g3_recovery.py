"""G3 -- analytic recovery, plus Gate 0.1, 0.2 and 0.3 from 00-VALIDATION-FIRST.

Synthetic data is the only data where the right answer is known in advance, so it
is the only data that can tell you the code is correct. Real data can say a
strategy was profitable; it can never say the Sharpe function has a `ddof` bug.

Three claims, and the third is the one people skip:

  0.1  The engine does not invent edge. Buy-and-hold on GBM recovers the known
       parameters.
  0.2  The estimator error falls as 1/sqrt(M). This is the *estimator* error,
       reducible by simulation -- not the estimate error, which shrinks only with
       history. Conflating them is the mistake 00's "Two kinds of error" exists
       to prevent.
  0.3  The engine does not destroy edge that exists. A framework which fails to
       find signal in a series that provably contains it is broken in a way no
       real-data test reveals, because on real data "found nothing" is a
       plausible answer.

A CORRECTION TO 00 GATE 0.1, and it matters because the gate as written would
fail correct code. 00 states the true annualised Sharpe of buy-and-hold is
(mu - sigma^2/2)/sigma = 0.30 and warns that 0.40 means you used mu instead of
mu - sigma^2/2. Both numbers are right, for different estimators:

    Sharpe of LOG returns    -> (mu - sigma^2/2)/sigma = 0.30
    Sharpe of SIMPLE returns -> mu/sigma               = 0.40

because E[exp(g) - 1] = mu/252 exactly when g is the log return. Part E's equity
recursion is multiplicative on *simple* returns, so the engine measures 0.40 and
is correct to. Asserting 0.30 against it would reject a working engine. Rather
than pick one, both are asserted against their own target below -- so confusing
them fails one direction or the other, which is what 00 wanted the gate to catch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import linregress, norm

from falsify.core.event import run_event
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST
from falsify.ledger import Ledger
from falsify.metrics import (
    annualise_sharpe,
    annualised_vol,
    cagr,
    elapsed_years,
    gbm_log_return_sharpe,
    gbm_simple_return_sharpe,
    gbm_simple_return_vol,
    max_drawdown,
    sharpe,
    sharpe_se,
    summarise,
)
from falsify.strategies.simple import BuyAndHold, CausalZScore
from falsify.synthetic import ar1, bars_from_close, gbm, half_life

# B3: the engines take a ledger, always. In-memory and non-persisting here --
# every invocation is still counted, which is what lets a test assert its own
# search size, but the gate suite does not write to the shipped ledger.
LEDGER = Ledger.memory()

MU = 0.08
SIGMA = 0.20
BARS = 2520  # ten trading years
PATHS = 200
BPY = 252
CAPITAL = 10_000.0
MASTER_SEED = 50_000

# Exact lognormal values, not the first-order approximations. The approximations
# mu/sigma = 0.400000 and sigma = 0.200000 are off by 0.020% and 0.035%
# respectively -- immaterial against sampling error at 200 paths, but a gate whose
# job is known-truth recovery should compare against the truth.
TRUE_SIMPLE_SR = gbm_simple_return_sharpe(MU, SIGMA)  # 0.399921
TRUE_LOG_SR = gbm_log_return_sharpe(MU, SIGMA)  # 0.300000, exact by construction
TRUE_SIMPLE_VOL = gbm_simple_return_vol(MU, SIGMA)  # 0.200071
TRUE_LOG_DRIFT = MU - 0.5 * SIGMA * SIGMA  # 0.06
# mu/sigma - (mu - sigma^2/2)/sigma = sigma/2, exactly.
TRUE_SHARPE_GAP = 0.5 * SIGMA  # 0.10


def mean_se(x: NDArray[np.float64]) -> tuple[float, float]:
    return float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(len(x)))


@pytest.fixture(scope="module")
def ensemble() -> dict[str, NDArray[np.float64]]:
    """200 independent GBM paths through the full engine, buy-and-hold, zero cost.

    Seeds are threaded per path (B9). One RNG reused across paths, or worse
    `np.random.seed` outside the loop, makes the paths correlated and flattens
    Gate 0.2's slope -- which is exactly the failure 00 warns about there.
    """
    simple_sr, log_sr, log_drift, vol, cagrs = [], [], [], [], []
    for i in range(PATHS):
        prices = gbm(MU, SIGMA, BARS, np.random.default_rng(MASTER_SEED + i))
        bars = bars_from_close(prices)
        result = run_vectorized(
            bars, BuyAndHold(), ZERO_COST, CAPITAL, "close_to_close", ledger=LEDGER
        )

        net = result.net_ret[1:]  # index 0 is the anchor bar, it earns nothing
        simple_sr.append(annualise_sharpe(sharpe(net)))
        vol.append(annualised_vol(net))

        logret = np.diff(np.log(prices))
        log_sr.append(annualise_sharpe(sharpe(logret)))
        log_drift.append(float(logret.sum() / len(logret) * BPY))

        window_ts = bars.ts[len(bars) - len(result) :]
        cagrs.append(cagr(result.equity, elapsed_years(window_ts)))

    return {
        "simple_sr": np.asarray(simple_sr),
        "log_sr": np.asarray(log_sr),
        "log_drift": np.asarray(log_drift),
        "vol": np.asarray(vol),
        "cagr": np.asarray(cagrs),
    }


# ------------------------------------------------------ 0.1 known-truth recovery


def test_g3_recovers_simple_return_sharpe(ensemble: dict[str, NDArray[np.float64]]) -> None:
    """The engine compounds simple returns, so it must recover the exact lognormal
    simple-return Sharpe -- 0.399921, not the first-order mu/sigma = 0.400000."""
    mean, se = mean_se(ensemble["simple_sr"])
    gap = abs(mean - TRUE_SIMPLE_SR) / se
    print(
        f"simple-return SR: {mean:+.6f} +/- {se:.6f}  exact {TRUE_SIMPLE_SR:+.6f}  "
        f"gap {gap:.2f} SE  (first-order mu/sigma = {MU / SIGMA:.6f})"
    )
    assert gap < 2.0, (
        f"ensemble mean {mean:+.6f} is {gap:.1f} SE from the exact value {TRUE_SIMPLE_SR:.6f}. "
        "If it landed near 0.30 the Sharpe is being computed on log returns while the "
        "equity path compounds simple ones."
    )


def test_g3_recovers_log_return_sharpe(ensemble: dict[str, NDArray[np.float64]]) -> None:
    """And the log-return Sharpe must recover (mu - sigma^2/2)/sigma = 0.30.

    Both targets are pinned so that swapping the two estimators fails one of these
    tests rather than passing quietly -- which is what 00 Gate 0.1 was reaching
    for when it warned about 0.40.
    """
    mean, se = mean_se(ensemble["log_sr"])
    gap = abs(mean - TRUE_LOG_SR) / se
    print(f"log-return SR:    {mean:+.6f} +/- {se:.6f}  exact {TRUE_LOG_SR:+.6f}  gap {gap:.2f} SE")
    assert gap < 2.0, f"ensemble mean {mean:+.6f} is {gap:.1f} SE from {TRUE_LOG_SR}"

    separation = abs(TRUE_SIMPLE_SR - TRUE_LOG_SR)
    assert separation > 3.0 * se, (
        f"the two targets differ by {separation:.3f} but SE is {se:.3f}; at this sample "
        "size the tests cannot tell them apart and neither is evidence of anything"
    )


def test_g3_the_two_sharpe_conventions_differ_by_sigma_over_two(
    ensemble: dict[str, NDArray[np.float64]],
) -> None:
    """The sharpest check in this file, and the one that stops the two conventions
    ever being confused.

        mu/sigma - (mu - sigma^2/2)/sigma = sigma/2,  exactly.

    Measured as a PAIRED difference on the same paths, so the path-to-path noise
    that dominates each level test cancels almost entirely -- the SE here is two
    orders of magnitude smaller than on either Sharpe alone. That makes it a far
    tighter constraint on the pair than the individual assertions are on either
    member, and it is why asserting both conventions is what *prevents* a mix-up
    rather than inviting one: swapping them, or applying one estimator's
    annualisation to the other's returns, moves this difference off sigma/2 and
    fails here even when both level tests still pass.
    """
    diff = ensemble["simple_sr"] - ensemble["log_sr"]
    mean, se = mean_se(diff)
    gap = abs(mean - TRUE_SHARPE_GAP) / se
    print(
        f"paired SR gap:    {mean:+.6f} +/- {se:.6f}  exact sigma/2 = {TRUE_SHARPE_GAP:+.6f}  "
        f"gap {gap:.2f} SE  ({se / mean_se(ensemble['simple_sr'])[1]:.1%} of the level SE)"
    )
    assert gap < 4.0, (
        f"the two Sharpe conventions differ by {mean:+.6f}, not sigma/2 = {TRUE_SHARPE_GAP}. "
        "Either they are being computed on the same returns, or an annualisation is "
        "being applied to the wrong series."
    )
    assert se < 0.1 * mean_se(ensemble["simple_sr"])[1], (
        "the paired difference should be far less noisy than either level; if it is not, "
        "the two are not being measured on the same paths and the pairing buys nothing"
    )


def test_g3_recovers_volatility(ensemble: dict[str, NDArray[np.float64]]) -> None:
    """Annualised vol against the exact lognormal value, not against sigma.

    `sigma` is the volatility of the LOG returns. The simple returns the engine
    compounds are lognormal and right-skewed, so their volatility is slightly
    higher -- 0.200071 against 0.200000. Both the spec's 1% relative band and a
    3 SE statistical bound are asserted; the second is roughly five times tighter
    and is what actually constrains the number.
    """
    mean, se = mean_se(ensemble["vol"])
    rel = abs(mean - TRUE_SIMPLE_VOL) / TRUE_SIMPLE_VOL
    gap = abs(mean - TRUE_SIMPLE_VOL) / se
    print(
        f"annualised vol:   {mean:.6f} +/- {se:.6f}  exact {TRUE_SIMPLE_VOL:.6f}  "
        f"rel {rel:.3%}  gap {gap:.2f} SE  (log-return sigma = {SIGMA:.6f})"
    )
    assert rel < 0.01, f"annualised vol {mean:.6f} is {rel:.2%} from {TRUE_SIMPLE_VOL:.6f}"
    assert gap < 3.0, f"annualised vol is {gap:.1f} SE from the exact lognormal value"

    wrong = SIGMA * math.sqrt(365.0 / 252.0)
    assert abs(mean - wrong) > 10.0 * se, "a sqrt(365) annualisation would not be distinguishable"


def test_g3_recovers_growth_rate(ensemble: dict[str, NDArray[np.float64]]) -> None:
    """Annualised log drift recovers mu - sigma^2/2 = 0.06.

    Growth is asserted in trading time rather than as a calendar CAGR, on purpose.
    A business-day calendar holds 260.97 bars per calendar year while the
    generator is parameterised at 252, so a calendar CAGR carries a ~3.5% wedge
    that has nothing to do with whether the engine is right. The CAGR *formula* is
    checked exactly in test_cagr_uses_elapsed_time, and the CAGR the engine
    actually reports is recorded below for the record.
    """
    mean, se = mean_se(ensemble["log_drift"])
    gap = abs(mean - TRUE_LOG_DRIFT) / se
    print(
        f"log drift:        {mean:+.5f} +/- {se:.5f}  target {TRUE_LOG_DRIFT:+.5f}  gap {gap:.2f} SE"
    )
    assert gap < 2.0, (
        f"annualised log drift {mean:+.5f} is {gap:.1f} SE from mu - sigma^2/2 = {TRUE_LOG_DRIFT}"
    )
    assert abs(mean - MU) > 3.0 * se, (
        "the drift is indistinguishable from mu itself, so this cannot catch the "
        "missing -sigma^2/2 term that 00 Gate 0.1 exists to catch"
    )

    cagr_mean, cagr_se = mean_se(ensemble["cagr"])
    print(
        f"engine CAGR:      {cagr_mean:+.5f} +/- {cagr_se:.5f}  "
        f"(calendar years, 260.97 bars/yr; exp(0.06)-1 = {math.exp(TRUE_LOG_DRIFT) - 1:+.5f})"
    )


def test_g3_holds_for_the_event_engine_too() -> None:
    """G2 says the engines agree, but G3 is cheap enough on a subsample to check
    the reference engine recovers the truth directly rather than by transitivity."""
    sr = []
    for i in range(25):
        bars = bars_from_close(gbm(MU, SIGMA, 400, np.random.default_rng(MASTER_SEED + 900 + i)))
        result = run_event(bars, BuyAndHold(), ZERO_COST, CAPITAL, "close_to_close", ledger=LEDGER)
        sr.append(annualise_sharpe(sharpe(result.net_ret[1:])))
    mean, se = mean_se(np.asarray(sr))
    print(f"event engine SR:  {mean:+.4f} +/- {se:.4f}  target {TRUE_SIMPLE_SR:+.4f}")
    assert abs(mean - TRUE_SIMPLE_SR) < 3.0 * se


# --------------------------------------------------- 0.2 Monte Carlo convergence


def test_gate_02_standard_error_falls_as_one_over_root_m() -> None:
    """Fit log(SE) against log(M) and require a slope in [-0.55, -0.45].

    This measures the ESTIMATOR error, which more simulation genuinely reduces. It
    says nothing about the estimate error on a real series, which shrinks only with
    history -- backtesting SPY a hundred times returns the same number a hundred
    times. A flat slope here means the paths are correlated, almost always a seed
    reused outside the loop.
    """
    counts = (10, 50, 100, 500, 1000, 5000)
    draws = np.asarray(
        [
            annualise_sharpe(
                sharpe(
                    np.diff(
                        np.log(gbm(MU, SIGMA, 260, np.random.default_rng(MASTER_SEED + 10_000 + i)))
                    )
                )
            )
            for i in range(max(counts))
        ]
    )
    ses = [float(np.std(draws[:m], ddof=1) / math.sqrt(m)) for m in counts]
    fit = linregress(np.log(counts), np.log(ses))
    for m, se in zip(counts, ses, strict=True):
        print(f"  M={m:>5}  SE={se:.6f}")
    print(f"log-log slope = {fit.slope:+.4f} (r^2 = {fit.rvalue**2:.4f})")
    assert -0.55 <= fit.slope <= -0.45, (
        f"SE scaling slope {fit.slope:+.4f} outside [-0.55, -0.45]; a flat slope means "
        "the paths are correlated, most likely one RNG shared across them"
    )


# ------------------------------------------------------- 0.3 the power test


def test_gate_03_finds_edge_that_exists_and_none_that_does_not() -> None:
    """The power test, and the one people skip.

    On GBM a mean-reversion rule has no edge by construction, so a positive result
    would be a bug. On a stationary AR(1) it has a real one. A framework that
    cannot find signal in a series that provably contains signal is broken in a way
    no real-data test will reveal.
    """
    strategy = CausalZScore(20)
    phi = 0.95
    n_paths, n_bars = 60, 800

    gbm_sr: list[float] = []
    ar1_sr: list[float] = []
    for i in range(n_paths):
        flat = bars_from_close(
            gbm(0.0, SIGMA, n_bars, np.random.default_rng(MASTER_SEED + 2_000 + i))
        )
        mean_reverting = bars_from_close(
            ar1(phi, 0.02, n_bars, np.random.default_rng(MASTER_SEED + 3_000 + i))
        )
        for bars, bucket in ((flat, gbm_sr), (mean_reverting, ar1_sr)):
            result = run_vectorized(
                bars, strategy, ZERO_COST, CAPITAL, "close_to_close", ledger=LEDGER
            )
            bucket.append(annualise_sharpe(sharpe(result.net_ret[1:])))

    gbm_mean, gbm_se = mean_se(np.asarray(gbm_sr))
    ar1_mean, ar1_se = mean_se(np.asarray(ar1_sr))
    print(
        f"mean reversion on GBM:            SR = {gbm_mean:+.4f} +/- {gbm_se:.4f}\n"
        f"mean reversion on AR(1) phi={phi}:  SR = {ar1_mean:+.4f} +/- {ar1_se:.4f} "
        f"(half-life {half_life(phi):.1f} bars)"
    )

    assert abs(gbm_mean) < 2.0 * gbm_se, (
        f"found edge on GBM: SR = {gbm_mean:+.4f} +/- {gbm_se:.4f}. There is none to find, "
        "so this is a leak or an accounting error, not a discovery"
    )
    assert ar1_mean > 3.0 * ar1_se, (
        f"found no edge on AR(1): SR = {ar1_mean:+.4f} +/- {ar1_se:.4f}. The series provably "
        "mean-reverts, so the engine is destroying signal that exists"
    )


# ------------------------------------------------------------ metrics identities


def test_sharpe_se_collapses_to_the_iid_expression() -> None:
    """With normal returns g3 = 0 and g4 = 3, so the non-normal correction must
    reduce to Lo (2002)'s sqrt((1 + SR^2/2)/(T-1)). A kurtosis convention bug
    shows up here and nowhere else (B10)."""
    returns = np.random.default_rng(11).normal(0.0004, 0.01, 4000)
    sr = sharpe(returns)
    expected = math.sqrt((1.0 + 0.5 * sr * sr) / (len(returns) - 1))
    got = sharpe_se(returns)
    print(
        f"SE(SR) = {got:.8f}  iid expression = {expected:.8f}  rel = {abs(got - expected) / expected:.3%}"
    )
    assert abs(got - expected) / expected < 0.05, (
        f"SE {got:.8f} differs from the i.i.d. form {expected:.8f} by more than sampling "
        "noise in skew and kurtosis explains; check fisher=False"
    )


def test_excess_kurtosis_would_change_the_answer() -> None:
    """Proof the fisher=False choice is load-bearing rather than decorative."""
    returns = np.random.default_rng(12).standard_t(4, size=3000) * 0.01
    sr = sharpe(returns)
    from scipy.stats import kurtosis

    non_excess = float(kurtosis(returns, fisher=False, bias=False))
    excess = float(kurtosis(returns, fisher=True, bias=False))
    assert abs(non_excess - excess - 3.0) < 1e-9, "scipy's two conventions differ by exactly 3"
    right = math.sqrt((1.0 - 0.0 * sr + ((non_excess - 1.0) / 4.0) * sr * sr) / (len(returns) - 1))
    wrong = math.sqrt(abs(1.0 - 0.0 * sr + ((excess - 1.0) / 4.0) * sr * sr) / (len(returns) - 1))
    print(
        f"non-excess g4 = {non_excess:.3f} -> SE {right:.8f};  excess g4 = {excess:.3f} -> SE {wrong:.8f}"
    )
    assert abs(right - wrong) / right > 1e-6, "the convention must actually matter"


def test_cagr_uses_elapsed_time_not_a_nominal_bar_count() -> None:
    """The naive baseline divides by `len(portfolio_values) / 252`, so every
    holiday-heavy stretch silently inflates the annualisation. Exact here."""
    equity = np.asarray([100.0, 100.0 * math.exp(0.06 * 10.0)])
    assert cagr(equity, 10.0) == pytest.approx(math.exp(0.06) - 1.0, abs=1e-12)

    n = 2520
    ts = np.busday_offset(np.datetime64("2020-01-01", "D"), np.arange(n), roll="forward").astype(
        "datetime64[ns]"
    )
    years = elapsed_years(ts)
    print(f"{n} business days = {years:.4f} calendar years ({n / years:.2f} bars/yr)")
    assert 9.5 < years < 9.8, "business-day calendar should be close to but under 10 years"
    assert abs(n / years - 252.0) > 5.0, (
        "if a business-day calendar gave exactly 252 bars/year there would be no reason "
        "to prefer elapsed time over a bar count"
    )


def test_degenerate_inputs_do_not_silently_produce_numbers() -> None:
    """Gate 0.4. A zero Sharpe and an undefined Sharpe are different claims and
    only one of them is true; the naive baseline returns 0.0 here."""
    assert math.isnan(sharpe(np.zeros(50))), "constant returns must give NaN, not 0.0"
    assert math.isnan(sharpe(np.asarray([0.01]))), "a single observation has no Sharpe"
    assert max_drawdown(np.asarray([100.0, 100.0, 100.0])) == 0.0
    assert math.isnan(cagr(np.asarray([100.0, 110.0]), 0.0)), "zero elapsed time has no CAGR"
    assert math.isnan(cagr(np.asarray([0.0, 110.0]), 1.0)), "zero starting equity has no CAGR"


def test_summarise_reports_a_sharpe_with_its_error_bar() -> None:
    """B2 is structural here: `Performance` cannot be built without the SE, so a
    bare Sharpe cannot leave the metrics layer."""
    bars = bars_from_close(gbm(MU, SIGMA, 1200, np.random.default_rng(99)))
    result = run_vectorized(bars, BuyAndHold(), ZERO_COST, CAPITAL, "close_to_close", ledger=LEDGER)
    window_ts = bars.ts[len(bars) - len(result) :]
    perf = summarise(result.net_ret[1:], result.equity, result.weights, elapsed_years(window_ts))
    print(perf.describe())
    lo, hi = perf.sharpe_ci95()
    assert lo < perf.sharpe_annual < hi
    assert math.isfinite(perf.sharpe_annual_se) and perf.sharpe_annual_se > 0.0
    assert perf.exposure == 1.0, "buy-and-hold is invested every bar"
    assert norm.cdf(0.0) == pytest.approx(0.5)  # scipy sanity, cheap
