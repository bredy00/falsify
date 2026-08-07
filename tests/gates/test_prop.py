"""Gate 0.0 -- reproduce the theory before building anything.

Experiments A, B and C from 00-VALIDATION-FIRST.md Part 0.0, after Bailey,
Borwein, Lopez de Prado & Zhu (2014), Notices of the AMS 61(5), 458-471.

  A  E[max of N standard normals] matches the two-term Gumbel approximation
     and grows like sqrt(2 ln N) from below.
  B  Memoryless random walks: in-sample Sharpe carries no information about
     out-of-sample Sharpe. Selection buys nothing -- and costs nothing.
  C  Compensation effects (common-mean constraint; stationary AR(1) memory):
     the in-sample vs out-of-sample slope turns negative. The in-sample
     winner is systematically the wrong pick (Propositions 3 and 5).

numpy and scipy only; matplotlib appears solely to save the deliverable
figure. No engine, no data layer, no network (B1). Seeds are threaded
explicitly (B9). Sharpes are per-observation internally and annualised only
at the display boundary (B8).

B and C are each other's controls (F7): the identical slope machinery must
stay silent on B and fire on C.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.stats import linregress, norm

EULER = 0.5772156649015329  # Euler-Mascheroni constant
ANN = math.sqrt(252.0)      # annualisation for daily bars; display boundary only (B8)
MASTER_SEED = 20140458      # Notices of the AMS 61(5), May 2014, p. 458

N_PATHS = 1000              # paths per experiment
T_BARS = 1000               # daily bars per path
SPLIT = T_BARS // 2         # in-sample / out-of-sample midpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_PATH = REPO_ROOT / "docs" / "figures" / "compensation_effect.png"

# Experiment A. The headline case is drawn brute force exactly as specified:
# N = 1000 standard normals, take the max, 10,000 repetitions.
HEADLINE_N, HEADLINE_REPS = 1_000, 10_000

# The sweep spans N = 2 to 10^6 with no exclusions. Two references, because
# there are two different claims to check and only one of them is about this
# code (see test_experiment_a_monte_carlo_matches_exact):
#
#   expected_max_normal_exact  the true E[max z], to quadrature precision.
#                              Monte Carlo must match THIS at every N.
#   expected_max_normal        the two-term Gumbel approximation the paper and
#                              the DSR actually use. Its distance from the
#                              exact value is a property of the formula, so it
#                              is characterised rather than asserted at 1%.
SWEEP_A = (2, 3, 5, 10, 20, 50, 100, 200, 500, 1_000, 10_000, 100_000, 1_000_000)
SWEEP_REPS = 200_000

# Measured error budget of the approximation against exact truth. The formula
# understates at N = 2, crosses over, then overstates for every N >= 3 and
# decays monotonically from N = 5 -- so its bias in the DSR is conservative
# (a stricter SR_0 benchmark) everywhere the build will use it.
GUMBEL_MONOTONE_FROM = 5
GUMBEL_OVER_ONE_PCT = (5, 10, 20, 50)     # approximation error > 1% here
GUMBEL_UNDER_ONE_PCT = (100, 1_000, 100_000)  # and < 1% from N = 100 upward

LOG_HALF = -math.log(2.0)


# ------------------------------------------------------------------- results
#
# Frozen, per B7: an experiment's result has exactly one state, so a later test
# cannot quietly reinterpret what an earlier one measured.


@dataclass(frozen=True, slots=True)
class SplitFit:
    """Sharpes either side of the midpoint, their OLS fit, and the IS winner."""

    sr_is: NDArray[np.float64]
    sr_oos: NDArray[np.float64]
    fit: Any  # scipy does not export LinregressResult
    pick: int


@dataclass(frozen=True, slots=True)
class SelectionRun:
    """One batch's fit, plus what repeating the selection many times produced."""

    base: SplitFit
    winner_is: NDArray[np.float64]
    winner_oos: NDArray[np.float64]


# ------------------------------------------------------------------ helpers


def child_rng(index: int) -> np.random.Generator:
    """Independent, reproducible stream #index off the master seed (B9)."""
    child = np.random.SeedSequence(MASTER_SEED).spawn(index + 1)[index]
    return np.random.default_rng(child)


def sharpe(returns: np.ndarray, axis: int = -1) -> np.ndarray:
    """Per-observation Sharpe ratio: mean / std(ddof=1). Unit: per bar."""
    return returns.mean(axis=axis) / returns.std(axis=axis, ddof=1)


def expected_max_normal(n: int) -> float:
    """Two-term Gumbel approximation of E[max of n std normals] (Prop 1).

    This is the expression the source paper uses and the one `01` Part B3 feeds
    into the Deflated Sharpe Ratio. It is an approximation; for the truth see
    `expected_max_normal_exact`.
    """
    a = norm.ppf(1.0 - 1.0 / n)
    b = norm.ppf(1.0 - 1.0 / (n * math.e))
    return float((1.0 - EULER) * a + EULER * b)


@cache
def expected_max_normal_exact(n: int) -> float:
    """E[max of n std normals] to quadrature precision -- no approximation.

    Uses E[X] = int_0^inf P(X>x) dx - int_-inf^0 P(X<=x) dx with
    P(max <= x) = Phi(x)^n. Both integrands are bounded in [0, 1], which is why
    this form is used rather than integrating x*n*phi*Phi^(n-1) directly: that
    integrand spikes near sqrt(2 ln n) and its tail underflows for large n.
    Phi(x)^n is evaluated as exp(n*logcdf) throughout to keep it exact.

    Validated against the closed forms at n = 1, 2, 3 by
    test_exact_reference_matches_closed_form.
    """
    upper, upper_err = quad(
        lambda x: -math.expm1(n * norm.logcdf(x)), 0.0, np.inf, limit=400
    )
    lower, lower_err = quad(
        lambda x: math.exp(n * norm.logcdf(x)), -np.inf, 0.0, limit=400
    )
    total_err = upper_err + lower_err
    assert total_err < 1e-6, f"quadrature error {total_err:.2e} too large at n={n}"
    return float(upper - lower)


def mean_se(x: np.ndarray) -> tuple[float, float]:
    """Sample mean and the standard error of that mean."""
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def max_normal_bruteforce(rng: np.random.Generator, n: int, reps: int) -> np.ndarray:
    """max(z_1..z_n) over `reps` repetitions, drawn literally. Chunked to cap
    peak memory at ~80 MB of float64."""
    out = np.empty(reps)
    done = 0
    chunk = max(1, 10_000_000 // n)
    while done < reps:
        k = min(chunk, reps - done)
        out[done : done + k] = rng.standard_normal((k, n)).max(axis=1)
        done += k
    return out


def max_normal_exact(rng: np.random.Generator, n: int, reps: int) -> np.ndarray:
    """Same distribution as `max_normal_bruteforce`, at one draw per repetition
    instead of n.

    P(max <= x) = Phi(x)^n, so with E ~ Exp(1) we have ln Phi(x) = -E/n and the
    maximum is the corresponding normal quantile. Each tail is evaluated on the
    side where it is well conditioned: `ppf` on the lower probability when
    Phi(x) < 1/2, `isf` on the survival probability otherwise. Doing it with one
    branch loses the small tail to cancellation -- fatal at n = 2, where half
    the draws land below the median.

    A mathematical identity, not an approximation -- and not taken on trust:
    test_experiment_a_headline checks it against brute force before use.
    """
    log_q = -rng.exponential(size=reps) / n  # = ln Phi(x), always negative
    out = np.empty(reps)
    lower = log_q < LOG_HALF
    out[lower] = norm.ppf(np.exp(log_q[lower]))
    out[~lower] = norm.isf(-np.expm1(log_q[~lower]))
    return out


def split_and_fit(returns: NDArray[np.float64]) -> SplitFit:
    """Halve each path, Sharpe both halves (annualised for display), OLS of
    out-of-sample on in-sample, and locate the in-sample winner."""
    sr_is = sharpe(returns[:, :SPLIT]) * ANN
    sr_oos = sharpe(returns[:, SPLIT:]) * ANN
    return SplitFit(
        sr_is=sr_is,
        sr_oos=sr_oos,
        fit=linregress(sr_is, sr_oos),
        pick=int(np.argmax(sr_is)),
    )


# ----------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def exp_b() -> SelectionRun:
    """Experiment B: driftless Gaussian random walks (memoryless)."""
    base = split_and_fit(child_rng(1).standard_normal((N_PATHS, T_BARS)))

    # Selection repeated on fresh batches: in- and out-of-sample Sharpe of each
    # batch's in-sample winner.
    rng = child_rng(2)
    n_reps, chunk_reps = 200, 10  # 10 x 1000 x 1000 float64 = 80 MB per chunk
    winner_is = np.empty(n_reps)
    winner_oos = np.empty(n_reps)
    done = 0
    while done < n_reps:
        k = min(chunk_reps, n_reps - done)
        r = rng.standard_normal((k, N_PATHS, T_BARS))
        s_is = sharpe(r[:, :, :SPLIT], axis=2)
        s_oos = sharpe(r[:, :, SPLIT:], axis=2)
        rows, pick = np.arange(k), s_is.argmax(axis=1)
        winner_is[done : done + k] = ANN * s_is[rows, pick]
        winner_oos[done : done + k] = ANN * s_oos[rows, pick]
        done += k
    return SelectionRun(base=base, winner_is=winner_is, winner_oos=winner_oos)


@pytest.fixture(scope="module")
def exp_c_recentred() -> SplitFit:
    """Experiment C1: every path recentred to a common mean (Prop 3)."""
    r = child_rng(3).standard_normal((N_PATHS, T_BARS))
    return split_and_fit(r - r.mean(axis=1, keepdims=True))


@pytest.fixture(scope="module")
def exp_c_ar1() -> SplitFit:
    """Experiment C2: stationary AR(1) log-price, phi = 0.995 (Prop 5).

    Half-life -ln(2)/ln(phi) ~ 138 bars, comfortably inside the 500-bar
    out-of-sample window. Returns are the level increments.
    """
    rng = child_rng(4)
    phi = 0.995
    x = np.empty((N_PATHS, T_BARS + 1))
    x[:, 0] = rng.normal(0.0, 1.0 / math.sqrt(1.0 - phi * phi), N_PATHS)
    eps = rng.standard_normal((N_PATHS, T_BARS))
    for t in range(1, T_BARS + 1):
        x[:, t] = phi * x[:, t - 1] + eps[:, t - 1]
    return split_and_fit(np.diff(x, axis=1))


# -------------------------------------------------------------- experiment A


def test_exact_reference_matches_closed_form() -> None:
    """The quadrature reference is validated against closed forms before any
    other test leans on it.

    E[max of 1] = 0, of 2 = 1/sqrt(pi), of 3 = 3/(2 sqrt(pi)). Nothing here is
    approximate, so the tolerance is machine precision rather than a judgement.
    """
    cases = (
        (1, 0.0, "0"),
        (2, 1.0 / math.sqrt(math.pi), "1/sqrt(pi)"),
        (3, 3.0 / (2.0 * math.sqrt(math.pi)), "3/(2 sqrt(pi))"),
    )
    for n, closed, name in cases:
        got = expected_max_normal_exact(n)
        print(f"n={n}: quad={got:.15f}  closed={closed:.15f} ({name})  diff={abs(got - closed):.2e}")
        assert abs(got - closed) < 1e-10, (
            f"n={n}: quadrature {got:.15f} != closed form {closed:.15f} ({name}); "
            "the exact reference is not exact"
        )


def test_experiment_a_headline() -> None:
    """N = 1000 standard normals, max, 10,000 repetitions, exactly as specified.

    Also validates the order-statistic sampler the sweep relies on, so the sweep
    never rests on an identity this file has not checked.
    """
    emp, se = mean_se(max_normal_bruteforce(child_rng(0), HEADLINE_N, HEADLINE_REPS))
    exact = expected_max_normal_exact(HEADLINE_N)
    gumbel = expected_max_normal(HEADLINE_N)
    print(
        f"brute force N={HEADLINE_N:,} over {HEADLINE_REPS:,} reps: "
        f"emp={emp:.4f} +/- {se:.4f}  exact={exact:.4f} ({abs(emp - exact) / se:.2f} SE)  "
        f"gumbel={gumbel:.4f} ({abs(gumbel - exact) / exact:.3%} high)"
    )
    assert abs(emp - exact) < 4.0 * se, (
        f"brute-force mean {emp:.4f} is {abs(emp - exact) / se:.1f} SE from the "
        f"exact value {exact:.4f}"
    )

    fast, fast_se = mean_se(max_normal_exact(child_rng(5), HEADLINE_N, HEADLINE_REPS))
    gap_se = math.hypot(se, fast_se)
    print(
        f"order-statistic sampler: {fast:.4f} +/- {fast_se:.4f}  "
        f"(brute force - sampler = {emp - fast:+.4f}, {abs(emp - fast) / gap_se:.2f} SE)"
    )
    assert abs(emp - fast) < 4.0 * gap_se, (
        f"order-statistic sampler mean {fast:.4f} disagrees with brute force "
        f"{emp:.4f} by {abs(emp - fast) / gap_se:.1f} combined SE"
    )


def test_experiment_a_monte_carlo_matches_exact() -> None:
    """N = 2 to 10^6, no exclusions: the empirical mean must match exact truth.

    This is the assertion that tests *this code*, and it is valid at every N
    because both sides are the same quantity. The earlier version of this test
    compared Monte Carlo against the Gumbel approximation and so had to skip
    N < 500, where the approximation's own 2%+ error swamped the 1% band. That
    was an untested region hiding behind a comment; there is none now.
    """
    rng = child_rng(6)
    worst_n, worst_gap = 0, 0.0
    for n in SWEEP_A:
        emp, se = mean_se(max_normal_exact(rng, n, SWEEP_REPS))
        exact = expected_max_normal_exact(n)
        gap = abs(emp - exact) / se
        if gap > worst_gap:
            worst_n, worst_gap = n, gap
        print(
            f"N={n:>9,}  emp={emp:.5f} +/- {se:.5f}  exact={exact:.5f}  "
            f"gap={gap:.2f} SE"
        )
        assert gap < 4.0, (
            f"N={n}: empirical {emp:.5f} is {gap:.1f} SE from exact {exact:.5f}"
        )
    print(f"worst deviation across {len(SWEEP_A)} values of N: {worst_gap:.2f} SE at N={worst_n:,}")


def test_experiment_a_growth_tracks_sqrt_2_log_n() -> None:
    """E[max z] grows with N and approaches sqrt(2 ln N) from below.

    Asserted on the exact values, so this is a statement about the mathematics
    rather than about a particular Monte Carlo run.
    """
    exacts = [expected_max_normal_exact(n) for n in SWEEP_A]
    bounds = [math.sqrt(2.0 * math.log(n)) for n in SWEEP_A]
    ratios = [e / b for e, b in zip(exacts, bounds, strict=True)]
    for n, e, b, r in zip(SWEEP_A, exacts, bounds, ratios, strict=True):
        print(f"N={n:>9,}  exact={e:.4f}  sqrt(2lnN)={b:.4f}  ratio={r:.4f}")
        assert e < b, f"N={n}: exact {e:.4f} not below the sqrt(2 ln N) bound {b:.4f}"
    assert all(b > a for a, b in pairwise(exacts)), (
        f"E[max] must grow with N, got {exacts}"
    )
    assert all(b > a for a, b in pairwise(ratios)), (
        f"E[max] must approach sqrt(2 ln N) from below as N grows, got {ratios}"
    )


def test_gumbel_approximation_error_budget() -> None:
    """Characterise the approximation the DSR uses, rather than assume it.

    `01` Part B3 computes SR_0 -- the benchmark the Deflated Sharpe is measured
    against -- with `expected_max_normal`. How far that sits from the truth is
    therefore a bias in the DSR itself, and its direction matters: the formula
    overstates for every N >= 3, so SR_0 is too strict rather than too lax, and
    the resulting DSR is conservative. This test pins that down, including the
    small-N region where the error exceeds 1%.
    """
    signed = {n: expected_max_normal(n) - expected_max_normal_exact(n) for n in SWEEP_A}
    rel = {n: abs(signed[n]) / expected_max_normal_exact(n) for n in SWEEP_A}
    for n in SWEEP_A:
        print(
            f"N={n:>9,}  exact={expected_max_normal_exact(n):.6f}  "
            f"gumbel={expected_max_normal(n):.6f}  signed={signed[n]:+.6f}  "
            f"rel={rel[n]:.3%}"
        )

    assert signed[2] < 0.0, (
        f"the approximation is expected to understate at N=2, got {signed[2]:+.6f}"
    )
    for n in SWEEP_A:
        if n >= 3:
            assert signed[n] > 0.0, (
                f"N={n}: approximation understates by {signed[n]:+.6f}; SR_0 would be "
                "too lax and the DSR optimistic, which reverses the documented bias"
            )

    tail = [n for n in SWEEP_A if n >= GUMBEL_MONOTONE_FROM]
    tail_rel = [rel[n] for n in tail]
    assert all(b < a for a, b in pairwise(tail_rel)), (
        f"approximation error must decay monotonically from N={GUMBEL_MONOTONE_FROM}, "
        f"got {list(zip(tail, tail_rel, strict=True))}"
    )

    for n in GUMBEL_OVER_ONE_PCT:
        assert rel[n] > 0.01, (
            f"N={n}: approximation error {rel[n]:.2%} is no longer above 1%. The "
            "small-N caveat has changed -- update the DSR's documented error budget"
        )
    for n in GUMBEL_UNDER_ONE_PCT:
        assert rel[n] < 0.01, (
            f"N={n}: approximation error {rel[n]:.2%} exceeds 1% in the regime the "
            "build reports N in; the DSR bias is larger than documented"
        )


# -------------------------------------------------------------- experiment B


def test_experiment_b_slope_is_zero(exp_b: SelectionRun) -> None:
    """Memoryless: OLS slope of OOS on IS Sharpe indistinguishable from zero."""
    fit = exp_b.base.fit
    print(
        f"slope = {fit.slope:+.4f} +/- {fit.stderr:.4f} "
        f"(|slope/SE| = {abs(fit.slope) / fit.stderr:.2f}), p = {fit.pvalue:.3f}"
    )
    assert abs(fit.slope) < 2.0 * fit.stderr, (
        f"memoryless slope {fit.slope:+.4f} exceeds 2*SE = {2 * fit.stderr:.4f}: "
        "in-sample Sharpe should carry no out-of-sample information"
    )


def test_experiment_b_winner_oos_is_zero(exp_b: SelectionRun) -> None:
    """The in-sample winner's OOS Sharpe averages to zero across repetitions."""
    mean, se = mean_se(exp_b.winner_oos)
    pick = exp_b.base.pick
    print(
        f"winner OOS mean over {len(exp_b.winner_oos)} repetitions = "
        f"{mean:+.4f} +/- {se:.4f} (annualised); base-batch winner: "
        f"IS = {exp_b.base.sr_is[pick]:.2f}, OOS = {exp_b.base.sr_oos[pick]:+.2f}"
    )
    assert abs(mean) < 2.0 * se, (
        f"selected-path OOS Sharpe mean {mean:+.4f} is not within "
        f"2*SE = {2 * se:.4f} of zero"
    )


def test_experiment_a_predicts_the_experiment_b_winner(exp_b: SelectionRun) -> None:
    """Prop 1 predicts, quantitatively, how good the winner of Experiment B
    looks in sample -- so A and B are one result, not two.

    Under the null the in-sample Sharpe of each path is approximately normal
    with standard error 1/sqrt(T_is), so the best of N_PATHS of them should sit
    at expected_max_normal(N_PATHS)/sqrt(T_is), annualised. This is the
    checkable form of the informal "roughly 1.2 to 2.6" in 00 Part 0.0: that
    range describes single draws from a Gumbel, whose mean is the number
    asserted here.
    """
    mean, se = mean_se(exp_b.winner_is)
    pred = expected_max_normal(N_PATHS) / math.sqrt(SPLIT) * ANN
    lo, hi = exp_b.winner_is.min(), exp_b.winner_is.max()
    print(
        f"winner IS mean = {mean:.4f} +/- {se:.4f}  predicted = {pred:.4f}  "
        f"({abs(mean - pred) / se:.2f} SE, {abs(mean - pred) / pred:.2%});  "
        f"single-draw range over {len(exp_b.winner_is)} reps = [{lo:.2f}, {hi:.2f}]"
    )
    assert abs(mean - pred) < 3.0 * se, (
        f"winner's in-sample Sharpe mean {mean:.4f} is {abs(mean - pred) / se:.1f} SE "
        f"from the Proposition 1 prediction {pred:.4f}; a wrong annualisation "
        f"factor or in-sample length would land here"
    )


# -------------------------------------------------------------- experiment C


def _assert_negative_slope(result: SplitFit, label: str) -> None:
    fit = result.fit
    p_txt = "<1e-300" if fit.pvalue == 0.0 else f"{fit.pvalue:.2e}"
    print(
        f"{label}: slope = {fit.slope:+.4f} +/- {fit.stderr:.4f} (p = {p_txt}), "
        f"intercept = {fit.intercept:+.4f} +/- {fit.intercept_stderr:.4f}"
    )
    assert fit.slope < 0.0, f"{label}: slope {fit.slope:+.4f} is not negative"
    assert fit.pvalue < 0.01, f"{label}: slope p-value {fit.pvalue:.3g} >= 0.01"


def test_experiment_c_recentred(exp_c_recentred: SplitFit) -> None:
    """Common-mean constraint: the exact reversal of Proposition 3."""
    _assert_negative_slope(exp_c_recentred, "recentred")


def test_experiment_c_ar1(exp_c_ar1: SplitFit) -> None:
    """Stationary AR(1) memory: the anticorrelation of Proposition 5."""
    _assert_negative_slope(exp_c_ar1, "AR(1) phi=0.995")


# ----------------------------------------------------- negative controls (F7)
#
# 03 Part F, F7: a gate that has never failed is not a test. These prove the
# assertions above discriminate, rather than passing because they are loose.


def test_control_slope_detector_responds_to_the_treatment() -> None:
    """The same paths, the same seed, the same estimator: slope ~ 0 before
    recentring and ~ -1 after.

    Experiments B and C differ by one transformation. If the negative slope in C
    came from anything else -- the estimator, the split, the seed -- it would
    show up here too, and this control would fail.
    """
    r = child_rng(7).standard_normal((N_PATHS, T_BARS))
    before = split_and_fit(r).fit
    after = split_and_fit(r - r.mean(axis=1, keepdims=True)).fit
    print(
        f"identical paths: slope before recentring = {before.slope:+.4f} "
        f"+/- {before.stderr:.4f}, after = {after.slope:+.4f} +/- {after.stderr:.4f}"
    )
    assert abs(before.slope) < 2.0 * before.stderr, (
        f"untreated slope {before.slope:+.4f} is already significant; the "
        "detector is responding to something other than the treatment"
    )
    assert after.slope < -0.5, (
        f"treated slope {after.slope:+.4f} is not the near-exact reversal "
        "Proposition 3 requires"
    )


def test_control_one_percent_tolerance_rejects_the_loose_bound() -> None:
    """The 1% error budget must be tight enough to tell the sharp two-term
    formula from the crude sqrt(2 ln N) bound.

    Both are 'about right'. Only one is usable as SR_0, and a tolerance that
    accepted either would be measuring nothing. Judged against the exact value,
    so no Monte Carlo noise enters a control.
    """
    exact = expected_max_normal_exact(HEADLINE_N)
    sharp = abs(expected_max_normal(HEADLINE_N) - exact) / exact
    loose = abs(math.sqrt(2.0 * math.log(HEADLINE_N)) - exact) / exact
    print(
        f"N={HEADLINE_N:,} vs exact {exact:.4f}: two-term formula off by "
        f"{sharp:.3%} (accepted), sqrt(2 ln N) off by {loose:.3%} (must be rejected)"
    )
    assert sharp < 0.01 <= loose, (
        f"the 1% budget does not discriminate: two-term formula {sharp:.2%}, "
        f"loose bound {loose:.2%}"
    )


def test_control_monte_carlo_detects_a_wrong_reference() -> None:
    """The 4-SE band in test_experiment_a_monte_carlo_matches_exact must reject
    a reference that is merely close.

    At SWEEP_REPS the standard error is small enough that the Gumbel
    approximation -- wrong by only 0.4% at N = 1000 -- is many SE away. That is
    what makes the exact reference necessary rather than fussy, and it is why
    the old formulation could not be repaired by widening the band: at small N
    no band both accepts correct code and rejects a wrong reference.
    """
    emp, se = mean_se(max_normal_exact(child_rng(8), HEADLINE_N, SWEEP_REPS))
    exact = expected_max_normal_exact(HEADLINE_N)
    good = abs(emp - exact) / se
    bad = abs(emp - expected_max_normal(HEADLINE_N)) / se
    print(
        f"N={HEADLINE_N:,}, {SWEEP_REPS:,} reps, SE={se:.5f}: "
        f"exact reference is {good:.2f} SE away (accepted), "
        f"Gumbel reference {bad:.2f} SE away (must be rejected)"
    )
    assert good < 4.0 <= bad, (
        f"the 4-SE band does not discriminate: exact {good:.1f} SE, "
        f"approximation {bad:.1f} SE"
    )


# ----------------------------------------------------------------- the figure


def test_deliverable_figure(
    exp_b: SelectionRun, exp_c_recentred: SplitFit, exp_c_ar1: SplitFit
) -> None:
    """Save the project's thesis in one image (Session 1 deliverable)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        (exp_b.base, "Random walk — no memory\nselection is free"),
        (exp_c_recentred, "Common-mean constraint\nProposition 3: exact reversal"),
        (exp_c_ar1, "Stationary AR(1), φ = 0.995\nProposition 5: the winner reverts"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4), sharex=True, sharey=True)
    for ax, (d, title) in zip(axes, panels, strict=True):
        f = d.fit
        ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
        ax.axvline(0.0, color="0.85", lw=0.8, zorder=0)
        ax.scatter(d.sr_is, d.sr_oos, s=9, alpha=0.35, color="#33526e", linewidths=0)
        xs = np.array([d.sr_is.min(), d.sr_is.max()])
        ax.plot(
            xs,
            f.intercept + f.slope * xs,
            color="#b0322b",
            lw=2.0,
            label=f"OLS slope = {f.slope:.2f} ± {f.stderr:.2f}",
        )
        ax.scatter(
            [d.sr_is[d.pick]],
            [d.sr_oos[d.pick]],
            marker="*",
            s=300,
            color="#e8a600",
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
            label="best in-sample",
        )
        p_txt = "p < 1e-300" if f.pvalue == 0.0 else f"p = {f.pvalue:.1e}"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("in-sample Sharpe (annualised)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.text(
            0.03,
            0.03,
            p_txt,
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"),
        )
    axes[0].set_ylabel("out-of-sample Sharpe (annualised)")
    fig.suptitle(
        "What selecting the best backtest buys you: "
        "nothing without memory, less than nothing with it",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.005,
        "Gate 0.0, Experiments B & C — 1,000 paths × 1,000 daily bars each, "
        "split at the midpoint; ★ = highest in-sample Sharpe. "
        "After Bailey–Borwein–López de Prado–Zhu (2014).",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.97))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=160, metadata={"Software": None})
    plt.close(fig)

    assert FIGURE_PATH.exists(), f"figure not written to {FIGURE_PATH}"
    assert FIGURE_PATH.stat().st_size > 20_000, "figure suspiciously small"
