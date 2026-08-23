"""B1's autocorrelation half -- the Newey-West corrected t-statistic. 01 Part B1.

`sharpe_se` already implements Lo (2002) with the non-normal correction, which handles
skew and kurtosis. It still assumes the observations are independent. 01 B1 is explicit
that they are not, and that serial correlation inflates the true standard error above
both analytic formulas -- so a significance claim uses the HAC t-statistic, not the
naive one.

Measured on AR(1) at T=3000, ratio of HAC SE to naive SE against the theoretical
long-run inflation `sqrt((1+phi)/(1-phi))`:

    phi      -0.5    0.0     0.2     0.5     0.8
    measured  0.607  1.009   1.211   1.624   2.288
    theory    0.577  1.000   1.225   1.732   3.000

Close through moderate persistence and visibly short at 0.8, because the automatic lag
rule truncates at L=8 while an AR(1) at 0.8 still carries 0.17 autocorrelation there.
The estimator under-corrects for strongly persistent series. That is asserted below
rather than hidden, because it means a HAC t-statistic on a very persistent series is
still optimistic and the bootstrap is the cross-check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from falsify.metrics import newey_west_lag, newey_west_se, newey_west_t

T = 3_000


def ar1(phi: float, n: int = T, seed: int = 11, sigma: float = 0.01) -> np.ndarray:
    e = np.random.default_rng(seed).normal(0.0, sigma, n)
    out = np.empty(n)
    out[0] = e[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + e[i]
    return out


def naive_se(x: np.ndarray) -> float:
    return float(x.std(ddof=1) / math.sqrt(len(x)))


# --------------------------------------------------------------------------------------
# The lag rule, including a discrepancy in the spec itself.
# --------------------------------------------------------------------------------------


def test_the_lag_rule_follows_the_formula_not_01_b1s_worked_example() -> None:
    """01 B1 writes `L = floor(4*(T/100)^(2/9))` and then says "For T = 1008, L = 7".

    Those disagree. The arithmetic is

        4 * (1008/100)^(2/9) = 4 * 1.6710 = 6.684  ->  floor 6

    Seven is what rounding gives, so the worked example reads as a rounding slip rather
    than a different intended rule -- and floor is the standard statement of Newey-West
    (1994) automatic lag selection. The code follows the formula.

    This test exists so the discrepancy is recorded where someone checking the code
    against the spec will find it, instead of looking like a bug in the implementation.
    """
    assert newey_west_lag(1_008) == 6, "01 B1's example says 7; the formula it prints says 6"
    assert math.floor(4.0 * (1008 / 100.0) ** (2.0 / 9.0)) == 6


@pytest.mark.parametrize(("n_obs", "expected"), [(100, 4), (252, 4), (1_008, 6), (2_516, 8)])
def test_the_lag_grows_slowly_with_sample_size(n_obs: int, expected: int) -> None:
    """A decade of daily bars still only reaches into the second week."""
    assert newey_west_lag(n_obs) == expected


def test_a_nonsensical_sample_size_raises() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        newey_west_lag(0)


# --------------------------------------------------------------------------------------
# What the correction does.
# --------------------------------------------------------------------------------------


def test_with_no_dependence_the_hac_se_matches_the_naive_one() -> None:
    """The null case. If HAC and naive disagreed on i.i.d. data the estimator would be
    manufacturing a correction out of nothing."""
    for t in (500, 2_000):
        x = np.random.default_rng(3).normal(0.0, 0.01, t)
        ratio = newey_west_se(x) / naive_se(x)
        print(f"T={t}: HAC/naive = {ratio:.4f}")
        assert 0.95 < ratio < 1.10, f"HAC/naive = {ratio:.4f} on i.i.d. data"


@pytest.mark.parametrize(("phi", "lo", "hi"), [(0.2, 1.10, 1.35), (0.5, 1.45, 1.85)])
def test_positive_autocorrelation_inflates_the_standard_error(
    phi: float, lo: float, hi: float
) -> None:
    """Persistent returns carry less information per observation, so the SE must grow.

    Bounds bracket the theoretical `sqrt((1+phi)/(1-phi))` -- 1.225 and 1.732 -- with room
    for the truncation and for sampling noise. Measured 1.211 and 1.624.
    """
    x = ar1(phi)
    ratio = newey_west_se(x) / naive_se(x)
    theory = math.sqrt((1 + phi) / (1 - phi))
    print(f"phi={phi}: HAC/naive = {ratio:.3f}, theory {theory:.3f}")
    assert lo < ratio < hi


def test_negative_autocorrelation_deflates_it() -> None:
    """Mean reversion is the other direction and the estimator has to follow it. A
    correction that could only ever inflate would be a fudge factor, not an estimator.
    Measured 0.607 against a theoretical 0.577.
    """
    ratio = newey_west_se(ar1(-0.5, seed=12)) / naive_se(ar1(-0.5, seed=12))
    print(f"phi=-0.5: HAC/naive = {ratio:.3f}, theory {math.sqrt(0.5 / 1.5):.3f}")
    assert 0.50 < ratio < 0.75


def test_the_correction_falls_short_on_strongly_persistent_series() -> None:
    """The known limit, asserted so it stays documented.

    At phi=0.8 the theoretical inflation is 3.0 and the estimator reaches 2.29, because
    the automatic lag truncates at 8 while the autocorrelation is still 0.17 there. A HAC
    t-statistic on a very persistent series is therefore still optimistic -- which is a
    reason to cross-check with the bootstrap, not a reason to raise the lag by hand.
    """
    x = ar1(0.8)
    ratio = newey_west_se(x) / naive_se(x)
    theory = math.sqrt((1 + 0.8) / (1 - 0.8))
    print(f"phi=0.8: HAC/naive = {ratio:.3f}, theory {theory:.3f} -- short by design")
    assert ratio > 1.8, "the correction is barely moving on a highly persistent series"
    assert ratio < theory, (
        "the truncation bias has vanished; if the estimator now reaches the theoretical "
        "inflation at phi=0.8, the docstrings that warn about it need rewriting"
    )


def test_a_longer_lag_captures_more_of_the_persistence() -> None:
    """Confirms the shortfall above is the truncation and not something else."""
    x = ar1(0.8)
    short = newey_west_se(x, lag=2)
    auto = newey_west_se(x)
    long = newey_west_se(x, lag=60)
    print(f"lag 2: {short:.3e}  auto({newey_west_lag(T)}): {auto:.3e}  lag 60: {long:.3e}")
    assert short < auto < long


# --------------------------------------------------------------------------------------
# The reported statistic.
# --------------------------------------------------------------------------------------


def test_the_t_statistic_is_scale_free() -> None:
    """`newey_west_t` is a ratio, so it needs no annualisation (B8). Rescaling the
    returns must leave it unchanged."""
    x = ar1(0.3)
    assert math.isclose(newey_west_t(x), newey_west_t(x * 1_000.0), rel_tol=1e-9)


def test_the_t_statistic_is_smaller_than_the_naive_one_under_persistence() -> None:
    """The practical consequence, and the reason 01 B1 says to use this one: the naive
    t-statistic overstates significance on exactly the returns a strategy produces."""
    x = ar1(0.6) + 0.0004
    hac = newey_west_t(x)
    naive = float(np.mean(x)) / naive_se(x)
    print(f"naive t = {naive:.3f}, HAC t = {hac:.3f}")
    assert abs(hac) < abs(naive)


def test_a_short_sample_returns_nan_rather_than_a_number() -> None:
    """Two observations cannot support a HAC estimate. Gate 0.4's rule: undefined and
    zero are different claims."""
    assert math.isnan(newey_west_se(np.array([0.01, -0.01])))
    assert math.isnan(newey_west_t(np.array([0.01])))


def test_zero_lag_reduces_to_the_naive_estimator() -> None:
    """`L = 0` drops every cross term, leaving `sqrt(gamma(0)/T)` -- the naive SE up to
    the ddof convention. A useful anchor: it shows the correction is entirely in the
    autocovariance terms."""
    x = np.random.default_rng(5).normal(0.0, 0.01, 1_000)
    biased = float(np.std(x, ddof=0) / math.sqrt(len(x)))
    assert math.isclose(newey_west_se(x, lag=0), biased, rel_tol=1e-12)
