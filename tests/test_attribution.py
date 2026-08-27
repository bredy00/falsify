"""Factor attribution. PLAYBOOK Phase 7.

Two tests here carry the weight.

The first is the identity: at one factor, `fit_factors` must reproduce `cov / var`
exactly -- the same quantity `regression.fit_bivariate` computes and the same one
`beta_from_covariance` computes directly. Measured agreement is 2.2e-16 to 5.6e-16, i.e.
machine epsilon. A multivariate solver that disagreed with the bivariate one in the case
they share would mean at least one is wrong with no way to tell which.

The second is the known answer. Regressing SPY's own excess return on the market factor
must give a beta near 1 and an R-squared near 1, because SPY is very nearly the market.
That check is not decoration: it caught a real error the first time it ran. The strategy
returns were being taken from the `next_open` convention, which measures open-to-open,
while Fama-French factors are close-to-close. The two correlate 0.40 at daily frequency,
so every beta in the first table was wrong and the R-squared for a market index came out
at 0.159. On the close-to-close clock it is 0.9896.

**Attribution requires the `close_to_close` convention.** There is no way to detect the
mismatch from a return series alone -- an open-to-open series is perfectly well formed --
so the requirement is stated here, enforced by the known-answer test, and repeated in the
module docstring of `falsify/attribution.py`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from falsify.attribution import (
    DegenerateDesign,
    beta_from_covariance,
    fit_factors,
)
from falsify.metrics import newey_west_lag
from falsify.regression import fit_bivariate

RNG_SEED = 0


def make_factors(n: int, k: int, seed: int = RNG_SEED) -> np.ndarray:
    """Correlated factors, because uncorrelated ones would hide the whole point of a
    multivariate fit."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.01, (n, k))
    mixing = np.eye(k) + 0.3 * rng.normal(0.0, 1.0, (k, k)) / max(k, 1)
    return np.asarray(base @ mixing, dtype=np.float64)


# --------------------------------------------------------------------------------------
# The identity. A factor beta IS cov/var.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [200, 1_000, 3_000])
def test_one_factor_beta_is_exactly_covariance_over_variance(n: int) -> None:
    """The answer to "does this also give us the betas": yes, and they are the same
    object. `b = (X'X)^-1 X'r` at K = 1 is `cov(x, y) / var(x)`, and the three routes to
    it agree to machine epsilon."""
    rng = np.random.default_rng(RNG_SEED)
    x = rng.normal(0.0, 0.01, n)
    y = 0.7 * x + rng.normal(0.0, 0.005, n) + 0.0002

    fit = fit_factors(y, x.reshape(-1, 1), ("MKT",))
    bivariate = fit_bivariate(x, y)
    direct = beta_from_covariance(y, x)

    print(f"n={n}: factor {fit.loading('MKT'):.15f}  bivariate {bivariate.beta_yx:.15f}")
    assert abs(fit.loading("MKT") - bivariate.beta_yx) < 1e-14
    assert abs(fit.loading("MKT") - direct) < 1e-14


def test_the_intercept_is_the_bivariate_intercept() -> None:
    """Alpha is what the bivariate fit calls the intercept. Same regression, same answer."""
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 0.01, 1_500)
    y = 0.5 * x + rng.normal(0.0, 0.004, 1_500) + 0.0003
    fit = fit_factors(y, x.reshape(-1, 1), ("MKT",))
    assert abs(fit.alpha - fit_bivariate(x, y).intercept_yx) < 1e-15


def test_multivariate_betas_are_not_a_stack_of_bivariate_ones() -> None:
    """The reason the multivariate fit exists.

    When factors are correlated, fitting each separately attributes the same return to
    several of them at once. The joint fit apportions it. If these agreed, the factors
    would be orthogonal and the multivariate machinery would be unnecessary.
    """
    rng = np.random.default_rng(2)
    n = 2_000
    factors = make_factors(n, 3, seed=7)
    truth = np.array([0.8, -0.3, 0.4])
    y = factors @ truth + rng.normal(0.0, 0.003, n)

    joint = fit_factors(y, factors, ("F1", "F2", "F3"))
    separate = [beta_from_covariance(y, factors[:, k]) for k in range(3)]

    print(
        f"joint {tuple(round(b, 4) for b in joint.betas)}  separate {tuple(round(b, 4) for b in separate)}"
    )
    assert np.allclose(joint.betas, truth, atol=0.02), "the joint fit should recover the truth"
    assert not np.allclose(joint.betas, separate, atol=0.01), (
        "the separate bivariate fits agreed with the joint one, so these factors are "
        "orthogonal and this test is not exercising what it claims to"
    )


def test_the_fit_recovers_a_known_alpha() -> None:
    """Power. A regression that cannot find an alpha that is there would make every
    zero-alpha result meaningless."""
    rng = np.random.default_rng(3)
    n = 4_000
    factors = make_factors(n, 4)
    alpha_per_bar = 0.0004
    y = alpha_per_bar + factors @ np.array([0.9, 0.2, -0.1, 0.3]) + rng.normal(0.0, 0.002, n)

    fit = fit_factors(y, factors, ("Mkt-RF", "SMB", "HML", "UMD"))
    print(f"alpha {fit.alpha:.6f} vs {alpha_per_bar:.6f}, t {fit.alpha_t:+.2f}")
    assert abs(fit.alpha - alpha_per_bar) < 1e-4
    assert fit.alpha_t > 3.0, "a real alpha this size should be detected"
    assert fit.survives


def test_no_alpha_is_reported_when_there_is_none() -> None:
    """The null. Returns that are pure factor exposure must show alpha indistinguishable
    from zero, or every positive result is suspect."""
    rng = np.random.default_rng(4)
    n = 4_000
    factors = make_factors(n, 4)
    y = factors @ np.array([1.0, 0.1, 0.0, 0.2]) + rng.normal(0.0, 0.002, n)

    fit = fit_factors(y, factors, ("Mkt-RF", "SMB", "HML", "UMD"))
    print(f"alpha {fit.alpha_annual:+.4%}/yr, t {fit.alpha_t:+.2f}")
    assert abs(fit.alpha_t) < 3.0
    assert not fit.survives


# --------------------------------------------------------------------------------------
# Standard errors.
# --------------------------------------------------------------------------------------


def test_hac_standard_errors_exceed_the_naive_ones_under_persistence() -> None:
    """01 Part B1's reason for using them. Serially correlated residuals carry less
    information per observation, so the honest standard error is larger."""
    rng = np.random.default_rng(5)
    n = 3_000
    factors = make_factors(n, 2)
    noise = np.empty(n)
    innovation = rng.normal(0.0, 0.002, n)
    noise[0] = innovation[0]
    for t in range(1, n):
        noise[t] = 0.7 * noise[t - 1] + innovation[t]
    y = factors @ np.array([0.8, 0.2]) + noise

    hac = fit_factors(y, factors, ("F1", "F2"))
    white = fit_factors(y, factors, ("F1", "F2"), lag=0)
    print(f"HAC alpha SE {hac.alpha_stderr:.3e} vs lag-0 {white.alpha_stderr:.3e}")
    assert hac.alpha_stderr > white.alpha_stderr
    assert hac.lag == newey_west_lag(n)


def test_lag_zero_is_white_not_ordinary_least_squares() -> None:
    """Recorded because it looks like a discrepancy and is not.

    Dropping every lag removes the serial-correlation correction but keeps the
    heteroskedasticity one -- `sum e^2 x x'` makes no constant-variance assumption where
    `sigma^2 (X'X)^-1` does. Measured at 0.958 of the OLS standard error on one sample.
    There is no setting of `lag` that recovers the homoskedastic estimator.
    """
    rng = np.random.default_rng(6)
    n = 1_000
    x = rng.normal(0.0, 0.01, n)
    y = 0.6 * x + rng.normal(0.0, 0.004, n)
    fit = fit_factors(y, x.reshape(-1, 1), ("MKT",), lag=0)

    residuals = y - (fit.alpha + fit.loading("MKT") * x)
    centred = x - x.mean()
    ols = math.sqrt(float(residuals @ residuals) / (n - 2) / float(centred @ centred))
    ratio = fit.beta_stderrs[0] / ols
    print(f"HAC(lag=0)/OLS = {ratio:.4f}")
    assert 0.8 < ratio < 1.2, "the two should be close but are not the same estimator"


# --------------------------------------------------------------------------------------
# Refusals.
# --------------------------------------------------------------------------------------


def test_collinear_factors_are_refused_rather_than_pseudo_inverted() -> None:
    """A rank-deficient design still yields a solution under the pseudo-inverse, and it
    looks like a fit. The loadings are then one arbitrary point on a line of equally good
    answers, and reporting them as exposures reports a choice the data did not make."""
    rng = np.random.default_rng(7)
    n = 500
    f1 = rng.normal(0.0, 0.01, n)
    factors = np.column_stack([f1, 2.0 * f1])  # exactly collinear
    y = rng.normal(0.0, 0.01, n)
    with pytest.raises(DegenerateDesign, match="rank deficient"):
        fit_factors(y, factors, ("F1", "F2"))


def test_too_few_observations_are_refused() -> None:
    with pytest.raises(DegenerateDesign, match="cannot support"):
        fit_factors(np.zeros(5), make_factors(5, 4), ("a", "b", "c", "d"))


def test_non_finite_input_is_refused() -> None:
    y = np.random.default_rng(8).normal(0.0, 0.01, 200)
    y[10] = np.nan
    with pytest.raises(DegenerateDesign, match="NaN or infinity"):
        fit_factors(y, make_factors(200, 2), ("F1", "F2"))


def test_a_name_for_every_column_is_required() -> None:
    with pytest.raises(ValueError, match="factor columns against"):
        fit_factors(np.zeros(200), make_factors(200, 3), ("F1", "F2"))
