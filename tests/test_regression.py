"""Unit tests for the bivariate decomposition.

The identity `rho^2 = beta_yx * beta_xy` is the reason this module exists as a module
rather than three inline expressions, so it is asserted to machine precision on
arbitrary inputs rather than checked once on a fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from falsify.regression import fit_bivariate

Series = NDArray[np.float64]

PROPERTY = settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def paired_series(draw: st.DrawFn, min_n: int = 5, max_n: int = 200) -> tuple[Series, Series]:
    """Two correlated series with guaranteed non-zero variance.

    Built as `y = a*x + noise` so the correlation spans the full range rather than
    hovering near zero, and constructed rather than filtered for the reason the
    selection tests document: filtering on raw draws starves the property test.
    """
    n = draw(st.integers(min_n, max_n))
    base = draw(
        arrays(np.float64, n, elements=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False))
    )
    noise = draw(
        arrays(np.float64, n, elements=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False))
    )
    a = draw(st.floats(-3.0, 3.0))
    # A deterministic alternating carrier guarantees dispersion in both series
    # whatever the draw happens to contain.
    carrier = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    x = base + 2.0 * carrier
    y = a * x + noise + 2.0 * np.where(np.arange(n) % 3 == 0, 1.0, -0.5)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


@PROPERTY
@given(pair=paired_series())
def test_rho_squared_equals_the_product_of_both_slopes(pair: tuple[Series, Series]) -> None:
    """The identity, to machine precision, on arbitrary inputs.

    `rho^2 = beta_yx * beta_xy` follows from the definitions the moment all three use
    the same `ddof`. Mixing sample and population conventions breaks it by a factor of
    n/(n-1): small, plausible-looking, and wrong -- which is exactly the class of bug
    this asserts away.
    """
    fit = fit_bivariate(*pair)
    assert fit.rho_squared_identity_error() < 1e-12, fit.describe()


@PROPERTY
@given(pair=paired_series())
def test_slope_is_correlation_times_the_dispersion_ratio(pair: tuple[Series, Series]) -> None:
    """`beta_yx = rho * sd_y / sd_x`, exactly.

    This is why a slope and a correlation are not interchangeable: the slope carries
    the units of both variables. Comparing slopes across panels with different
    dispersions compares scale as much as association.
    """
    fit = fit_bivariate(*pair)
    assert fit.attenuation_error() < 1e-12, fit.describe()


@PROPERTY
@given(pair=paired_series())
def test_correlation_is_bounded_and_matches_numpy(pair: tuple[Series, Series]) -> None:
    """Cross-checked against numpy's own implementation, which is written differently."""
    x, y = pair
    fit = fit_bivariate(x, y)
    assert -1.0 - 1e-12 <= fit.rho <= 1.0 + 1e-12
    assert abs(fit.rho - float(np.corrcoef(x, y)[0, 1])) < 1e-10


@PROPERTY
@given(pair=paired_series())
def test_both_lines_pass_through_the_centroid(pair: tuple[Series, Series]) -> None:
    """The anchor of the picture: the one point the regression cannot get wrong."""
    fit = fit_bivariate(*pair)
    centre = np.asarray([fit.mean_x])
    assert abs(float(fit.line_yx(centre)[0]) - fit.mean_y) < 1e-9
    if fit.beta_xy != 0.0:
        assert abs(float(fit.line_xy(centre)[0]) - fit.mean_y) < 1e-9


def test_slope_matches_scipy_linregress() -> None:
    """Agreement with an independent implementation, including the standard error."""
    from scipy.stats import linregress

    rng = np.random.default_rng(20260820)
    x = rng.normal(0.0, 1.0, 500)
    y = 0.7 * x + rng.normal(0.0, 0.5, 500)
    fit = fit_bivariate(x, y)
    ref = linregress(x, y)
    assert abs(fit.beta_yx - float(ref.slope)) < 1e-12
    assert abs(fit.intercept_yx - float(ref.intercept)) < 1e-12
    assert abs(fit.rho - float(ref.rvalue)) < 1e-12
    assert abs(fit.beta_yx_stderr - float(ref.stderr)) < 1e-10


def test_perfect_correlation_collapses_the_two_lines() -> None:
    """When `rho^2 = 1` the relationship is deterministic and both regressions agree.

    The angle between the lines is the geometric reading of how much information one
    variable carries about the other: zero here, ninety degrees when rho is zero.
    """
    x = np.linspace(-2.0, 2.0, 200)
    fit = fit_bivariate(x, 3.0 * x + 1.0)
    assert abs(abs(fit.rho) - 1.0) < 1e-12
    assert abs(fit.beta_yx - 3.0) < 1e-12
    assert abs(fit.beta_xy - 1.0 / 3.0) < 1e-12
    assert fit.angle_between_lines_degrees() < 1e-6, "the lines must coincide"


def test_exact_reversal_gives_both_slopes_minus_one() -> None:
    """Proposition 3's geometry, isolated.

    Under a common-mean constraint the out-of-sample Sharpe is the exact negative of
    the in-sample one, so `rho = -1` and BOTH slopes are -1 -- the one case where the
    two regressions agree while pointing downhill.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, 400)
    fit = fit_bivariate(x, -x)
    assert abs(fit.rho + 1.0) < 1e-12
    assert abs(fit.beta_yx + 1.0) < 1e-12
    assert abs(fit.beta_xy + 1.0) < 1e-12
    assert fit.angle_between_lines_degrees() < 1e-6


def test_independent_series_give_near_perpendicular_lines() -> None:
    """`rho ~ 0` means the two lines are close to perpendicular: knowing x tells you
    almost nothing about y, and the regression says so geometrically."""
    rng = np.random.default_rng(11)
    fit = fit_bivariate(rng.normal(size=4000), rng.normal(size=4000))
    assert abs(fit.rho) < 0.05
    assert 80.0 < fit.angle_between_lines_degrees() <= 90.0


@pytest.mark.parametrize(
    ("x", "y", "match"),
    [
        (np.zeros(5), np.arange(5.0), "non-zero variance"),
        (np.arange(3.0), np.arange(4.0), "same shape"),
        (np.arange(2.0), np.arange(2.0), "at least 3"),
        (np.zeros((2, 2)), np.zeros((2, 2)), "1-D"),
    ],
)
def test_malformed_input_is_rejected(x: Series, y: Series, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        fit_bivariate(x, y)


def test_reversing_the_arguments_swaps_the_slopes() -> None:
    """`fit(y, x)` must give the mirror image, with `rho` unchanged."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=300)
    y = 0.4 * x + rng.normal(size=300)
    forward, reverse = fit_bivariate(x, y), fit_bivariate(y, x)
    assert abs(forward.beta_yx - reverse.beta_xy) < 1e-12
    assert abs(forward.beta_xy - reverse.beta_yx) < 1e-12
    assert abs(forward.rho - reverse.rho) < 1e-12


def test_identity_holds_on_the_gate_00_geometry() -> None:
    """The three Gate 0.0 regimes, checked through the same decomposition.

    Random walk: rho near zero. Recentred: rho exactly -1. AR(1): in between. The
    identity has to hold in all three or the comparison figure is reporting numbers
    that do not describe the same scatter.
    """
    rng = np.random.default_rng(20140458)
    n, t = 800, 400
    for label in ("memoryless", "recentred", "ar1"):
        r = rng.standard_normal((n, t))
        if label == "recentred":
            r = r - r.mean(axis=1, keepdims=True)
        half = t // 2
        sr_is = r[:, :half].mean(axis=1) / r[:, :half].std(axis=1, ddof=1)
        sr_oos = r[:, half:].mean(axis=1) / r[:, half:].std(axis=1, ddof=1)
        fit = fit_bivariate(sr_is, sr_oos)
        assert fit.rho_squared_identity_error() < 1e-12, f"{label}: {fit.describe()}"
        assert fit.attenuation_error() < 1e-12, label
