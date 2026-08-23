"""Bivariate regression, decomposed. Supports the Gate 0.0 comparison and Phase 7.

Three quantities describe the linear relationship between two variables, and the
whole point of keeping them together is that people routinely quote one while
meaning another:

    beta_yx = cov(X, Y) / var(X)        regress Y on X -- the slope Gate 0.0 reports
    beta_xy = cov(X, Y) / var(Y)        regress X on Y -- the *other* slope
    rho     = cov(X, Y) / (sd(X) sd(Y)) correlation -- scale-free, and neither of the above

They are bound by an exact identity:

    rho^2 = beta_yx * beta_xy

which is asserted numerically in `test_regression.py` rather than quoted. Two
consequences follow and both matter for reading a scatter plot honestly.

**The two regression lines are different lines.** `beta_yx` is not `1 / beta_xy`
unless `|rho| = 1`. Regressing Y on X minimises vertical distance and regressing X on
Y minimises horizontal distance, so each line is pulled toward the axis it is
predicting. The gap between them is a direct visual measure of how far the
relationship is from deterministic -- they coincide exactly when `rho^2 = 1` and are
perpendicular when `rho = 0`.

**Both lines pass through the centroid.** `(mean(X), mean(Y))` satisfies both
equations exactly, so the two lines always intersect there and nowhere else (unless
they coincide). That intersection is the anchor of the whole picture: it is the point
the regression cannot get wrong.

**Attenuation.** `beta_yx = rho * sd(Y) / sd(X)`, so a slope carries the units of the
two variables while `rho` does not. Comparing slopes across panels with different
dispersions compares scale as much as association, which is why the comparison figure
reports both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Series = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BivariateFit:
    """The full linear description of a scatter. Frozen (B7)."""

    n: int
    mean_x: float
    mean_y: float
    var_x: float
    var_y: float
    covariance: float
    beta_yx: float
    beta_xy: float
    rho: float
    beta_yx_stderr: float
    intercept_yx: float

    @property
    def sd_x(self) -> float:
        return math.sqrt(self.var_x)

    @property
    def sd_y(self) -> float:
        return math.sqrt(self.var_y)

    def rho_squared_identity_error(self) -> float:
        """|rho^2 - beta_yx * beta_xy|. Exactly zero in exact arithmetic.

        Kept as a method rather than an assertion so the figure can print it: a
        number a reader can check beats a claim they have to trust.
        """
        return abs(self.rho * self.rho - self.beta_yx * self.beta_xy)

    def attenuation_error(self) -> float:
        """|beta_yx - rho * sd_y / sd_x|. Also exactly zero."""
        if self.sd_x == 0.0:
            return float("nan")
        return abs(self.beta_yx - self.rho * self.sd_y / self.sd_x)

    def line_yx(self, x: Series) -> Series:
        """The Y-on-X line: minimises vertical distance."""
        return np.asarray(self.mean_y + self.beta_yx * (x - self.mean_x), dtype=np.float64)

    def line_xy(self, x: Series) -> Series:
        """The X-on-Y line, expressed as y(x): minimises horizontal distance.

        Undefined as a function of x when `beta_xy == 0`, which is the case of a
        perfectly vertical fit -- returned as NaN rather than an exception, since a
        plot should degrade rather than crash.
        """
        if self.beta_xy == 0.0:
            return np.full_like(x, np.nan)
        return np.asarray(self.mean_y + (x - self.mean_x) / self.beta_xy, dtype=np.float64)

    def angle_between_lines_degrees(self) -> float:
        """Angle between the two regression lines.

        Zero when `rho^2 = 1` (the lines coincide and the relationship is
        deterministic), ninety degrees when `rho = 0`. A geometric reading of how much
        information one variable carries about the other.
        """
        if self.beta_xy == 0.0 or self.beta_yx == 0.0:
            return 90.0
        a = math.atan(self.beta_yx)
        b = math.atan(1.0 / self.beta_xy)
        return abs(math.degrees(a - b))

    def describe(self) -> str:
        return (
            f"n={self.n}  beta_yx={self.beta_yx:+.6f}  beta_xy={self.beta_xy:+.6f}  "
            f"rho={self.rho:+.6f}  rho^2={self.rho**2:.6f}  "
            f"identity_err={self.rho_squared_identity_error():.2e}"
        )


def fit_bivariate(x: Series, y: Series) -> BivariateFit:
    """Both regression slopes, the correlation, and the OLS standard error.

    Everything uses `ddof=1`, consistently. Mixing sample and population conventions
    between the covariance and the variances breaks the `rho^2 = beta_yx beta_xy`
    identity by a factor of `n/(n-1)` -- small, plausible-looking, and wrong.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    if x.ndim != 1:
        raise ValueError(f"expected 1-D series, got {x.ndim}-D")
    n = x.size
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}")

    mean_x, mean_y = float(x.mean()), float(y.mean())
    dx, dy = x - mean_x, y - mean_y
    denom = n - 1
    var_x = float(dx @ dx) / denom
    var_y = float(dy @ dy) / denom
    cov = float(dx @ dy) / denom

    if var_x <= 0.0 or var_y <= 0.0:
        raise ValueError("both series need non-zero variance for a bivariate fit")

    beta_yx = cov / var_x
    beta_xy = cov / var_y
    rho = cov / math.sqrt(var_x * var_y)
    intercept = mean_y - beta_yx * mean_x

    # OLS standard error of beta_yx: sqrt( residual variance / (n-2) / Sxx ).
    residuals = dy - beta_yx * dx
    resid_var = float(residuals @ residuals) / (n - 2) if n > 2 else float("nan")
    sxx = float(dx @ dx)
    stderr = math.sqrt(resid_var / sxx) if sxx > 0.0 and resid_var >= 0.0 else float("nan")

    return BivariateFit(
        n=n,
        mean_x=mean_x,
        mean_y=mean_y,
        var_x=var_x,
        var_y=var_y,
        covariance=cov,
        beta_yx=beta_yx,
        beta_xy=beta_xy,
        rho=rho,
        beta_yx_stderr=stderr,
        intercept_yx=intercept,
    )


__all__ = ["BivariateFit", "fit_bivariate"]
