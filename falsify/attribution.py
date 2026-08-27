"""Factor attribution. PLAYBOOK Phase 7.

    "Factor attribution: regress strategy excess returns on Mkt-RF, SMB, HML, UMD from
     Ken French. Report alpha with Newey-West SEs. If your alpha t-stat drops below 2
     after controlling for momentum, say so in the README."

**The betas here are the same object as `regression.beta_yx`.** That module computes
`cov(x, y) / var(x)` for one explanatory series; this solves

    b = (X'X)^-1 X'r

for `X = [1, f_1, ..., f_K]`, which at `K = 1` reduces to exactly `cov / var` -- the
intercept absorbs the means, the slope is the covariance over the variance, and the two
routines agree to machine precision. `test_attribution.py` asserts that rather than
describing it, because a multivariate solver that quietly disagreed with the bivariate
one in the case they share would mean at least one of them is wrong and there would be
no way to tell which.

So a factor regression answers both questions at once. Alpha is what is left after the
known exposures are removed; the betas are what those exposures were, each one a
covariance over a variance, generalised to account for the factors being correlated with
each other. That last part is the whole difference: running four separate bivariate
`cov/var` fits and calling the results factor loadings would attribute the same return
to several factors at once, because the factors overlap.

Standard errors are Newey-West throughout (01 Part B1). An OLS standard error assumes
independent residuals, and strategy residuals are not -- so the naive alpha t-statistic
overstates significance on exactly the series anyone would want to test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from falsify.core.types import BARS_PER_YEAR
from falsify.metrics import newey_west_lag

Series = NDArray[np.float64]
Matrix = NDArray[np.float64]

# Below this the design matrix is treated as rank-deficient. Scaled against the largest
# singular value, so it is a relative test and does not care about the units of a factor.
CONDITION_TOL = 1e-10


class DegenerateDesign(ValueError):
    """A factor set that cannot support a regression.

    Raised rather than pseudo-inverted. A rank-deficient design still yields *a* solution
    under the pseudo-inverse, and it looks like a fit -- but the loadings are then one
    arbitrary point on a line of equally good answers, and reporting them as exposures
    would be reporting a choice the data did not make.
    """


@dataclass(frozen=True, slots=True)
class FactorFit:
    """Alpha, the loadings, and their HAC standard errors. Frozen (B7).

    `alpha` is per observation, matching B8; `alpha_annual` is the reporting-boundary
    figure. Both are carried because the t-statistic is computed on the per-bar quantity
    and annualising a t-statistic is meaningless -- it is already unit-free.
    """

    names: tuple[str, ...]
    alpha: float
    alpha_stderr: float
    betas: tuple[float, ...]
    beta_stderrs: tuple[float, ...]
    n_obs: int
    lag: int
    r_squared: float
    residual_vol: float
    bars_per_year: int

    @property
    def alpha_annual(self) -> float:
        return self.alpha * self.bars_per_year

    @property
    def alpha_t(self) -> float:
        """HAC t-statistic on alpha. The number PLAYBOOK asks be reported."""
        if not math.isfinite(self.alpha_stderr) or self.alpha_stderr <= 0.0:
            return float("nan")
        return self.alpha / self.alpha_stderr

    def beta_t(self, name: str) -> float:
        index = self.names.index(name)
        stderr = self.beta_stderrs[index]
        if not math.isfinite(stderr) or stderr <= 0.0:
            return float("nan")
        return self.betas[index] / stderr

    def loading(self, name: str) -> float:
        return self.betas[self.names.index(name)]

    @property
    def survives(self) -> bool:
        """Whether alpha clears the bar PLAYBOOK sets: a HAC t above 2.

        Stated as a property so the README claim and the test read the same expression.
        PLAYBOOK's instruction is explicit that when this is False it goes in the README
        anyway.
        """
        return abs(self.alpha_t) > 2.0

    def describe(self) -> str:
        loadings = "  ".join(
            f"{n}={b:+.3f}(t{self.beta_t(n):+.1f})"
            for n, b in zip(self.names, self.betas, strict=True)
        )
        return (
            f"alpha {self.alpha_annual:+.2%}/yr (t {self.alpha_t:+.2f})  "
            f"R2 {self.r_squared:.3f}  {loadings}"
        )


def _hac_covariance(design: Matrix, residuals: Series, lag: int) -> Matrix:
    """Newey-West covariance of the OLS coefficient vector.

        S     = Gamma_0 + sum_j w_j (Gamma_j + Gamma_j')
        Var(b) = (X'X)^-1 S (X'X)^-1

    with Bartlett weights `w_j = 1 - j/(L+1)`. The taper is what keeps `S` positive
    semi-definite; an untapered truncation can produce negative variance estimates, which
    is the reason Newey-West exists rather than a plain sum of autocovariances.

    At `lag = 0` this is White's HC0, not the textbook OLS standard error -- measured at
    0.958 of it on one sample. That is not a discrepancy to fix. Dropping every lag
    removes the serial-correlation correction but keeps the heteroskedasticity one, since
    `sum e_t^2 x_t x_t'` makes no constant-variance assumption where `sigma^2 (X'X)^-1`
    does. There is no setting of `lag` that recovers the homoskedastic estimator, and
    there should not be: it is the weaker of the two.
    """
    n, k = design.shape
    weighted = design * residuals[:, None]

    s_matrix = weighted.T @ weighted
    for j in range(1, min(lag, n - 1) + 1):
        gamma = weighted[j:].T @ weighted[:-j]
        s_matrix = s_matrix + (1.0 - j / (lag + 1.0)) * (gamma + gamma.T)

    xtx_inv = np.linalg.inv(design.T @ design)
    covariance = xtx_inv @ s_matrix @ xtx_inv
    return np.asarray(covariance, dtype=np.float64).reshape(k, k)


def fit_factors(
    excess_returns: Series,
    factors: Matrix,
    names: tuple[str, ...],
    *,
    lag: int | None = None,
    bars_per_year: int = BARS_PER_YEAR,
) -> FactorFit:
    """Regress excess returns on the factors. Alpha is the intercept.

    `excess_returns` must already be in excess of the risk-free rate -- the factors are
    excess quantities and mixing conventions puts the risk-free rate into alpha, which is
    the single most common way this regression is got wrong. `attribution` does not
    subtract it for you, because it cannot tell whether it already has been.

    `lag` defaults to the automatic Newey-West rule from 01 Part B1.
    """
    r = np.asarray(excess_returns, dtype=np.float64)
    f = np.atleast_2d(np.asarray(factors, dtype=np.float64))
    if f.shape[0] != r.size:
        f = f.T
    if f.shape[0] != r.size:
        raise ValueError(f"{r.size} returns against {f.shape[0]} factor rows")
    if f.shape[1] != len(names):
        raise ValueError(f"{f.shape[1]} factor columns against {len(names)} names")
    n, k = f.shape
    if n < k + 3:
        raise DegenerateDesign(f"{n} observations cannot support {k} factors plus an intercept")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(f)):
        raise DegenerateDesign("returns or factors contain NaN or infinity")

    design = np.column_stack([np.ones(n), f])
    singular = np.linalg.svd(design, compute_uv=False)
    if singular[-1] <= CONDITION_TOL * singular[0]:
        raise DegenerateDesign(
            f"the design matrix is rank deficient (condition {singular[0] / singular[-1]:.2e}); "
            "two factors are collinear and their loadings would not be identified"
        )

    coefficients, *_ = np.linalg.lstsq(design, r, rcond=None)
    residuals = r - design @ coefficients

    resolved_lag = newey_west_lag(n) if lag is None else lag
    if resolved_lag < 0:
        raise ValueError(f"lag must be non-negative, got {resolved_lag}")
    covariance = _hac_covariance(design, residuals, resolved_lag)
    stderrs = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    centred = r - r.mean()
    total = float(centred @ centred)
    r_squared = 1.0 - float(residuals @ residuals) / total if total > 0.0 else float("nan")

    return FactorFit(
        names=tuple(names),
        alpha=float(coefficients[0]),
        alpha_stderr=float(stderrs[0]),
        betas=tuple(float(b) for b in coefficients[1:]),
        beta_stderrs=tuple(float(s) for s in stderrs[1:]),
        n_obs=n,
        lag=resolved_lag,
        r_squared=r_squared,
        residual_vol=float(np.std(residuals, ddof=k + 1)) * math.sqrt(bars_per_year),
        bars_per_year=bars_per_year,
    )


def beta_from_covariance(y: Series, x: Series) -> float:
    """`cov(x, y) / var(x)`, computed directly.

    Here so the identity `fit_factors` relies on can be checked against something that
    is obviously the definition rather than against another matrix solve. Mirrors
    `regression.fit_bivariate.beta_yx`, which is the same quantity reached by the same
    route.
    """
    dx = x - x.mean()
    dy = y - y.mean()
    var_x = float(dx @ dx)
    if var_x <= 0.0:
        return float("nan")
    return float(dx @ dy) / var_x


__all__ = [
    "CONDITION_TOL",
    "DegenerateDesign",
    "FactorFit",
    "beta_from_covariance",
    "fit_factors",
]
