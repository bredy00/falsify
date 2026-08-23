"""Effective number of trials. Specified by 01 Part B4, decided by 03 Part H decision 2.

Raw `N` overstates a search whenever the configurations are correlated, and on a
parameter grid they always are: `MACrossover(20, 50)` and `MACrossover(21, 50)` are not
two independent looks at the data, they are almost the same look twice. Deflating a
Sharpe by a raw `N` of 800 when the grid really contains about 47 distinct bets is a
correction applied at the wrong magnitude -- and, being too harsh, it is at least
conservative, which is why raw `N` still gets reported as one end of the interval.

Two estimators, per 01 B4. Part H decision 2 settles which leads: the participation ratio
is primary because it has no threshold to justify, and clustering is reported as a
secondary estimate. The revisit condition is written into `EffectiveTrials.diverges`:
when the two differ by more than a factor of two, report the smaller and say why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from falsify.selection import noise_floor

Matrix = NDArray[np.float64]

# 01 B4: "cut at a fixed height (0.5 is a reasonable default, stated in the README)".
DEFAULT_CUT_HEIGHT = 0.5

# 03 Part H decision 2's revisit condition.
DIVERGENCE_FACTOR = 2.0


class DegenerateGrid(ValueError):
    """A trial grid that admits no correlation matrix.

    A configuration with zero variance has no correlation with anything -- not a
    correlation of zero, an undefined one. Gate 0.4's rule applies: those are different
    claims and only one of them is true, so this raises rather than filling in a zero and
    letting the eigenvalues absorb it.
    """


def trial_correlation(grid: Matrix) -> Matrix:
    """`(T, N)` trial returns -> `(N, N)` correlation matrix.

    Symmetrised explicitly. `np.corrcoef` is symmetric in exact arithmetic but returns
    last-bit asymmetries in practice, and `scipy.spatial.distance.squareform` rejects a
    matrix that is not exactly symmetric -- so this is load-bearing, not hygiene.
    """
    if grid.ndim != 2:
        raise ValueError(f"expected a (T, N) matrix, got shape {grid.shape}")
    t, n = grid.shape
    if t < 3:
        raise ValueError(f"need at least 3 observations to correlate, got {t}")
    if n < 1:
        raise ValueError("need at least one trial")
    if not np.all(np.isfinite(grid)):
        raise DegenerateGrid("trial returns contain NaN or infinity")

    # The floor, not `sd == 0.0`. A column holding one value T times has
    # `std(ddof=1) ~ 1e-19` rather than exactly zero, because `sum/n` does not round-trip
    # to the original float -- so an exact comparison lets it through and `np.corrcoef`
    # divides by dust, producing correlations that are pure rounding noise. This is the
    # same failure `selection.noise_floor` exists for at Gate 0.4, so it uses that
    # function rather than a second, subtly different threshold.
    sd = grid.std(axis=0, ddof=1)
    scale = np.max(np.abs(grid), axis=0)
    floors = np.array([noise_floor(float(m), t) for m in scale], dtype=np.float64)
    dead = ~np.isfinite(sd) | (sd <= floors)
    if np.any(dead):
        bad = np.flatnonzero(dead)
        raise DegenerateGrid(
            f"trials {bad.tolist()} have zero variance (sd={sd[bad].tolist()}, "
            f"floor={floors[bad].tolist()}), so their correlation is undefined rather "
            "than zero"
        )

    corr = np.corrcoef(grid, rowvar=False)
    corr = np.atleast_2d(np.asarray(corr, dtype=np.float64))
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def participation_ratio(corr: Matrix) -> float:
    """`N_eff = (sum lambda)^2 / sum lambda^2` over the correlation eigenvalues.

    The primary estimator (Part H decision 2). Bounded by construction: `N` when the
    trials are mutually uncorrelated, 1 when they are all the same trial.

    A correlation matrix has unit diagonal, so `sum lambda = trace = N` exactly and this
    reduces to `N^2 / sum lambda^2`. The general form is computed anyway -- the identity
    is asserted in the tests instead, where a violation means the input was not a
    correlation matrix and the caller should hear about it.

    Eigenvalues of a positive semi-definite matrix are non-negative in exact arithmetic
    and slightly negative in floating point; they are clipped at zero, which changes the
    sum by ~1e-16 and prevents a negative from cancelling a real contribution.
    """
    n = int(corr.shape[0])
    if corr.shape != (n, n):
        raise ValueError(f"expected a square matrix, got shape {corr.shape}")
    if n == 1:
        return 1.0

    eigenvalues = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    total = float(eigenvalues.sum())
    sum_sq = float(np.square(eigenvalues).sum())
    if sum_sq <= 0.0 or not math.isfinite(sum_sq):
        raise DegenerateGrid("correlation matrix has no positive eigenvalues")
    return float(total * total / sum_sq)


def cluster_count(corr: Matrix, cut_height: float = DEFAULT_CUT_HEIGHT) -> int:
    """Number of clusters at `cut_height` under average linkage. Secondary estimate.

    Distance is `d_ij = sqrt(0.5 * (1 - rho_ij))` (01 B4), which maps perfect correlation
    to 0 and perfect anticorrelation to 1.

    That mapping is where this estimator is weakest, and the hedge case shows it exactly.
    A trial and its exact negative sit at maximum distance, so clustering counts two,
    while the participation ratio returns 1.000 -- correctly, because `rho = -1` and
    `rho = +1` collapse the same dimension and the pair spans one direction between them.
    Measured on that input the two estimates differ by a factor of two, which is
    precisely Part H decision 2's revisit condition, and its rule (report the smaller)
    lands on the right answer without needing a special case. The disagreement is the
    signal; do not read the clustering number as "independent bets" on its own.
    """
    n = int(corr.shape[0])
    if n == 1:
        return 1
    if not (0.0 < cut_height <= 1.0):
        raise ValueError(f"cut_height must lie in (0, 1], got {cut_height}")

    distance = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    labels = fcluster(linkage(condensed, method="average"), t=cut_height, criterion="distance")
    return int(np.unique(labels).size)


@dataclass(frozen=True, slots=True)
class EffectiveTrials:
    """Both estimates of `N_eff`, with the raw count. Frozen (B7).

    Reported as a pair because 01 B4 requires it, and because the pair is informative:
    when they agree the number is trustworthy, and when they do not the grid has
    structure that one of them is missing.
    """

    n_raw: int
    n_eff: float  # participation ratio -- primary, Part H decision 2
    n_eff_clustering: int  # secondary
    cut_height: float

    @property
    def diverges(self) -> bool:
        """Part H decision 2's revisit condition: a factor of two between the estimates.

        Part H says "diverge by more than 2x", which read literally excludes a ratio of
        exactly 2. The boundary is inclusive here, and that is a deliberate edge-case
        ruling rather than a change to the decision. The hedge pair -- one trial and its
        negative -- lands on exactly 2.000, so the literal reading leaves the canonical
        divergence case unflagged. Being inclusive costs nothing when the estimates agree
        and prevents `reportable` returning the larger estimate at the one ratio the rule
        was written for.
        """
        lo = min(self.n_eff, float(self.n_eff_clustering))
        hi = max(self.n_eff, float(self.n_eff_clustering))
        return lo > 0.0 and hi / lo >= DIVERGENCE_FACTOR

    @property
    def reportable(self) -> float:
        """The number to deflate by. The smaller of the two when they diverge, per
        Part H; the participation ratio otherwise."""
        if self.diverges:
            return min(self.n_eff, float(self.n_eff_clustering))
        return self.n_eff

    @property
    def compression(self) -> float:
        """`N_eff / N`. How much of the search was distinct rather than repeated."""
        return self.n_eff / self.n_raw if self.n_raw else float("nan")

    def describe(self) -> str:
        note = "  DIVERGENT (>2x): reporting the smaller" if self.diverges else ""
        return (
            f"N={self.n_raw}  N_eff={self.n_eff:.1f} (participation ratio)  "
            f"N_eff={self.n_eff_clustering} (clustering @ {self.cut_height})  "
            f"compression={self.compression:.2f}{note}"
        )


def effective_trials(grid: Matrix, cut_height: float = DEFAULT_CUT_HEIGHT) -> EffectiveTrials:
    """`(T, N)` trial returns -> both estimates of the effective number of trials."""
    corr = trial_correlation(grid)
    return EffectiveTrials(
        n_raw=int(grid.shape[1]),
        n_eff=participation_ratio(corr),
        n_eff_clustering=cluster_count(corr, cut_height),
        cut_height=cut_height,
    )


__all__ = [
    "DEFAULT_CUT_HEIGHT",
    "DIVERGENCE_FACTOR",
    "DegenerateGrid",
    "EffectiveTrials",
    "cluster_count",
    "effective_trials",
    "participation_ratio",
    "trial_correlation",
]
