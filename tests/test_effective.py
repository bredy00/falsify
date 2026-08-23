"""B4 -- effective number of trials. 01 Part B4, decided by 03 Part H decision 2.

The headline measurement, and the reason this module exists: an MA-crossover lattice run
through the certified engine on 2,500 bars of GBM gives

    N=16   N_eff=2.0    compression 0.12
    N=54   N_eff=1.8    compression 0.03
    N=270  N_eff=1.8    compression 0.01

`N_eff` does not move as `N` grows sixteen-fold. Those 270 configurations are not 270
looks at the data; they are about two, evaluated repeatedly. Deflating a Sharpe by 270
would be roughly 150x too harsh -- conservative, but wrong, and wrong in a way that
makes the correction look rigorous while being unmoored from what the search actually
covered.
"""

from __future__ import annotations

import numpy as np
import pytest

from falsify.costs import ZERO_COST
from falsify.effective import (
    DIVERGENCE_FACTOR,
    DegenerateGrid,
    cluster_count,
    effective_trials,
    participation_ratio,
    trial_correlation,
)
from falsify.evaluation import build_grid
from falsify.strategies.simple import MACrossover
from falsify.synthetic import bars_from_close, gbm

T = 1_000


def independent(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(T, n))


def identical(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=T)
    return np.tile(base[:, None], (1, n)) + rng.normal(0.0, 1e-9, (T, n))


# --------------------------------------------------------------------------------------
# The two bounds the estimator must hit exactly.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 20, 60])
def test_independent_trials_give_back_the_raw_count(n: int) -> None:
    """`N_eff -> N` when nothing is correlated with anything. Measured 4.97, 19.61, 56.63
    for N = 5, 20, 60 -- short of N by the sampling noise in the correlation matrix, which
    is real and should not be tuned away."""
    e = effective_trials(independent(n))
    assert 0.85 * n <= e.n_eff <= n, f"N_eff {e.n_eff:.2f} for {n} independent trials"
    assert e.n_eff_clustering == n


@pytest.mark.parametrize("n", [5, 20, 60])
def test_identical_trials_collapse_to_one(n: int) -> None:
    """`N_eff -> 1` when every column is the same trial. This is the case that makes the
    correction matter: a grid of near-duplicates should not be allowed to claim credit
    for a wide search."""
    e = effective_trials(identical(n))
    assert abs(e.n_eff - 1.0) < 0.01, f"N_eff {e.n_eff:.4f} for {n} identical trials"
    assert e.n_eff_clustering == 1


def test_n_eff_never_exceeds_n() -> None:
    """The correction can only ever shrink the trial count. If it grew one, the
    deflation would be anti-conservative and the DSR would overstate significance."""
    for n in (3, 12, 40):
        for builder in (independent, identical):
            e = effective_trials(builder(n))
            assert 1.0 <= e.n_eff <= n + 1e-9


def test_the_eigenvalues_sum_to_n_because_a_correlation_matrix_has_unit_diagonal() -> None:
    """`sum(lambda) = trace = N` exactly, so the participation ratio reduces to
    `N^2 / sum(lambda^2)`. Asserted rather than used: if this fails the input was not a
    correlation matrix, and the caller should hear about it rather than get a number."""
    for n in (5, 30):
        corr = trial_correlation(independent(n, seed=1))
        total = float(np.linalg.eigvalsh(corr).sum())
        assert abs(total - n) < 1e-10, f"trace {total} != N {n}"


# --------------------------------------------------------------------------------------
# On a grid the engine actually produced.
# --------------------------------------------------------------------------------------


def test_a_parameter_lattice_contains_far_fewer_bets_than_configurations() -> None:
    """The finding this module exists to surface, measured through the real engine.

    A 54-configuration MA-crossover lattice on GBM has an effective size under 4. The
    assertion is deliberately loose on the upper side -- what matters is that the
    compression is an order of magnitude, not that it hits a particular decimal.
    """
    bars = bars_from_close(gbm(mu=0.08, sigma=0.20, n_bars=2_500, rng=np.random.default_rng(4)))
    strategies = [MACrossover(f, s) for f in range(5, 35, 5) for s in range(40, 130, 10) if f < s]
    grid = build_grid(bars, strategies, ZERO_COST)
    e = effective_trials(grid.returns)
    print(e.describe())

    assert e.n_raw >= 40, "the lattice shrank; this test is no longer about a real grid"
    assert e.n_eff < 6.0, (
        f"N_eff came out {e.n_eff:.2f} on a lattice of {e.n_raw} overlapping moving-average "
        "configurations. They share most of their signal; if this is now large, the "
        "correlation matrix is not seeing that."
    )
    assert e.compression < 0.2


def test_adding_configurations_to_a_lattice_adds_almost_no_independent_information() -> None:
    """N grows 16 -> 54, N_eff does not. Measured 2.0 -> 1.8.

    This is the practical claim: densifying a grid inflates raw `N` and the apparent
    severity of the deflation while adding nothing to the search.
    """
    bars = bars_from_close(gbm(mu=0.08, sigma=0.20, n_bars=2_500, rng=np.random.default_rng(4)))
    coarse = [MACrossover(f, s) for f in (5, 10, 20, 30) for s in (40, 60, 90, 120) if f < s]
    dense = [MACrossover(f, s) for f in range(5, 35, 5) for s in range(40, 130, 10) if f < s]

    c = effective_trials(build_grid(bars, coarse, ZERO_COST).returns)
    d = effective_trials(build_grid(bars, dense, ZERO_COST).returns)
    print(f"coarse N={c.n_raw} N_eff={c.n_eff:.2f}  ->  dense N={d.n_raw} N_eff={d.n_eff:.2f}")

    assert d.n_raw > 2 * c.n_raw, "the two lattices are not different enough to compare"
    assert d.n_eff < 2.0 * c.n_eff, (
        f"N_eff scaled with N ({c.n_eff:.2f} -> {d.n_eff:.2f}) when the raw count more than "
        "doubled. The whole point is that it should not."
    )


# --------------------------------------------------------------------------------------
# Where the two estimators disagree, and why the disagreement is the useful part.
# --------------------------------------------------------------------------------------


def test_a_hedge_pair_is_where_the_two_estimators_diverge() -> None:
    """A trial and its exact negative. Measured: participation ratio 1.000, clustering 2.

    The participation ratio is right -- `rho = -1` and `rho = +1` collapse the same
    dimension, and the pair spans one direction. Clustering puts them at maximum distance
    under `d = sqrt(0.5(1 - rho))` and counts two. The estimates differ by exactly the
    factor of two that is Part H decision 2's revisit condition, and its rule (report the
    smaller) lands on the correct answer with no special case.
    """
    x = np.random.default_rng(1).normal(size=(T, 1))
    e = effective_trials(np.hstack([x, -x]))
    print(e.describe())

    assert abs(e.n_eff - 1.0) < 1e-6, f"participation ratio gave {e.n_eff:.6f} for a hedge pair"
    assert e.n_eff_clustering == 2
    assert e.diverges, "a factor of two is exactly Part H decision 2's revisit condition"
    assert abs(e.reportable - 1.0) < 1e-6, "the rule must report the smaller estimate"


def test_agreement_is_not_flagged_as_divergence() -> None:
    """The flag has to be quiet when the estimators agree, or it means nothing when
    they do not."""
    e = effective_trials(independent(20, seed=5))
    assert not e.diverges
    assert e.reportable == e.n_eff
    assert "DIVERGENT" not in e.describe()
    ratio = max(e.n_eff, e.n_eff_clustering) / min(e.n_eff, e.n_eff_clustering)
    assert ratio < DIVERGENCE_FACTOR


# --------------------------------------------------------------------------------------
# Degenerate input is refused, per Gate 0.4.
# --------------------------------------------------------------------------------------


def test_a_constant_trial_has_an_undefined_correlation_not_a_zero_one() -> None:
    """Gate 0.4's rule. Filling in a zero would let a dead configuration count as an
    independent bet and inflate `N_eff`, which is the anti-conservative direction."""
    grid = independent(5)
    grid[:, 2] = 0.004
    with pytest.raises(DegenerateGrid, match="zero variance"):
        effective_trials(grid)


def test_non_finite_input_is_refused() -> None:
    grid = independent(4)
    grid[3, 1] = np.inf
    with pytest.raises(DegenerateGrid, match="NaN or infinity"):
        trial_correlation(grid)


def test_a_single_trial_is_one_effective_trial() -> None:
    e = effective_trials(independent(1))
    assert e.n_eff == 1.0
    assert e.n_eff_clustering == 1
    assert participation_ratio(np.ones((1, 1))) == 1.0
    assert cluster_count(np.ones((1, 1))) == 1
