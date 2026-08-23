"""B5 -- the stationary bootstrap. 01 Part B5, parameterised by 03 Part H decision 3.

Every bound here came from measurement first. What was measured, before these tests
existed -- the ratio of the bootstrap SE of the mean to `sd/sqrt(T)` on i.i.d. normal
data, 40 samples per cell, 600 replicates each:

                  block    T=500              T=2000
    T^(-1/3)       8-13    0.9961 +/- 0.0105  1.0141 +/- 0.0081
    1/sqrt(T)     22-45    0.9780 +/- 0.0191  0.9963 +/- 0.0170
    T^(-2/3)      63-159   0.8961 +/- 0.0288  0.9186 +/- 0.0262
    i.i.d. control (p=1)                      0.9960 +/- 0.0049

At Part H's default of `p = 1/sqrt(T)` the resampler is unbiased -- within 1.2 SE of 1.0
at both sample sizes. At the long-block end it is not: 0.896 and 0.919 are 3.1 to 3.6 SE
*below* 1.0, because a block of 159 in a sample of 2000 leaves only about a dozen
effective blocks and the resampled mean comes out under-dispersed. That is a real limit
of the estimator at that setting, so it is asserted as a known bias rather than papered
over -- and it is a reason to read the long end of the sensitivity grid as diagnostic
rather than as an equally good alternative.
"""

from __future__ import annotations

import math
import statistics as st

import numpy as np
import pytest

from falsify.bootstrap import (
    BootstrapCI,
    _reference,
    bootstrap_ci,
    default_p,
    p_sensitivity,
    sensitivity_grid,
    stationary_bootstrap,
    width_dispersion,
)

SAMPLES = 12
N_BOOT = 600


def ar1(phi: float, n: int, seed: int, sigma: float = 0.01) -> np.ndarray:
    e = np.random.default_rng(seed).normal(0.0, sigma, n)
    out = np.empty(n)
    out[0] = e[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + e[i]
    return out


def lag1_autocorr(x: np.ndarray) -> float:
    c = x - x.mean()
    return float((c[:-1] * c[1:]).sum() / (c * c).sum())


# --------------------------------------------------------------------------------------
# The specification is a loop. The fast path must be that loop.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_obs", "p", "n_boot"),
    [(50, 0.2, 20), (200, 1 / math.sqrt(200), 30), (1000, 0.05, 10)],
)
def test_the_fast_path_is_bitwise_identical_to_the_01_b5_reference(
    n_obs: int, p: float, n_boot: int
) -> None:
    """The same discipline as G2's twin engines, for the same reason.

    `_reference` is the loop exactly as 01 Part B5 writes it, and is what a reader will
    check the code against. `stationary_bootstrap` removes the inner scan over T and is
    11x faster at T=2516. Nothing but this test stops the optimisation drifting away from
    the specification, and a drift would be invisible -- both versions return plausible
    resampled paths.
    """
    x = np.random.default_rng(0).normal(size=n_obs)
    slow = _reference(x, p, n_boot, np.random.default_rng(42))
    fast = stationary_bootstrap(x, p, n_boot, np.random.default_rng(42))
    assert np.array_equal(slow, fast), (
        "the vectorised resampler no longer reproduces 01 Part B5's reference loop"
    )


# --------------------------------------------------------------------------------------
# 01 B5's own validation test.
# --------------------------------------------------------------------------------------


def test_bootstrap_se_of_the_mean_matches_the_analytic_one_on_iid_data() -> None:
    """01 B5: "bootstrap an i.i.d. normal sample. The bootstrap SE of the mean must match
    sigma/sqrt(T) within Monte Carlo error. If it doesn't, the resampler is broken."

    Measured 0.9963 +/- 0.0170 over 40 samples at these settings. The tolerance is 4 SE
    of the mean of SAMPLES draws from a per-sample sd of 0.107 -- what 12 samples can
    resolve, not a number chosen to look tight.
    """
    t = 2_000
    ratios = []
    for s in range(SAMPLES):
        x = np.random.default_rng(1_000 + s).normal(0.0, 1.5, t)
        paths = stationary_bootstrap(x, default_p(t), N_BOOT, np.random.default_rng(5_000 + s))
        ratios.append(paths.mean(axis=1).std(ddof=1) / (x.std(ddof=1) / math.sqrt(t)))

    mean = st.mean(ratios)
    tolerance = 4.0 * 0.107 / math.sqrt(SAMPLES)
    print(f"bootstrap SE / analytic SE = {mean:.4f} over {SAMPLES} samples (tol {tolerance:.3f})")
    assert abs(mean - 1.0) < tolerance, (
        f"the resampler puts the SE of the mean at {mean:.4f} of its analytic value on "
        "i.i.d. data, where there is no dependence to capture. That is 01 B5's stated "
        "check for a broken resampler."
    )


def test_long_blocks_under_disperse_the_mean() -> None:
    """The measured limit, asserted so it cannot silently disappear or worsen.

    At `T^(-2/3)` the mean block is 159 bars of a 2000-bar sample, leaving about a dozen
    effective blocks, and the ratio falls to 0.919 -- 3.1 SE below 1.0. This is why the
    sensitivity grid is diagnostic: its ends are not three equally good parameter choices,
    and a narrower interval at the long end is the estimator degrading rather than the
    data speaking.
    """
    t = 2_000
    long_p = sensitivity_grid(t)[2]
    ratios = []
    for s in range(SAMPLES):
        x = np.random.default_rng(1_000 + s).normal(0.0, 1.5, t)
        paths = stationary_bootstrap(x, long_p, N_BOOT, np.random.default_rng(5_000 + s))
        ratios.append(paths.mean(axis=1).std(ddof=1) / (x.std(ddof=1) / math.sqrt(t)))
    mean = st.mean(ratios)
    print(f"long-block (p={long_p:.4f}, block~{1 / long_p:.0f}) ratio = {mean:.4f}")
    assert mean < 0.97, (
        f"the long-block downward bias has gone (ratio {mean:.4f}). It was 0.919; if the "
        "resampler improved, the docstrings that warn about it need rewriting."
    )


# --------------------------------------------------------------------------------------
# What the stationary bootstrap is FOR.
# --------------------------------------------------------------------------------------


def test_serial_dependence_survives_resampling_and_iid_resampling_destroys_it() -> None:
    """The whole justification for blocks, in one comparison.

    Measured on AR(1) with phi=0.6: the original series has lag-1 autocorrelation 0.611,
    resampled paths retain 0.599, and `p = 1` -- which makes every observation its own
    block, i.e. the i.i.d. bootstrap -- returns 0.001. An i.i.d. bootstrap of a serially
    correlated series produces confident intervals about a process that is not the one
    being studied.
    """
    x = ar1(0.6, 4_000, seed=3)
    original = lag1_autocorr(x)
    blocks = stationary_bootstrap(x, default_p(4_000), 300, np.random.default_rng(5))
    iid = stationary_bootstrap(x, 1.0, 300, np.random.default_rng(5))

    kept = float(np.mean([lag1_autocorr(path) for path in blocks]))
    destroyed = float(np.mean([lag1_autocorr(path) for path in iid]))
    print(f"lag-1 autocorr: original {original:.3f}, blocks {kept:.3f}, iid {destroyed:.3f}")

    assert abs(kept - original) < 0.10, (
        f"block resampling kept only {kept:.3f} of an original {original:.3f}; the "
        "dependence the method exists to preserve is being lost"
    )
    assert abs(destroyed) < 0.05, "the p=1 control should behave like an i.i.d. bootstrap"


def test_the_interval_widens_when_returns_are_persistent() -> None:
    """A serially correlated series carries less information per observation, so its
    interval must be wider. Same variance, same length, different dependence."""
    rng = np.random.default_rng(9)
    independent = np.random.default_rng(4).normal(0.0, 0.01, 3_000)
    persistent = ar1(0.7, 3_000, seed=4)

    wide = bootstrap_ci(persistent, lambda v: float(np.mean(v)), rng, n_boot=400)
    narrow = bootstrap_ci(independent, lambda v: float(np.mean(v)), rng, n_boot=400)
    print(f"persistent {wide.width:.3e} vs independent {narrow.width:.3e}")
    assert wide.width > narrow.width


# --------------------------------------------------------------------------------------
# Reporting surface.
# --------------------------------------------------------------------------------------


def test_the_sensitivity_grid_runs_short_blocks_to_long() -> None:
    grid = sensitivity_grid(1_000)
    assert grid[0] > grid[1] > grid[2], "p decreasing means block length increasing"
    assert math.isclose(grid[1], default_p(1_000)), "the middle of the grid is the default"
    assert math.isclose(1.0 / grid[1], math.sqrt(1_000), rel_tol=1e-12)


def test_p_sensitivity_reports_three_intervals_and_their_dispersion() -> None:
    """01 B5 requires the sensitivity be reported, and sets 20% as the level above which
    the block length is doing real work. This asserts the machinery, not a value: on
    i.i.d. data the dispersion is small, and on real returns it is the number to look at.
    """
    x = np.random.default_rng(2).normal(0.0, 0.01, 1_200)
    intervals = p_sensitivity(x, lambda v: float(np.mean(v)), np.random.default_rng(2), n_boot=300)
    assert len(intervals) == 3
    assert all(isinstance(i, BootstrapCI) for i in intervals)
    assert all(i.lo < i.point < i.hi for i in intervals)
    dispersion = width_dispersion(intervals)
    print(f"width dispersion across the grid: {dispersion:.1%} (01 B5 flags >20%)")
    assert math.isfinite(dispersion) and dispersion >= 0.0


def test_the_interval_carries_the_parameters_that_produced_it() -> None:
    """B2 is about numbers being checkable. An interval without its `p` and `n_boot` is
    not reproducible, so `BootstrapCI` refuses to exist without them."""
    x = np.random.default_rng(6).normal(0.0, 0.01, 500)
    ci = bootstrap_ci(x, lambda v: float(np.mean(v)), np.random.default_rng(6), n_boot=200)
    assert ci.n_boot == 200
    assert math.isclose(ci.p, default_p(500))
    assert math.isclose(ci.mean_block_length, math.sqrt(500), rel_tol=1e-12)
    assert "p=" in ci.describe() and "B=" in ci.describe()


def test_the_bootstrap_is_deterministic_given_a_seed() -> None:
    """B9. G10 depends on it."""
    x = np.random.default_rng(8).normal(size=300)
    a = stationary_bootstrap(x, 0.1, 50, np.random.default_rng(1))
    b = stationary_bootstrap(x, 0.1, 50, np.random.default_rng(1))
    assert np.array_equal(a, b)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"p": 0.0}, "p must lie"),
        ({"p": 1.5}, "p must lie"),
        ({"n_boot": 0}, "n_boot must be"),
    ],
)
def test_bad_parameters_raise_rather_than_return_something_plausible(
    kwargs: dict[str, float], match: str
) -> None:
    x = np.random.default_rng(0).normal(size=100)
    args = {"p": 0.1, "n_boot": 10} | kwargs
    with pytest.raises(ValueError, match=match):
        stationary_bootstrap(x, float(args["p"]), int(args["n_boot"]), np.random.default_rng(0))


def test_a_series_with_nan_is_refused() -> None:
    x = np.random.default_rng(0).normal(size=100)
    x[7] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        stationary_bootstrap(x, 0.1, 10, np.random.default_rng(0))
