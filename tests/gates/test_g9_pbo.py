"""G9 -- Probability of Backtest Overfitting via CSCV.

The gate PLAYBOOK states as "PBO computed and reported; ship only if < 0.5", and the
hardest code in the project by 03's own estimate. Everything asserted here was measured
first; the numbers quoted in each docstring are the measurements the bound came from,
not targets chosen in advance.

What was measured, before any of these tests existed:

  Null calibration, `PBO(ArgMax)` on grids with no differential merit, 80 grids each:
      S= 8 (70 splits)      0.4930 +/- 0.0247
      S=10 (252 splits)     0.4850 +/- 0.0251
      S=12 (924 splits)     0.4798 +/- 0.0247
      S=16 (12,870 splits)  0.4809 +/- 0.0242
  All four within 0.9 SE of 0.5, which is what CSCV claims and is the reason the
  0.5 ship/no-ship threshold means anything.

  The grid-to-grid standard deviation is 0.22 and barely moves with S. That is the
  single most important number in this file: PBO is a statement about a grid, and
  running more splits on one grid does not shrink it. An earlier 16-grid run put the
  null at 0.345 and looked like a 3.2-SE bias; it was an unlucky draw of seeds. Every
  bound below is therefore set against a per-grid sd of 0.22, over many grids.
"""

from __future__ import annotations

import math
import statistics as st
from collections.abc import Callable, Sequence
from functools import cache

import numpy as np
import pytest
from numpy.typing import NDArray

from falsify.cscv import CSCVResult, cscv, n_splits_for
from falsify.selection import ArgMax, EqualWeight, SelectionRule, Softmax, TopK
from falsify.synthetic import compensation_grid, merit_grid, noise_grid

BLOCKS = 10  # C(10,5) = 252 splits: the null calibrates identically at S=16, far cheaper
GRIDS = 40
PER_GRID_SD = 0.22  # measured over 80 grids at each of S = 8, 10, 12, 16

# Seeds 5000+ are the block characterised by the 80-grid calibration quoted above, so
# the tests draw from a distribution that was measured rather than one assumed. Worth
# stating plainly, because picking a seed block that flatters a result is the exact sin
# this project exists to catch: a neighbouring block (1000-1023) put the null at 0.374,
# which is 2.9 SE low and entirely consistent with a per-grid sd of 0.22. That spread is
# the finding, not a reason to shop for seeds -- which is why the tolerance below stays
# at 4 SE and why nothing here is asserted on a single grid.
SEED0 = 5_000

Builder = Callable[..., NDArray[np.float64]]

# Singletons, because `functools.cache` keys on object identity for these: a fresh
# `ArgMax()` at each call site would miss the cache every time and quietly undo it.
ARGMAX = ArgMax()
EQUAL_WEIGHT = EqualWeight()


@cache
def sweep(
    builder: Builder, rule: SelectionRule, n_grids: int = GRIDS, seed0: int = SEED0
) -> tuple[CSCVResult, ...]:
    """One CSCV sweep per (builder, rule), memoised across tests.

    Five tests share three sweeps; recomputing them cost 11s of a 26s gate. Caching is
    safe because the inputs are seeds rather than state and `cscv` is deterministic
    by B9 -- the results are identical either way.

    Whole `CSCVResult`s rather than bare PBOs, so the F3 test can look at the ranks of
    the same sweep the calibration test took its mean from instead of running its own.
    """
    out = []
    for s in range(n_grids):
        rng = np.random.default_rng(seed0 + s)
        grid = builder(rng, n_blocks=BLOCKS) if builder is compensation_grid else builder(rng)
        out.append(cscv(grid, rule, n_blocks=BLOCKS))
    return tuple(out)


def pbo_over_grids(
    builder: Builder, rule: SelectionRule, n_grids: int = GRIDS, seed0: int = SEED0
) -> tuple[float, ...]:
    return tuple(r.pbo() for r in sweep(builder, rule, n_grids, seed0))


def summarise(values: Sequence[float]) -> tuple[float, float]:
    """Mean and its standard error. B2: no number without an error bar."""
    return st.mean(values), st.stdev(values) / math.sqrt(len(values))


# --------------------------------------------------------------------------------------
# Structure. These would catch a rank bug that no statistical test would notice.
# --------------------------------------------------------------------------------------


def test_g9_enumerates_every_symmetric_half_split() -> None:
    """C(S, S/2) splits, no more and no fewer. Off-by-one here is silent.

    Checked at two block counts, because one could agree with the formula by accident.
    S=16 and its 12,870 splits are deliberately not run here: it is one code path with
    a different loop bound, it costs 4s, and `scripts/g9_temperature.py` exercises it
    at full size for the figure. What is asserted here is that the enumeration matches
    `n_splits_for` where it is cheap to check, plus the arithmetic at 16.
    """
    grid = noise_grid(np.random.default_rng(0))
    assert cscv(grid, ARGMAX, n_blocks=10).n_splits == n_splits_for(10) == 252
    assert cscv(grid, ARGMAX, n_blocks=12).n_splits == n_splits_for(12) == 924
    assert n_splits_for(16) == 12_870


def test_g9_ranks_stay_strictly_inside_the_open_unit_interval() -> None:
    """`omega/(N+1)` exists so no rank is 0 or 1, because logit(0) is infinite and an
    infinite logit silently drops a split from the average -- which would bias PBO by
    exactly the splits that are most extreme."""
    result = cscv(noise_grid(np.random.default_rng(3)), ArgMax(), n_blocks=BLOCKS)
    assert np.all(result.relative_ranks > 0.0) and np.all(result.relative_ranks < 1.0)
    assert np.all(np.isfinite(result.logits)), "an infinite logit means a dropped split"
    assert 0.0 <= result.pbo() <= 1.0


def test_g9_a_saturated_pbo_is_investigated_rather_than_trusted() -> None:
    """F3: PBO of exactly 0.0 or 1.0 is the signature of broken rank bookkeeping.

    It is not proof of one. On a grid with real persistent merit the winner genuinely
    never lands in the out-of-sample bottom half, and PBO is then exactly 0 for honest
    reasons. The two cases are told apart by looking at the ranks: broken bookkeeping
    pins them to one value, while a real sweep moves them around and simply never
    crosses the boundary.

    Measured on the grid that produces 0.0000: five distinct ranks, minimum 0.6154
    against a boundary of 0.5. A sweep that stops short, not a stuck pointer.
    """
    saturated = [r for r in sweep(merit_grid, ARGMAX) if r.pbo() in (0.0, 1.0)]
    assert saturated, (
        "no merit grid saturated, so this test is not exercising the F3 path it exists "
        "for -- the generator drifted and the check has quietly stopped checking"
    )
    for result in saturated:
        distinct = np.unique(result.relative_ranks)
        assert distinct.size >= 2, (
            f"PBO={result.pbo()} with every split at rank {distinct[0]:.4f}. That is not a "
            "confident selection, it is a rank computation that stopped varying."
        )
        assert np.all(np.isfinite(result.logits))


# --------------------------------------------------------------------------------------
# Calibration. The threshold is only meaningful if the null actually sits at 0.5.
# --------------------------------------------------------------------------------------


def test_g9_pbo_is_one_half_when_no_configuration_is_better_than_another() -> None:
    """The null, and the justification for the 0.5 ship/no-ship line.

    With no differential merit the in-sample winner is a coin flip to land in either
    out-of-sample half, so PBO must be 0.5. Measured over 80 grids: 0.4850 +/- 0.0251
    at these settings, 0.60 SE from 0.5.

    The tolerance is 4 SE of the mean of GRIDS draws from a per-grid sd of 0.22 -- what
    40 grids can actually resolve, not what would look impressive.
    """
    values = pbo_over_grids(noise_grid, ARGMAX)
    mean, se = summarise(values)
    tolerance = 4.0 * PER_GRID_SD / math.sqrt(GRIDS)
    print(f"null PBO(ArgMax) = {mean:.4f} +/- {se:.4f} over {GRIDS} grids (tol {tolerance:.3f})")
    assert abs(mean - 0.5) < tolerance, (
        f"the null sits at {mean:.4f}, {abs(mean - 0.5) / se:.1f} SE from 0.5. Either the "
        "rank bookkeeping is biased or the generator is not the null it claims to be. "
        "Do not widen this tolerance without finding out which."
    )


def test_g9_pbo_is_low_when_the_differences_between_configurations_are_real() -> None:
    """Power. A gate that fires on everything is as useless as one that never fires.

    Measured: 0.090 +/- 0.018 over 40 grids, worst grid 0.409 -- every one below the
    0.5 threshold, so selection on this grid ships and should.
    """
    values = pbo_over_grids(merit_grid, ARGMAX)
    mean, se = summarise(values)
    print(f"merit PBO(ArgMax) = {mean:.4f} +/- {se:.4f}, worst grid {max(values):.4f}")
    assert mean < 0.25, f"PBO {mean:.4f} on a grid whose best column is genuinely best"
    assert all(v < 0.5 for v in values), (
        f"{sum(v >= 0.5 for v in values)} grids with real merit failed the 0.5 gate"
    )


# --------------------------------------------------------------------------------------
# F7. The gate has to be shown firing, on a grid built to defeat it.
# --------------------------------------------------------------------------------------


def test_g9_fires_on_a_grid_where_selection_is_a_trap() -> None:
    """F7: a gate that has never failed is not a gate.

    `compensation_grid` centres each configuration's per-block means to sum to zero, so
    over complementary halves the in-sample mean is the negative of the out-of-sample
    mean and the in-sample winner is mechanically the out-of-sample loser. Propositions
    3 and 5, constructed rather than hoped for.

    Measured: 0.793 +/- 0.026 over 40 grids, 11.3 SE above the 0.5 threshold. Not every
    individual grid clears it -- the best was 0.337 -- which is the per-grid sd of 0.22
    showing up again and is exactly why the assertion is on the mean.
    """
    values = pbo_over_grids(compensation_grid, ARGMAX)
    mean, se = summarise(values)
    print(f"trap PBO(ArgMax) = {mean:.4f} +/- {se:.4f}, best grid {min(values):.4f}")
    assert mean > 0.6, (
        f"the trap only reached PBO {mean:.4f}. G9 is supposed to catch this grid; if it "
        "no longer does, the gate has stopped gating."
    )
    assert (mean - 0.5) / se > 3.0, f"only {(mean - 0.5) / se:.1f} SE above the threshold"


def test_g9_separates_the_trap_from_the_genuine_grid() -> None:
    """The whole claim in one comparison: same rule, same block count, same number of
    configurations, opposite verdicts. Measured 0.090 (merit) < 0.496 (null) < 0.793 (trap)."""
    trap, _ = summarise(pbo_over_grids(compensation_grid, ARGMAX))
    real, _ = summarise(pbo_over_grids(merit_grid, ARGMAX))
    null, _ = summarise(pbo_over_grids(noise_grid, ARGMAX))
    print(f"PBO: merit {real:.3f} < null {null:.3f} < trap {trap:.3f}")
    assert real < null < trap, (
        f"PBO failed to order the three grids (merit {real:.3f}, null {null:.3f}, "
        f"trap {trap:.3f}). It is then measuring something other than overfitting."
    )


# --------------------------------------------------------------------------------------
# Temperature. 01 Part E3's headline figure -- and where the spec's expectation is wrong.
# --------------------------------------------------------------------------------------


def test_g9_softmax_approaches_argmax_as_temperature_falls() -> None:
    """The tau -> 0 endpoint of the dial, asserted on the weights rather than on PBO."""
    grid = compensation_grid(np.random.default_rng(11), n_blocks=BLOCKS)
    block = grid[: grid.shape[0] // 2]
    cold = Softmax(1e-3).weights(block)
    assert np.array_equal(np.argmax(cold), np.argmax(ArgMax().weights(block)))
    assert cold.max() > 0.99, f"Softmax(1e-3) spread {1 - cold.max():.2e} off the winner"


def test_g9_cooling_the_temperature_raises_the_price_of_selectivity() -> None:
    """The robust half of 01 Part E3's expectation.

    E3 predicts PBO decreases monotonically in tau. Measured paired across grids, the
    high-temperature half of that holds firmly -- PBO(tau=16) < PBO(tau=1) in 14 of 16
    grids at these settings and 8 of 8 at S=12 -- so this asserts that half, paired,
    because the grid-to-grid sd of 0.22 dwarfs the effect and an unpaired test cannot
    see it.

    The low-temperature half does NOT hold; see the test below.
    """
    diffs = []
    for s in range(16):
        grid = compensation_grid(np.random.default_rng(2_000 + s), n_blocks=BLOCKS)
        hot = cscv(grid, Softmax(16.0), n_blocks=BLOCKS).pbo()
        cool = cscv(grid, Softmax(1.0), n_blocks=BLOCKS).pbo()
        diffs.append(cool - hot)
    mean, se = st.mean(diffs), st.stdev(diffs) / math.sqrt(len(diffs))
    print(
        f"PBO(tau=1) - PBO(tau=16) = {mean:+.4f} +/- {se:.4f}, t = {mean / se:.1f}, "
        f"{sum(d > 0 for d in diffs)}/16 grids"
    )
    assert mean / se > 3.0, (
        f"diluting selection stopped reducing PBO (t = {mean / se:.1f}). That is the one "
        "direction 01 Part E3 predicts and the measurement supported."
    )


def test_g9_pbo_is_not_monotone_in_temperature() -> None:
    """Recorded because it contradicts 01 Part E3, and the measurement wins.

    E3 says "expect it to decrease monotonically". It does not. Across 8 grids at S=12
    the mean curve runs 0.766 -> 0.810 -> 0.811 -> 0.565 -> 0.425 for
    tau = 0.05, 0.25, 1, 4, 16: a hump, not a slope. PBO(tau=1) exceeded PBO(tau=0.05)
    in 5 of 8 grids, which is a coin flip, so the low-temperature half of the
    prediction is unsupported rather than merely weak.

    The mechanism is that a mild softmax still concentrates on the top few in-sample
    performers, and on a compensation grid those are precisely the columns that
    reverse -- while blending them cuts the portfolio's volatility and lifts its
    in-sample Sharpe. Selectivity is not diluted until tau is large enough to reach
    well down the ranking.

    This test asserts only that the curve is not monotone decreasing, so that if a
    future change makes it monotone, someone has to come back and read this.
    """
    grid = compensation_grid(np.random.default_rng(7), n_blocks=12)
    taus = (0.05, 0.25, 1.0, 4.0, 16.0)
    curve = [cscv(grid, Softmax(t), n_blocks=12).pbo() for t in taus]
    print("PBO vs tau: " + "  ".join(f"{t}={p:.3f}" for t, p in zip(taus, curve, strict=True)))
    assert curve != sorted(curve, reverse=True), (
        "PBO came out monotone decreasing in tau on this grid. That is what 01 Part E3 "
        "predicted and what repeated measurement contradicted; if it now holds, the "
        "measurement above needs redoing rather than this test deleting."
    )
    assert max(curve) > curve[0], "the hump is the finding; it has gone"


def test_g9_equal_weight_pbo_is_a_property_of_the_grid_not_of_overfitting() -> None:
    """Why the tau -> infinity endpoint carries no information, reported not asserted.

    EqualWeight returns the same weights on every split, so all 252 splits are near
    perfectly dependent and the effective sample size is about one grid. Measured over
    12 grids: mean 0.400 with sd 0.296, spanning 0.048 to 0.837.

    A single EqualWeight PBO is therefore one draw from something close to uniform, and
    reading it as "the asymptote" -- in either direction -- would be reading noise. The
    figure plots it as a reference line with its spread shown, and nothing here asserts
    a bound on it.
    """
    values = pbo_over_grids(noise_grid, EQUAL_WEIGHT, n_grids=12)
    mean, se = summarise(values)
    spread = st.stdev(values)
    print(
        f"EqualWeight PBO = {mean:.3f} +/- {se:.3f}, sd {spread:.3f}, "
        f"range {min(values):.3f}..{max(values):.3f}"
    )
    assert all(0.0 <= v <= 1.0 for v in values)
    assert spread > 3.0 * PER_GRID_SD / math.sqrt(12), (
        f"EqualWeight's PBO spread has collapsed to {spread:.3f}. It was 0.321, and the "
        "reason it is excluded from every bound in this file is that it is that wide."
    )


@pytest.mark.parametrize("rule", [ArgMax(), TopK(3), Softmax(0.25), Softmax(4.0), EqualWeight()])
def test_g9_every_rule_produces_a_usable_pbo(rule: SelectionRule) -> None:
    """The gate is defined for the whole interface, not just argmax -- which is the
    point of having built `SelectionRule` before CSCV (01 Part E3, build-order note)."""
    result = cscv(compensation_grid(np.random.default_rng(5), n_blocks=BLOCKS), rule, BLOCKS)
    assert result.n_splits == n_splits_for(BLOCKS)
    assert np.all(np.isfinite(result.logits))
    assert 0.0 <= result.pbo() <= 1.0
    assert math.isfinite(result.performance_degradation())
    assert result.rule == rule.name
