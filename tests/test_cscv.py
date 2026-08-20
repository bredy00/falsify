"""G9 phase 1 -- the CSCV groundwork, tested before the gate is built on it.

`03` Part C budgets two full sessions for G9 and says to write the test first, because
CSCV has more moving parts than anything else in the build -- block partitioning,
combination enumeration, per-split argmax across N columns, rank computation, logit
transform -- and *none of them throws an exception when it is off by one*.

So this file covers the bookkeeping in isolation, at sizes small enough to check by
hand, before phase 2 puts a gate on top of it. The three that would silently corrupt
every downstream number:

  - block sums must equal a direct recomputation (the optimisation must be exact)
  - the generalised rank must reduce to the textbook rank for ArgMax
  - the logit must stay finite, or splits vanish from the average without a trace
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest
from numpy.typing import NDArray

from falsify.core.types import BARS_PER_YEAR
from falsify.cscv import (
    InsufficientBlocks,
    block_moments,
    block_partition,
    cscv,
    n_splits_for,
    relative_rank,
)
from falsify.selection import ArgMax, EqualWeight, SelectionRule, Softmax, TopK

Matrix = NDArray[np.float64]

SEED = 90_901
RULES: tuple[SelectionRule, ...] = (ArgMax(), TopK(3), Softmax(1.0), EqualWeight())


@pytest.fixture(scope="module")
def grid() -> Matrix:
    """A small grid with genuinely different columns, so ranking is not a tie."""
    rng = np.random.default_rng(SEED)
    n_obs, n_cfg = 480, 8
    base = rng.normal(0.0, 0.01, (n_obs, n_cfg))
    # A spread of true edges, so the columns are not exchangeable.
    return np.asarray(base + np.linspace(-0.0006, 0.0006, n_cfg), dtype=np.float64)


# ------------------------------------------------------------- block partition


def test_blocks_are_contiguous_and_tile_the_period() -> None:
    """Contiguous by requirement: shuffling rows would destroy the serial correlation
    that makes a price series what it is."""
    blocks = block_partition(100, 16)
    assert len(blocks) == 16
    joined = np.concatenate(blocks)
    assert np.array_equal(joined, np.arange(100)), "blocks must tile without gaps"
    for b in blocks:
        assert np.array_equal(b, np.arange(b[0], b[-1] + 1)), "a block is not contiguous"


@pytest.mark.parametrize(("n_blocks", "match"), [(3, "even"), (0, "even"), (17, "even")])
def test_odd_block_counts_are_rejected(n_blocks: int, match: str) -> None:
    """CSCV's symmetry needs the blocks to split into equal halves."""
    with pytest.raises(ValueError, match=match):
        block_partition(200, n_blocks)


def test_too_few_observations_raises() -> None:
    with pytest.raises(InsufficientBlocks):
        block_partition(20, 16)


def test_split_count_is_the_binomial_coefficient() -> None:
    """C(16, 8) = 12,870 -- the number `01` Part B6 quotes."""
    assert n_splits_for(16) == 12_870
    assert n_splits_for(8) == 70
    assert n_splits_for(4) == 6


# ------------------------------------------------- the block-sum optimisation


def test_block_sums_reproduce_a_direct_recomputation(grid: Matrix) -> None:
    """The optimisation must be exact, not merely fast.

    Accumulating per-block sums and adding the chosen halves costs about sixty times
    less than recomputing from raw rows at S = 16. It is only worth anything if it
    agrees, so every one of the 70 half-splits at S = 8 is checked against a two-pass
    mean and variance over the same rows.
    """
    n_blocks = 8
    moments = block_moments(grid, n_blocks)
    blocks = block_partition(grid.shape[0], n_blocks)
    worst = 0.0

    for chosen in combinations(range(n_blocks), n_blocks // 2):
        rows = np.concatenate([blocks[b] for b in chosen])
        block = grid[rows]
        direct = (
            block.mean(axis=0) / block.std(axis=0, ddof=1) * math.sqrt(BARS_PER_YEAR)
        )
        fast = moments.sharpe_over(chosen, BARS_PER_YEAR)
        worst = max(worst, float(np.max(np.abs(direct - fast))))

    print(f"worst |block-sum Sharpe - direct Sharpe| over 70 splits: {worst:.3e}")
    assert worst < 1e-9, (
        f"the block-sum shortcut disagrees with a direct recomputation by {worst:.3e}; "
        "at that size it is an accounting error, not rounding"
    )


def test_block_moments_reject_non_finite_input(grid: Matrix) -> None:
    bad = grid.copy()
    bad[5, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        block_moments(bad, 8)


# ----------------------------------------------------------- rank bookkeeping


def test_relative_rank_stays_strictly_inside_zero_and_one() -> None:
    """The logit of 0 or 1 is infinite, and an infinite logit drops that split from
    the average with no trace in the output. `r = omega/(N+1)` with omega in 1..N is
    what keeps every split countable."""
    columns = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    for portfolio in (-99.0, 0.05, 0.25, 0.55, 99.0):
        r = relative_rank(portfolio, columns)
        assert 0.0 < r < 1.0, f"portfolio {portfolio} gave r = {r}"
        assert math.isfinite(math.log(r / (1.0 - r)))


def test_relative_rank_orders_correctly() -> None:
    columns = np.asarray([0.1, 0.2, 0.3, 0.4])
    assert relative_rank(0.05, columns) < relative_rank(0.25, columns)
    assert relative_rank(0.25, columns) < relative_rank(0.95, columns)


def test_relative_rank_matches_the_textbook_rank_for_argmax(grid: Matrix) -> None:
    """The generalisation must reduce to the standard definition.

    Standard CSCV ranks the argmax column among the N out-of-sample Sharpes. This
    ranks the rule's *portfolio*, which for ArgMax is that same column -- so the two
    must agree exactly, or `PBO(ArgMax)` is not the number the literature reports.
    """
    n_blocks = 8
    blocks = block_partition(grid.shape[0], n_blocks)
    moments = block_moments(grid, n_blocks)
    rule = ArgMax()

    for chosen in combinations(range(n_blocks), n_blocks // 2):
        complement = tuple(b for b in range(n_blocks) if b not in chosen)
        train = np.concatenate([blocks[b] for b in chosen])

        weights = rule.weights(grid[train])
        picked = int(np.argmax(weights))

        oos_columns = moments.sharpe_over(complement, BARS_PER_YEAR)
        # Via the same pooled moments the implementation uses. Recomputing the
        # portfolio from raw rows here would reintroduce exactly the inconsistency
        # this test exists to rule out: the two Sharpes would then differ in the last
        # bits and the strict `<` would count the chosen column as beaten by itself
        # about half the time. Whether the block sums match a direct recomputation is
        # a separate question, covered by test_block_sums_reproduce_a_direct_recomputation.
        portfolio_oos = moments.portfolio_sharpe(complement, weights, BARS_PER_YEAR)

        textbook = 1 + int(np.sum(oos_columns < oos_columns[picked]))
        generalised_r = relative_rank(portfolio_oos, oos_columns)
        assert abs(generalised_r - textbook / (len(oos_columns) + 1)) < 1e-12, (
            "the generalised rank does not reduce to the textbook rank for ArgMax"
        )


# --------------------------------------------------------------- end to end


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.name)
def test_cscv_runs_and_reports_a_valid_probability(rule: SelectionRule, grid: Matrix) -> None:
    """Every rule must produce a finite PBO over every split, at a size that runs fast."""
    result = cscv(grid, rule, n_blocks=8)
    print(result.describe())
    assert result.n_splits == n_splits_for(8)
    assert 0.0 <= result.pbo() <= 1.0
    assert np.all(np.isfinite(result.logits)), "an infinite logit silently drops a split"
    assert np.all((result.relative_ranks > 0.0) & (result.relative_ranks < 1.0))
    assert result.n_configs == grid.shape[1]


def test_cscv_is_deterministic(grid: Matrix) -> None:
    """No RNG anywhere in the path (B9), so two runs must agree bitwise."""
    a = cscv(grid, Softmax(1.0), n_blocks=8)
    b = cscv(grid, Softmax(1.0), n_blocks=8)
    assert np.array_equal(a.logits, b.logits)
    assert a.pbo() == b.pbo()


def test_equal_weight_has_no_selection_to_overfit(grid: Matrix) -> None:
    """EqualWeight ignores in-sample evidence entirely, so its PBO is a baseline.

    It cannot overfit the selection because it does not select -- whatever PBO it
    reports is the geometry of the split rather than the cost of choosing, and it is
    the number the other rules have to be read against.
    """
    result = cscv(grid, EqualWeight(), n_blocks=8)
    print(f"EqualWeight baseline: {result.describe()}")
    assert 0.0 <= result.pbo() <= 1.0
    # Zero degradation is the signature of a rule that does not select: it makes the
    # same choice in and out of sample, so there is nothing for the split to reveal.
    assert abs(result.performance_degradation()) < 1e-9, (
        "EqualWeight ignores in-sample evidence, so its in-sample and out-of-sample "
        "portfolios are the same object and cannot degrade"
    )
    # At S = 16 on a symmetric grid this number reaches ~0.90, and it must not be read
    # as overfitting -- see the note in falsify/cscv.py. It is where a diversified
    # blend sits among its own constituents, which is a different question.
    assert np.all(np.isfinite(result.logits))


def test_pbo_reacts_to_a_grid_with_no_true_differences() -> None:
    """On exchangeable columns the in-sample winner is chosen by noise alone, so the
    rank should be close to uniform and PBO close to a coin flip.

    Reported rather than tightly bounded: with 70 splits the standard error on a
    proportion near 0.5 is about 0.06, so a narrow assertion here would be an
    assertion on noise -- the same mistake the G6 and G8 bounds had to be rescued
    from. Phase 2 runs the full C(16,8) = 12,870 and can say more.
    """
    rng = np.random.default_rng(SEED + 1)
    exchangeable = rng.normal(0.0, 0.01, (480, 8))
    result = cscv(exchangeable, ArgMax(), n_blocks=8)
    print(f"exchangeable columns: {result.describe()}  (SE on PBO ~ {0.5 / math.sqrt(70):.3f})")
    assert 0.1 < result.pbo() < 0.9, (
        f"PBO = {result.pbo():.3f} on exchangeable columns. Exactly 0.0 or 1.0 would mean "
        "the rank computation is broken or every column is identical (failure mode F3)."
    )


def test_pbo_is_not_degenerate_at_zero_or_one(grid: Matrix) -> None:
    """Failure mode F3 in `03`: 'PBO comes out at exactly 0.0 or 1.0. The rank
    computation is broken, or every column is identical.'"""
    for rule in RULES:
        pbo = cscv(grid, rule, n_blocks=8).pbo()
        assert pbo not in (0.0, 1.0), f"{rule.name} produced a degenerate PBO of {pbo}"


def test_cscv_rejects_a_grid_that_cannot_be_split() -> None:
    with pytest.raises(InsufficientBlocks):
        cscv(np.random.default_rng(1).normal(size=(20, 4)), ArgMax(), n_blocks=16)
