"""G9 phase 1 -- CSCV groundwork. Specified by 01 Part B6, generalised by 01 Part E3.

Combinatorially Symmetric Cross-Validation answers a sharper question than "does this
strategy work": **when my selection rule picks a winner in sample, how often does that
winner land in the bottom half out of sample?** That is a property of the *procedure*,
not of any strategy, and publishing it about your own work is the credibility move.

    1. M is a (T, N) matrix of configuration return series
    2. split the rows into S contiguous blocks, S even
    3. for each of C(S, S/2) ways to choose half the blocks as in-sample:
           w   <- the selection rule's weights on the in-sample block
           r   <- relative rank of the rule's portfolio among the N out-of-sample Sharpes
           lam <- ln(r / (1 - r))
    4. PBO = fraction of splits with lam <= 0

`C(16, 8) = 12,870`, which is why the Sharpe computation is vectorised across all N
columns and the loop runs over splits only.

**Generalised from the start.** The textbook definition ranks the argmax column. This
ranks *the rule's portfolio*, which reduces to the textbook case exactly when the rule
is `ArgMax` -- asserted in the tests rather than argued here -- and makes `PBO(rule)`
meaningful for `Softmax`, `TopK` and `EqualWeight` too. `01` Part E3 is explicit that
retrofitting CSCV from a hardcoded argmax to a rule interface means rewriting the rank
bookkeeping, which is the fiddliest code in the project, so it is written against the
interface on the first pass.

**Why block sums.** Each split needs the mean and variance of every column over an
arbitrary half of the blocks. Recomputing from the raw rows costs `C(S,S/2) * T * N`
operations; accumulating per-block counts, sums and sums-of-squares once and adding
the eight chosen blocks costs `C(S,S/2) * S * N`, which is roughly sixty times less at
S = 16. The two agree to floating-point rounding and `test_cscv.py` asserts it against
a direct recomputation rather than assuming it.

How to read the result (`01` Part B6): PBO near 0.5 means the selection rule is a coin
flip. Above 0.5 is worse than a coin flip and is the empirical signature of the
compensation effect -- the in-sample winner is systematically the wrong pick. Below 0.2
means the procedure carries real information. It does *not* say the strategy works.

**And it only means that for a rule that selects.** Measured at S = 16 on a grid whose
true edges are spread symmetrically about zero, `PBO(EqualWeight)` comes out near 0.90.
That is not a claim that equal weighting is ninety per cent overfit. EqualWeight ignores
in-sample evidence entirely, so its blend carries the grid's average edge -- zero, by
construction, on a symmetric grid -- while diversification shrinks its volatility. The
blend therefore lands near the middle of the column distribution every time, and which
side of the median it falls on is decided by noise rather than by overfitting.

PBO answers "when my rule picks a winner, how often is that winner in the bottom half
out of sample". A rule that picks nothing has no winner, so the number it produces is a
statement about where a diversified portfolio sits among its own constituents, which is
a different question wearing the same name. Report `PBO(EqualWeight)` as the geometric
baseline the selecting rules are read against -- never as its overfitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from falsify.core.types import BARS_PER_YEAR
from falsify.selection import SelectionRule

Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]

DEFAULT_BLOCKS = 16


class InsufficientBlocks(ValueError):
    """Too few observations to cut the requested number of blocks."""


@dataclass(frozen=True, slots=True)
class BlockMoments:
    """Per-block count, sum and cross-product for every column. Frozen (B7).

    The whole point of CSCV's cost profile: computed once, then any subset of blocks
    is a row-sum away from its own mean and variance.

    **Why the full cross-product and not just a sum of squares.** The rank compares
    the rule's *portfolio* against the N *columns*, and if those two Sharpes come from
    different arithmetic they disagree in the last bits. For `ArgMax` the portfolio IS
    one of the columns, so a strict `<` then counts the chosen column as beaten by
    itself roughly half the time and the rank moves by one -- a silent off-by-one in
    the exact place `03` Part C warns about, throwing no exception and shifting PBO.

    Storing `X'X` per block fixes it at the root: a portfolio's sum of squares is the
    quadratic form `w' C w`, and for a one-hot `w` that is literally `C[j, j]`, the
    same float the column's own variance uses. The reduction is then exact rather
    than approximate, which `test_cscv.py` asserts.

    Cost is `S x N^2` floats -- negligible at the N this runs at now, and a phase-2
    concern at the full grid, where an `ArgMax` fast path can skip the quadratic form
    entirely since the portfolio is a column.
    """

    count: NDArray[np.int64]  # (S,)
    total: Matrix  # (S, N)
    cross: NDArray[np.float64]  # (S, N, N) -- X'X per block

    @property
    def n_blocks(self) -> int:
        return int(self.count.size)

    @property
    def n_configs(self) -> int:
        return int(self.total.shape[1])

    def _pooled(self, blocks: tuple[int, ...]) -> tuple[int, Vector, NDArray[np.float64]]:
        idx = list(blocks)
        return (
            int(self.count[idx].sum()),
            self.total[idx].sum(axis=0),
            self.cross[idx].sum(axis=0),
        )

    def sharpe_over(self, blocks: tuple[int, ...], bars_per_year: int) -> Vector:
        """Annualised Sharpe of every column over the union of `blocks`.

        Variance from `E[x^2] - E[x]^2` in the `ddof=1` form. Safe at these
        magnitudes -- returns of order 1e-3 against a mean square of order 1e-4 leave
        no catastrophic cancellation -- and checked against a direct two-pass
        computation in the tests rather than assumed safe.
        """
        n, s1, cross = self._pooled(blocks)
        if n < 2:
            return np.full(self.n_configs, np.nan)
        mean = s1 / n
        var = (np.diag(cross) - n * mean * mean) / (n - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_bar = np.where(var > 0.0, mean / np.sqrt(var), np.nan)
        return np.asarray(per_bar * math.sqrt(bars_per_year), dtype=np.float64)

    def portfolio_sharpe(
        self, blocks: tuple[int, ...], weights: Vector, bars_per_year: int
    ) -> float:
        """Annualised Sharpe of `returns @ weights` over the union of `blocks`.

        Derived from the same pooled moments as `sharpe_over`, which is the whole
        point: the portfolio and the columns are then commensurable and the rank
        comparison is stable.
        """
        n, s1, cross = self._pooled(blocks)
        if n < 2:
            return float("nan")
        mean = float(s1 @ weights) / n
        sum_sq = float(weights @ cross @ weights)
        var = (sum_sq - n * mean * mean) / (n - 1)
        if not math.isfinite(var) or var <= 0.0:
            return float("nan")
        return mean / math.sqrt(var) * math.sqrt(bars_per_year)


def block_partition(n_obs: int, n_blocks: int) -> list[NDArray[np.int64]]:
    """Cut `range(n_obs)` into `n_blocks` contiguous, near-equal blocks.

    Contiguous by requirement, not convenience: shuffling rows would destroy the
    serial correlation that makes a financial time series what it is, and CSCV's
    symmetry argument rests on the blocks being exchangeable *as blocks*.
    """
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be even and at least 2, got {n_blocks}")
    if n_obs < n_blocks * 2:
        raise InsufficientBlocks(
            f"{n_obs} observations cannot be cut into {n_blocks} blocks of at least 2"
        )
    bounds = np.linspace(0, n_obs, n_blocks + 1).astype(int)
    return [np.arange(bounds[k], bounds[k + 1], dtype=np.int64) for k in range(n_blocks)]


def block_moments(returns: Matrix, n_blocks: int = DEFAULT_BLOCKS) -> BlockMoments:
    """Accumulate the per-block moments CSCV reuses across every split."""
    if returns.ndim != 2:
        raise ValueError(f"expected a (T, N) matrix, got shape {returns.shape}")
    if not np.all(np.isfinite(returns)):
        raise ValueError("returns contain non-finite values; trim the warm-up first")

    blocks = block_partition(returns.shape[0], n_blocks)
    count = np.array([b.size for b in blocks], dtype=np.int64)
    total = np.stack([returns[b].sum(axis=0) for b in blocks])
    cross = np.stack([returns[b].T @ returns[b] for b in blocks])
    return BlockMoments(count=count, total=total, cross=cross)


@dataclass(frozen=True, slots=True)
class CSCVResult:
    """The outcome of one CSCV run. Frozen (B7)."""

    rule: str
    n_blocks: int
    n_configs: int
    logits: Vector  # (n_splits,)
    relative_ranks: Vector  # (n_splits,)
    is_sharpe: Vector  # (n_splits,) the rule's portfolio, in sample
    oos_sharpe: Vector  # (n_splits,) the rule's portfolio, out of sample

    @property
    def n_splits(self) -> int:
        return int(self.logits.size)

    def pbo(self) -> float:
        """Probability of Backtest Overfitting: the fraction of splits whose
        in-sample choice landed in the bottom half out of sample."""
        finite = self.logits[np.isfinite(self.logits)]
        if finite.size == 0:
            return float("nan")
        return float(np.mean(finite <= 0.0))

    def median_logit(self) -> float:
        finite = self.logits[np.isfinite(self.logits)]
        return float(np.median(finite)) if finite.size else float("nan")

    def performance_degradation(self) -> float:
        """Mean in-sample minus mean out-of-sample Sharpe of the rule's portfolio."""
        return float(np.nanmean(self.is_sharpe) - np.nanmean(self.oos_sharpe))

    def describe(self) -> str:
        return (
            f"{self.rule}: PBO={self.pbo():.4f}  median lambda={self.median_logit():+.4f}  "
            f"degradation={self.performance_degradation():+.4f}  "
            f"splits={self.n_splits:,}  N={self.n_configs}"
        )


def relative_rank(portfolio_oos: float, column_oos: Vector) -> float:
    """Relative rank of the rule's portfolio among the N column Sharpes.

    `omega` counts how many columns the portfolio beats, plus one, so it runs 1..N and
    `r = omega / (N + 1)` stays strictly inside (0, 1) -- which matters because the
    logit of 0 or 1 is infinite and would silently drop splits from the average.

    For `ArgMax` the portfolio *is* one of the columns, so this reduces to the textbook
    rank of the selected column. `test_cscv.py` asserts that equivalence rather than
    relying on the reader to see it.
    """
    finite = column_oos[np.isfinite(column_oos)]
    n = finite.size
    if n == 0 or not math.isfinite(portfolio_oos):
        return float("nan")
    beaten = int(np.sum(finite < portfolio_oos))
    omega = min(max(beaten + 1, 1), n)
    return omega / (n + 1)


def cscv(
    returns: Matrix,
    rule: SelectionRule,
    n_blocks: int = DEFAULT_BLOCKS,
    bars_per_year: int = BARS_PER_YEAR,
) -> CSCVResult:
    """Run CSCV over every symmetric half-split of the block partition."""
    moments = block_moments(returns, n_blocks)
    all_blocks = tuple(range(n_blocks))
    half = n_blocks // 2
    blocks = block_partition(returns.shape[0], n_blocks)

    logits, ranks, is_sr, oos_sr = [], [], [], []

    for chosen in combinations(all_blocks, half):
        complement = tuple(b for b in all_blocks if b not in chosen)

        # The rule sees only the in-sample rows. Concatenating the block indices keeps
        # the rows in chronological order within the half, which matters for any rule
        # that is not purely cross-sectional.
        train_rows = np.concatenate([blocks[b] for b in chosen])
        weights = rule.weights(returns[train_rows])

        # Both the portfolio and the columns come from `moments`, so the rank below
        # compares like with like. Computing the portfolio from raw rows instead
        # leaves the two disagreeing in the last bits, which moves the ArgMax rank by
        # one about half the time.
        is_sr.append(moments.portfolio_sharpe(chosen, weights, bars_per_year))
        portfolio_oos_sr = moments.portfolio_sharpe(complement, weights, bars_per_year)
        oos_sr.append(portfolio_oos_sr)

        r = relative_rank(portfolio_oos_sr, moments.sharpe_over(complement, bars_per_year))
        ranks.append(r)
        logits.append(math.log(r / (1.0 - r)) if math.isfinite(r) else float("nan"))

    return CSCVResult(
        rule=rule.name,
        n_blocks=n_blocks,
        n_configs=int(returns.shape[1]),
        logits=np.asarray(logits, dtype=np.float64),
        relative_ranks=np.asarray(ranks, dtype=np.float64),
        is_sharpe=np.asarray(is_sr, dtype=np.float64),
        oos_sharpe=np.asarray(oos_sr, dtype=np.float64),
    )


def n_splits_for(n_blocks: int) -> int:
    """C(S, S/2) -- the number of symmetric half-splits. 12,870 at S = 16."""
    return math.comb(n_blocks, n_blocks // 2)


__all__ = [
    "DEFAULT_BLOCKS",
    "BlockMoments",
    "CSCVResult",
    "InsufficientBlocks",
    "block_moments",
    "block_partition",
    "cscv",
    "n_splits_for",
    "relative_rank",
]
