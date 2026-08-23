"""Stationary bootstrap. Specified by 01 Part B5, parameterised by 03 Part H decision 3.

Politis-Romano (1994). Resample geometric-length blocks with mean `1/p`, wrapping at
the end, which preserves the autocorrelation an i.i.d. bootstrap destroys.

Why this and not the i.i.d. bootstrap: strategy returns are serially correlated, and
01 Part B1 is explicit that this inflates the true standard error above both the Lo
(2002) expression and its non-normal correction. Those formulas are analytic and assume
independence; this makes no distributional assumption at all and lets the block length
carry the dependence. Where the two disagree, the bootstrap is the one to believe.

Two implementations live here on purpose, in the spirit of the twin engines at G2. The
reference is the loop as written in 01 Part B5, kept because it is obviously correct and
is what a reader will check against. The fast path removes the inner Python loop over T
and `test_bootstrap.py` asserts the two are bitwise identical, so the optimisation cannot
drift away from the specification without a test failing.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Series = NDArray[np.float64]
Statistic = Callable[[Series], float]

# 03 Part H decision 3. Mean block length sqrt(T); revisit if CI width moves >20%
# across the sensitivity range below.
DEFAULT_N_BOOT = 1_000


def default_p(n_obs: int) -> float:
    """`p = 1/sqrt(T)`, so the mean block length is `sqrt(T)`."""
    if n_obs < 1:
        raise ValueError(f"need at least one observation, got {n_obs}")
    return 1.0 / math.sqrt(n_obs)


def sensitivity_grid(n_obs: int) -> tuple[float, float, float]:
    """`{T^(-1/3), T^(-1/2), T^(-2/3)}` -- the range 01 B5 requires be reported.

    Short blocks at one end, long at the other. If the interval width moves more than
    20% across it, the autocorrelation structure is doing real work and that has to be
    said rather than resolved by picking the flattering end.
    """
    if n_obs < 1:
        raise ValueError(f"need at least one observation, got {n_obs}")
    t = float(n_obs)
    return (t ** (-1.0 / 3.0), t ** (-1.0 / 2.0), t ** (-2.0 / 3.0))


def _validate(x: Series, p: float, n_boot: int) -> int:
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {x.shape}")
    t = int(x.size)
    if t < 2:
        raise ValueError(f"need at least 2 observations, got {t}")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains NaN or infinity; trim the warm-up first")
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must lie in (0, 1], got {p}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    return t


def _reference(x: Series, p: float, n_boot: int, rng: np.random.Generator) -> Series:
    """01 Part B5 verbatim. Slow, and kept exactly because it is the specification."""
    t = _validate(x, p, n_boot)
    out = np.empty((n_boot, t), dtype=np.float64)
    for b in range(n_boot):
        idx = np.empty(t, dtype=np.int64)
        idx[0] = rng.integers(t)
        new_block = rng.random(t) < p
        jumps = rng.integers(0, t, size=t)
        for i in range(1, t):
            idx[i] = jumps[i] if new_block[i] else (idx[i - 1] + 1) % t
        out[b] = x[idx]
    return out


def stationary_bootstrap(x: Series, p: float, n_boot: int, rng: np.random.Generator) -> Series:
    """`(n_boot, T)` resampled paths preserving serial dependence.

    Identical output to `_reference` for the same seed, bitwise. The draw order per
    replicate is deliberately unchanged -- one `integers` for the seed index, one
    `random(T)` for the block starts, one `integers(T)` for the jump targets -- because
    reordering the RNG calls would consume the stream differently and silently produce a
    different, equally valid, non-comparable sample. Only the sequential scan over `T` is
    replaced, and that is where the cost was.

    The scan is a forward-fill: every observation belongs to the most recent block start,
    and its value is that block's starting index plus how far into the block it sits.
    `np.maximum.accumulate` over the start positions computes exactly that.
    """
    t = _validate(x, p, n_boot)
    out = np.empty((n_boot, t), dtype=np.float64)
    positions = np.arange(t, dtype=np.int64)

    for b in range(n_boot):
        first = int(rng.integers(t))
        new_block = rng.random(t) < p
        jumps = rng.integers(0, t, size=t)

        is_start = new_block.copy()
        is_start[0] = True  # position 0 always starts a block, seeded by `first`
        start_value = np.where(positions == 0, first, jumps)

        # Index of the most recent block start at or before each position.
        start_pos = np.maximum.accumulate(np.where(is_start, positions, 0))
        idx = (start_value[start_pos] + (positions - start_pos)) % t
        out[b] = x[idx]
    return out


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """A statistic with a bootstrap interval. Frozen (B7).

    Carries `p` and `n_boot` because an interval without the resampling parameters that
    produced it is not reproducible, and B2 is about numbers being checkable rather than
    merely decorated.
    """

    point: float
    lo: float
    hi: float
    alpha: float
    p: float
    n_boot: int
    mean_block_length: float

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def describe(self) -> str:
        return (
            f"{self.point:+.4f}  [{self.lo:+.4f}, {self.hi:+.4f}] "
            f"{100 * (1 - self.alpha):.0f}%  (p={self.p:.4f}, "
            f"block~{self.mean_block_length:.1f}, B={self.n_boot:,})"
        )


def bootstrap_ci(
    x: Series,
    statistic: Statistic,
    rng: np.random.Generator,
    *,
    p: float | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
) -> BootstrapCI:
    """Percentile interval for `statistic` under the stationary bootstrap.

    `p` defaults to 03 Part H decision 3. Percentile rather than BCa: BCa's acceleration
    term needs a jackknife over a serially dependent series, where the leave-one-out
    resample is not the right object, and pretending otherwise would buy a
    sophisticated-looking interval with a worse justification.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    resolved_p = default_p(int(x.size)) if p is None else p
    paths = stationary_bootstrap(x, resolved_p, n_boot, rng)

    values = np.array([statistic(path) for path in paths], dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        raise ValueError(
            f"only {finite.size} of {n_boot} bootstrap replicates gave a finite "
            "statistic; the interval would be meaningless"
        )
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(
        point=float(statistic(x)),
        lo=float(lo),
        hi=float(hi),
        alpha=alpha,
        p=resolved_p,
        n_boot=n_boot,
        mean_block_length=1.0 / resolved_p,
    )


def p_sensitivity(
    x: Series,
    statistic: Statistic,
    rng: np.random.Generator,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
) -> tuple[BootstrapCI, ...]:
    """The three intervals 01 B5 requires be reported, short blocks to long."""
    return tuple(
        bootstrap_ci(x, statistic, rng, p=p, n_boot=n_boot, alpha=alpha)
        for p in sensitivity_grid(int(x.size))
    )


def width_dispersion(intervals: tuple[BootstrapCI, ...]) -> float:
    """Relative spread of interval width across the sensitivity grid.

    01 B5 sets 20% as the level above which the block length is doing real work and the
    dependence has to be discussed rather than parameterised away. Returned as a fraction
    so a caller can compare against 0.20 without re-deriving the convention.
    """
    widths = [i.width for i in intervals]
    if not widths or min(widths) <= 0.0:
        return float("nan")
    return (max(widths) - min(widths)) / min(widths)


__all__ = [
    "DEFAULT_N_BOOT",
    "BootstrapCI",
    "bootstrap_ci",
    "default_p",
    "p_sensitivity",
    "sensitivity_grid",
    "stationary_bootstrap",
    "width_dispersion",
]
