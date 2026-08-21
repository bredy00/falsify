"""Selection rules. Specified by 02-ENGINE-SPEC.md Part H and 01 Part E2/E3.

Built before G9, deliberately. CSCV's rank bookkeeping is the fiddliest code in
the project, and hardcoding it to argmax means rewriting that bookkeeping later
(01 Part E3, build-order note). So the interface exists first and G9 will be
written against it.

Why the interface matters at all: argmax is not a neutral choice. Under the null
it hands you E[max of N draws] ~ sqrt(2 ln N), which is the entire overfitting
problem. Under a compensation effect it is worse than neutral -- Propositions 3
and 5 say the in-sample winner is systematically the out-of-sample loser, so
argmax reliably selects the configuration most likely to reverse. A temperature
dial between argmax and the grid mean makes that cost measurable instead of
assumed, and `tau` is a shrinkage parameter in the same family as James-Stein,
ridge, and Bayesian model averaging.

`tau` is itself a trial (01 Part E2). Every temperature evaluated earns a ledger
row; sweeping temperature and keeping the best is the original sin one level up.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

Returns = NDArray[np.float64]
Weights = NDArray[np.float64]

# The contract's tolerance on sum(w) == 1, per 02 Part H.
SUM_TOL = 1e-12


class DegenerateTrial(ValueError):
    """A configuration's in-sample returns admit no finite Sharpe.

    Raised rather than patched. A column of constant returns has an undefined
    Sharpe, not a zero one, and Gate 0.4 is explicit that those are different
    claims and only one of them is true. Silently scoring it zero would let a
    dead configuration take a weight.
    """


def noise_floor(magnitude: float, count: int) -> float:
    """The spread below which a dispersion is rounding dust, not information.

    Testing `sd == 0.0` is not enough and the failure is silent. A column holding
    the same value 60 times has `std(ddof=1) ~ 2e-19` rather than exactly zero,
    because `sum/n` does not round-trip to the original float -- so an exact
    comparison passes it through and the Sharpe comes out around 1e16. That is a
    finite, enormous, meaningless number, and under ArgMax or Softmax it takes
    essentially the whole portfolio. Gate 0.4 exists to stop precisely this.

    Scales with both the data magnitude and the observation count, since
    summation error accumulates with n.
    """
    return float(np.finfo(np.float64).eps * max(magnitude, 0.0) * max(count, 1) * 4.0)


def in_sample_sharpe(is_returns: Returns) -> NDArray[np.float64]:
    """Per-observation Sharpe of each column. Unit: per bar (B8).

    No annualisation anywhere in this module: every rule is scale-invariant in
    the Sharpe, so annualising would change nothing but the chance of a units
    bug.
    """
    if is_returns.ndim != 2:
        raise ValueError(f"expected a (T_is, N) matrix, got shape {is_returns.shape}")
    if is_returns.shape[0] < 2:
        raise ValueError(f"need at least 2 in-sample observations, got {is_returns.shape[0]}")
    if not np.all(np.isfinite(is_returns)):
        raise DegenerateTrial("in-sample returns contain NaN or infinity")

    t_is = is_returns.shape[0]
    mu = is_returns.mean(axis=0)
    sd = is_returns.std(axis=0, ddof=1)

    # Per-column floor: a column's own magnitude sets its dust level. Computed for
    # every column in one pass rather than a comprehension over `is_returns.T` --
    # identical values, but the loop version issued one `np.max` call per column and
    # profiling put it at 90% of CSCV's runtime, 154,440 calls for a single
    # C(16,8) sweep.
    scale = np.max(np.abs(is_returns), axis=0)
    floors = np.finfo(np.float64).eps * np.maximum(scale, 0.0) * max(t_is, 1) * 4.0
    degenerate = ~np.isfinite(sd) | (sd <= floors)
    if np.any(degenerate):
        bad = np.flatnonzero(degenerate)
        raise DegenerateTrial(
            f"columns {bad.tolist()} have zero or non-finite in-sample volatility "
            f"(sd={sd[bad].tolist()}, floor={floors[bad].tolist()})"
        )

    sharpe = mu / sd
    if not np.all(np.isfinite(sharpe)):
        bad = np.flatnonzero(~np.isfinite(sharpe))
        raise DegenerateTrial(f"columns {bad.tolist()} have non-finite in-sample Sharpe")
    return np.asarray(sharpe, dtype=np.float64)


class SelectionRule(ABC):
    """Turns in-sample evidence into a portfolio over configurations.

    CONTRACT, enforced by tests rather than by convention:
      - returns a non-negative vector summing to 1.0 within SUM_TOL
      - depends ONLY on the in-sample block passed in
      - deterministic: same input, same output, no RNG (B9)

    One documented limit. Where two configurations tie on in-sample Sharpe, the
    discrete rules (`ArgMax`, `TopK`) still return exactly one selection, and
    which one is decided by floating-point rounding rather than by anything
    meaningful. It is *not* the lowest column index: the tie is exact only in
    exact arithmetic, so `argmax` compares values that differ in their last bits
    and may pick any member of the tied set. The choice is stable for a given
    input but changes under rescaling or column permutation.

    The continuous rules have no such ambiguity -- `Softmax` splits tied columns
    evenly. G9 must not read significance into which member of a tied set argmax
    happened to pick.
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def weights(self, is_returns: Returns) -> Weights:
        """(T_is, N) in-sample returns -> (N,) weights."""

    def __repr__(self) -> str:
        return f"{self.name}()"


class ArgMax(SelectionRule):
    """All weight on the best in-sample Sharpe. The baseline everyone uses, and
    the maximum-selection-bias end of the dial."""

    def weights(self, is_returns: Returns) -> Weights:
        sharpe = in_sample_sharpe(is_returns)
        out = np.zeros(sharpe.size)
        out[int(np.argmax(sharpe))] = 1.0
        return out


class EqualWeight(SelectionRule):
    """1/N regardless of evidence. The tau -> infinity asymptote, and the only
    rule here that incurs no selection bias at all."""

    def weights(self, is_returns: Returns) -> Weights:
        n = in_sample_sharpe(is_returns).size
        return np.full(n, 1.0 / n)


class TopK(SelectionRule):
    """1/k across the best k configurations. A discrete middle ground."""

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = k

    @property
    def name(self) -> str:
        return f"TopK(k={self.k})"

    def weights(self, is_returns: Returns) -> Weights:
        sharpe = in_sample_sharpe(is_returns)
        n = sharpe.size
        if self.k > n:
            raise ValueError(f"k={self.k} exceeds the number of configurations {n}")
        # argpartition would be faster but ties break by position rather than
        # deterministically across numpy versions; a full argsort is stable and
        # N here is hundreds, not millions.
        chosen = np.argsort(-sharpe, kind="stable")[: self.k]
        out = np.zeros(n)
        out[chosen] = 1.0 / self.k
        return out

    def __repr__(self) -> str:
        return f"TopK(k={self.k})"


class Softmax(SelectionRule):
    """Exponential weights on the cross-sectional z-scores of in-sample Sharpe.

    Standardising first is not cosmetic. Softmax on raw Sharpe values is
    scale-dependent and `tau` becomes uninterpretable; on z-scores, `tau = 1`
    means exactly one cross-sectional standard deviation.

    `tau -> 0` recovers ArgMax, `tau -> infinity` recovers EqualWeight, so the
    temperature is a continuous dial between maximum selection bias and none.

    The connection worth knowing: exponential weighting over N experts is the
    Hedge algorithm, whose regret against the best expert in hindsight is
    O(sqrt(T ln N)) -- the same ln N that appears in sqrt(2 ln N) from extreme
    value theory. Two unrelated fields, one measuring the danger of selection and
    the other the price of learning, land on the same logarithm.
    """

    def __init__(self, temperature: float) -> None:
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"temperature must be finite and positive, got {temperature}")
        self.temperature = float(temperature)

    @property
    def name(self) -> str:
        return f"Softmax(tau={self.temperature:g})"

    def weights(self, is_returns: Returns) -> Weights:
        sharpe = in_sample_sharpe(is_returns)
        n = sharpe.size
        if n == 1:
            return np.ones(1)

        # The floor uses T_is, not n. Each Sharpe carries rounding error that
        # accumulated over its own T_is observations, so that -- not the number
        # of columns -- sets how far apart two mathematically equal Sharpes can
        # land. Using n here left the floor ~40x too tight at T_is=80 and the
        # dust survived.
        spread = float(sharpe.std(ddof=1))
        if spread <= noise_floor(float(np.max(np.abs(sharpe))), is_returns.shape[0]):
            # No evidence to exponentiate. Equal weight is the honest answer.
            #
            # The floor, rather than `spread == 0.0`, is load-bearing. z-scoring
            # divides by this spread, so when it sits at the rounding floor the
            # z-scores blow up to +/-1 and softmax collapses into an arbitrary
            # argmax chosen by floating-point dust: two columns whose true
            # Sharpes are identical (one being a rescaling of the other) differ
            # in the last bits, and at tau=0.25 that 1e-17 gap becomes a
            # 0.9997/0.0003 allocation. CSCV in-sample blocks produce
            # near-identical Sharpes routinely, so without this the temperature
            # sweep at G9 would be noise wherever the grid agrees.
            return np.full(n, 1.0 / n)

        z = (sharpe - sharpe.mean()) / spread
        scaled = z / self.temperature
        # Subtract the max before exponentiating or tau -> 0 overflows.
        exp_z = np.exp(scaled - scaled.max())
        total = exp_z.sum()
        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError(f"softmax underflowed at tau={self.temperature}")
        return np.asarray(exp_z / total, dtype=np.float64)

    def __repr__(self) -> str:
        return f"Softmax(temperature={self.temperature!r})"


__all__ = [
    "SUM_TOL",
    "ArgMax",
    "DegenerateTrial",
    "EqualWeight",
    "SelectionRule",
    "Softmax",
    "TopK",
    "Weights",
    "in_sample_sharpe",
]
