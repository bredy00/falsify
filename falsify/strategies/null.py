"""The calibrated null. Supports G6.

A thousand coin flips through the whole pipeline, whose Sharpe distribution is the
empirical null a real result has to beat. 03 Part C is explicit that the compute is
trivial and the one genuinely hard part is **turnover matching**: a null that flips
every bar has enormous turnover, gets destroyed by costs, and makes the real strategy
look good for entirely the wrong reason. The null has to trade at the same rate as
the thing it is testing.

Two design points that are not obvious and both matter.

**The path is generated once, in the constructor.** `signals` must be deterministic
and prefix-consistent -- G1 requires bitwise stability, and the event engine calls
`signals` again on a growing prefix at every bar. A strategy that drew random numbers
inside `signals` would return different weights on every call: G1 would fail, the
twin engines would disagree, and neither failure would look like an RNG problem. So
the chain is drawn up front from an explicitly threaded seed (B9) and `signals` only
slices it.

**The null matches exposure as well as turnover.** Weights are +/-`scale` rather than
+/-1, so mean absolute exposure equals `scale` and can be matched to the strategy
under test. Matching turnover alone leaves the null fully invested against a target
that may sit at 30% average exposure, and the Sharpe comparison then silently
contrasts two different amounts of risk-taking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from falsify.core.types import BARS_PER_YEAR, Bars
from falsify.strategies.base import Strategy


@dataclass(frozen=True, slots=True)
class TurnoverSpec:
    """What a null is being asked to imitate. Frozen (B7)."""

    turnover_annual: float
    exposure: float

    def __post_init__(self) -> None:
        if not 0.0 < self.exposure <= 1.0:
            raise ValueError(f"exposure must be in (0, 1], got {self.exposure}")
        if self.turnover_annual < 0.0:
            raise ValueError(f"turnover_annual must be non-negative, got {self.turnover_annual}")


def flip_probability(spec: TurnoverSpec, bars_per_year: int = BARS_PER_YEAR) -> float:
    """Per-bar flip probability that reproduces `spec.turnover_annual`.

    For a two-state chain on +/-`scale`, a flip moves the weight by `2*scale` and
    nothing else does, so

        E[|dw|] per bar   = 2 * scale * p
        turnover per year = 2 * scale * p * bars_per_year

    which inverts to `p = turnover_annual / (2 * scale * bars_per_year)`. Solved in
    closed form rather than by search -- and then *verified empirically* in G6 to
    within 5%, because a closed form is a claim about the chain and the assertion is
    about the realised path.

    Raises when the requested turnover is unreachable: at `p = 1` the chain flips
    every bar, so `2 * scale * bars_per_year` is the hard ceiling. Clamping silently
    would produce a null that trades less than the strategy it is meant to imitate,
    which biases the comparison in the strategy's favour -- exactly the error the
    gate exists to prevent.
    """
    ceiling = 2.0 * spec.exposure * bars_per_year
    if spec.turnover_annual > ceiling:
        raise ValueError(
            f"turnover {spec.turnover_annual:.2f}/yr is unreachable at exposure "
            f"{spec.exposure:.4f}: a chain flipping every bar delivers at most "
            f"{ceiling:.2f}/yr. Lower the target or raise the exposure."
        )
    return spec.turnover_annual / ceiling


class RandomSign(Strategy):
    """A two-state Markov chain on +/-`scale`. No edge by construction.

    Not `rng.choice([-1, 0, 1], size=T)`: an i.i.d. draw fixes turnover at whatever
    the alphabet implies (2/3 of the maximum for three symbols) and offers no way to
    tune it. The chain's flip probability is the dial that makes turnover matching
    possible at all.
    """

    lookback = 1

    def __init__(
        self,
        n_bars: int,
        rng: np.random.Generator,
        flip_prob: float = 0.5,
        scale: float = 1.0,
    ) -> None:
        if n_bars < 2:
            raise ValueError(f"need at least 2 bars, got {n_bars}")
        if not 0.0 <= flip_prob <= 1.0:
            raise ValueError(f"flip_prob must be in [0, 1], got {flip_prob}")
        if not 0.0 < scale <= 1.0:
            raise ValueError(f"scale must be in (0, 1], got {scale}")

        self.flip_prob = flip_prob
        self.scale = scale

        sign = np.where(rng.random(n_bars) < 0.5, -1.0, 1.0)
        flips = rng.random(n_bars) < flip_prob
        # Fold the flips into a running sign. Vectorising this with a cumulative
        # XOR is possible but the loop is clearer and G6 runs it once per null.
        current = float(sign[0])
        path = np.empty(n_bars)
        for t in range(n_bars):
            if t > 0 and flips[t]:
                current = -current
            path[t] = current

        weights = path * scale
        weights[: self.lookback] = np.nan
        self._weights = weights

    @property
    def name(self) -> str:
        return f"RandomSign(p={self.flip_prob:.4f},scale={self.scale:.3f})"

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        n = len(bars)
        if n > len(self._weights):
            raise ValueError(
                f"{self.name} was generated for {len(self._weights)} bars but asked for {n}; "
                "the path is fixed at construction so that signals() is deterministic"
            )
        return np.asarray(self._weights[:n], dtype=np.float64)


def realised_turnover_annual(
    turnover: NDArray[np.float64], bars_per_year: int = BARS_PER_YEAR
) -> float:
    """Annualised sum of |dw| over a reported window."""
    if len(turnover) == 0:
        return float("nan")
    return float(np.sum(turnover) / len(turnover) * bars_per_year)


def realised_exposure(weights: NDArray[np.float64]) -> float:
    """Mean absolute weight -- how much risk was actually taken."""
    if len(weights) == 0:
        return float("nan")
    return float(np.mean(np.abs(weights)))


def spec_from_result(
    turnover: NDArray[np.float64],
    weights: NDArray[np.float64],
    bars_per_year: int = BARS_PER_YEAR,
) -> TurnoverSpec:
    """Read the imitation target off a real engine run.

    Measured from the strategy's own `Result` rather than assumed, so the null
    inherits whatever the strategy actually did -- including the effect of any
    overlay. That is the difference between a calibrated null and a plausible one.
    """
    return TurnoverSpec(
        turnover_annual=realised_turnover_annual(turnover, bars_per_year),
        exposure=realised_exposure(weights),
    )


__all__ = [
    "RandomSign",
    "TurnoverSpec",
    "flip_probability",
    "realised_exposure",
    "realised_turnover_annual",
    "spec_from_result",
]
