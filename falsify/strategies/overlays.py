"""Composable overlays that wrap a strategy and reshape its weights.

Each overlay is itself a `Strategy`, so they compose and the engines cannot tell
the difference. Both exist because of what G5 showed: `CausalZScore(5)` trades ~134
times a year and is unprofitable *even at zero cost*, while `CausalZScore(60)` trades
~52 times and survives to 90 bps. Turnover was doing more damage than the signal was
doing good, and there was no dial to turn.

Causality is preserved by construction and verified by G1. Both overlays read only
`bars[0:t+1]` to set the weight at `t`, and `TurnoverBuffer` is a strict left-to-right
fold, so evaluating it on a prefix gives the same answer as evaluating it on the whole
series -- which is what lets the event engine and the vectorised engine agree bitwise
at G2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from falsify.core.types import BARS_PER_YEAR, Bars
from falsify.features import rolling_std, shift_one
from falsify.strategies.base import Strategy


def simple_returns(close: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-bar simple returns with NaN at bar 0. Causal: `r[t]` uses `close[t-1:t+1]`."""
    out = np.full(len(close), np.nan)
    out[1:] = close[1:] / close[:-1] - 1.0
    return out


class VolTarget(Strategy):
    """Scale a base strategy's weight toward a constant volatility target.

        w[t] = clip( base[t] * target / vol_hat[t], -cap, +cap )

    `vol_hat` is the annualised trailing realised volatility over `window` bars, and
    it is **lagged one bar, matching the base signal's own lag**. That is not a
    detail. The base strategies shift their decision by a bar, so sizing on
    same-bar volatility would set the position from an information set the signal
    itself is not allowed to use -- you would be sizing on a bar you had not yet
    seen when you decided to trade. G1's strict execution-alignment cut catches it,
    and did: the first version of this overlay failed that check while passing the
    Part A1 causality contract, which is exactly the distinction the two cut modes
    exist to separate.

    The effect is to lean out when the market is wild and lean in when it is calm.
    Note that with `cap = 1.0` on a volatile series the scaling only ever binds
    downward, so it *reduces* turnover here rather than adding it -- the textbook
    claim that vol targeting adds turnover assumes it can scale up, which the cap
    prevents.

    `cap` defaults to 1.0, which keeps weights inside the [-1, 1] contract in Part B
    and means the overlay can only ever reduce exposure. A cap above 1 implies
    borrowing, so it must be asked for explicitly.
    """

    def __init__(
        self,
        base: Strategy,
        target_annual_vol: float = 0.15,
        window: int = 60,
        cap: float = 1.0,
        bars_per_year: int = BARS_PER_YEAR,
    ) -> None:
        if target_annual_vol <= 0.0:
            raise ValueError(f"target_annual_vol must be positive, got {target_annual_vol}")
        if window < 2:
            raise ValueError(f"window must be at least 2, got {window}")
        if cap <= 0.0:
            raise ValueError(f"cap must be positive, got {cap}")
        self.base = base
        self.target_annual_vol = target_annual_vol
        self.window = window
        self.cap = cap
        self.bars_per_year = bars_per_year
        # +2: one because the vol estimate is built on returns, which start at bar 1,
        # and one more for the lag that keeps its information set aligned with the
        # base signal's.
        self.lookback = max(base.lookback, window + 2)

    @property
    def name(self) -> str:
        return f"VolTarget({self.base.name},{self.target_annual_vol:g},{self.window})"

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        raw = self.base.signals(bars)
        vol = shift_one(
            rolling_std(simple_returns(bars.close), self.window) * np.sqrt(self.bars_per_year)
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled = raw * (self.target_annual_vol / vol)
        out = np.clip(scaled, -self.cap, self.cap)
        # A NaN target or a NaN vol estimate must stay NaN rather than clip to a
        # number: the warm-up has to remain visible to G1, which compares NaN
        # patterns as part of the causality check.
        out[np.isnan(raw) | np.isnan(vol)] = np.nan
        out[: self.lookback] = np.nan
        return np.asarray(out, dtype=np.float64)


class TurnoverBuffer(Strategy):
    """Hold the current position until the base target moves outside a band.

        hold[t] = base[t]       if |base[t] - hold[t-1]| > band
                  hold[t-1]     otherwise

    The dial G5 showed was missing. Turnover is the tax on a signal, and a rule that
    re-optimises every bar pays it every bar whether or not its view has actually
    changed. The band converts small view changes into no trade at all, which cuts
    turnover with no change to the underlying signal -- so the comparison of a
    strategy with and without a buffer isolates exactly what turnover was costing.

    Implemented as a left-to-right fold, which is what keeps it causal *and*
    prefix-consistent: `hold[t]` is a function of `base[0:t+1]` and nothing else, so
    the event engine's hard-sliced prefix produces the identical value.
    """

    def __init__(self, base: Strategy, band: float = 0.25) -> None:
        if band < 0.0:
            raise ValueError(f"band must be non-negative, got {band}")
        self.base = base
        self.band = band
        self.lookback = base.lookback

    @property
    def name(self) -> str:
        return f"TurnoverBuffer({self.base.name},{self.band:g})"

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        raw = self.base.signals(bars)
        out = np.full(len(raw), np.nan)
        held: float | None = None
        for t in range(len(raw)):
            target = float(raw[t])
            if not np.isfinite(target):
                continue
            if held is None or abs(target - held) > self.band:
                held = target
            out[t] = held
        return out


__all__ = ["TurnoverBuffer", "VolTarget", "simple_returns"]
