"""Time-series momentum. PLAYBOOK Phase 6, after Moskowitz, Ooi and Pedersen (2012).

    "Time-series momentum (Moskowitz-Ooi-Pedersen 2012: 12-month lookback, 1-month
     hold). Published Sharpe ~ 0.8 on a diversified futures basket. If yours comes out
     at 3.0 on SPY, you have a bug -- this is a free calibration check against the
     literature."

Read that carefully, because it is easy to misread as a target. The 0.8 is for a
*diversified* basket of 58 futures across equities, bonds, currencies and commodities,
each scaled to constant volatility and equally weighted. Most of that Sharpe is
diversification: roughly uncorrelated bets stacked together. A single instrument cannot
reproduce it and should not be expected to.

So this is an upper sanity bound, not a benchmark to hit. A single-asset TSMOM landing
near 0.8 would be luck; landing at 3.0 would be a bug; landing somewhere well below with
a wide interval is what the literature actually predicts, and is what `test_momentum.py`
asserts.

**What is here and what is not.** MOP's position is `sign(past return)` scaled to a
constant ex-ante volatility. Only the signal and the holding period live here, because
`overlays.VolTarget` already does the scaling and composing them is the architecture
(B4: strategies emit weights, overlays wrap strategies). The full MOP construction is

    VolTarget(TimeSeriesMomentum(12, 1), target_annual_vol=0.40, window=60)

with 0.40 being the paper's own target. Composing rather than reimplementing means the
vol scaling is the same code G1 already certified as causal, and the lag alignment that
overlay had to get right is not re-derived here and re-broken.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from falsify.core.types import BARS_PER_YEAR, Bars
from falsify.features import shift_one
from falsify.strategies.base import Strategy

# 252 / 12. The paper works in calendar months; a daily engine needs a bar count, and
# this is the conventional one. Stated here rather than inlined so a reader can see the
# assumption and a caller can override it.
BARS_PER_MONTH = BARS_PER_YEAR // 12


class TimeSeriesMomentum(Strategy):
    """Long if the trailing return is positive, short if negative, held for a month.

        signal[t] = sign( close[t] / close[t - lookback] - 1 ),  lagged one bar
        weight[t] = signal at the most recent rebalance bar

    Two things distinguish this from `MACrossover`, and both are the point.

    **The signal is a return, not a crossing.** A moving-average crossover fires on the
    relationship between two smoothed prices; this fires on the sign of the actual return
    over the lookback. They agree often and disagree at turning points, where the
    crossover is slower.

    **The position is held.** MOP rebalance monthly, and a daily-rebalanced version is a
    different strategy with several times the turnover -- which matters enormously once
    costs are on, and is exactly the kind of detail that makes a published Sharpe
    irreproducible when it is quietly dropped. The hold is implemented as sample-and-hold
    on the signal: the weight is the signal as it stood at the most recent rebalance bar.

    That forward-fill is a position being held, not a gap being filled, so it is not the
    thing B6 forbids. B6 is about fabricating data you did not have; this is about not
    trading on data you did have. The distinction is that no `close` is invented here --
    only the decision is carried forward, which is what holding a position means.
    """

    def __init__(
        self,
        lookback_months: int = 12,
        hold_months: int = 1,
        bars_per_month: int = BARS_PER_MONTH,
    ) -> None:
        if lookback_months < 1:
            raise ValueError(f"lookback_months must be at least 1, got {lookback_months}")
        if hold_months < 1:
            raise ValueError(f"hold_months must be at least 1, got {hold_months}")
        if bars_per_month < 1:
            raise ValueError(f"bars_per_month must be at least 1, got {bars_per_month}")

        self.lookback_months = lookback_months
        self.hold_months = hold_months
        self.bars_per_month = bars_per_month
        self.hold = hold_months * bars_per_month
        self.window = lookback_months * bars_per_month

        # The `+ 1` is `shift_one`, and it is not cosmetic. `close[t] / close[t - window]`
        # is first computable at `t = window`; the decision lag then pushes the first
        # usable weight to `window + 1`. The engine slices `signals[start - lag :]`, so it
        # reads from exactly `lookback` -- declare `window` here and the engine reads one
        # NaN and refuses the run, which is what it did on the first attempt.
        #
        # `MACrossover` looks like it declares `lookback = slow` with no such term, but it
        # is the same arithmetic: `rolling_mean(x, w)` is first finite at `w - 1`, and
        # `shift_one` moves that to `w`. The offset is hidden there and explicit here
        # because the underlying windows differ by one.
        self.lookback = self.window + 1

    @property
    def name(self) -> str:
        return f"TSMomentum({self.lookback_months}m,{self.hold_months}m)"

    def signals(self, bars: Bars) -> NDArray[np.float64]:
        close = bars.close
        n = close.size
        raw = np.full(n, np.nan)

        # close[t] / close[t - lookback] - 1 uses only bars[0:t+1], which is the Part A1
        # contract. `shift_one` then supplies the decision lag every strategy here takes.
        window = self.window
        if n > window:
            raw[window:] = np.sign(close[window:] / close[:-window] - 1.0)
        lagged = shift_one(raw)

        return self._hold(lagged)

    def _hold(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sample the signal on rebalance bars and carry it between them.

        Rebalance bars are anchored at the first bar carrying a valid signal and spaced
        `hold` apart, so the schedule is a deterministic function of the lookback and the
        holding period -- never of the data, and never of where a caller happened to slice
        the series. A schedule that moved with the data would be a second, undeclared
        signal.

        `np.maximum.accumulate` over the rebalance positions is a forward-fill: every bar
        takes the most recent rebalance at or before it. Same trick as the stationary
        bootstrap's block scan, and for the same reason -- the loop it replaces is a
        sequential carry that numpy can do in one pass.
        """
        n = signal.size
        first = np.flatnonzero(np.isfinite(signal))
        if first.size == 0:
            return signal

        start = int(first[0])
        positions = np.arange(n)
        is_rebalance = (positions >= start) & ((positions - start) % self.hold == 0)

        held = np.full(n, np.nan)
        source = np.maximum.accumulate(np.where(is_rebalance, positions, 0))
        held[start:] = signal[source[start:]]
        return held

    def rebalance_count(self, n_bars: int) -> int:
        """How many decisions this actually makes over `n_bars`. Reported alongside
        turnover, because the holding period is the whole difference between this and a
        daily-rebalanced trend follower."""
        usable = n_bars - self.lookback
        return 0 if usable <= 0 else 1 + (usable - 1) // self.hold


__all__ = ["BARS_PER_MONTH", "TimeSeriesMomentum"]
