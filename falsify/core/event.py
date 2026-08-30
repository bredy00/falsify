"""The event engine -- the reference. Specified by 02-ENGINE-SPEC.md Part F1.

Slow and obviously correct. Its whole value is that it is *structurally* unable
to cheat: signals come from `bars.slice(0, k + 1)`, a hard prefix, so the code
cannot see bar `k + 1` even if it wanted to. Everything after that is scalar
arithmetic in a Python loop, one bar at a time, in the order Part E writes it.

This file deliberately shares no accounting code with `vectorized.py`. Sharing a
helper would guarantee the two agree and prove nothing -- G2 exists to catch a
vectorised implementation that is subtly different, and it can only do that if
the two are written independently. B5 makes the equations canonical, not the
implementation.

Cost: O(T^2), because the strategy is recomputed on a growing prefix at every
bar. That is intended. This engine certifies the fast one on a subsample; it is
not the product.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from falsify.core.conventions import (
    DEFAULT_CONVENTION,
    Convention,
    fill_prices,
    signal_lag,
)
from falsify.core.trial import record_trial
from falsify.core.types import BARS_PER_YEAR, Bars, InsufficientHistory, Result
from falsify.costs import CostModel
from falsify.ledger import Ledger
from falsify.strategies.base import Strategy


def warmup_start(lookback: int, lag: int) -> int:
    """First return index with a valid weight.

    A weight at return index t comes from the signal at t - lag, and signals are
    NaN before `lookback`, so t must be at least lookback + lag. Getting this
    wrong by one bar is the classic way to leak a single observation, which is
    why it is computed in one place and shared by both engines.
    """
    return lookback + lag


def run_event(
    bars: Bars,
    strategy: Strategy,
    costs: CostModel,
    initial_capital: float,
    convention: Convention = DEFAULT_CONVENTION,
    *,
    ledger: Ledger,
) -> Result:
    """Bar-by-bar reference implementation of the Part E equations."""
    if initial_capital <= 0.0:
        raise ValueError(f"initial_capital must be positive, got {initial_capital}")

    n = len(bars)
    lag = signal_lag(convention)
    start = warmup_start(strategy.lookback, lag)
    if n - start < 2:
        raise InsufficientHistory(
            f"{n} bars is too few: lookback={strategy.lookback} and "
            f"convention={convention!r} (lag={lag}) leave {max(n - start, 0)} reported bars, "
            "and at least 2 are needed"
        )

    price = fill_prices(bars, convention)
    m = n - start

    weights = np.empty(m)
    for k in range(m):
        signal_index = start + k - lag
        # Hard prefix: this call cannot reach past `signal_index`.
        window = bars.slice(0, signal_index + 1)
        signals = strategy.signals(window)
        if len(signals) != signal_index + 1:
            raise ValueError(
                f"{strategy.name}.signals returned {len(signals)} values for a "
                f"{signal_index + 1}-bar window; a strategy must return one weight per bar"
            )
        weights[k] = signals[-1]

    if not np.all(np.isfinite(weights)):
        bad = int(np.flatnonzero(~np.isfinite(weights))[0])
        raise ValueError(
            f"{strategy.name} produced a non-finite weight at reported index {bad} "
            f"(bar {start + bad}); NaN must not survive past the declared lookback"
        )

    cost_rate = costs.cost_rate()
    cash_per_bar = costs.cash_rate_per_bar(BARS_PER_YEAR)
    borrow_per_bar = costs.borrow_rate_per_bar(BARS_PER_YEAR)

    equity = np.empty(m)
    gross_ret = np.zeros(m)
    net_ret = np.zeros(m)
    cost_paid = np.zeros(m)
    turnover = np.zeros(m)

    equity[0] = initial_capital  # anchor bar: no return, no cost, no turnover

    for k in range(1, m):
        t = start + k
        r = price[t] / price[t - 1] - 1.0

        w = weights[k]
        w_prev = weights[k - 1]

        # Part E, gross return: exposure, plus yield on the unallocated
        # fraction, minus borrow on the short leg.
        gross = w * r + (1.0 - abs(w)) * cash_per_bar - max(-w, 0.0) * borrow_per_bar

        # Part E, cost on traded notional -- not on portfolio return.
        traded = abs(w - w_prev)
        charge = traded * equity[k - 1] * cost_rate

        # Part E, multiplicative equity recursion. Not net = gross - cost_rate:
        # the additive form is a first-order approximation and there is no reason
        # to accept its error when the exact form is one line.
        equity[k] = equity[k - 1] * (1.0 + gross) - charge

        gross_ret[k] = gross
        turnover[k] = traded
        cost_paid[k] = charge
        net_ret[k] = equity[k] / equity[k - 1] - 1.0

    outcome = Result(
        equity=equity,
        weights=weights,
        gross_ret=gross_ret,
        net_ret=net_ret,
        costs=cost_paid,
        turnover=turnover,
    )
    # B3 rule 1, and B5: the twin engines record identically or they are not twins.
    record_trial(bars, strategy, costs, outcome, ledger)
    return outcome


def benchmark_equity(
    bars: Bars,
    lookback: int,
    initial_capital: float,
    convention: Convention = DEFAULT_CONVENTION,
) -> NDArray[np.float64]:
    """Buy-and-hold on the *reported* window. Part E, final block.

    Sliced before compounding, so `bench[0] == initial_capital` exactly and the
    two curves are comparable. The naive baseline compounds the market through
    the warm-up while the strategy curve restarts after a `dropna`, then plots
    them against different bases -- a bug that makes the strategy look better or
    worse purely as a function of warm-up length.
    """
    lag = signal_lag(convention)
    start = warmup_start(lookback, lag)
    price = fill_prices(bars, convention)
    n = len(bars)
    if n - start < 2:
        raise InsufficientHistory(f"{n} bars leaves {max(n - start, 0)} reported bars")

    bench = np.empty(n - start)
    bench[0] = initial_capital
    for k in range(1, n - start):
        t = start + k
        bench[k] = bench[k - 1] * (1.0 + (price[t] / price[t - 1] - 1.0))
    return bench
