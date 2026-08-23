"""The vectorised engine -- the product. Specified by 02-ENGINE-SPEC.md Part F2.

Same equations as `event.py`, written independently. Signals are computed once
over the whole series with rolling operations instead of once per bar over a
growing prefix, and turnover and gross return are array expressions rather than
scalar ones.

The equity path is the one genuine subtlety: it is *not* a cumprod, because
`cost[t]` depends on `equity[t-1]`. Part F2 offers an approximate route -- express
cost as a return deduction and cumprod it -- and rejects it, because G2 demands
agreement to 1e-12 and an error of order cost^2 will not deliver that. So the
equity recursion is an explicit loop here too. The vectorisation win lives in the
signal computation, which is where the time actually goes; the accounting loop
over a few thousand bars is free.

What this file must NOT do is import the event engine's accounting. Two engines
that share their arithmetic agree by construction and certify nothing. The
divergence G2 is built to find is precisely the kind that appears when the same
equation is expressed twice -- an accumulation-order difference in the equity
recursion, a rolling mean computed by cumsum rather than by window, an off-by-one
in the weight alignment.
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
from falsify.core.event import warmup_start
from falsify.core.trial import record_trial
from falsify.core.types import BARS_PER_YEAR, Bars, InsufficientHistory, Result
from falsify.costs import CostModel
from falsify.ledger import Ledger
from falsify.strategies.base import Strategy


def run_vectorized(
    bars: Bars,
    strategy: Strategy,
    costs: CostModel,
    initial_capital: float,
    convention: Convention = DEFAULT_CONVENTION,
    *,
    ledger: Ledger,
) -> Result:
    """Array implementation of the Part E equations."""
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

    signals = strategy.signals(bars)
    if len(signals) != n:
        raise ValueError(
            f"{strategy.name}.signals returned {len(signals)} values for {n} bars; "
            "a strategy must return one weight per bar"
        )

    # Weight active over return index t is the signal from t - lag.
    weights: NDArray[np.float64] = np.asarray(signals[start - lag : n - lag], dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        bad = int(np.flatnonzero(~np.isfinite(weights))[0])
        raise ValueError(
            f"{strategy.name} produced a non-finite weight at reported index {bad} "
            f"(bar {start + bad}); NaN must not survive past the declared lookback"
        )

    m = n - start
    returns = np.zeros(m)
    returns[1:] = price[start + 1 : n] / price[start : n - 1] - 1.0

    cost_rate = costs.cost_rate()
    cash_per_bar = costs.cash_rate_per_bar(BARS_PER_YEAR)
    borrow_per_bar = costs.borrow_rate_per_bar(BARS_PER_YEAR)

    exposure = np.abs(weights)
    gross_ret = (
        weights * returns
        + (1.0 - exposure) * cash_per_bar
        - np.maximum(-weights, 0.0) * borrow_per_bar
    )
    gross_ret[0] = 0.0  # anchor bar earns nothing

    # prepend=weights[0] makes turnover[0] exactly zero: the anchor position is
    # established, not traded into.
    turnover = np.abs(np.diff(weights, prepend=weights[0]))

    equity = np.empty(m)
    cost_paid = np.zeros(m)
    net_ret = np.zeros(m)
    equity[0] = initial_capital

    for k in range(1, m):
        prev = equity[k - 1]
        charge = turnover[k] * prev * cost_rate
        equity[k] = prev * (1.0 + gross_ret[k]) - charge
        cost_paid[k] = charge
        net_ret[k] = equity[k] / prev - 1.0

    outcome = Result(
        equity=equity,
        weights=weights,
        gross_ret=gross_ret,
        net_ret=net_ret,
        costs=cost_paid,
        turnover=turnover,
    )
    # B3 rule 1: every engine invocation writes a row. Unconditional, at the single
    # point of return, so there is no path out of this function that skips it.
    record_trial(bars, strategy, costs, outcome, ledger)
    return outcome
