"""Turning one engine run into one ledger row. B3 rule 1.

Both engines call `record_trial` at the point they return, so "every engine invocation
writes a row" is true by construction rather than by discipline. It lives here rather
than in either engine because B5 requires the two to stay identical, and a recording step
duplicated in two files is a recording step that will eventually differ in one.

It lives here rather than in `falsify.ledger` because the ledger should not need to know
what a `Bars` or a `Result` is. The ledger stores trials; this translates a run into one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from falsify.core.types import BARS_PER_YEAR, Bars, Result
from falsify.costs import CostModel
from falsify.ledger import Ledger, ParamValue, Recording, make_record, series_digest
from falsify.metrics import annualise_sharpe, sharpe, sharpe_se

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing the strategy layer
    from falsify.strategies.base import Strategy


def strategy_params(strategy: Strategy) -> dict[str, ParamValue]:
    """The strategy's scalar attributes, machine-read.

    Never hand-typed (B3). `vars()` covers ordinary instances and `__slots__` covers the
    frozen ones, and anything that is not a JSON scalar is left out -- an array of
    precomputed signal, say, which is state rather than a parameter.

    Dropping non-scalars is safe here only because `strategy.name` is recorded alongside
    and already encodes the configuration: `MACrossover(20,50)` names its own parameters.
    Identity rests on the pair, so a strategy whose name did not distinguish it would
    need its parameters made scalar rather than this relaxed.
    """
    raw: dict[str, object] = {}
    if hasattr(strategy, "__dict__"):
        raw.update(vars(strategy))
    for slot in getattr(type(strategy), "__slots__", ()):
        if hasattr(strategy, slot):
            raw[slot] = getattr(strategy, slot)
    return {
        k: v
        for k, v in raw.items()
        if not k.startswith("_") and isinstance(v, (str, int, float, bool, type(None)))
    }


def record_trial(
    bars: Bars,
    strategy: Strategy,
    costs: CostModel,
    result: Result,
    ledger: Ledger,
    *,
    bars_per_year: int = BARS_PER_YEAR,
) -> None:
    """Write one row for this run. Unconditional -- no debug flag, no early return.

    The Sharpe is annualised here because the ledger is a reporting surface (B8), and it
    carries its standard error because B2 admits no bare performance number, including
    into a file nobody reads until they need `N`.

    Returns are taken from index 1 onward: index 0 is the anchor bar, which carries no
    return by construction, and including it would drag every recorded Sharpe toward zero
    by one observation.
    """
    returns = result.net_ret[1:]

    # A `NONE` ledger persists nothing; the record exists only for its content address,
    # and the address does not depend on performance. Computing a Sharpe and its standard
    # error for a row that is immediately discarded cost 0.72 ms of every engine
    # invocation across the whole gate suite, which is thousands of calls.
    #
    # This is not a B2 exemption. B2 governs REPORTED numbers, and a `NONE` ledger reports
    # none -- `n_trials()` returns a count, not a performance claim. The moment a row is
    # persisted, and so becomes readable by someone deciding something, it carries its
    # error bar. A counting token is not a measurement, and `sharpe = NaN` says so
    # honestly rather than offering a number nobody computed.
    measured = ledger.recording is not Recording.NONE
    if measured:
        per_bar = sharpe(returns) if returns.size >= 2 else float("nan")
        annual = annualise_sharpe(per_bar, bars_per_year) if np.isfinite(per_bar) else per_bar
        standard_error = sharpe_se(returns, bars_per_year) if returns.size >= 3 else float("nan")
    else:
        annual = standard_error = float("nan")

    ledger.observe(
        make_record(
            strategy=strategy.name,
            params=strategy_params(strategy),
            sharpe=annual,
            sharpe_se=standard_error,
            n_obs=int(returns.size),
            cost_bps=costs.total_bps,  # a property, not a method -- mypy caught the call
            series_digest=series_digest(bars.close),
            date_range=(str(bars.ts[0]), str(bars.ts[-1])),
            recording=ledger.recording,
        )
    )


__all__ = ["record_trial", "strategy_params"]
