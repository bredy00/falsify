"""PSR, the Deflated Sharpe Ratio, and MinBTL. Specified by 01 Part B2/B3.

The move that makes a Sharpe honest: instead of testing against zero, test against
the expected best-of-N result under the null. A strategy with a 2.1 Sharpe and a DSR
of 0.4 is a strategy you do not trade.

Two conventions matter here and both are easy to get wrong silently:

  * Everything is PER-OBSERVATION. Annualise only at the reporting boundary (B8).
    Feeding an annualised Sharpe to `psr` inflates it by sqrt(252) and the answer
    still looks like a probability.
  * Kurtosis is NON-EXCESS. `scipy.stats.kurtosis` defaults to `fisher=True`, which
    returns excess, and every number downstream is then wrong by a factor that looks
    plausible (B10). `test_g6_null` asserts the normal-case collapse that catches it.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, norm, skew

Series = NDArray[np.float64]

EULER = 0.5772156649015329


def expected_max_sharpe(n_trials: int, var_across_trials: float) -> float:
    """SR_0 -- the expected maximum per-observation Sharpe under the null.

        SR_0 = sqrt(V) * [ (1 - g)*Phi^-1(1 - 1/N) + g*Phi^-1(1 - 1/(N e)) ]

    The two-term Gumbel approximation from the source paper. Its own error against
    the exact order statistic is characterised in `test_prop.py`: it OVERSTATES for
    every N >= 3, so SR_0 is too strict rather than too lax and the resulting DSR is
    conservative. Below N = 100 that error exceeds 1% and should be quoted alongside
    any DSR computed there.
    """
    if n_trials < 2:
        return 0.0
    if var_across_trials < 0.0:
        raise ValueError(f"variance across trials must be non-negative, got {var_across_trials}")
    a = float(norm.ppf(1.0 - 1.0 / n_trials))
    b = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return math.sqrt(var_across_trials) * ((1.0 - EULER) * a + EULER * b)


def psr(returns: Series, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true Sharpe > benchmark).

        PSR = Phi[ (SR - SR*) * sqrt(T - 1) / sqrt(1 - g3*SR + ((g4 - 1)/4)*SR^2) ]

    `returns` and `sr_benchmark` are both per-observation. `PSR(0)` is the
    probability the strategy has any edge at all *ignoring selection*, which is the
    number to report for a strategy you did not search for.
    """
    t = len(returns)
    if t < 3:
        return float("nan")
    sd = float(np.std(returns, ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return float("nan")

    sr = float(np.mean(returns)) / sd
    g3 = float(skew(returns, bias=False))
    g4 = float(kurtosis(returns, fisher=False, bias=False))  # NON-excess (B10)
    variance = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr
    if not np.isfinite(variance) or variance <= 0.0:
        return float("nan")
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(t - 1) / math.sqrt(variance)))


def deflated_sharpe(returns: Series, all_trial_sharpes: Series) -> float:
    """DSR: PSR measured against the expected best-of-N noise result.

    `all_trial_sharpes` are the per-observation Sharpes of every trial evaluated --
    read from the ledger, never hand-typed, and including everything abandoned.
    Deleting the failures from a notebook does not delete them from the statistics.
    """
    trials = np.asarray(all_trial_sharpes, dtype=np.float64)
    if trials.size < 2:
        return psr(returns, 0.0)
    sr0 = expected_max_sharpe(trials.size, float(np.var(trials, ddof=1)))
    return psr(returns, sr0)


def min_backtest_length_years(n_trials: int, target_annual_sharpe: float) -> float:
    """MinBTL: years of daily history needed before an in-sample annualised Sharpe
    of `target_annual_sharpe` is evidence rather than arithmetic.

        MinBTL ~ [ (1-g)Phi^-1(1-1/N) + g*Phi^-1(1-1/(Ne)) ]^2 / SR*^2

    The quadratic is the punishing part: halving the Sharpe target quadruples the
    history required. Report it next to the history actually available -- when the
    two do not reconcile, the honest headline is that the result is uninterpretable
    at this sample length, and printing that is the point.
    """
    if n_trials < 2:
        return 0.0
    if target_annual_sharpe <= 0.0:
        raise ValueError(f"target Sharpe must be positive, got {target_annual_sharpe}")
    z = expected_max_sharpe(n_trials, 1.0)
    return (z * z) / (target_annual_sharpe * target_annual_sharpe)


def empirical_p_value(observed: float, null_distribution: Series) -> float:
    """One-sided p-value of `observed` against an EMPIRICAL null.

    The point of G6. A real strategy's Sharpe must sit in the tail of the null its
    own pipeline produces -- same costs, same turnover, same conventions -- not in
    the tail of a textbook normal. Uses the (r + 1)/(n + 1) form so the result is
    never exactly zero: with n draws you cannot resolve a p-value below 1/(n+1), and
    reporting 0.0 would claim resolution the sample does not have.
    """
    null = np.asarray(null_distribution, dtype=np.float64)
    finite = null[np.isfinite(null)]
    if finite.size == 0:
        return float("nan")
    exceeded = int(np.sum(finite >= observed))
    return (exceeded + 1) / (finite.size + 1)


__all__ = [
    "EULER",
    "deflated_sharpe",
    "empirical_p_value",
    "expected_max_sharpe",
    "min_backtest_length_years",
    "psr",
]
