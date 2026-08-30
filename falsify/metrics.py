"""Performance metrics. Per-observation internally, annualised only here (B8).

Every function that takes or returns a Sharpe names its unit, because mixing
per-bar and annualised figures is the single most common bug in this area and it
produces numbers that look plausible either way.

No bare Sharpe ever leaves this module without a standard error available
alongside it (B2). `sharpe_se` is the Lo (2002) expression, and the non-normal
correction is written in terms of NON-excess kurtosis -- `scipy.stats.kurtosis`
defaults to `fisher=True`, which returns excess, and passing it here silently
rescales everything downstream (B10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew

from falsify.core.types import BARS_PER_YEAR

Series = NDArray[np.float64]

DAYS_PER_YEAR = 365.25


def sharpe(returns: Series, risk_free_per_bar: float = 0.0) -> float:
    """Per-observation Sharpe: mean excess return over sample std (ddof=1).

    Unit: per bar. Returns NaN when the sample has no dispersion -- a zero Sharpe
    and an undefined Sharpe are different claims and only one of them is true
    (Gate 0.4). The naive baseline returns 0.0 here.
    """
    if len(returns) < 2:
        return float("nan")
    sd = float(np.std(returns, ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return float("nan")
    return float((np.mean(returns) - risk_free_per_bar) / sd)


def annualise_sharpe(per_bar: float, bars_per_year: int = BARS_PER_YEAR) -> float:
    """Per-bar Sharpe -> annualised. The reporting boundary, and the only place
    the sqrt(252) belongs."""
    return per_bar * math.sqrt(bars_per_year)


def sample_moments(returns: Series) -> tuple[float, float]:
    """Bias-corrected sample skewness and NON-excess kurtosis (B10).

    Identical to `scipy.stats.skew(x, bias=False)` and
    `scipy.stats.kurtosis(x, fisher=False, bias=False)` -- agreement is 2.7e-16 and
    4.4e-16 relative across sample sizes 10 to 2,500 on heavy-tailed draws, i.e. machine
    epsilon, and `test_hac.py` asserts it rather than trusting this note.

    It exists because B3 puts `sharpe_se` on the path of every engine invocation, and
    scipy's versions cost 5.2 ms against 0.43 ms here -- twelve times, on a function the
    gate suite calls tens of thousands of times. The arithmetic is the same; scipy spends
    the difference on input validation and NaN policy that this caller has already done.

    Below four observations the bias correction divides by `n - 3`, so scipy's own path
    is used instead: it returns a finite value at n = 3 by a different route, and this
    is not the place to invent a third convention.
    """
    x = np.asarray(returns, dtype=np.float64)
    n = x.size
    if n < 4:
        return (
            float(skew(x, bias=False)),
            float(kurtosis(x, fisher=False, bias=False)),
        )
    d = x - x.mean()
    m2 = float(np.mean(d * d))
    if m2 <= 0.0:
        return (float("nan"), float("nan"))
    m3 = float(np.mean(d**3))
    m4 = float(np.mean(d**4))
    g1 = m3 / m2**1.5
    g2 = m4 / (m2 * m2) - 3.0
    skewness = g1 * math.sqrt(n * (n - 1.0)) / (n - 2.0)
    excess = ((n + 1.0) * g2 + 6.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0))
    return (skewness, excess + 3.0)


def sharpe_se(returns: Series, bars_per_year: int | None = None) -> float:
    """Standard error of the Sharpe estimate, Lo (2002) with the non-normal
    correction.

        SE(SR) ~ sqrt( (1 - g3*SR + ((g4 - 1)/4)*SR^2) / (T - 1) )

    Per bar by default; pass `bars_per_year` to get the annualised SE. With normal
    returns g3 = 0 and g4 = 3, so the numerator collapses to 1 + SR^2/2 and this
    agrees with the i.i.d. expression -- asserted in the G3 suite.
    """
    t = len(returns)
    if t < 3:
        return float("nan")
    sr = sharpe(returns)
    if not np.isfinite(sr):
        return float("nan")
    g3, g4 = sample_moments(returns)  # g4 is NON-excess (B10)
    variance = (1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr) / (t - 1)
    if not np.isfinite(variance) or variance < 0.0:
        return float("nan")
    se = math.sqrt(variance)
    return se * math.sqrt(bars_per_year) if bars_per_year else se


def gbm_simple_return_sharpe(mu: float, sigma: float, bars_per_year: int = BARS_PER_YEAR) -> float:
    """Exact annualised Sharpe of the SIMPLE returns of a GBM. No approximation.

    A simple return is `exp(g) - 1` for a normal log return `g ~ N(m, s^2)`, i.e.
    lognormal, so its moments are known in closed form:

        E[r]  = exp(m + s^2/2) - 1
        sd[r] = exp(m + s^2/2) * sqrt(exp(s^2) - 1)

    The convenient first-order answer is `mu/sigma`, which is what most write down
    and what this gate used to assert. It is off by 0.02% at mu=0.08, sigma=0.20 --
    immaterial against sampling error, but a known-truth gate should compare
    against truth rather than against a good approximation to it, and the exact
    form costs three lines.
    """
    m = (mu - 0.5 * sigma * sigma) / bars_per_year
    s2 = sigma * sigma / bars_per_year
    scale = math.exp(m + 0.5 * s2)
    mean = scale - 1.0
    sd = scale * math.sqrt(math.expm1(s2))
    return mean / sd * math.sqrt(bars_per_year)


def gbm_log_return_sharpe(mu: float, sigma: float, bars_per_year: int = BARS_PER_YEAR) -> float:
    """Exact annualised Sharpe of the LOG returns of a GBM: (mu - sigma^2/2)/sigma.

    Exact with no correction needed, because log returns are exactly normal. The
    gap to `gbm_simple_return_sharpe` is exactly sigma/2 -- see
    test_g3_recovery.test_g3_the_two_sharpe_conventions_differ_by_sigma_over_two,
    which is the sharpest check in the file precisely because the difference is
    measured on the same paths and the sampling noise cancels.
    """
    del bars_per_year  # the ratio is scale-free in the annualisation
    return (mu - 0.5 * sigma * sigma) / sigma


def gbm_simple_return_vol(mu: float, sigma: float, bars_per_year: int = BARS_PER_YEAR) -> float:
    """Exact annualised volatility of the SIMPLE returns of a GBM.

    Slightly above `sigma` -- 0.200071 against 0.200000 at sigma = 0.20 -- because
    the lognormal is right-skewed. `sigma` is the volatility of the LOG returns.
    """
    m = (mu - 0.5 * sigma * sigma) / bars_per_year
    s2 = sigma * sigma / bars_per_year
    return math.exp(m + 0.5 * s2) * math.sqrt(math.expm1(s2)) * math.sqrt(bars_per_year)


def annualised_vol(returns: Series, bars_per_year: int = BARS_PER_YEAR) -> float:
    if len(returns) < 2:
        return float("nan")
    return float(np.std(returns, ddof=1)) * math.sqrt(bars_per_year)


def elapsed_years(ts: NDArray[np.datetime64]) -> float:
    """Calendar years between the first and last timestamp.

    Elapsed time, not `len(bars) / 252`. The naive baseline divides by a nominal
    bar count, so every holiday-heavy stretch silently inflates the annualisation.
    """
    span = (ts[-1] - ts[0]) / np.timedelta64(1, "D")
    return float(span) / DAYS_PER_YEAR


def cagr(equity: Series, years: float) -> float:
    """Compound annual growth rate over `years` of elapsed calendar time."""
    if years <= 0.0 or len(equity) < 2 or equity[0] <= 0.0 or equity[-1] <= 0.0:
        return float("nan")
    return float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)


def max_drawdown(equity: Series) -> float:
    """Largest peak-to-trough decline, as a non-positive fraction."""
    if len(equity) < 2:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def exposure(weights: Series) -> float:
    """Fraction of bars holding a non-zero position.

    Reported alongside Sharpe because a long/cash strategy's exact zeros shrink
    the sample standard deviation without shrinking the mean proportionally, which
    flatters its Sharpe against a fully-invested one. Without exposure the two
    numbers are not comparable.
    """
    if len(weights) == 0:
        return float("nan")
    return float(np.mean(weights != 0.0))


@dataclass(frozen=True, slots=True)
class Performance:
    """A reported result and its error bar. Frozen (B7).

    B2 says every performance number carries an error bar; this type makes that
    structural rather than a habit, so a bare Sharpe cannot be constructed here.
    """

    sharpe_annual: float
    sharpe_annual_se: float
    vol_annual: float
    cagr: float
    max_drawdown: float
    exposure: float
    n_obs: int

    def sharpe_ci95(self) -> tuple[float, float]:
        half = 1.959963984540054 * self.sharpe_annual_se
        return (self.sharpe_annual - half, self.sharpe_annual + half)

    def describe(self) -> str:
        lo, hi = self.sharpe_ci95()
        return (
            f"SR={self.sharpe_annual:+.3f} +/- {self.sharpe_annual_se:.3f} "
            f"(95% CI [{lo:+.3f}, {hi:+.3f}])  vol={self.vol_annual:.3f}  "
            f"CAGR={self.cagr:+.4%}  MDD={self.max_drawdown:.4%}  "
            f"exposure={self.exposure:.3f}  T={self.n_obs}"
        )


def summarise(
    net_returns: Series,
    equity: Series,
    weights: Series,
    years: float,
    bars_per_year: int = BARS_PER_YEAR,
) -> Performance:
    """Assemble the reported figures from one engine run.

    `net_returns` and `equity` must already be sliced to the reported window --
    slice first, compound second (Part E).
    """
    return Performance(
        sharpe_annual=annualise_sharpe(sharpe(net_returns), bars_per_year),
        sharpe_annual_se=sharpe_se(net_returns, bars_per_year=bars_per_year),
        vol_annual=annualised_vol(net_returns, bars_per_year),
        cagr=cagr(equity, years),
        max_drawdown=max_drawdown(equity),
        exposure=exposure(weights),
        n_obs=len(net_returns),
    )


def newey_west_lag(n_obs: int) -> int:
    """`L = floor(4 * (T/100)^(2/9))`, the rule 01 Part B1 specifies.

    **01 B1's worked example is wrong and this follows the formula instead.** The spec
    writes the floor explicitly and then says "For T = 1008, L = 7". It is 6:

        4 * (1008/100)^(2/9) = 4 * 10.08^0.2222 = 4 * 1.6710 = 6.684  ->  floor 6

    Seven is what rounding gives, so the example looks like a rounding slip rather than a
    different intended rule. Floor is also the standard statement of Newey-West (1994)
    automatic lag selection, so the formula is right and the arithmetic beside it is not.
    Recorded here rather than silently matched, because a reader checking the code against
    the spec will hit this and deserves to find the discrepancy already accounted for.

    The rule grows slowly: T=252 gives 4, T=1008 gives 6, T=2516 gives 8. A decade of
    daily bars still only reaches into the second week.
    """
    if n_obs < 1:
        raise ValueError(f"need at least one observation, got {n_obs}")
    scale: float = (n_obs / 100.0) ** (2.0 / 9.0)
    return math.floor(4.0 * scale)


def newey_west_se(returns: Series, lag: int | None = None) -> float:
    """HAC standard error of the MEAN return. Unit: per bar (B8).

    Bartlett kernel, `w_k = 1 - k/(L+1)`:

        S = gamma(0) + 2 * sum_{k=1..L} w_k * gamma(k)
        SE(mean) = sqrt(S / T)

    Why this exists at all: `sharpe_se` implements Lo (2002) with the non-normal
    correction, which handles skew and kurtosis but still assumes the observations are
    independent. Strategy returns are not -- momentum and vol clustering both put weight
    at low lags -- and 01 Part B1 is explicit that serial correlation inflates the true
    standard error above both analytic formulas. So a significance claim uses this, not
    the naive one.

    The Bartlett weights are not decoration. An unweighted sum of autocovariances can
    produce a negative variance estimate; the triangular taper is what makes `S`
    positive semi-definite, which is the entire point of Newey-West over a plain
    truncated estimator. If `S` still comes out non-positive the sample is too short for
    the chosen lag, and this returns NaN rather than the square root of a negative
    number.

    **Known limit, measured rather than assumed.** On AR(1) data the ratio of this SE to
    the naive one should approach `sqrt((1+phi)/(1-phi))`. Measured at T=3000:

        phi     0.0     0.2     0.5     0.8      -0.5
        HAC     1.009   1.211   1.624   2.288    0.607
        theory  1.000   1.225   1.732   0.577 (phi=-0.5), 3.000 (phi=0.8)

    Close through moderate persistence and visibly short at `phi = 0.8`, because the
    automatic lag rule truncates at L=8 while an AR(1) at 0.8 still has autocorrelation
    0.17 at lag 8. The estimator under-corrects for strongly persistent series; that is a
    property of the truncation, not a bug, and it means a HAC t-statistic on a very
    persistent series is still optimistic. The bootstrap in `falsify.bootstrap` makes no
    such truncation and is the cross-check when persistence is high.
    """
    t = len(returns)
    if t < 3:
        return float("nan")
    resolved = newey_west_lag(t) if lag is None else lag
    if resolved < 0:
        raise ValueError(f"lag must be non-negative, got {resolved}")
    resolved = min(resolved, t - 1)

    centred = np.asarray(returns, dtype=np.float64) - float(np.mean(returns))
    gamma0 = float(centred @ centred) / t
    total = gamma0
    for k in range(1, resolved + 1):
        gamma_k = float(centred[:-k] @ centred[k:]) / t
        total += 2.0 * (1.0 - k / (resolved + 1.0)) * gamma_k

    if not math.isfinite(total) or total <= 0.0:
        return float("nan")
    return math.sqrt(total / t)


def newey_west_t(returns: Series, lag: int | None = None) -> float:
    """t-statistic on the mean return, HAC-corrected. `newey_west_t` in 01 Part D.

    Scale-free and unit-free, so it needs no annualisation (B8). This is the number a
    significance claim rests on; `sharpe` alone is not one.
    """
    se = newey_west_se(returns, lag)
    if not math.isfinite(se) or se <= 0.0:
        return float("nan")
    return float(np.mean(returns)) / se
