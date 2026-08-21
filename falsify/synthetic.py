"""Synthetic generators with known truth. Specified by 00-VALIDATION-FIRST.md.

Synthetic data is the only data where the right answer is known in advance, which
makes it the only data that can tell you whether the code is correct. Real data
can tell you a strategy is profitable; it can never tell you the Sharpe function
has a `ddof` bug, because there is nothing to compare against.

Seeds are threaded explicitly as a `Generator` argument (B9). No module-level RNG,
no `np.random.seed`, ever -- G10 depends on it and so does Gate 0.2's 1/sqrt(M)
scaling, which goes flat the moment two paths share state.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from falsify.core.types import BARS_PER_YEAR, Bars

Prices = NDArray[np.float64]


@lru_cache(maxsize=32)
def _business_days(start: str, n: int) -> NDArray[np.datetime64]:
    """Cached business-day index. Pure in (start, n), so caching is safe.

    Worth caching rather than premature: G1 rebuilds the pipeline 10,000 times per
    strategy, and `np.busday_offset` over the full index each time dominated the
    suite -- 33 s of a 2:11 run. The returned array is marked read-only so a shared
    reference cannot be mutated by one caller and observed by another.
    """
    ts: NDArray[np.datetime64] = np.busday_offset(
        np.datetime64(start, "D"), np.arange(n), roll="forward"
    ).astype("datetime64[ns]")
    ts.setflags(write=False)
    return ts


def gbm(
    mu: float,
    sigma: float,
    n_bars: int,
    rng: np.random.Generator,
    s0: float = 100.0,
    bars_per_year: int = BARS_PER_YEAR,
) -> Prices:
    """Geometric Brownian motion, discretised exactly.

        S_t = S_0 * exp[(mu - sigma^2/2) t + sigma W_t]

    `mu` and `sigma` are annual. Log returns are exactly
    N((mu - sigma^2/2)/bars_per_year, sigma^2/bars_per_year), so several
    quantities are known in closed form -- see `tests/gates/test_g3_recovery.py`,
    which pins two of them that are easy to confuse.

    Returns `n_bars` prices with `prices[0] == s0`.
    """
    if n_bars < 2:
        raise ValueError(f"need at least 2 bars, got {n_bars}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    dt = 1.0 / bars_per_year
    drift = (mu - 0.5 * sigma * sigma) * dt
    shock = sigma * np.sqrt(dt) * rng.standard_normal(n_bars - 1)
    log_path = np.concatenate(([0.0], np.cumsum(drift + shock)))
    return np.asarray(s0 * np.exp(log_path), dtype=np.float64)


def ar1(
    phi: float,
    sigma: float,
    n_bars: int,
    rng: np.random.Generator,
    s0: float = 100.0,
) -> Prices:
    """Stationary AR(1) log-price -- a process with a genuine, exploitable edge.

        x_t = phi * x_{t-1} + eps_t,   eps ~ N(0, sigma^2),   0 < phi < 1
        price_t = s0 * exp(x_t)

    `x_0` is drawn from the stationary distribution N(0, sigma^2/(1 - phi^2)) so
    the series does not spend its first hundred bars converging.

    This is the counterpart to `gbm`: GBM has no exploitable structure, so a
    strategy that beats buy-and-hold on it is a bug, while here a z-score
    mean-reversion rule genuinely should find something. Gate 0.1 shows the engine
    does not invent edge; Gate 0.3 shows it does not destroy edge that exists.
    """
    if not 0.0 < phi < 1.0:
        raise ValueError(f"phi must be in (0, 1) for a stationary AR(1), got {phi}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if n_bars < 2:
        raise ValueError(f"need at least 2 bars, got {n_bars}")

    x = np.empty(n_bars)
    x[0] = rng.normal(0.0, sigma / np.sqrt(1.0 - phi * phi))
    eps = rng.normal(0.0, sigma, n_bars - 1)
    for t in range(1, n_bars):
        x[t] = phi * x[t - 1] + eps[t - 1]
    return np.asarray(s0 * np.exp(x), dtype=np.float64)


def half_life(phi: float) -> float:
    """Bars for an AR(1) deviation to decay by half: -ln 2 / ln phi."""
    return float(-np.log(2.0) / np.log(phi))


def bars_from_close(close: Prices, start: str = "2020-01-01") -> Bars:
    """Build `Bars` from a close series, strictly per bar.

    Every field is a function of bar t and t-1 only, so this stage cannot leak --
    deliberately, because if the fixture leaked then every G1 result computed
    through it would be meaningless. `open[t] = close[t-1]` is the causal choice;
    high and low bracket the two. Timestamps are a fixed calendar independent of
    price, which is what a real `align` stage produces (02 Part G).

    Bars sit on business days, not consecutive calendar days. Daily equity bars are
    business days in reality, and the difference is not cosmetic: consecutive
    calendar days put 365 bars in a year, so `elapsed_years` would report 6.9 years
    for what the generator produced as 10 trading years and every CAGR would be
    inflated by 45%. Business days give 260.97 bars per calendar year -- still not
    252, because real calendars lose ~9 days a year to holidays, which is exactly
    why CAGR must be computed from elapsed time rather than from a nominal bar
    count, and why `test_g3_recovery` asserts growth recovery in trading time and
    the CAGR formula separately.
    """
    n = len(close)
    if n < 2:
        raise ValueError(f"need at least 2 bars, got {n}")
    ts = _business_days(start, n)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    return Bars(
        ts=ts,
        open=open_,
        high=np.maximum(open_, close),
        low=np.minimum(open_, close),
        close=np.asarray(close, dtype=np.float64).copy(),
        volume=np.full(n, 1_000_000.0),
        adjustment="total_return",
    )


def noise_grid(
    rng: np.random.Generator, n_obs: int = 1200, n_configs: int = 12, vol: float = 0.01
) -> NDArray[np.float64]:
    """A configuration grid with no differential merit whatsoever.

    Every column has the same true mean and volatility, so any apparent winner is
    sampling noise. G9's null: `PBO(ArgMax)` here must sit at 0.5, because the
    in-sample winner is a coin flip to land in either half out of sample.
    """
    return np.asarray(rng.normal(0.0, vol, (n_obs, n_configs)), dtype=np.float64)


def merit_grid(
    rng: np.random.Generator,
    n_obs: int = 1200,
    n_configs: int = 12,
    vol: float = 0.01,
    spread: float = 8e-4,
) -> NDArray[np.float64]:
    """Real, persistent differences between configurations.

    Column means run linearly from `-spread` to `+spread` and stay there. The best
    column is genuinely best in every block, so selection is doing its job and PBO
    should be low. The counterpart to `compensation_grid`, and the reason a low PBO
    is evidence of something rather than an artefact of the construction.
    """
    drift = np.linspace(-spread, spread, n_configs)
    return np.asarray(rng.normal(0.0, vol, (n_obs, n_configs)) + drift, dtype=np.float64)


def compensation_grid(
    rng: np.random.Generator,
    n_obs: int = 1200,
    n_configs: int = 12,
    n_blocks: int = 12,
    vol: float = 0.01,
    amp: float = 1.2e-3,
) -> NDArray[np.float64]:
    """The overfitting trap: a grid where the in-sample winner is mechanically the
    out-of-sample loser.

    Each configuration gets a per-block mean, and those means are centred to sum to
    zero across blocks -- so whatever a configuration earns in one block it borrowed
    from another. Over two complementary halves the in-sample mean is then the
    negative of the out-of-sample mean, which is Propositions 3 and 5 made concrete
    rather than assumed.

    This exists because of F7: a gate that has never failed is not a gate. G9 has to
    be shown firing on a grid built to defeat it, not merely passing on grids that
    were never going to trouble it. The block count must match the `n_blocks` CSCV
    is run with, or the compensation is smeared across block boundaries and the trap
    is only partly armed.
    """
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be even and at least 2, got {n_blocks}")
    if n_obs < n_blocks * 2:
        raise ValueError(f"{n_obs} observations cannot carry {n_blocks} blocks")

    out = rng.normal(0.0, vol, (n_obs, n_configs))
    means = rng.normal(0.0, amp, (n_blocks, n_configs))
    means -= means.mean(axis=0)  # zero-sum: every gain is borrowed
    bounds = np.linspace(0, n_obs, n_blocks + 1).astype(int)
    for k in range(n_blocks):
        out[bounds[k] : bounds[k + 1]] += means[k]
    return np.asarray(out, dtype=np.float64)
