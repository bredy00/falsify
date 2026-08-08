"""G1 -- the causality cut. Specified by 02-ENGINE-SPEC.md Part A.

The contract. For a price series of length T and a cut tau, the scrambled series
P^tau agrees with P on [0, tau] and is independent noise on (tau, T). For any
strategy S and any tau:

    S(P)[0:tau+1] == S(P^tau)[0:tau+1]      exactly, bitwise

Everything at index s <= tau must be invariant to arbitrary mutilation of
everything at s > tau. If you can change the future and the past moves, you have
leakage.

Why this exists when `shift(1)` already does: `shift(1)` enforces one alignment
convention, and it is a claim. This is the proof of the claim, and it catches
whole classes of leakage that no amount of shifting can express -- a scaler
fitted on the full series, a `dropna()` whose row count depends on future rows, a
global-percentile outlier clip, a rolling window with `center=True`. Every one of
those passes code review and fails here.

Run it on the widest pipeline you have. 02 Part A2 is explicit that leakage lives
in the data layer more often than in the strategy layer, because the strategy
layer is where attention goes. Testing `signals` alone will pass while a `bfill`
upstream quietly cheats.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

Prices = NDArray[np.float64]
Pipeline = Callable[[Prices], NDArray[np.float64]]

# `sample_taus` returns an int64 array; a hand-written list is equally valid.
TauLike = Sequence[int] | NDArray[np.integer[Any]]

# 02 Part A4: 500 values of tau, 20 seeds each.
DEFAULT_N_TAUS = 500
DEFAULT_N_SEEDS = 20


class CausalityViolation(AssertionError):
    """Raised when signals at or before the cut respond to the future."""


def scramble_future(
    prices: Prices, tau: int, rng: np.random.Generator, *, include_cut: bool = False
) -> Prices:
    """Return a copy of `prices` with the future replaced by noise.

    `include_cut=False` (the Part A1 default) preserves [0, tau] and scrambles
    (tau, T). `include_cut=True` preserves [0, tau) and scrambles [tau, T), so
    that bar `tau` itself is mutilated -- see `causality_cut_test` for why that
    second mode exists and what it proves.

    The replacement is a geometric random walk continuing from the last preserved
    price, with per-step volatility matched to the observed series, so the
    scrambled tail is a *plausible* future rather than an obviously broken one. A
    pipeline that rejected the tail on validation grounds would otherwise pass
    this test for the wrong reason.
    """
    first_scrambled = tau if include_cut else tau + 1
    if not 1 <= first_scrambled < len(prices):
        raise ValueError(
            f"tau={tau} (include_cut={include_cut}) gives no valid cut for T={len(prices)}"
        )

    scrambled = prices.copy()
    tail = len(prices) - first_scrambled
    sigma = float(np.std(np.diff(np.log(prices))))
    shocks = rng.normal(0.0, sigma, size=tail)
    scrambled[first_scrambled:] = prices[first_scrambled - 1] * np.exp(np.cumsum(shocks))
    return scrambled


def causality_cut_test(
    pipeline: Pipeline,
    prices: Prices,
    taus: TauLike,
    rng: np.random.Generator,
    n_seeds: int = DEFAULT_N_SEEDS,
    *,
    include_cut: bool = False,
) -> None:
    """G1. Raises `CausalityViolation` on any leakage.

    Two modes, because 02 Part A specifies two different properties and only one
    of them is causality:

    `include_cut=False` -- the Part A1 contract, and the default. Scrambles
        (tau, T) and requires signals[0:tau+1] to be bitwise identical. This is
        the *causality* property: the signal at bar t is a functional of the past
        light cone, bars[0:t+1] inclusive. It catches every leakage class named in
        Part A3 -- a scaler fitted on the whole series, a `dropna()` whose row
        count depends on future rows, a global-percentile clip, a rolling window
        with `center=True`.

    `include_cut=True` -- scrambles [tau, T) instead, so bar `tau` is itself
        mutilated, and therefore requires signals[0:tau+1] to depend only on
        bars[0:t] *strictly*. This is not causality; it is the *execution
        alignment* claim that `shift(1)` makes -- that the weight held during bar
        t was decided before bar t opened.

    RULING on Part A4 versus Part A1, taken 2026-08-08 and settled by measurement
    rather than by argument. **A1 stands as the causality contract.**

    A4 offers `LeakyOracle` -- `sign(diff(close))` -- and asserts G1 must catch it.
    It must not, because it does not leak. Two independent reasons:

    1. Structurally: close[t] lies inside bars[0:t+1], which A1 permits. And every
       convention in Part D applies the weight with a lag of at least one bar, so
       the weight earning return t was decided from bars strictly older than t.
       There is no convention under which that strategy trades on information it
       could not have had.
    2. Empirically: run through the engine over 40 GBM paths it earns a mean
       annualised Sharpe of +0.054 against buy-and-hold's +0.372. A strategy that
       actually saw one bar ahead earns +21.10. `sign(diff(close))` is not an
       oracle; the name was aspirational.

    So the trap was mis-specified, not the harness, and tightening A1 to
    bars[0:t] strictly would have failed every legitimate `close_to_close`
    strategy -- which Part D explicitly permits -- while catching nothing real.
    `test_g1_causality.py` traps a genuine look-ahead oracle instead, and keeps
    the A4 strategy as a documented non-violator so the reasoning cannot be lost.

    `include_cut=True` is retained, but as what it actually is: a check on
    execution alignment, not on causality. It answers "was this weight decided
    before the bar it earns opened", which is the engine's responsibility.

    `equal_nan=True` because warm-up NaNs are legitimate -- and must themselves
    be stable. A pipeline whose warm-up length depends on the future is leaking
    just as surely as one whose values do.
    """
    baseline = pipeline(prices)
    n_preserved = tau_preserved = 0  # rebound per tau; declared for clarity

    for raw_tau in taus:
        tau = int(raw_tau)  # normalise: np.int64 from sample_taus, or a plain int
        # Indices required to be invariant, and indices actually scrambled.
        tau_preserved = tau + 1
        n_preserved = tau if include_cut else tau + 1

        for seed_index in range(n_seeds):
            scrambled = scramble_future(prices, tau, rng, include_cut=include_cut)

            # Guard against a vacuous test. If the scramble were a no-op, every
            # pipeline on earth would pass and G1 would certify nothing (F7).
            if not np.array_equal(scrambled[:n_preserved], prices[:n_preserved]):
                raise AssertionError(f"scramble corrupted the past at tau={tau}")
            if np.array_equal(scrambled[n_preserved:], prices[n_preserved:]):
                raise AssertionError(
                    f"scramble left the future unchanged at tau={tau}; the test would be vacuous"
                )

            out = pipeline(scrambled)
            if out.shape != baseline.shape:
                raise CausalityViolation(
                    f"output length changed under scrambling at tau={tau}: "
                    f"{out.shape} vs baseline {baseline.shape}"
                )
            if not np.array_equal(out[:tau_preserved], baseline[:tau_preserved], equal_nan=True):
                bad = int(
                    np.argmax(~_elementwise_equal(out[:tau_preserved], baseline[:tau_preserved]))
                )
                raise CausalityViolation(
                    f"causality violated at tau={tau} (seed index {seed_index}, "
                    f"include_cut={include_cut}): first divergence at index {bad}, "
                    f"baseline={baseline[bad]!r} scrambled={out[bad]!r}"
                )


def _elementwise_equal(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Equality treating NaN as equal to NaN, for locating the first divergence."""
    both_nan = np.isnan(a) & np.isnan(b)
    return np.asarray((a == b) | both_nan)


def sample_taus(
    n_bars: int, lookback: int, rng: np.random.Generator, n_taus: int = DEFAULT_N_TAUS
) -> NDArray[np.int64]:
    """`n_taus` cut points drawn uniformly over [lookback, n_bars - 1).

    The upper end is exclusive of the final bar: a cut at T-1 leaves no future to
    scramble, so it would test nothing.
    """
    low, high = lookback, n_bars - 1
    if high <= low:
        raise ValueError(f"no valid tau for n_bars={n_bars}, lookback={lookback}")
    return rng.integers(low, high, size=n_taus, dtype=np.int64)
