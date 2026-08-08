"""Health check -- stress the gates far past what their own tests cover.

    uv run python scripts/health_check.py

The gate suite proves each property on a fixture. This proves the same properties
across a wide parameter grid, which is a different question: a gate can be green
because the property holds, or green because the fixture happened to be kind. Run
it when a result looks too good, before trusting a refactor, or when someone
reports a failure the suite does not reproduce.

Deliberately separate from `pytest`:
  - it sweeps hundreds of combinations and takes minutes, so it would dominate CI
  - it reports the worst case it found rather than asserting a threshold, which is
    what you want when diagnosing rather than gating

Exits non-zero if any invariant is violated, so it can still be wired into a
pre-release check.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from falsify.analysis import sweep_costs
from falsify.core.conventions import CONVENTIONS
from falsify.core.event import benchmark_equity, run_event
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST, CostModel
from falsify.metrics import (
    annualise_sharpe,
    annualised_vol,
    gbm_log_return_sharpe,
    gbm_simple_return_sharpe,
    gbm_simple_return_vol,
    sharpe,
)
from falsify.strategies.base import Strategy
from falsify.strategies.simple import BuyAndHold, CausalZScore, Flat, MACrossover
from falsify.synthetic import ar1, bars_from_close, gbm

CAPITAL = 10_000.0
G2_TOLERANCE = 1e-12

RESULT_FIELDS = ("equity", "weights", "gross_ret", "net_ret", "costs", "turnover")

STRATEGIES: tuple[Strategy, ...] = (
    BuyAndHold(),
    Flat(),
    MACrossover(3, 7),
    MACrossover(20, 50),
    CausalZScore(5),
    CausalZScore(30),
    CausalZScore(60),
)

COST_MODELS = (
    ZERO_COST,
    CostModel(commission_bps=2.5),
    CostModel(commission_bps=7.0, half_spread_bps=3.0, slippage_bps=1.5),
    CostModel(commission_bps=40.0, cash_yield_annual=0.05, borrow_bps_annual=200.0),
    CostModel(
        commission_bps=250.0, half_spread_bps=90.0, cash_yield_annual=0.11,
        borrow_bps_annual=900.0,
    ),
)


def relative_deviation(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Max |a - b| / |a|, treating all-NaN slices as agreement."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(a - b) / np.abs(a)
    if np.all(np.isnan(rel)):
        return 0.0
    value = float(np.nanmax(rel))
    return value if math.isfinite(value) else 0.0


def rule(title: str) -> None:
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def check_g2() -> tuple[bool, str]:
    """Twin-engine agreement across strategies, conventions, costs, processes."""
    rule("G2 -- twin engines, full grid, every Result field")
    worst, where, runs = 0.0, "", 0

    for length in (120, 400, 900):
        for seed in (1, 2, 3):
            series = (
                ("gbm", gbm(0.08, 0.20, length, np.random.default_rng(seed))),
                ("ar1", ar1(0.95, 0.02, length, np.random.default_rng(seed + 77))),
            )
            for name, prices in series:
                bars = bars_from_close(prices)
                for strategy in STRATEGIES:
                    if length - (strategy.lookback + 2) < 2:
                        continue
                    for convention in CONVENTIONS:
                        for costs in COST_MODELS:
                            a = run_event(bars, strategy, costs, CAPITAL, convention)
                            b = run_vectorized(bars, strategy, costs, CAPITAL, convention)
                            runs += 1
                            for field in RESULT_FIELDS:
                                dev = relative_deviation(getattr(a, field), getattr(b, field))
                                if dev > worst:
                                    worst = dev
                                    where = (
                                        f"{name} T={length} seed={seed} {strategy.name} "
                                        f"{convention} {costs.total_bps:g}bps {field}"
                                    )

    print(f"combinations: {runs}   worst relative deviation: {worst:.3e}")
    if worst > 0.0:
        print(f"  worst at: {where}")
    ok = worst < G2_TOLERANCE
    return ok, f"worst {worst:.3e} across {runs} combinations"


def check_g4() -> tuple[bool, str]:
    """Zero-cost identity, exact float equality, wide grid."""
    rule("G4 -- zero-cost identity, bitwise")
    checked, broken, worst, where = 0, 0, 0.0, ""

    for length in (50, 137, 400, 1500):
        for seed in range(6):
            series = (
                ("gbm", gbm(0.10, 0.30, length, np.random.default_rng(400 + seed))),
                ("ar1", ar1(0.90, 0.05, length, np.random.default_rng(500 + seed))),
            )
            for name, prices in series:
                bars = bars_from_close(prices)
                for convention in CONVENTIONS:
                    bench = benchmark_equity(bars, BuyAndHold().lookback, CAPITAL, convention)
                    for engine in (run_event, run_vectorized):
                        result = engine(bars, BuyAndHold(), ZERO_COST, CAPITAL, convention)
                        checked += 1
                        if not np.array_equal(result.equity, bench):
                            broken += 1
                            dev = relative_deviation(bench, result.equity)
                            if dev > worst:
                                worst = dev
                                where = (
                                    f"{name} T={length} seed={seed} {convention} "
                                    f"{engine.__name__}"
                                )

    print(f"combinations: {checked}   not bitwise identical: {broken}")
    if broken:
        print(f"  worst relative deviation {worst:.3e} at {where}")
    return broken == 0, f"{checked - broken}/{checked} bitwise identical"


def check_g3(paths: int = 800) -> tuple[bool, str]:
    """Known-truth recovery against EXACT lognormal values, at scale.

    Run at 800 paths rather than the gate's 200 because the point here is to see
    whether the estimator converges to truth or to something near it. At 200 paths
    a 1 SE gap is unremarkable noise; at 800 a persistent gap would be bias.
    """
    rule(f"G3 -- recovery vs exact lognormal targets, {paths} paths")
    mu, sigma, bars_count = 0.08, 0.20, 2520

    simple, logs, vols = [], [], []
    for i in range(paths):
        prices = gbm(mu, sigma, bars_count, np.random.default_rng(50_000 + i))
        result = run_vectorized(
            bars_from_close(prices), BuyAndHold(), ZERO_COST, CAPITAL, "close_to_close"
        )
        net = result.net_ret[1:]
        simple.append(annualise_sharpe(sharpe(net)))
        vols.append(annualised_vol(net))
        logs.append(annualise_sharpe(sharpe(np.diff(np.log(prices)))))

    targets = (
        ("simple SR", simple, gbm_simple_return_sharpe(mu, sigma), mu / sigma),
        ("log SR   ", logs, gbm_log_return_sharpe(mu, sigma), (mu - sigma**2 / 2) / sigma),
        ("vol      ", vols, gbm_simple_return_vol(mu, sigma), sigma),
    )
    ok = True
    worst_gap = 0.0
    for label, values, exact, first_order in targets:
        arr = np.asarray(values)
        mean = float(arr.mean())
        se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
        gap = abs(mean - exact) / se
        worst_gap = max(worst_gap, gap)
        ok = ok and gap < 3.0
        print(
            f"  {label} mean={mean:+.6f} +/- {se:.6f}  exact={exact:+.6f}  "
            f"gap={gap:4.2f} SE   (first-order {first_order:+.6f}, "
            f"off by {abs(exact - first_order) / exact:.4%})"
        )

    # The paired difference: sigma/2 exactly, and far less noisy than either level.
    diff = np.asarray(simple) - np.asarray(logs)
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(len(diff)))
    gap = abs(mean - sigma / 2) / se
    ok = ok and gap < 4.0
    print(
        f"  paired gap mean={mean:+.6f} +/- {se:.6f}  "
        f"exact sigma/2={sigma / 2:+.6f}  gap={gap:4.2f} SE"
    )
    return ok, f"worst level gap {worst_gap:.2f} SE, paired gap {gap:.2f} SE"


def check_g5() -> tuple[bool, str]:
    """Cost monotonicity across strategies and processes."""
    rule("G5 -- cost monotonicity and break-even")
    grid = np.linspace(0.0, 100.0, 41)
    violations = 0
    rows = 0
    for seed in (5_050, 5_051, 5_052):
        bars = bars_from_close(ar1(0.95, 0.02, 1200, np.random.default_rng(seed)))
        for strategy in (CausalZScore(5), CausalZScore(20), CausalZScore(60), MACrossover(5, 15)):
            result = sweep_costs(bars, strategy, grid, CAPITAL, "next_open")
            rows += 1
            monotone = result.is_monotone()
            violations += 0 if monotone else 1
            print(
                f"  seed={seed} {strategy.name:18s} SR {result.sharpe_annual[0]:+.4f} -> "
                f"{result.sharpe_annual[-1]:+.4f}  turnover {result.turnover_annual:6.2f}/yr  "
                f"c*={result.break_even_bps():7.2f} bps  monotone={monotone}"
            )
    return violations == 0, f"{rows - violations}/{rows} sweeps monotone"


def check_invariants() -> tuple[bool, str]:
    """Structural invariants that no single gate owns."""
    rule("Invariants")
    problems: list[str] = []

    bars = bars_from_close(gbm(0.08, 0.20, 200, np.random.default_rng(1)))
    result = run_vectorized(bars, BuyAndHold(), ZERO_COST, CAPITAL, "next_open")

    # B7 -- frozen dataclasses, no in-place mutation.
    for obj, field in ((bars, "close"), (result, "equity")):
        try:
            setattr(obj, field, np.zeros(3))
            problems.append(f"B7: {type(obj).__name__}.{field} is mutable")
        except (AttributeError, TypeError):
            pass
    print(f"  B7 frozen dataclasses:              {'ok' if not problems else 'FAILED'}")

    # B8 -- per-observation internally, annualised only at the boundary.
    per_bar = sharpe(result.net_ret[1:])
    annual = annualise_sharpe(per_bar)
    if not math.isclose(annual / per_bar, math.sqrt(252), rel_tol=1e-12):
        problems.append("B8: annualisation is not sqrt(252)")
    print(f"  B8 annualisation factor sqrt(252):  {annual / per_bar:.6f}")

    # B9 -- seeds threaded; the same seed must reproduce, different seeds must not.
    a = gbm(0.08, 0.2, 50, np.random.default_rng(7))
    b = gbm(0.08, 0.2, 50, np.random.default_rng(7))
    c = gbm(0.08, 0.2, 50, np.random.default_rng(8))
    if not np.array_equal(a, b):
        problems.append("B9: same seed did not reproduce")
    if np.array_equal(a, c):
        problems.append("B9: different seeds produced identical paths")
    print(f"  B9 seeds threaded explicitly:       {'ok' if np.array_equal(a, b) else 'FAILED'}")

    # B10 -- non-excess kurtosis. The two conventions must differ by exactly 3.
    from scipy.stats import kurtosis

    sample = np.random.default_rng(3).standard_t(5, size=2000)
    if abs(
        float(kurtosis(sample, fisher=False, bias=False))
        - float(kurtosis(sample, fisher=True, bias=False))
        - 3.0
    ) > 1e-9:
        problems.append("B10: kurtosis conventions do not differ by 3")
    print("  B10 non-excess kurtosis available:   ok")

    # Gate 0.4 -- degenerate inputs must not silently produce numbers.
    if not math.isnan(sharpe(np.zeros(50))):
        problems.append("0.4: constant returns gave a number instead of NaN")
    print(f"  0.4 degenerate inputs give NaN:     {'ok' if not problems else 'FAILED'}")

    for problem in problems:
        print(f"  ! {problem}")
    return not problems, f"{len(problems)} violation(s)"


def main() -> int:
    print("falsify health check -- stressing the gates beyond their own fixtures")
    checks = (
        ("G2 twin engines", check_g2),
        ("G3 recovery", check_g3),
        ("G4 zero-cost identity", check_g4),
        ("G5 cost monotonicity", check_g5),
        ("Invariants B7-B10", check_invariants),
    )
    outcomes: list[tuple[str, bool, str]] = []
    for name, fn in checks:
        ok, detail = fn()
        outcomes.append((name, ok, detail))

    rule("SUMMARY")
    width = max(len(name) for name, _, _ in outcomes)
    for name, ok, detail in outcomes:
        print(f"  {name:<{width}}  {'PASS' if ok else 'FAIL'}   {detail}")

    failed = [name for name, ok, _ in outcomes if not ok]
    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("all checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
