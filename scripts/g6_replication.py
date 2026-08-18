"""G6 replication study -- 15 independent runs, to set bounds from evidence.

    uv run python scripts/g6_replication.py [n_replications]

A single null calibration tells you the gate passed once. It says nothing about how
much the calibration statistics move between independent worlds, and the gate's
thresholds have to be set against that spread or they are guesses. This runs the
whole G6 construction end to end N times -- fresh price series, fresh strategy,
fresh calibration, fresh thousand nulls -- and reports the observed range of every
quantity the gate asserts on.

Deliberately a script rather than a test: it takes minutes, and its output is the
evidence used to CHOOSE thresholds. Baking it into the suite would mean the suite
both sets and checks its own bounds, which is circular.
"""

from __future__ import annotations

import math
import sys

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kstest

from falsify.core.conventions import Convention
from falsify.core.types import BARS_PER_YEAR
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST, CostModel
from falsify.deflated import deflated_sharpe, empirical_p_value, psr
from falsify.metrics import annualise_sharpe, sharpe
from falsify.strategies.null import (
    RandomSign,
    flip_probability,
    realised_exposure,
    realised_turnover_annual,
    spec_from_result,
)
from falsify.strategies.simple import CausalZScore
from falsify.synthetic import ar1, bars_from_close

N_BARS = 1000
N_NULLS = 1000
N_DSR = 200
CAPITAL = 10_000.0
CONVENTION: Convention = "next_open"
ALPHAS = (0.10, 0.05, 0.01)


def replication(index: int) -> dict[str, float]:
    """One complete, independent G6 construction."""
    # A different world each time: the price series, the null seeds and the DSR
    # subset all move together, so the replications are genuinely independent
    # rather than the same paths re-labelled.
    bars = bars_from_close(ar1(0.95, 0.02, N_BARS, np.random.default_rng(70_000 + index)))
    reference = CausalZScore(20)

    ref_run = run_vectorized(bars, reference, ZERO_COST, CAPITAL, CONVENTION)
    spec = spec_from_result(ref_run.turnover, ref_run.weights)
    p = flip_probability(spec)

    seed_base = 5_000_000 + index * 100_000
    sharpes, per_obs, turnover, exposure, psr_values = [], [], [], [], []
    kept_returns: list[NDArray[np.float64]] = []

    for i in range(N_NULLS):
        null = RandomSign(len(bars), np.random.default_rng(seed_base + i), p, spec.exposure)
        result = run_vectorized(bars, null, ZERO_COST, CAPITAL, CONVENTION)
        net = result.net_ret[1:]
        per_bar = sharpe(net)
        per_obs.append(per_bar)
        sharpes.append(annualise_sharpe(per_bar))
        turnover.append(realised_turnover_annual(result.turnover))
        exposure.append(realised_exposure(result.weights))
        psr_values.append(psr(net, 0.0))
        if i < N_DSR:
            kept_returns.append(net)

    sr = np.asarray(sharpes)
    trials = np.asarray(per_obs)
    psr_arr = np.asarray(psr_values)
    dsr = np.asarray([deflated_sharpe(r, trials) for r in kept_returns])

    se = float(sr.std(ddof=1) / math.sqrt(len(sr)))
    theory_sd = math.sqrt(BARS_PER_YEAR / (N_BARS - 3))
    ks = kstest(psr_arr, "uniform")

    out: dict[str, float] = {
        "turnover_rel_err": abs(float(np.mean(turnover)) - spec.turnover_annual)
        / spec.turnover_annual,
        "exposure_rel_err": abs(float(np.mean(exposure)) - spec.exposure) / spec.exposure,
        "sr_mean_abs_se": abs(float(sr.mean())) / se,
        "sr_sd_ratio": float(sr.std(ddof=1)) / theory_sd,
        "psr_mean": float(psr_arr.mean()),
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "dsr_max": float(dsr.max()),
        "dsr_frac_survive": float(np.mean(dsr > 0.95)),
        "flip_prob": p,
        "target_turnover": spec.turnover_annual,
        "target_exposure": spec.exposure,
    }
    for alpha in ALPHAS:
        frac = float(np.mean(psr_arr > 1.0 - alpha))
        se_a = math.sqrt(alpha * (1.0 - alpha) / len(psr_arr))
        out[f"rej_{alpha:.2f}"] = frac
        out[f"rej_gap_se_{alpha:.2f}"] = abs(frac - alpha) / se_a

    for label, costs in (("gross", ZERO_COST), ("net20", CostModel(commission_bps=20.0))):
        run = run_vectorized(bars, reference, costs, CAPITAL, CONVENTION)
        observed = annualise_sharpe(sharpe(run.net_ret[1:]))
        out[f"ref_sr_{label}"] = observed
        out[f"ref_p_{label}"] = empirical_p_value(observed, sr)
    return out


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    print(f"G6 replication study: {n} independent worlds x {N_NULLS} nulls x {N_BARS} bars")
    print("=" * 96)

    rows = []
    for i in range(n):
        row = replication(i)
        rows.append(row)
        print(
            f"  rep {i:2d}  p={row['flip_prob']:.5f}  turn_err={row['turnover_rel_err']:.4%}  "
            f"expo_err={row['exposure_rel_err']:.4%}  |SR|={row['sr_mean_abs_se']:.2f}SE  "
            f"sd_ratio={row['sr_sd_ratio']:.4f}  rej05={row['rej_0.05']:.4f}  "
            f"KSp={row['ks_p']:.3f}  dsr_max={row['dsr_max']:.4f}",
            flush=True,
        )

    print("=" * 96)
    print(f"{'statistic':<26} {'min':>10} {'max':>10} {'mean':>10} {'sd':>10}   proposed bound")
    print("-" * 96)

    def summarise(key: str, fmt: str = ".4f", bound: str = "") -> tuple[float, float]:
        values = np.asarray([r[key] for r in rows])
        print(
            f"{key:<26} {values.min():>10{fmt}} {values.max():>10{fmt}} "
            f"{values.mean():>10{fmt}} {values.std(ddof=1):>10{fmt}}   {bound}"
        )
        return float(values.min()), float(values.max())

    _, turn_max = summarise("turnover_rel_err", bound="spec: 5%")
    _, expo_max = summarise("exposure_rel_err", bound="spec: 5%")
    _, sr_se_max = summarise("sr_mean_abs_se", bound="gate: < 3 SE")
    sd_lo, sd_hi = summarise("sr_sd_ratio", bound="gate: 0.8-1.2")
    psr_lo, psr_hi = summarise("psr_mean", bound="gate: 0.45-0.55")
    for alpha in ALPHAS:
        summarise(f"rej_{alpha:.2f}", bound=f"nominal {alpha:.2f}")
        summarise(f"rej_gap_se_{alpha:.2f}", bound="gate: < 3 SE")
    ks_lo, _ = summarise("ks_p", bound="uniformity")
    _, dsr_hi = summarise("dsr_max", bound="gate: frac>0.95 <= 5%")
    summarise("dsr_frac_survive")
    summarise("ref_sr_gross")
    summarise("ref_p_gross")
    summarise("ref_sr_net20")
    summarise("ref_p_net20")

    # ---- verify the bounds the gate actually asserts, world by world.
    # The study sets the thresholds, so it must also show they hold everywhere --
    # otherwise the gate is calibrated on the same run that judges it.
    print("=" * 96)
    print("GATE CHECK -- the bounds in tests/gates/test_g6_null.py, applied to every world")
    print("-" * 96)
    failures = 0
    for i, r in enumerate(rows):
        checks = {
            "turnover<5%": r["turnover_rel_err"] < 0.05,
            "exposure<5%": r["exposure_rel_err"] < 0.05,
            "|mean|/sd<0.25": r["sr_mean_abs_se"] / math.sqrt(N_NULLS) < 0.25,
            "sd_ratio 0.8-1.2": 0.8 < r["sr_sd_ratio"] < 1.2,
            "psr_mean 0.42-0.58": 0.42 < r["psr_mean"] < 0.58,
            "rej10 0.05-0.17": 0.05 <= r["rej_0.10"] <= 0.17,
            "rej05 0.02-0.09": 0.02 <= r["rej_0.05"] <= 0.09,
            "dsr_survive<=5%": r["dsr_frac_survive"] <= 0.05,
        }
        bad = [k for k, ok in checks.items() if not ok]
        failures += bool(bad)
        status = "PASS" if not bad else "FAIL " + ",".join(bad)
        print(f"  world {i:2d}: {status}")
    print("-" * 96)
    print(f"gate holds in {len(rows) - failures}/{len(rows)} worlds")

    print("=" * 96)
    print("MAXIMAL BOUNDS OBSERVED (use these to set the gate, with margin):")
    headroom = 0.05 / max(turn_max, 1e-12)
    print(
        f"  turnover match error   <= {turn_max:.4%}   "
        f"(spec band 5%; headroom {headroom:.0f}x)"
    )
    print(f"  exposure match error   <= {expo_max:.4%}")
    print(f"  |null SR| / SE         <= {sr_se_max:.2f} SE")
    print(f"  SR sd / theory         in [{sd_lo:.4f}, {sd_hi:.4f}]")
    print(f"  PSR(0) mean            in [{psr_lo:.4f}, {psr_hi:.4f}]")
    print(f"  KS p vs uniform        >= {ks_lo:.4f}")
    print(f"  DSR max over nulls     <= {dsr_hi:.4f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
