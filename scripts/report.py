"""Write `outputs/metrics.json` -- 01 Part D's contract for one strategy.

    uv run python scripts/report.py

Deterministic by construction: fixed seeds (B9), no wall-clock field, and provenance
from the git SHA and the data manifest digest. `make reproduce` runs it twice and
asserts the bytes match, which is the metrics half of G10.

Runs on synthetic GBM rather than the SPY cache so it works in a clean checkout with no
network and no populated cache. That is a deliberate limitation and not a claim: the
numbers below describe a moving-average crossover on a random walk, which is a process
with no edge to find, and the report says so in its own headline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from falsify.analysis import sweep_costs
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST
from falsify.cscv import cscv
from falsify.evaluation import build_grid
from falsify.reporting import build_report, write_metrics
from falsify.selection import ArgMax
from falsify.strategies.simple import MACrossover
from falsify.synthetic import bars_from_close, gbm

OUTPUT = Path("outputs/metrics.json")
MANIFEST = Path("data/MANIFEST.json")

SEED = 4
BOOTSTRAP_SEED = 1
N_BARS = 2_500
BLOCKS = 10
CHOSEN = MACrossover(20, 60)
COST_GRID = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


def main() -> int:
    bars = bars_from_close(gbm(mu=0.08, sigma=0.20, n_bars=N_BARS, rng=np.random.default_rng(SEED)))

    # The full search, so `n_trials_raw` counts what was actually evaluated rather than
    # what was kept. B3's ledger will widen this to every run; today it is this run.
    strategies = [MACrossover(f, s) for f in range(5, 35, 5) for s in range(40, 130, 10) if f < s]
    grid = build_grid(bars, strategies, ZERO_COST)

    result = run_vectorized(bars, CHOSEN, ZERO_COST, 10_000.0, "next_open")
    returns = result.net_ret[max(CHOSEN.lookback + 2, 130) :]

    pbo = cscv(grid.returns, ArgMax(), n_blocks=BLOCKS).pbo()
    sweep = sweep_costs(bars, CHOSEN, COST_GRID)

    report = build_report(
        bars,
        returns,
        grid,
        pbo,
        sweep.break_even_bps(),
        np.random.default_rng(BOOTSTRAP_SEED),
        manifest_path=MANIFEST,
    )
    write_metrics(OUTPUT, report)

    print(f"strategy: {CHOSEN.name} chosen from {grid.n_configs} configurations")
    print(report.headline())
    print(f"\nwrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
