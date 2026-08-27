"""Write `outputs/metrics.json` -- 01 Part D's contract for one strategy.

    uv run python scripts/report.py

Deterministic by construction: fixed seeds (B9), no wall-clock field, and provenance
from the git SHA and the data manifest digest. `make reproduce` runs it twice and
asserts the bytes match, which is the metrics half of G10.

The chosen strategy is `TimeSeriesMomentum(12, 1)` -- Phase 8's, and the one this project
reports on. Buy-and-hold remains the benchmark it has to beat and, on real SPY, does not.

Runs on synthetic GBM rather than the SPY cache so it works in a clean checkout with no
network and no populated cache. That is a deliberate limitation and not a claim, and the
choice of process matters here more than usual: GBM is momentum's NULL. A trend follower
on a random walk has nothing to find, and the report says so in its own headline rather
than being pointed at a process picked to flatter it. The real-data number lives in
`tests/live/test_momentum_live.py`, where it can be checked against a manifest.
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
from falsify.ledger import Ledger
from falsify.reporting import build_report, write_metrics
from falsify.selection import ArgMax
from falsify.strategies.momentum import TimeSeriesMomentum
from falsify.synthetic import bars_from_close, gbm

# B3: the engines take a ledger, always. In-memory and non-persisting here --
# every invocation is still counted, which is what lets a test assert its own
# search size, but the gate suite does not write to the shipped ledger.
LEDGER = Ledger.memory()

OUTPUT = Path("outputs/metrics.json")
MANIFEST = Path("data/MANIFEST.json")

SEED = 4
BOOTSTRAP_SEED = 1
N_BARS = 2_500
BLOCKS = 10
CHOSEN = TimeSeriesMomentum(12, 1)
COST_GRID = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


def main() -> int:
    bars = bars_from_close(gbm(mu=0.08, sigma=0.20, n_bars=N_BARS, rng=np.random.default_rng(SEED)))

    # The full search, so `n_trials_raw` counts what was actually evaluated rather than
    # what was kept. B3's ledger will widen this to every run; today it is this run.
    strategies = [
        TimeSeriesMomentum(lookback, hold)
        for lookback in (3, 6, 9, 12, 15, 18)
        for hold in (1, 2, 3, 6)
    ]
    grid = build_grid(bars, strategies, ZERO_COST, ledger=LEDGER)

    result = run_vectorized(bars, CHOSEN, ZERO_COST, 10_000.0, "next_open", ledger=LEDGER)
    returns = result.net_ret[1:]

    pbo = cscv(grid.returns, ArgMax(), n_blocks=BLOCKS).pbo()
    sweep = sweep_costs(bars, CHOSEN, COST_GRID, ledger=LEDGER)

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
