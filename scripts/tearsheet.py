"""The tearsheet. PLAYBOOK Phase 8.

    "Tearsheet: equity curve (log scale) vs benchmark, drawdown underwater plot, rolling
     Sharpe, monthly return heatmap, turnover. Break-even cost curve: net Sharpe vs cost
     bps."

    uv run --group data python scripts/tearsheet.py

Six panels, and the benchmark is in as many of them as it fits. A tearsheet showing only
the strategy invites the reader to judge a curve that goes up against nothing, and every
curve here goes up -- the decade did. What the buy-and-hold line does is take that away:
the question stops being "did it make money" and becomes "did it beat the thing you could
have bought instead", which on this window it does not.

The equity panel is log-scaled because a linear axis on a decade of compounding devotes
most of its height to the last two years, and a drawdown early in the sample becomes
invisible. The underwater panel exists for the same reason from the other direction.

Runs on real SPY, so it needs the cache. Not part of `make reproduce`, which must pass in
a clean checkout with no network -- see `parameter_surface.py` for the same note.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from falsify.analysis import sweep_costs
from falsify.core.types import Bars, Result
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST
from falsify.data.loaders import DEFAULT_CACHE, DEFAULT_MANIFEST, FetchSpec, load
from falsify.ledger import Ledger, Recording
from falsify.metrics import annualise_sharpe, max_drawdown, newey_west_t, sharpe, sharpe_se
from falsify.strategies.momentum import TimeSeriesMomentum
from falsify.strategies.simple import BuyAndHold

FIGURE_PATH = Path("docs/figures/tearsheet.png")
SPEC = FetchSpec("SPY", "2015-01-01", "2025-01-01", "total_return")
CAPITAL = 10_000.0
CHOSEN = TimeSeriesMomentum(12, 1)
# Out to 900 bps, which looks absurd until you notice the strategy turns over 1.56 times
# a year. The first grid stopped at 100 and `break_even_bps()` returned NaN because the
# curve had not crossed zero -- correct behaviour, uninformative figure. A monthly-held
# position is very nearly cost-insensitive, and showing where it finally breaks is the
# honest way to say so.
COST_GRID = (0.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 300.0, 450.0, 600.0, 750.0, 900.0)

INK, SLATE, BRICK, MOSS, GREY = "#16202b", "#33526e", "#b0322b", "#2f6b4f", "#5b6a79"


def underwater(equity: NDArray[np.float64]) -> NDArray[np.float64]:
    """Drawdown from the running peak, as a non-positive fraction."""
    return np.asarray(equity / np.maximum.accumulate(equity) - 1.0, dtype=np.float64)


def rolling_sharpe(returns: NDArray[np.float64], window: int = 252) -> NDArray[np.float64]:
    """Trailing annualised Sharpe. NaN until the window fills -- a partial window is a
    different statistic, not an early estimate of this one."""
    out = np.full(returns.size, np.nan)
    for t in range(window, returns.size + 1):
        out[t - 1] = annualise_sharpe(sharpe(returns[t - window : t]))
    return out


def monthly_table(ts: NDArray[np.datetime64], returns: NDArray[np.float64]) -> pd.DataFrame:
    """Compound daily returns into calendar months, years down and months across.

    Compounded rather than summed: a month is a product of daily gross returns, and
    adding log-approximations would quietly understate a volatile month.
    """
    frame = pd.DataFrame({"r": returns}, index=pd.DatetimeIndex(ts))
    monthly = frame["r"].add(1.0).groupby([frame.index.year, frame.index.month]).prod().sub(1.0)
    table = monthly.unstack(level=-1)
    return table.reindex(columns=range(1, 13))


def draw(
    bars: Bars,
    strategy_run: Result,
    benchmark_run: Result,
    sweep: object,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = bars.ts[len(bars) - len(strategy_run) :]
    returns = strategy_run.net_ret[1:]
    bench_returns = benchmark_run.net_ret[1:]

    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.0))
    fig.suptitle(
        f"{CHOSEN.name} on SPY, 2015-2024, zero cost unless stated\n"
        f"SR {annualise_sharpe(sharpe(returns)):+.3f} +/- {sharpe_se(returns, 252):.3f}   "
        f"HAC t {newey_west_t(returns):+.2f}   "
        f"max drawdown {max_drawdown(strategy_run.equity):.1%}   "
        f"benchmark SR {annualise_sharpe(sharpe(bench_returns)):+.3f}",
        fontsize=12,
    )

    # 1. equity, log scale, against the thing you could have bought instead
    ax = axes[0, 0]
    ax.plot(ts, strategy_run.equity, color=SLATE, lw=1.6, label=CHOSEN.name)
    ax.plot(ts, benchmark_run.equity, color=GREY, lw=1.3, ls="--", label="buy and hold")
    ax.set_yscale("log")
    ax.set_title("equity, log scale", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    # 2. underwater
    ax = axes[0, 1]
    ax.fill_between(ts, underwater(strategy_run.equity) * 100, 0.0, color=BRICK, alpha=0.35)
    ax.plot(ts, underwater(benchmark_run.equity) * 100, color=GREY, lw=1.0, ls="--")
    ax.set_title("drawdown from peak, %", fontsize=10.5)
    ax.grid(alpha=0.25)

    # 3. rolling one-year Sharpe
    ax = axes[1, 0]
    ax.plot(ts[1:], rolling_sharpe(returns), color=SLATE, lw=1.3)
    ax.plot(ts[1:], rolling_sharpe(bench_returns), color=GREY, lw=1.0, ls="--")
    ax.axhline(0.0, color=INK, lw=0.9, alpha=0.6)
    ax.set_title("rolling 1-year Sharpe", fontsize=10.5)
    ax.grid(alpha=0.25)

    # 4. monthly returns
    ax = axes[1, 1]
    table = monthly_table(ts[1:], returns)
    limit = float(np.nanmax(np.abs(table.to_numpy(dtype=float))))
    image = ax.imshow(
        table.to_numpy(dtype=float) * 100,
        cmap="RdBu_r",
        vmin=-limit * 100,
        vmax=limit * 100,
        aspect="auto",
    )
    ax.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_yticks(range(len(table.index)), [str(y) for y in table.index])
    ax.set_title("monthly return, %", fontsize=10.5)
    fig.colorbar(image, ax=ax, fraction=0.046)

    # 5. turnover
    ax = axes[2, 0]
    ax.plot(ts, np.cumsum(strategy_run.turnover), color=MOSS, lw=1.5)
    annual = float(np.sum(strategy_run.turnover) / len(strategy_run) * 252)
    ax.set_title(f"cumulative turnover — {annual:.2f} turns a year", fontsize=10.5)
    ax.grid(alpha=0.25)

    # 6. break-even cost
    ax = axes[2, 1]
    bps = np.asarray(sweep.bps)  # type: ignore[attr-defined]
    net = np.asarray(sweep.sharpe_annual)  # type: ignore[attr-defined]
    ax.plot(bps, net, color=SLATE, marker="o", ms=4, lw=1.5)
    ax.axhline(0.0, color=INK, lw=0.9, alpha=0.6)
    break_even = sweep.break_even_bps()  # type: ignore[attr-defined]
    if np.isfinite(break_even):
        ax.axvline(break_even, color=BRICK, ls="--", lw=1.2)
        ax.annotate(
            f"break-even {break_even:.0f} bps",
            xy=(break_even, 0.0),
            xytext=(break_even * 0.55, float(np.nanmax(net)) * 0.45),
            fontsize=8.5,
            color=BRICK,
        )
    ax.set_xlabel("round-trip cost, bps")
    ax.set_title("net Sharpe against cost", fontsize=10.5)
    ax.grid(alpha=0.25)

    for row in axes:
        for ax in row:
            ax.tick_params(labelsize=8)

    fig.tight_layout(rect=(0.0, 0.015, 1.0, 0.96))
    fig.text(
        0.5,
        0.004,
        "The dashed grey line is buy-and-hold in every panel it fits. It beats the strategy "
        "on Sharpe over this window, and saying so is the point of putting it there.",
        ha="center",
        va="bottom",
        fontsize=8,
        alpha=0.75,
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, metadata={"Software": None})
    plt.close(fig)


def main() -> int:
    try:
        bars = load(SPEC, cache_dir=DEFAULT_CACHE, manifest_path=DEFAULT_MANIFEST)
    except Exception as exc:
        print(f"cache unavailable ({type(exc).__name__}); run scripts/fetch_data.py")
        return 1

    ledger = Ledger.memory(Recording.NONE)
    strategy_run = run_vectorized(bars, CHOSEN, ZERO_COST, CAPITAL, "next_open", ledger=ledger)
    benchmark_run = run_vectorized(
        bars, BuyAndHold(), ZERO_COST, CAPITAL, "next_open", ledger=ledger
    )

    # The benchmark must cover the same window or the curves start from different bases.
    offset = len(benchmark_run) - len(strategy_run)
    if offset > 0:
        benchmark_run = run_vectorized(
            bars.slice(offset, len(bars)),
            BuyAndHold(),
            ZERO_COST,
            CAPITAL,
            "next_open",
            ledger=ledger,
        )

    sweep = sweep_costs(bars, CHOSEN, COST_GRID, ledger=ledger)

    returns = strategy_run.net_ret[1:]
    print(f"{CHOSEN.name} on SPY, {len(strategy_run)} reported bars")
    print(
        f"  Sharpe            {annualise_sharpe(sharpe(returns)):+.4f}"
        f" +/- {sharpe_se(returns, 252):.4f}"
    )
    print(f"  HAC t             {newey_west_t(returns):+.3f}")
    print(
        f"  max drawdown      {max_drawdown(strategy_run.equity):.2%}   "
        f"benchmark {max_drawdown(benchmark_run.equity):.2%}"
    )
    print(f"  turnover          {np.sum(strategy_run.turnover) / len(strategy_run) * 252:.2f}/yr")
    print(f"  break-even cost   {sweep.break_even_bps():.1f} bps   monotone {sweep.is_monotone()}")
    print(f"  benchmark Sharpe  {annualise_sharpe(sharpe(benchmark_run.net_ret[1:])):+.4f}")

    draw(bars, strategy_run, benchmark_run, sweep)
    print(f"\nwrote {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
