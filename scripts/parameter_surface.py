"""In-sample against out-of-sample parameter surfaces. PLAYBOOK Phase 8.

    "Parameter surface: in-sample Sharpe heatmap next to the out-of-sample heatmap for
     the same grid. When the second one is flat noise, you've just shown overfitting
     visually. Best figure in the repo."

    uv run --group data python scripts/parameter_surface.py

The argument the figure makes is not that one panel looks worse than the other. It is
that the *structure* does not survive: the ridge a reader's eye finds in the left panel,
and would naturally read as "the good region of the parameter space", does not appear in
the right one. If the ridge were a property of the strategy it would be in both.

Two things keep this honest rather than rhetorical.

The **shared colour scale** is one. Plotting each panel on its own scale would make the
out-of-sample noise look as structured as the in-sample signal -- same picture, opposite
conclusion -- so both use one symmetric scale and the reader can compare heights rather
than patterns.

The **rank correlation** is the other. "Looks like noise" is an invitation to see what
you expect, so the figure also carries Spearman's rho between the two surfaces. If
selecting on in-sample rank told you anything about out-of-sample rank, that number would
be positive and large. It is printed whatever it is.

Runs on real SPY, so it needs the cache. `make reproduce` deliberately does not hash this
figure: G10 must pass in a clean checkout with no network, and a figure that cannot be
built without a populated cache cannot be part of that guarantee. It is reproducible given
the manifest, which is a weaker claim honestly stated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from falsify.core.types import Bars
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST
from falsify.data.loaders import DEFAULT_CACHE, DEFAULT_MANIFEST, FetchSpec, load
from falsify.ledger import Ledger, Recording
from falsify.metrics import annualise_sharpe, sharpe
from falsify.strategies.momentum import TimeSeriesMomentum

FIGURE_PATH = Path("docs/figures/parameter_surface.png")
SPEC = FetchSpec("SPY", "2015-01-01", "2025-01-01", "total_return")
CAPITAL = 10_000.0

LOOKBACKS = (1, 2, 3, 4, 6, 8, 10, 12, 15, 18)
HOLDS = (1, 2, 3, 6, 9, 12)


def split_sharpes(bars: Bars, ledger: Ledger) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Annualised Sharpe of every configuration, in the first half and the second.

    One split rather than a walk-forward, because the figure's claim is about a single
    pair of surfaces a reader can hold side by side. G8 and G9 make the repeated-split
    version of the same argument with rather more machinery.

    Each configuration is run once over the whole sample and its return series is then
    cut, so both halves come from one engine invocation and one ledger row. Running the
    engine twice on two slices would give a strategy with a 253-bar lookback a different
    warm-up in each half, and the surfaces would not be comparable.
    """
    in_sample = np.full((len(LOOKBACKS), len(HOLDS)), np.nan)
    out_sample = np.full_like(in_sample, np.nan)

    for i, lookback in enumerate(LOOKBACKS):
        for j, hold in enumerate(HOLDS):
            strategy = TimeSeriesMomentum(lookback, hold)
            try:
                result = run_vectorized(
                    bars, strategy, ZERO_COST, CAPITAL, "next_open", ledger=ledger
                )
            except Exception:
                continue
            returns = result.net_ret[1:]
            midpoint = returns.size // 2
            in_sample[i, j] = annualise_sharpe(sharpe(returns[:midpoint]))
            out_sample[i, j] = annualise_sharpe(sharpe(returns[midpoint:]))
    return in_sample, out_sample


def spearman(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Rank correlation between two surfaces, over the cells where both are finite.

    Rank rather than level: the question is whether in-sample ORDERING predicts
    out-of-sample ordering, which is exactly what a reader does when they look at the
    left panel and pick the bright cell.
    """
    both = np.isfinite(a) & np.isfinite(b)
    if int(both.sum()) < 3:
        return float("nan")
    x, y = a[both], b[both]
    rank_x = np.argsort(np.argsort(x)).astype(float)
    rank_y = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def draw(in_sample: NDArray[np.float64], out_sample: NDArray[np.float64], rho: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limit = float(np.nanmax(np.abs(np.concatenate([in_sample.ravel(), out_sample.ravel()]))))
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.9), width_ratios=[1.0, 1.0, 0.85])

    for ax, surface, title in (
        (axes[0], in_sample, "in sample — first half"),
        (axes[1], out_sample, "out of sample — second half"),
    ):
        image = ax.imshow(
            surface, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", origin="lower"
        )
        ax.set_xticks(range(len(HOLDS)), [f"{h}m" for h in HOLDS])
        ax.set_yticks(range(len(LOOKBACKS)), [f"{lb}m" for lb in LOOKBACKS])
        ax.set_xlabel("holding period")
        ax.set_title(title, fontsize=10.5)
        for i in range(surface.shape[0]):
            for j in range(surface.shape[1]):
                if np.isfinite(surface[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{surface[i, j]:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.4,
                        color="#16202b",
                    )
    axes[0].set_ylabel("lookback")
    fig.colorbar(image, ax=axes[1], label="annualised Sharpe", fraction=0.046)

    scatter = axes[2]
    both = np.isfinite(in_sample) & np.isfinite(out_sample)
    scatter.scatter(in_sample[both], out_sample[both], s=26, color="#33526e", alpha=0.75, zorder=3)
    span = [-limit, limit]
    scatter.plot(span, span, ls="--", lw=1.0, color="#5b6a79", alpha=0.7, label="if rank persisted")
    scatter.axhline(0.0, lw=0.9, color="#c9d2dc", zorder=0)
    scatter.axvline(0.0, lw=0.9, color="#c9d2dc", zorder=0)
    scatter.set_xlabel("in-sample Sharpe")
    scatter.set_ylabel("out-of-sample Sharpe")
    scatter.set_title(f"Spearman rho = {rho:+.3f}", fontsize=10.5)
    scatter.legend(fontsize=8, loc="upper left", framealpha=0.9)
    scatter.grid(alpha=0.25)

    fig.suptitle(
        "The ridge does not survive the split\n"
        f"TimeSeriesMomentum on SPY 2015-2024, {int(both.sum())} configurations, "
        "one shared colour scale",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.99))
    fig.text(
        0.5,
        0.005,
        "Both panels share one scale, so heights are comparable rather than patterns. "
        "The scatter is the same data without the spatial layout: if picking the best "
        "in-sample cell told you anything, the points would follow the dashed line.",
        ha="center",
        va="bottom",
        fontsize=7.8,
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
    in_sample, out_sample = split_sharpes(bars, ledger)
    rho = spearman(in_sample, out_sample)
    both = np.isfinite(in_sample) & np.isfinite(out_sample)

    print(f"{int(both.sum())} configurations, {ledger.n_trials()} distinct trials recorded")
    print(f"  in sample     best {np.nanmax(in_sample):+.3f}  mean {np.nanmean(in_sample):+.3f}")
    print(f"  out of sample best {np.nanmax(out_sample):+.3f}  mean {np.nanmean(out_sample):+.3f}")

    best = np.unravel_index(int(np.nanargmax(in_sample)), in_sample.shape)
    print(
        f"  the in-sample winner is ({LOOKBACKS[best[0]]}m, {HOLDS[best[1]]}m) at "
        f"{in_sample[best]:+.3f}; out of sample it earns {out_sample[best]:+.3f}"
    )
    print(f"  Spearman rho between the surfaces: {rho:+.4f}")

    draw(in_sample, out_sample, rho)
    print(f"\nwrote {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
