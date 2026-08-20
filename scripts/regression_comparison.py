"""Regression comparison: session 1's synthetic propositions vs the built pipeline.

    uv run python scripts/regression_comparison.py

Gate 0.0's figure plotted in-sample against out-of-sample Sharpe and drew one OLS
line through each panel. That line answers one question -- "given x, predict y" -- and
a scatter supports three, which is what this figure adds.

    beta_yx = cov(X,Y) / var(X)          regress Y on X   (the original line)
    beta_xy = cov(X,Y) / var(Y)          regress X on Y   (the other line)
    rho     = cov(X,Y) / (sd_X sd_Y)     correlation

bound by  rho^2 = beta_yx * beta_xy,  verified to machine precision in each panel and
printed so a reader can check rather than trust.

Two lines per panel, not one. `beta_yx` is not `1/beta_xy` unless `|rho| = 1`, because
each regression minimises distance along a different axis. The scissor they open is a
direct visual reading of how far the relationship is from deterministic -- shut when
`rho^2 = 1`, right-angled when `rho = 0` -- and they cross at the centroid, the one
point neither regression can get wrong.

The top row is session 1's synthetic construction, unchanged. The bottom row is the
same three regimes measured through everything built since: the certified engine, real
transaction costs, purged walk-forward splits, and the overlay stack. The numbers do
not match the top row, and the figure exists to show where and why.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from falsify.costs import CostModel
from falsify.evaluation import build_grid, walk_forward_select
from falsify.regression import BivariateFit, fit_bivariate
from falsify.selection import ArgMax
from falsify.strategies.base import Strategy
from falsify.strategies.overlays import TurnoverBuffer, VolTarget
from falsify.strategies.simple import CausalZScore, MACrossover
from falsify.synthetic import ar1, bars_from_close, gbm
from falsify.walkforward import ExpandingWindow

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
FIGURE_PATH = REPO_ROOT / "docs" / "figures" / "regression_comparison.png"

MASTER_SEED = 20140458
N_PATHS, T_BARS = 1000, 1000
ANN = float(np.sqrt(252.0))

Series = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Panel:
    title: str
    subtitle: str
    x: Series
    y: Series
    fit: BivariateFit


def sharpe_rows(returns: NDArray[np.float64]) -> tuple[Series, Series]:
    """Annualised in-sample and out-of-sample Sharpe for each row, split at the mid."""
    half = returns.shape[1] // 2
    def sr(block: NDArray[np.float64]) -> Series:
        return np.asarray(
            block.mean(axis=1) / block.std(axis=1, ddof=1) * ANN, dtype=np.float64
        )
    return sr(returns[:, :half]), sr(returns[:, half:])


def child(index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(MASTER_SEED).spawn(12)[index])


def synthetic_panels() -> list[Panel]:
    """Session 1's construction, reproduced exactly."""
    out = []

    r = child(1).standard_normal((N_PATHS, T_BARS))
    x, y = sharpe_rows(r)
    out.append(Panel("Random walk", "memoryless — selection is free", x, y, fit_bivariate(x, y)))

    r = child(3).standard_normal((N_PATHS, T_BARS))
    r = r - r.mean(axis=1, keepdims=True)
    x, y = sharpe_rows(r)
    out.append(
        Panel("Common-mean constraint", "Proposition 3 — exact reversal", x, y, fit_bivariate(x, y))
    )

    rng = child(4)
    phi, sigma = 0.995, 1.0
    lvl = np.empty((N_PATHS, T_BARS + 1))
    lvl[:, 0] = rng.normal(0.0, sigma / np.sqrt(1 - phi**2), N_PATHS)
    eps = rng.normal(0.0, sigma, (N_PATHS, T_BARS))
    for t in range(T_BARS):
        lvl[:, t + 1] = phi * lvl[:, t] + eps[:, t]
    x, y = sharpe_rows(np.diff(lvl, axis=1))
    out.append(
        Panel("Stationary AR(1), φ = 0.995", "Proposition 5 — the winner reverts", x, y,
              fit_bivariate(x, y))
    )
    return out


def engine_panel(title: str, subtitle: str, prices: Series, use_overlays: bool) -> Panel:
    """The same question asked of the built pipeline: engine, costs, purged folds."""
    bars = bars_from_close(prices)
    base = CausalZScore(20)
    strategies: list[Strategy] = [CausalZScore(w) for w in (10, 20, 40, 80)]
    strategies += [MACrossover(5, 20), MACrossover(10, 50)]
    if use_overlays:
        strategies += [
            TurnoverBuffer(base, 0.25),
            VolTarget(base, 0.15, 60),
            TurnoverBuffer(VolTarget(base, 0.15, 60), 0.25),
        ]
    grid = build_grid(bars, strategies, CostModel(commission_bps=10.0))
    splitter = ExpandingWindow(n_splits=10, test_size=80, min_train=200, purge=10)
    result = walk_forward_select(grid, splitter, ArgMax())
    x = result.config_is_sharpe.ravel()
    y = result.config_oos_sharpe.ravel()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    return Panel(title, subtitle, x, y, fit_bivariate(x, y))


def pipeline_panels() -> list[Panel]:
    return [
        engine_panel(
            "Engine on GBM",
            "no config has a true edge",
            gbm(0.0, 0.20, 2000, np.random.default_rng(MASTER_SEED + 11)),
            use_overlays=False,
        ),
        engine_panel(
            "Engine on AR(1), φ = 0.95",
            "configs genuinely differ",
            ar1(0.95, 0.02, 2000, np.random.default_rng(MASTER_SEED + 12)),
            use_overlays=False,
        ),
        engine_panel(
            "AR(1) with the overlay stack",
            "vol target + turnover buffer",
            ar1(0.95, 0.02, 2000, np.random.default_rng(MASTER_SEED + 12)),
            use_overlays=True,
        ),
    ]


def draw(panels: list[Panel]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, slate, brick, amber, moss = "#16202b", "#33526e", "#b0322b", "#c8860d", "#2f6b4f"
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.4))

    for ax, panel in zip(axes.ravel(), panels, strict=True):
        f = panel.fit
        ax.axhline(0.0, color="0.88", lw=0.8, zorder=0)
        ax.axvline(0.0, color="0.88", lw=0.8, zorder=0)
        ax.scatter(panel.x, panel.y, s=8, alpha=0.28, color=slate, linewidths=0, zorder=1)

        span = np.array([panel.x.min(), panel.x.max()])
        ax.plot(span, f.line_yx(span), color=brick, lw=2.0, zorder=3,
                label=f"Y on X:  β$_{{yx}}$ = {f.beta_yx:+.3f}")
        line_xy = f.line_xy(span)
        if np.all(np.isfinite(line_xy)):
            ax.plot(span, line_xy, color=amber, lw=2.0, ls="--", zorder=3,
                    label=f"X on Y:  β$_{{xy}}$ = {f.beta_xy:+.3f}")
        ax.scatter([f.mean_x], [f.mean_y], s=90, marker="+", color=moss, lw=2.2, zorder=4,
                   label="centroid — both lines cross here")

        # Clip to the scatter, not to the lines. When rho is near zero the X-on-Y line
        # is almost vertical, and letting it set the limits squashes the data into an
        # invisible band -- the panel would then be a picture of the axis rather than
        # of the relationship. The line is allowed to run off the frame instead.
        pad = 0.08 * (panel.y.max() - panel.y.min())
        ax.set_ylim(panel.y.min() - pad, panel.y.max() + pad)

        ax.set_title(f"{panel.title}\n{panel.subtitle}", fontsize=11, color=ink)
        ax.set_xlabel("in-sample Sharpe (annualised)", fontsize=9)
        ax.set_ylabel("out-of-sample Sharpe (annualised)", fontsize=9)
        ax.legend(fontsize=7.6, loc="best", framealpha=0.92)
        ax.tick_params(labelsize=8)
        ax.text(
            0.02, 0.02,
            f"ρ = {f.rho:+.4f}    ρ² = {f.rho ** 2:.4f}\n"
            f"β$_{{yx}}$·β$_{{xy}}$ = {f.beta_yx * f.beta_xy:.4f}   "
            f"|Δ| = {f.rho_squared_identity_error():.1e}\n"
            f"angle = {f.angle_between_lines_degrees():.1f}°   n = {f.n:,}",
            transform=ax.transAxes, fontsize=7.4, va="bottom", ha="left", color=ink,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.8"},
        )

    fig.suptitle(
        "Two regressions, one scatter — Gate 0.0's synthetic propositions (top) "
        "against the built pipeline (bottom)",
        fontsize=13.5, color=ink,
    )
    fig.text(
        0.5, 0.008,
        "β$_{yx}$ = cov/var(X) minimises vertical error; β$_{xy}$ = cov/var(Y) minimises "
        "horizontal error; ρ² = β$_{yx}$·β$_{xy}$ exactly. The two lines coincide only when "
        "ρ² = 1 and are perpendicular when ρ = 0.",
        ha="center", fontsize=8.4, color="#5b6a79",
    )
    fig.tight_layout(rect=(0, 0.022, 1, 0.955))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, metadata={"Software": None})
    plt.close(fig)


def main() -> int:
    panels = synthetic_panels() + pipeline_panels()
    draw(panels)

    header = f"{'panel':<34}{'beta_yx':>10}{'beta_xy':>10}{'rho':>9}{'rho^2':>9}{'ident':>10}{'angle':>8}{'n':>8}"
    print(header)
    print("-" * len(header))
    for p in panels:
        f = p.fit
        print(
            f"{p.title:<34}{f.beta_yx:>+10.4f}{f.beta_xy:>+10.4f}{f.rho:>+9.4f}"
            f"{f.rho ** 2:>9.4f}{f.rho_squared_identity_error():>10.1e}"
            f"{f.angle_between_lines_degrees():>7.1f}°{f.n:>8,}"
        )
    print(f"\nwrote {FIGURE_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
