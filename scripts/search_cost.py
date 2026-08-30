"""What searching costs, in four pictures. Companion to the README's "Why your backtest
is probably wrong" section.

    uv run python scripts/search_cost.py

Offline, seeded, no cache, no network -- so it belongs to `make reproduce` in a way the
SPY figures cannot. Everything here is a property of arithmetic rather than of a dataset,
which is the point: none of it depends on which market you picked.

The four panels answer four objections a reader raises in order.

1. "My best configuration made money."  So would the best of N coin flips. The curve is
   what noise alone hands you.
2. "But my Sharpe was 1.5, that's significant."  It was, as one hypothesis. Panel two
   prices the same 1.5 against the search that produced it.
3. "Then I just need a better strategy."  Or more history than exists. Panel three is the
   sample length the significance you want would actually require.
4. "My win rate is 65%."  Panel four is why that sentence carries no information.

Panel one is Monte Carlo against the closed form rather than the closed form alone. The
Gumbel approximation in `deflated.expected_max_sharpe` is known to overstate for every
N >= 3 (characterised in `test_prop.py`), so drawing it unaccompanied would be asserting
the very thing this repository refuses to assert. The simulation is the check.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t

from falsify.deflated import expected_max_sharpe, min_backtest_length_years, psr
from falsify.metrics import annualise_sharpe, sharpe

FIGURE_PATH = Path("docs/figures/search_cost.png")

BARS_PER_YEAR = 252
YEARS = 4
T = YEARS * BARS_PER_YEAR
SEED = 20
N_PATHS = 20_000  # simulations per point
CHUNK = 2_000  # rows per draw, to keep the working array under ~40 MB

# The grid of "how many configurations did you try", log-spaced because the interesting
# behaviour is over orders of magnitude and nobody tries 337 things.
TRIAL_GRID = (2, 3, 5, 8, 12, 20, 35, 60, 100, 175, 300, 500, 800, 1_400, 2_500)

REPORTED_SHARPE = 1.5  # the number in the pitch deck
THIS_REPO = (24, 0.606)  # N and annualised Sharpe of TimeSeriesMomentum(12m,1m) on SPY
THIS_REPO_YEARS = 9.58  # 2015-01-02 to 2024-12-31, as metrics.json reports it

INK, SLATE, BRICK, MOSS, GREY = "#16202b", "#33526e", "#b0322b", "#2f6b4f", "#5b6a79"


def annualise_all(per_obs: NDArray[np.float64]) -> NDArray[np.float64]:
    """`annualise_sharpe` applied to a whole array.

    B8 keeps exactly one definition of the annualisation rule, so rather than restating
    the sqrt as though it were common knowledge this checks itself against the scalar
    function on every call. If the rule ever changes, this fails loudly instead of
    quietly reporting numbers on the old convention.
    """
    scaled = np.asarray(per_obs * math.sqrt(BARS_PER_YEAR), dtype=np.float64)
    if per_obs.size:
        reference = annualise_sharpe(float(per_obs.flat[0]), BARS_PER_YEAR)
        if not math.isclose(float(scaled.flat[0]), reference, rel_tol=1e-12):
            raise AssertionError(
                f"vectorised annualisation {scaled.flat[0]!r} disagrees with "
                f"annualise_sharpe {reference!r}"
            )
    return scaled


def best_of_n_on_noise(rng: np.random.Generator) -> dict[int, NDArray[np.float64]]:
    """Annualised Sharpe of the best of N pure-noise strategies, for every N in the grid.

    Nothing here has an edge by construction, so whatever comes back is the price of
    looking rather than a property of any market.

    The Sharpe is drawn from its own sampling distribution instead of by simulating
    return paths and measuring them. For T i.i.d. normal returns with zero true mean,
    `sqrt(T) * mean / sd` is exactly Student-t on T-1 degrees of freedom, so the
    per-observation Sharpe is `t / sqrt(T)` -- not an approximation of the path
    simulation but the same distribution written down. Checked against 60,000 simulated
    paths before it was used here: two-sample KS D = 0.0034, p = 0.88, and both match the
    asymptotic sd of 1/sqrt(T) to five decimals. Simulating the paths first was the
    obvious way to write this and it was projected to take over an hour; this takes
    seconds and is exact.

    One draw serves every N, read off as a running maximum. That makes the grid a nested
    search -- the best of 100 includes the best of 20 -- which is what widening a search
    actually looks like, and it removes the sampling noise that independent draws per N
    would put into a curve meant to be read as monotone.
    """
    widest = max(TRIAL_GRID)
    columns = np.array(TRIAL_GRID) - 1
    best = np.empty((N_PATHS, len(TRIAL_GRID)), dtype=np.float64)
    for start in range(0, N_PATHS, CHUNK):
        rows = min(CHUNK, N_PATHS - start)
        draws = student_t.rvs(df=T - 1, size=(rows, widest), random_state=rng) / np.sqrt(T)
        best[start : start + rows] = np.maximum.accumulate(draws, axis=1)[:, columns]
    return {n: annualise_all(best[:, j]) for j, n in enumerate(TRIAL_GRID)}


def returns_with_exact_sharpe(annual: float, rng: np.random.Generator) -> NDArray[np.float64]:
    """A return series whose annualised Sharpe is `annual` to floating point.

    Standardised first, then scaled: the point of the panel is to hold the reported
    number fixed while the search behind it varies, so the number has to be exact rather
    than approximately drawn.
    """
    draws = rng.standard_normal(T)
    draws = (draws - draws.mean()) / draws.std(ddof=1)
    return np.asarray(draws * 0.01 + 0.01 * annual / np.sqrt(BARS_PER_YEAR), dtype=np.float64)


def haircut_curve(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Deflated Sharpe of one fixed result, as the search behind it widens.

    `expected_max_sharpe` is fed the variance of a per-observation Sharpe across noise
    trials, which is 1/T -- the strategies are being compared on the same window, so the
    only spread between them is sampling.
    """
    return np.array([psr(returns, expected_max_sharpe(n, 1.0 / T)) for n in TRIAL_GRID])


def draw(
    simulated: dict[int, NDArray[np.float64]],
    reported: NDArray[np.float64],
    haircut: NDArray[np.float64],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6))
    fig.suptitle(
        "What searching costs\n"
        f"{YEARS} years of daily data, {N_PATHS:,} simulations per point, "
        "no strategy in panel one has any edge",
        fontsize=12.5,
    )
    grid = np.array(TRIAL_GRID, dtype=float)

    # 1. the best of N coin flips
    ax = axes[0, 0]
    means = np.array([simulated[n].mean() for n in TRIAL_GRID])
    lo = np.array([np.quantile(simulated[n], 0.10) for n in TRIAL_GRID])
    hi = np.array([np.quantile(simulated[n], 0.90) for n in TRIAL_GRID])
    theory = np.array([expected_max_sharpe(n, BARS_PER_YEAR / T) for n in TRIAL_GRID])
    ax.fill_between(grid, lo, hi, color=SLATE, alpha=0.18, label="10th-90th percentile")
    ax.plot(grid, means, color=SLATE, lw=1.8, marker="o", ms=3.5, label="simulated mean")
    ax.plot(grid, theory, color=BRICK, lw=1.4, ls="--", label="Gumbel closed form")
    ax.axhline(0.0, color=INK, lw=0.9, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("configurations tried")
    ax.set_ylabel("annualised Sharpe of the best one")
    ax.set_title("1. the best of N strategies that have no edge", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    # 2. the haircut
    ax = axes[0, 1]
    ax.plot(grid, haircut, color=SLATE, lw=1.8, marker="o", ms=3.5)
    ax.axhline(0.95, color=MOSS, lw=1.1, ls=":", label="0.95")
    ax.axhline(0.50, color=BRICK, lw=1.1, ls=":", label="0.50 — a coin flip")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("configurations tried before reporting it")
    ax.set_ylabel("deflated Sharpe")
    ax.set_title(
        f"2. the same annualised Sharpe of {REPORTED_SHARPE}, priced for the search",
        fontsize=10.5,
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25, which="both")

    # 3. minimum backtest length
    ax = axes[1, 0]
    for target, colour in ((0.5, BRICK), (1.0, SLATE), (2.0, MOSS)):
        years = [min_backtest_length_years(n, target) for n in TRIAL_GRID]
        ax.plot(grid, years, color=colour, lw=1.6, label=f"target Sharpe {target}")
    ax.axhline(THIS_REPO_YEARS, color=INK, lw=1.0, ls="--", alpha=0.75)
    needed = min_backtest_length_years(*THIS_REPO)
    ax.plot([THIS_REPO[0]], [needed], marker="*", ms=13, color=INK, zorder=5)
    ax.annotate(
        f"this repo: N={THIS_REPO[0]}, SR {THIS_REPO[1]:.3f}\nneeds {needed:.1f}y, "
        f"has {THIS_REPO_YEARS:.1f}y",
        xy=(THIS_REPO[0], needed),
        xytext=(2.4, 34),
        fontsize=8,
        color=INK,
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.9},
    )
    ax.set_xscale("log")
    ax.set_xlabel("configurations tried")
    ax.set_ylabel("years of daily history required")
    ax.set_title("3. the history that significance would need", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    # 4. hit rate carries no information on its own
    ax = axes[1, 1]
    hit = np.linspace(0.06, 0.94, 400)
    ax.plot(hit * 100, (1.0 - hit) / hit, color=INK, lw=1.8, label="break-even")
    for edge, colour, style in ((1.35, MOSS, "-"), (0.70, BRICK, "--")):
        ax.plot(
            hit * 100,
            edge * (1.0 - hit) / hit,
            color=colour,
            lw=1.3,
            ls=style,
            label=f"{'+' if edge > 1 else ''}{(edge - 1) * 100:.0f}% edge on the payoff",
        )
    for label, p, b in (("trend follower", 0.35, 1.857), ("mean reverter", 0.70, 0.4286)):
        ax.plot([p * 100], [b], marker="o", ms=7, color=SLATE, zorder=5)
        ax.annotate(
            label, xy=(p * 100, b), xytext=(p * 100 + 2, b + 0.45), fontsize=8.5, color=SLATE
        )
    ax.set_yscale("log")
    ax.set_xlabel("hit rate, %")
    ax.set_ylabel("average win ÷ average loss")
    ax.set_title("4. both marked strategies have exactly zero edge", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, which="both")

    for row in axes:
        for cell in row:
            cell.tick_params(labelsize=8)

    fig.tight_layout(rect=(0.0, 0.022, 1.0, 0.945))
    fig.text(
        0.5,
        0.006,
        "Panels 1-3 are properties of arithmetic, not of any market: they hold whatever "
        "you backtest. Panel 4 is why 'my win rate is 65%' is not a claim about edge — "
        "it describes the shape of the payoff, and every point on the black curve breaks "
        "even.",
        ha="center",
        va="bottom",
        fontsize=8,
        alpha=0.75,
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, metadata={"Software": None})
    plt.close(fig)


def main() -> int:
    rng = np.random.default_rng(SEED)
    simulated = best_of_n_on_noise(rng)
    reported = returns_with_exact_sharpe(REPORTED_SHARPE, np.random.default_rng(SEED + 1))
    haircut = haircut_curve(reported)

    print(f"{N_PATHS:,} simulations per point, {YEARS} years of daily data, seed {SEED}")
    print(f"  reported strategy annualised Sharpe: {annualise_sharpe(sharpe(reported)):.4f}")
    print("\n  best-of-N on pure noise, annualised Sharpe:")
    for n in (12, 100, 800, 2_500):
        theory = expected_max_sharpe(n, BARS_PER_YEAR / T)
        print(f"    N={n:>5}  simulated {simulated[n].mean():+.3f}   closed form {theory:+.3f}")
    print(f"\n  a reported {REPORTED_SHARPE} Sharpe, deflated:")
    for n in (2, 100, 800, 2_500):
        print(f"    N={n:>5}  DSR {haircut[TRIAL_GRID.index(n)]:.4f}")
    needed = min_backtest_length_years(*THIS_REPO)
    print(
        f"\n  this repo's own strategy: N={THIS_REPO[0]}, SR {THIS_REPO[1]:.3f} "
        f"-> needs {needed:.1f} years, has {THIS_REPO_YEARS:.1f}"
    )

    draw(simulated, reported, haircut)
    print(f"\nwrote {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
