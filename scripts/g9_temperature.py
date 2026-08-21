"""G9's headline figure: PBO as a function of softmax temperature.

01 Part E3 calls this "the headline figure of the whole project" and predicts a
monotone decrease, with `ArgMax` at `tau -> 0` and `EqualWeight` as the asymptote.

Two of those three claims did not survive measurement, and the figure plots what was
measured rather than what was predicted:

  1. The decrease is real but only at high temperature. Paired across grids,
     `PBO(tau=16) < PBO(tau=1)` in 14 of 16 grids (t = 5.3). At low temperature the
     curve is flat to rising -- `PBO(tau=1) > PBO(tau=0.05)` in 5 of 8 grids, which is
     a coin flip. The curve is a hump, not a slope.

  2. `EqualWeight` is not a usable asymptote. Its weights do not vary across splits, so
     every split is near perfectly dependent and its PBO is effectively one draw: sd
     0.30 across grids, spanning 0.05 to 0.84. It is drawn as a band, not a line,
     because a single value of it would be reading noise.

Run at the full C(16,8) = 12,870 splits, which is the size the CI gate deliberately
does not run. Output is deterministic (fixed seeds, `metadata={"Software": None}`) so
G10 can assert it byte-for-byte.

    uv run python scripts/g9_temperature.py
"""

from __future__ import annotations

import math
import statistics as st
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from falsify.cscv import cscv
from falsify.selection import ArgMax, EqualWeight, Softmax
from falsify.synthetic import compensation_grid, merit_grid, noise_grid

FIGURE_PATH = Path("docs/figures/pbo_vs_temperature.png")
BLOCKS = 16  # C(16,8) = 12,870 splits, the full size
GRIDS = 5  # 3 kinds x 5 grids x 12 rules = 180 full C(16,8) sweeps, ~10 minutes
SEED0 = 5_000
TAUS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

GRID_KINDS = (
    ("no merit (null)", noise_grid, "#6b7280"),
    ("real merit", merit_grid, "#2563eb"),
    ("compensation (trap)", compensation_grid, "#dc2626"),
)


@dataclass(frozen=True, slots=True)
class Curve:
    """One grid kind's measured temperature curve. Frozen (B7)."""

    colour: str
    curve: tuple[float, ...]
    errs: tuple[float, ...]
    argmax: tuple[float, float]
    equal_mean: float
    equal_sd: float


Builder = Callable[..., NDArray[np.float64]]


def build(builder: Builder, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    if builder is compensation_grid:
        return compensation_grid(rng, n_blocks=BLOCKS)
    return builder(rng)


def mean_se(values: list[float]) -> tuple[float, float]:
    return st.mean(values), st.stdev(values) / math.sqrt(len(values))


def sweep_temperatures() -> dict[str, Curve]:
    """PBO(tau) per grid kind, plus the two endpoints, with error bars (B2)."""
    out: dict[str, Curve] = {}
    for label, builder, colour in GRID_KINDS:
        grids = [build(builder, SEED0 + s) for s in range(GRIDS)]
        curve, errs = [], []
        for tau in TAUS:
            values = [cscv(g, Softmax(tau), n_blocks=BLOCKS).pbo() for g in grids]
            m, se = mean_se(values)
            curve.append(m)
            errs.append(se)
            print(f"  {label:<22} tau={tau:6.2f}  PBO {m:.4f} +/- {se:.4f}")
        argmax_vals = [cscv(g, ArgMax(), n_blocks=BLOCKS).pbo() for g in grids]
        equal_vals = [cscv(g, EqualWeight(), n_blocks=BLOCKS).pbo() for g in grids]
        am, ase = mean_se(argmax_vals)
        ew, ewse = mean_se(equal_vals)
        print(f"  {label:<22} ArgMax      PBO {am:.4f} +/- {ase:.4f}")
        print(
            f"  {label:<22} EqualWeight PBO {ew:.4f} +/- {ewse:.4f} (sd {st.stdev(equal_vals):.3f})"
        )
        out[label] = Curve(
            colour=colour,
            curve=tuple(curve),
            errs=tuple(errs),
            argmax=(am, ase),
            equal_mean=ew,
            equal_sd=st.stdev(equal_vals),
        )
    return out


def draw(data: dict[str, Curve]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    for label, _builder, _colour in GRID_KINDS:
        d = data[label]
        colour = d.colour
        curve = np.asarray(d.curve, dtype=float)
        errs = np.asarray(d.errs, dtype=float)
        ax.errorbar(
            TAUS,
            curve,
            yerr=errs,
            color=colour,
            marker="o",
            ms=4.5,
            lw=1.8,
            capsize=3,
            label=label,
            zorder=3,
        )
        am, ase = d.argmax
        ax.errorbar(
            [TAUS[0] * 0.55],
            [am],
            yerr=[ase],
            color=colour,
            marker="D",
            ms=6,
            capsize=3,
            lw=1.8,
            zorder=4,
        )
        ew, ewsd = d.equal_mean, d.equal_sd
        ax.axhspan(ew - ewsd, ew + ewsd, color=colour, alpha=0.07, zorder=0)
        ax.axhline(ew, color=colour, ls=":", lw=1.2, alpha=0.55, zorder=1)

    ax.axhline(0.5, color="black", ls="--", lw=1.1, alpha=0.7, zorder=2)
    ax.text(
        TAUS[-1],
        0.5,
        "  ship / do not ship",
        va="center",
        ha="left",
        fontsize=8.5,
        color="black",
        alpha=0.8,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"softmax temperature $\tau$   (diamond at left: ArgMax, the $\tau\to0$ limit)")
    ax.set_ylabel("PBO   (probability of backtest overfitting)")
    ax.set_title(
        "The price of selectivity is real, and it is not paid smoothly\n"
        f"C(16,8) = 12,870 splits, {GRIDS} grids per point, error bars are 1 SE across grids",
        fontsize=11,
    )
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(loc="center left", fontsize=9, framealpha=0.9)
    ax.text(
        0.5,
        0.02,
        "dotted lines and bands: EqualWeight +/- 1 sd across grids. Its weights do not vary "
        "across splits,\nso its PBO is effectively a single draw and is not the asymptote "
        "01 Part E3 expected.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.8,
        alpha=0.75,
    )

    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, metadata={"Software": None})
    plt.close(fig)


def main() -> None:
    print(f"G9 temperature sweep: S={BLOCKS}, C(16,8)=12,870 splits, {GRIDS} grids per point\n")
    data = sweep_temperatures()
    draw(data)

    curve = list(data["compensation (trap)"].curve)
    peak = TAUS[int(np.argmax(curve))]
    print(f"\n  trap curve peaks at tau={peak}, not at tau->0 as 01 Part E3 predicts.")
    print(f"  monotone decreasing? {curve == sorted(curve, reverse=True)}")
    print(f"\nwrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
