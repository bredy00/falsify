"""G9's headline figure: PBO as a function of softmax temperature.

01 Part E3 calls this "the headline figure of the whole project" and predicts a monotone
decrease, with `ArgMax` at `tau -> 0` and `EqualWeight` as the asymptote. Measured at the
full C(16,8) = 12,870 splits over 5 grids of each kind, none of that is quite right, and
what replaces it is more interesting than what was predicted.

  compensation trap   0.625  0.631  0.655  0.696  0.733  0.695  0.587  0.514  0.477  0.457
  no merit (null)     0.333  0.326  0.317  0.320  0.334  0.363  0.413  0.434  0.442  0.446
  real merit          0.024  0.018  0.008  0.002  0.000  0.000  0.025  0.145  0.257  0.324
  tau =               0.05   0.1    0.25   0.5    1      2      4      8      16     32

Three curves, three different shapes:

  1. On the trap, PBO *rises* from 0.625 to a peak of 0.733 at tau=1 before falling. A
     mild softmax still concentrates on the top few in-sample performers, which on a
     compensation grid are exactly the columns that reverse, while blending them cuts
     the portfolio's volatility and lifts its in-sample Sharpe. Dilution only helps once
     tau is large enough to reach well down the ranking.

  2. On a grid with real merit the curve runs the other way: PBO is ~0 around tau=1 and
     climbs to 0.324 by tau=32. Diluting selection when the differences are real throws
     away the information that made selection worth doing.

  3. So the safe temperature is not 0 and not infinity. It depends on whether the grid
     has genuine differential merit -- which is the thing you do not know in advance and
     the reason PBO has to be measured rather than assumed.

`EqualWeight` cannot serve as the `tau -> infinity` asymptote in any case. Its weights
are identical on every split, so every split is near perfectly dependent and its PBO is
effectively a single draw: sd 0.26 to 0.34 across grids. It is drawn with that sd rather
than a standard error, and excluded from every assertion in the gate.

Output is deterministic (fixed seeds, `metadata={"Software": None}`) so G10 can assert it
byte-for-byte. The measured numbers are cached alongside the figure, so redrawing costs a
second rather than the ~25 minutes the sweep itself takes.

    uv run python scripts/g9_temperature.py
"""

from __future__ import annotations

import json
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
# The sweep is 180 full C(16,8) runs and takes ~25 minutes. Its output is committed so
# the figure can be redrawn in a second, and so the numbers behind it are auditable
# without rerunning them. Delete this file to force a recompute.
DATA_PATH = Path("docs/figures/pbo_vs_temperature.json")
BLOCKS = 16  # C(16,8) = 12,870 splits, the full size
GRIDS = 5  # 3 kinds x 5 grids x 12 rules = 180 full C(16,8) sweeps, ~10 minutes
SEED0 = 5_000
TAUS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
EW_X = 96.0  # where the EqualWeight marker is parked: past the sweep, off the curve

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
        # EqualWeight as a point with its spread, parked past the last temperature.
        # It was drawn as three full-width `axhspan`s and they overlapped into an
        # opaque wash across the whole plot -- a band is the right way to show a wide
        # draw only when there is one of them.
        ax.errorbar(
            [EW_X],
            [d.equal_mean],
            yerr=[d.equal_sd],
            color=colour,
            marker="s",
            ms=5.5,
            capsize=4,
            lw=1.6,
            alpha=0.85,
            zorder=4,
        )

    ax.axhline(0.5, color="black", ls="--", lw=1.1, alpha=0.7, zorder=2)
    ax.text(
        0.3,
        0.5,
        "ship / do not ship",
        va="bottom",
        ha="left",
        fontsize=8.5,
        color="black",
        alpha=0.75,
    )
    ax.set_xscale("log")
    ax.set_xlim(TAUS[0] * 0.30, EW_X * 1.9)
    ax.set_xlabel(r"softmax temperature $\tau$")
    ax.set_ylabel("PBO   (probability of backtest overfitting)")
    ax.set_title(
        "There is no safe temperature -- only one that matches the grid\n"
        f"C(16,8) = 12,870 splits, {GRIDS} grids per point, error bars 1 SE across grids",
        fontsize=11,
    )
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)

    # Label the two endpoint markers on the axis itself rather than in a caption.
    for x, txt in ((TAUS[0] * 0.55, "ArgMax\n" + r"($\tau\to0$)"), (EW_X, "Equal\nweight")):
        ax.annotate(
            txt,
            xy=(x, 1.005),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=7.6,
            alpha=0.7,
            annotation_clip=False,
        )

    fig.text(
        0.5,
        0.015,
        "EqualWeight (squares) is plotted with +/- 1 sd across grids, not 1 SE: its weights are "
        "identical on every split, so its PBO is\neffectively a single draw and it is not the "
        r"$\tau\to\infty$ asymptote 01 Part E3 expected. Diamonds are ArgMax on the same grids.",
        ha="center",
        va="bottom",
        fontsize=7.6,
        alpha=0.75,
    )

    fig.tight_layout(rect=(0.0, 0.075, 1.0, 1.0))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, metadata={"Software": None})
    plt.close(fig)


def to_json(data: dict[str, Curve]) -> str:
    payload = {
        k: {
            "colour": c.colour,
            "curve": list(c.curve),
            "errs": list(c.errs),
            "argmax": list(c.argmax),
            "equal_mean": c.equal_mean,
            "equal_sd": c.equal_sd,
        }
        for k, c in data.items()
    }
    return json.dumps(payload, indent=2) + "\n"


def from_json(raw: str) -> dict[str, Curve]:
    return {
        k: Curve(
            colour=v["colour"],
            curve=tuple(v["curve"]),
            errs=tuple(v["errs"]),
            argmax=(v["argmax"][0], v["argmax"][1]),
            equal_mean=v["equal_mean"],
            equal_sd=v["equal_sd"],
        )
        for k, v in json.loads(raw).items()
    }


def load_or_sweep() -> dict[str, Curve]:
    """Reuse the committed measurements if they are there, otherwise measure.

    The sweep is 180 full C(16,8) runs and takes about 25 minutes. Caching it means the
    figure can be redrawn in a second and, more to the point, that the numbers behind
    the figure are readable in the repository rather than only inferable from a picture.
    """
    if DATA_PATH.exists():
        print(f"loaded {DATA_PATH} -- delete it to recompute the ~25 minute sweep\n")
        return from_json(DATA_PATH.read_text(encoding="utf-8"))
    data = sweep_temperatures()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(to_json(data), encoding="utf-8")
    return data


def main() -> None:
    print(f"G9 temperature sweep: S={BLOCKS}, C(16,8)=12,870 splits, {GRIDS} grids per point\n")
    data = load_or_sweep()
    draw(data)

    curve = list(data["compensation (trap)"].curve)
    peak = TAUS[int(np.argmax(curve))]
    print(f"\n  trap curve peaks at tau={peak}, not at tau->0 as 01 Part E3 predicts.")
    print(f"  monotone decreasing? {curve == sorted(curve, reverse=True)}")
    print(f"\nwrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
