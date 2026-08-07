"""EXPERIMENTAL -- accumulated error bands instead of point expectations.

Requested for trial. The idea: rather than asserting a statistic equals a set
expected value, pool many independent measurements and require the *accumulated*
quantity to fall inside a band. The argument for it is that a point expectation
invites tuning a run until it hits the point, which is an optimisation move
wearing a validation costume, while a band over pooled evidence cannot be hit by
tuning one number.

This module is ADDITIVE and nothing here replaces an existing assertion. The
exact checks stay exactly where they were -- the closed-form quadrature identity
at diff = 0.00e+00 and the per-N `< 4 SE` bounds both still run and still gate
the build. If the accumulation approach proves more useful we can lean on it
harder; if not, it is deleted and nothing else moves. That is the whole reason to
keep both while the experiment runs.

Why this keeps Monte Carlo central rather than replacing it: every quantity here
is computed *from* sampled measurements. An exact reference tells you where the
truth is; only sampling tells you whether your estimator finds it, and the
accumulated z-statistics below are meaningless without the sampling that
produced them.

The two statistics worth accumulating, both dimensionless so one band covers
every N:

    rms_z   root-mean-square of (measured - reference) / SE.
            Under correct code each z ~ N(0,1), so this pools to 1.0 with
            standard deviation 1/sqrt(2k) over k measurements. It detects bias
            and variance inflation together: a systematically wrong reference
            pushes it up, an overstated SE pulls it down.

    mean_abs_z  mean of |z|. Pools to sqrt(2/pi) = 0.7979 with standard
            deviation sqrt(1 - 2/pi)/sqrt(k). Less sensitive to a single
            outlier than rms_z, so the pair disagreeing is itself informative.

`mean_abs_diff` is also recorded. It is *not* banded, because it carries units
that change with N and with the sampling budget, so any fixed bound on it would
be a statement about the budget rather than about the code. It is reported so a
band can be chosen from evidence.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Under correct code: rms_z -> 1, mean_abs_z -> sqrt(2/pi).
RMS_Z_EXPECTED = 1.0
MEAN_ABS_Z_EXPECTED = math.sqrt(2.0 / math.pi)


@dataclass(frozen=True, slots=True)
class Accumulation:
    """Pooled deviation statistics over k independent measurements. Frozen (B7)."""

    label: str
    count: int
    rms_z: float
    mean_abs_z: float
    max_abs_z: float
    mean_abs_diff: float
    max_abs_diff: float

    def rms_z_band(self, sigmas: float = 3.0) -> tuple[float, float]:
        """Band for rms_z: 1 +/- sigmas/sqrt(2k)."""
        half = sigmas / math.sqrt(2.0 * self.count)
        return (max(0.0, RMS_Z_EXPECTED - half), RMS_Z_EXPECTED + half)

    def mean_abs_z_band(self, sigmas: float = 3.0) -> tuple[float, float]:
        """Band for mean_abs_z: sqrt(2/pi) +/- sigmas*sqrt(1 - 2/pi)/sqrt(k)."""
        half = sigmas * math.sqrt(1.0 - 2.0 / math.pi) / math.sqrt(self.count)
        return (max(0.0, MEAN_ABS_Z_EXPECTED - half), MEAN_ABS_Z_EXPECTED + half)

    def describe(self) -> str:
        rlo, rhi = self.rms_z_band()
        mlo, mhi = self.mean_abs_z_band()
        return (
            f"{self.label}: k={self.count}  "
            f"rms_z={self.rms_z:.4f} in [{rlo:.4f}, {rhi:.4f}]  "
            f"mean|z|={self.mean_abs_z:.4f} in [{mlo:.4f}, {mhi:.4f}]  "
            f"max|z|={self.max_abs_z:.4f}  "
            f"mean|diff|={self.mean_abs_diff:.3e}  max|diff|={self.max_abs_diff:.3e}"
        )


def accumulate(
    label: str,
    measured: Sequence[float] | NDArray[np.float64],
    reference: Sequence[float] | NDArray[np.float64],
    standard_errors: Sequence[float] | NDArray[np.float64],
) -> Accumulation:
    """Pool k measurements into one Accumulation.

    Every standard error must be strictly positive: a zero SE would make z
    infinite and silently dominate the pool, which is the same class of mistake
    as scoring a constant column's Sharpe at zero.
    """
    m = np.asarray(measured, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    if not (m.shape == r.shape == se.shape):
        raise ValueError(f"shape mismatch: measured {m.shape}, reference {r.shape}, se {se.shape}")
    if m.size == 0:
        raise ValueError("nothing to accumulate")
    if not np.all(np.isfinite(se)) or np.any(se <= 0.0):
        raise ValueError("standard errors must all be finite and strictly positive")

    diff = m - r
    z = diff / se
    return Accumulation(
        label=label,
        count=int(m.size),
        rms_z=float(np.sqrt(np.mean(z**2))),
        mean_abs_z=float(np.mean(np.abs(z))),
        max_abs_z=float(np.max(np.abs(z))),
        mean_abs_diff=float(np.mean(np.abs(diff))),
        max_abs_diff=float(np.max(np.abs(diff))),
    )


def assert_within_bands(acc: Accumulation, sigmas: float = 3.0) -> None:
    """Require both pooled statistics to sit inside their analytic bands."""
    rlo, rhi = acc.rms_z_band(sigmas)
    if not rlo <= acc.rms_z <= rhi:
        raise AssertionError(
            f"{acc.label}: accumulated rms_z={acc.rms_z:.4f} outside [{rlo:.4f}, {rhi:.4f}] "
            f"over k={acc.count}. Above the band means bias or an understated SE; "
            f"below means an overstated SE."
        )
    mlo, mhi = acc.mean_abs_z_band(sigmas)
    if not mlo <= acc.mean_abs_z <= mhi:
        raise AssertionError(
            f"{acc.label}: accumulated mean|z|={acc.mean_abs_z:.4f} outside "
            f"[{mlo:.4f}, {mhi:.4f}] over k={acc.count}"
        )


def append_history(path: Path, acc: Accumulation) -> list[dict[str, object]]:
    """Append to an append-only JSON-lines history and return every record.

    Append-only, echoing the trials ledger discipline in B3: a run that came out
    badly stays in the file. The point of accumulating across runs is defeated by
    a history that forgets.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(acc), sort_keys=True) + "\n")

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


__all__ = [
    "MEAN_ABS_Z_EXPECTED",
    "RMS_Z_EXPECTED",
    "Accumulation",
    "accumulate",
    "append_history",
    "assert_within_bands",
]
