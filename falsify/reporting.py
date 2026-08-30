"""The reporting contract. 01 Part D, PLAYBOOK Phase 8, and the second half of G10.

01 Part D: "Every performance claim in the README carries all six fields. No exceptions,
including for results you like." The block it prints has ten, and all ten are here.

The two that matter most are the pair Part D calls out as being in tension:

    "Note the two fields in tension: min_backtest_length_years: 10.2 against
     actual_history_years: 4.0. When those don't reconcile, the honest headline is that
     the result is uninterpretable at this sample length -- and printing that is the
     point of the project."

So `MetricsReport.interpretable` is a field of the report rather than a note in a
docstring, and `headline()` refuses to lead with a Sharpe when it is False.

**Why there is no timestamp in this file.** G10's pass condition is that two runs
produce byte-identical figures *and metrics JSON*. A wall-clock field would break that
on every run, for the same reason the trials ledger cannot use `uuid4` (see
docs/superpowers/specs/2026-08-21-trials-ledger-design.md, Conflict 2). Provenance comes
from `git_sha` and `data_manifest_hash` instead, which are content-derived: they change
when and only when the inputs change, which is what provenance is actually for.

**Where `n_trials_raw` comes from, and its current limit.** B3 says `N` is read from the
trials ledger and never hand-typed. The ledger exists -- `falsify/ledger.py`, gated by
`tests/gates/test_b3_ledger.py` -- but `build_report` does not read from it yet: it takes
`N` from the `StrategyGrid` actually evaluated, machine-derived from the object holding
every trial, never a literal. That satisfies "not hand-typed" but not
"cumulative across runs": today's `N` is the trials in *this* report's scope, and a
search spread over several sessions would undercount. `n_trials_source` records which
regime produced the number so a reader is not left guessing.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from falsify.bootstrap import DEFAULT_N_BOOT, bootstrap_ci
from falsify.core.types import BARS_PER_YEAR, Bars
from falsify.data.manifest import sha256_of
from falsify.deflated import deflated_sharpe, min_backtest_length_years
from falsify.effective import effective_trials
from falsify.evaluation import StrategyGrid
from falsify.metrics import annualise_sharpe, elapsed_years, newey_west_t, sharpe

Series = NDArray[np.float64]

UNKNOWN = "unknown"


def _json_number(value: float) -> float | None:
    """RFC 8259 has no Infinity or NaN. Non-finite becomes null."""
    return float(value) if np.isfinite(value) else None


def git_sha(repo: Path | None = None) -> str:
    """`<sha>` or `<sha>-dirty`. The same convention the ledger spec uses.

    A run on uncommitted code must not be confusable with one on shipped code, so the
    suffix is part of the identity rather than a warning printed somewhere.
    """
    root = repo or Path(__file__).resolve().parent.parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return UNKNOWN
    return f"{sha}-dirty" if dirty else sha


def manifest_hash(manifest_path: Path) -> str:
    """SHA256 of the data manifest -- one value standing for every input series.

    The manifest already records a digest per cached file, so hashing the manifest
    fingerprints the whole data state. If any series is refetched and differs, this
    changes, and G10 fails loudly rather than producing quietly different numbers.
    """
    return sha256_of(manifest_path) if manifest_path.exists() else UNKNOWN


@dataclass(frozen=True, slots=True)
class MetricsReport:
    """01 Part D's contract, with provenance. Frozen (B7).

    Field order is the order Part D prints them, so the JSON reads like the spec.
    """

    sharpe_annual: float
    sharpe_ci95: tuple[float, float]
    n_trials_raw: int
    n_trials_effective: float
    deflated_sharpe: float
    pbo: float
    min_backtest_length_years: float
    actual_history_years: float
    break_even_cost_bps: float
    newey_west_t: float

    # Provenance and honesty, not part of Part D's ten but required by Phase 8 and G10.
    git_sha: str
    data_manifest_hash: str
    n_obs: int
    n_trials_source: str
    ci_method: str
    ci_n_boot: int

    def __post_init__(self) -> None:
        """Refuse a report whose fields are not numbers.

        Not defensive clutter -- it caught a real caller error the first time it ran.
        `CostSweep.break_even_bps` is a method rather than a property, so passing
        `sweep.break_even_bps` without the parentheses hands over a bound method. Every
        arithmetic path here happens to be lazy, so it travelled the whole way to
        `json.dumps` before failing, with a message naming neither the field nor the
        caller. A contract whose entire job is that published numbers are checkable
        should check that they are numbers.
        """
        for name in (
            "sharpe_annual",
            "n_trials_effective",
            "deflated_sharpe",
            "pbo",
            "min_backtest_length_years",
            "actual_history_years",
            "break_even_cost_bps",
            "newey_west_t",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a number, got {type(value).__name__}. "
                    "A bound method here usually means a missing pair of parentheses."
                )
        if len(self.sharpe_ci95) != 2:
            raise ValueError(f"sharpe_ci95 must be a (lo, hi) pair, got {self.sharpe_ci95!r}")
        lo, hi = self.sharpe_ci95
        if np.isfinite(lo) and np.isfinite(hi):
            if lo > hi:
                raise ValueError(f"sharpe_ci95 is inverted: [{lo}, {hi}]")
            # The point estimate must lie inside its own interval. Caught a nonsense
            # test fixture -- a Sharpe of -0.119 carrying an interval of [0.31, 2.48] --
            # which `ships` then approved, because it reads the lower bound and the lower
            # bound was positive. An incoherent pair like that cannot arise from a real
            # bootstrap, so it always means the two were assembled from different runs.
            if np.isfinite(self.sharpe_annual) and not (lo <= self.sharpe_annual <= hi):
                raise ValueError(
                    f"sharpe_annual {self.sharpe_annual:+.4f} lies outside its own "
                    f"interval [{lo:+.4f}, {hi:+.4f}]. The point estimate and the interval "
                    "were not computed from the same returns."
                )
        if self.n_trials_raw < 1:
            raise ValueError(f"n_trials_raw must be at least 1, got {self.n_trials_raw}")
        if self.n_trials_effective > self.n_trials_raw + 1e-9:
            raise ValueError(
                f"n_trials_effective ({self.n_trials_effective:.2f}) exceeds n_trials_raw "
                f"({self.n_trials_raw}). The correction can only ever shrink the count; "
                "the other direction would understate the deflation."
            )

    @property
    def interpretable(self) -> bool:
        """False when the history is shorter than the minimum the search requires.

        Part D's central point. A Sharpe from a four-year backtest that needed ten years
        to be distinguishable is not a weak result, it is an uninterpretable one, and the
        difference matters.
        """
        if self.sharpe_annual <= 0.0:
            # A losing strategy is perfectly interpretable -- it says it loses. "How long
            # a backtest would make this significant" is not a question a non-positive
            # Sharpe poses, so an infinite requirement here is not the same failure as a
            # short sample and must not be reported as one.
            return True
        required = self.min_backtest_length_years
        if not np.isfinite(required):
            return False
        return self.actual_history_years >= required

    @property
    def ships(self) -> bool:
        """Three conditions, and PLAYBOOK's PBO rule is only one of them.

        PLAYBOOK states G9 as "ship only if PBO < 0.5". That is necessary and nowhere
        near sufficient, which the first run of this contract demonstrated: a grid whose
        best configuration earned an annualised Sharpe of -0.119 came back with
        PBO = 0.40 and would have shipped. PBO measures whether selection was a coin
        flip, not whether the thing selected makes money -- a grid of uniformly losing
        configurations can have a perfectly respectable PBO because the in-sample
        winner keeps landing in the top half out of sample. Top half of a losing field.

        So shipping also requires the result to be readable at this sample length, and
        the 95% interval to exclude zero. That last is the strictest of the three and
        the one most results fail; it is also the only one that asks whether there is
        an edge rather than whether the search was honest.
        """
        lo, _ = self.sharpe_ci95
        return (
            self.interpretable
            and np.isfinite(self.pbo)
            and self.pbo < 0.5
            and np.isfinite(lo)
            and lo > 0.0
        )

    def headline(self) -> str:
        """The sentence that goes at the top. It is not always the Sharpe."""
        if self.sharpe_annual <= 0.0:
            return (
                f"NEGATIVE at {self.sharpe_annual:+.3f} annualised over "
                f"{self.actual_history_years:.1f} years and {self.n_trials_raw} trials "
                f"({self.n_trials_effective:.1f} effective). There is nothing to deflate: "
                "the best configuration found did not beat zero."
            )
        if not self.interpretable:
            return (
                f"UNINTERPRETABLE at this sample length: {self.actual_history_years:.1f} years "
                f"of history against the {self.min_backtest_length_years:.1f} years a search "
                f"over {self.n_trials_raw} trials would need. The Sharpe of "
                f"{self.sharpe_annual:+.3f} is reported below, but it cannot be "
                "distinguished from selection luck on this sample."
            )
        lo, hi = self.sharpe_ci95
        verdict = "ships" if self.ships else "does not ship"
        return (
            f"SR {self.sharpe_annual:+.3f} [{lo:+.3f}, {hi:+.3f}] 95%, "
            f"DSR {self.deflated_sharpe:.3f}, PBO {self.pbo:.3f} over "
            f"{self.n_trials_raw} trials ({self.n_trials_effective:.1f} effective) -- {verdict}"
        )

    def to_json(self) -> dict[str, Any]:
        """JSON-safe payload. Non-finite values become `null`.

        `json.dumps` emits bare `Infinity` and `NaN` by default. Those are JavaScript
        literals, not JSON -- RFC 8259 has no such tokens, and a strict parser rejects
        the file. This project publishes `metrics.json` as the machine-readable record
        of its own results, so a file that only some parsers accept is not acceptable.

        `null` rather than a sentinel number: an infinite `min_backtest_length_years`
        means no sample length would do, and encoding that as 1e308 would invite
        arithmetic on a value that is not a length.
        """
        payload: dict[str, Any] = dict(asdict(self))
        payload["sharpe_ci95"] = [_json_number(v) for v in self.sharpe_ci95]
        for key, value in list(payload.items()):
            if isinstance(value, float):
                payload[key] = _json_number(value)
        payload["interpretable"] = self.interpretable
        payload["ships"] = self.ships
        return payload


def write_metrics(path: Path, report: MetricsReport) -> None:
    """Write `metrics.json`, deterministically.

    `sort_keys=False` so the file reads in Part D's order, a trailing newline so it is a
    well-formed text file, and no timestamp anywhere -- see the module docstring. Two
    runs from the same inputs produce identical bytes, which is what G10 asserts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_json(), indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


def build_report(
    bars: Bars,
    returns: Series,
    grid: StrategyGrid,
    pbo: float,
    break_even_cost_bps: float,
    rng: np.random.Generator,
    *,
    manifest_path: Path,
    n_boot: int = DEFAULT_N_BOOT,
    bars_per_year: int = BARS_PER_YEAR,
    repo: Path | None = None,
) -> MetricsReport:
    """Assemble Part D's contract from the certified components.

    `sharpe_ci95` comes from the stationary bootstrap rather than from `sharpe_se`. Both
    exist and they answer different questions: the analytic SE assumes independent
    observations, and 01 B1 is explicit that strategy returns are not. Where they
    disagree the bootstrap is the one to report, so it is the one wired in here.

    `pbo` and `break_even_cost_bps` are passed in rather than computed. A full CSCV sweep
    and a cost sweep are both expensive and both have their own parameters; making this
    function run them would bury those choices inside a reporting call.
    """
    annual = annualise_sharpe(sharpe(returns), bars_per_year)
    ci = bootstrap_ci(
        returns,
        lambda r: annualise_sharpe(sharpe(r), bars_per_year),
        rng,
        n_boot=n_boot,
    )
    eff = effective_trials(grid.returns)
    trial_sharpes = np.array(
        [
            annualise_sharpe(sharpe(grid.returns[:, j]), bars_per_year)
            for j in range(grid.n_configs)
        ],
        dtype=np.float64,
    )
    return MetricsReport(
        sharpe_annual=annual,
        sharpe_ci95=(ci.lo, ci.hi),
        n_trials_raw=grid.n_configs,
        n_trials_effective=eff.reportable,
        deflated_sharpe=deflated_sharpe(returns, trial_sharpes),
        pbo=pbo,
        # Only defined for a positive target: no sample length makes a non-positive
        # Sharpe significant, so the requirement is infinite rather than an error.
        min_backtest_length_years=(
            min_backtest_length_years(grid.n_configs, annual) if annual > 0.0 else float("inf")
        ),
        actual_history_years=elapsed_years(bars.ts),
        break_even_cost_bps=break_even_cost_bps,
        newey_west_t=newey_west_t(returns),
        git_sha=git_sha(repo),
        data_manifest_hash=manifest_hash(manifest_path),
        n_obs=int(returns.size),
        n_trials_source="StrategyGrid (not the B3 ledger; scope is this run only)",
        ci_method="stationary bootstrap, percentile",
        ci_n_boot=n_boot,
    )


__all__ = [
    "UNKNOWN",
    "MetricsReport",
    "build_report",
    "git_sha",
    "manifest_hash",
    "write_metrics",
]
