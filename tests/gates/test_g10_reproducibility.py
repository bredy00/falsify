"""G10 -- reproducibility. PLAYBOOK: "two runs produce byte-identical figures and
metrics JSON."

The figure half has held since Gate 0.0 and is exercised by `make reproduce`. The metrics
half could not exist until there was a `metrics.json`, which is why G10 sat at "partial"
through nine other gates. This is the half that closes it.

The thing that nearly broke it, recorded because it is not obvious: a wall-clock
timestamp in the report would fail this gate on every run, and a timestamp is the first
field anyone reaches for when writing a provenance block. `falsify.reporting` carries
`git_sha` and `data_manifest_hash` instead -- content-derived, so they change when and
only when the inputs change. This is the same conflict the trials ledger hit with
`uuid4`, resolved the same way (docs/superpowers/specs/2026-08-21-trials-ledger-design.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from falsify.core.types import Bars
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST
from falsify.cscv import cscv
from falsify.evaluation import StrategyGrid, build_grid
from falsify.ledger import Ledger
from falsify.reporting import MetricsReport, build_report, write_metrics
from falsify.selection import ArgMax
from falsify.strategies.simple import MACrossover
from falsify.synthetic import bars_from_close, gbm

# B3: the engines take a ledger, always. In-memory and non-persisting here --
# every invocation is still counted, which is what lets a test assert its own
# search size, but the gate suite does not write to the shipped ledger.
LEDGER = Ledger.memory()

Pieces = tuple[Bars, NDArray[np.float64], StrategyGrid, float]

MANIFEST = Path("data/MANIFEST.json")
N_BOOT = 120

# 01 Part D's contract, verbatim. Every one of these must appear in the file.
PART_D_FIELDS = (
    "sharpe_annual",
    "sharpe_ci95",
    "n_trials_raw",
    "n_trials_effective",
    "deflated_sharpe",
    "pbo",
    "min_backtest_length_years",
    "actual_history_years",
    "break_even_cost_bps",
    "newey_west_t",
)


@pytest.fixture(scope="module")
def pieces() -> Pieces:
    bars = bars_from_close(gbm(mu=0.08, sigma=0.20, n_bars=1_500, rng=np.random.default_rng(4)))
    strategies = [MACrossover(f, s) for f in (5, 10, 20) for s in (40, 60, 90)]
    grid = build_grid(bars, strategies, ZERO_COST, ledger=LEDGER)
    result = run_vectorized(
        bars, MACrossover(20, 60), ZERO_COST, 10_000.0, "next_open", ledger=LEDGER
    )
    returns = result.net_ret[100:]
    pbo = cscv(grid.returns, ArgMax(), n_blocks=8).pbo()
    return bars, returns, grid, pbo


def make_report(pieces: Pieces, seed: int = 1) -> MetricsReport:
    bars, returns, grid, pbo = pieces
    return build_report(
        bars,
        returns,
        grid,
        pbo,
        11.3,
        np.random.default_rng(seed),
        ledger=LEDGER,
        manifest_path=MANIFEST,
        n_boot=N_BOOT,
    )


def test_g10_two_runs_produce_byte_identical_metrics_json(pieces: Pieces, tmp_path: Path) -> None:
    """The gate. Same seeds in, same bytes out.

    Byte comparison rather than field-by-field: the point is that the artefact is
    reproducible, and a difference in float formatting or key order is a difference in
    the file even when every number agrees.
    """
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    write_metrics(first, make_report(pieces))
    write_metrics(second, make_report(pieces))
    assert first.read_bytes() == second.read_bytes(), (
        "two runs from identical inputs produced different metrics.json bytes. Something "
        "non-deterministic entered the report -- a timestamp, an unseeded RNG (B9), or "
        "dict ordering."
    )


def test_g10_the_gate_can_fail(pieces: Pieces, tmp_path: Path) -> None:
    """F7: a gate that has never failed is not a gate.

    Two demonstrations that the byte comparison is actually sensitive -- a different
    bootstrap seed moves the interval, and different returns move everything. If either
    of these produced identical bytes, the test above would be asserting nothing.
    """
    baseline = tmp_path / "base.json"
    write_metrics(baseline, make_report(pieces))

    reseeded = tmp_path / "reseeded.json"
    write_metrics(reseeded, make_report(pieces, seed=999))
    assert reseeded.read_bytes() != baseline.read_bytes(), (
        "changing the bootstrap seed left the file identical; the interval is not "
        "actually coming from the resampler"
    )

    bars, returns, grid, pbo = pieces
    perturbed = tmp_path / "perturbed.json"
    write_metrics(
        perturbed,
        build_report(
            bars,
            returns * 1.01,
            grid,
            pbo,
            11.3,
            np.random.default_rng(1),
            manifest_path=MANIFEST,
            ledger=LEDGER,
            n_boot=N_BOOT,
        ),
    )
    assert perturbed.read_bytes() != baseline.read_bytes()


def test_g10_the_report_carries_no_wall_clock_field(pieces: Pieces) -> None:
    """The trap this gate exists to catch, asserted directly rather than only through
    the byte comparison -- so the failure names the cause instead of leaving someone to
    diff two JSON files and work it out."""
    payload = make_report(pieces).to_json()
    forbidden = ("timestamp", "generated", "created", "fetched", "utc", "date", "time")
    offenders = [k for k in payload if any(word in k.lower() for word in forbidden)]
    assert not offenders, (
        f"{offenders} look like wall-clock fields. G10 requires two runs to be "
        "byte-identical, so provenance has to be content-derived: git_sha and "
        "data_manifest_hash, not the time of day."
    )


def test_g10_provenance_is_content_derived(pieces: Pieces) -> None:
    """`git_sha` and `data_manifest_hash` are the substitute for a timestamp, so they
    have to be present and stable across runs to be worth anything."""
    a, b = make_report(pieces), make_report(pieces)
    assert a.git_sha == b.git_sha and a.data_manifest_hash == b.data_manifest_hash
    assert a.git_sha, "git_sha is empty"
    if a.data_manifest_hash != "unknown":
        assert len(a.data_manifest_hash) == 64, "a sha256 is 64 hex characters"


def test_g10_every_part_d_field_is_present(pieces: Pieces, tmp_path: Path) -> None:
    """01 Part D: "Every performance claim carries all six fields. No exceptions,
    including for results you like." The printed block has ten."""
    path = tmp_path / "metrics.json"
    write_metrics(path, make_report(pieces))
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [f for f in PART_D_FIELDS if f not in payload]
    assert not missing, f"the reporting contract is missing {missing}"
    assert isinstance(payload["sharpe_ci95"], list) and len(payload["sharpe_ci95"]) == 2


def test_g10_the_file_is_valid_rfc_8259_json(pieces: Pieces, tmp_path: Path) -> None:
    """`json.dumps` emits bare `Infinity` and `NaN` by default, and neither is JSON.

    Measured, not hypothetical: the first report this contract produced had
    `min_backtest_length_years: Infinity`, because the best configuration on that grid
    had a negative Sharpe and no sample length makes a negative Sharpe significant. A
    strict parser rejects that file outright.
    """
    path = tmp_path / "metrics.json"
    write_metrics(path, make_report(pieces))
    raw = path.read_text(encoding="utf-8")

    def reject(token: str) -> float:
        raise AssertionError(f"{token!r} is a JavaScript literal, not JSON")

    json.loads(raw, parse_constant=reject)
    assert "Infinity" not in raw and "NaN" not in raw
