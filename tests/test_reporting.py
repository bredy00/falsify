"""The reporting contract's semantics. 01 Part D.

Part D's whole argument is that a number can be arithmetically fine and still not mean
what it appears to. Two fields carry that: `interpretable`, when the history is shorter
than the search required, and `ships`, which is not the Sharpe being positive.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from falsify.reporting import MetricsReport, write_metrics

MANIFEST = Path("data/MANIFEST.json")


def report(**overrides: object) -> MetricsReport:
    base: dict[str, object] = {
        "sharpe_annual": 1.42,
        "sharpe_ci95": (0.31, 2.48),
        "n_trials_raw": 800,
        "n_trials_effective": 47.0,
        "deflated_sharpe": 0.61,
        "pbo": 0.34,
        "min_backtest_length_years": 4.0,
        "actual_history_years": 10.0,
        "break_even_cost_bps": 11.3,
        "newey_west_t": 1.87,
        "git_sha": "abc123",
        "data_manifest_hash": "d" * 64,
        "n_obs": 2_516,
        "n_trials_source": "test",
        "ci_method": "stationary bootstrap, percentile",
        "ci_n_boot": 1_000,
    }
    return MetricsReport(**(base | overrides))  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Part D's headline case.
# --------------------------------------------------------------------------------------


def test_a_result_shorter_than_its_required_history_is_uninterpretable() -> None:
    """Part D's own example: 10.2 years required against 4.0 years available.

    "When those don't reconcile, the honest headline is that the result is
    uninterpretable at this sample length -- and printing that is the point of the
    project." So the headline leads with that, not with the Sharpe.
    """
    r = report(min_backtest_length_years=10.2, actual_history_years=4.0)
    assert not r.interpretable
    assert not r.ships
    assert r.headline().startswith("UNINTERPRETABLE")
    assert "10.2" in r.headline() and "4.0" in r.headline()


def test_enough_history_is_interpretable() -> None:
    r = report(min_backtest_length_years=4.0, actual_history_years=10.0)
    assert r.interpretable
    assert not r.headline().startswith("UNINTERPRETABLE")


def test_a_losing_strategy_is_interpretable_it_just_loses() -> None:
    """A negative Sharpe needs an infinite backtest to become significant, which is a
    different statement from "the sample is too short". Conflating them would report a
    clear loser as an open question."""
    r = report(
        sharpe_annual=-0.119,
        sharpe_ci95=(-0.75, 0.49),
        min_backtest_length_years=float("inf"),
    )
    assert r.interpretable, "a losing strategy is readable; it says it loses"
    assert not r.ships
    assert r.headline().startswith("NEGATIVE")


# --------------------------------------------------------------------------------------
# Shipping. PLAYBOOK's PBO rule is necessary and not sufficient.
# --------------------------------------------------------------------------------------


def test_a_low_pbo_alone_does_not_ship_a_losing_strategy() -> None:
    """The bug this test exists for was real, and it shipped on the first run.

    A grid of MA-crossover configurations whose best member earned an annualised Sharpe
    of -0.119 came back with PBO = 0.40 and passed a `pbo < 0.5` check. PBO measures
    whether selection was a coin flip, not whether the selected thing makes money: a
    field of uniformly losing configurations can have a fine PBO because the in-sample
    winner keeps landing in the top half out of sample. Top half of a losing field.
    """
    r = report(
        sharpe_annual=-0.119,
        sharpe_ci95=(-0.75, 0.49),
        pbo=0.40,
        min_backtest_length_years=float("inf"),
    )
    assert r.pbo < 0.5, "the premise: this clears PLAYBOOK's stated G9 condition"
    assert not r.ships, "and it must still not ship"


def test_an_interval_containing_zero_does_not_ship() -> None:
    """The strictest of the three conditions and the one most results fail. It is also
    the only one that asks whether there is an edge rather than whether the search was
    honest."""
    assert not report(sharpe_annual=1.42, sharpe_ci95=(-0.10, 2.48)).ships
    assert report(sharpe_ci95=(0.31, 2.48)).ships


def test_a_high_pbo_does_not_ship_however_good_the_sharpe() -> None:
    assert not report(pbo=0.62).ships
    assert not report(pbo=float("nan")).ships


def test_shipping_requires_all_three_conditions() -> None:
    assert report().ships
    assert not report(pbo=0.9).ships
    assert not report(sharpe_annual=1.42, sharpe_ci95=(-0.01, 3.0)).ships
    assert not report(min_backtest_length_years=99.0).ships


# --------------------------------------------------------------------------------------
# The contract refuses to hold things that are not numbers.
# --------------------------------------------------------------------------------------


def test_a_bound_method_is_refused_with_a_message_that_names_the_field() -> None:
    """`CostSweep.break_even_bps` is a method, not a property, so a missing pair of
    parentheses hands this a bound method. Every arithmetic path here is lazy, so
    without this check it travelled all the way to `json.dumps` and failed there naming
    neither the field nor the caller."""

    class Sweep:
        def break_even_bps(self) -> float:
            return 11.3

    with pytest.raises(TypeError, match="break_even_cost_bps must be a number"):
        report(break_even_cost_bps=Sweep().break_even_bps)


def test_effective_trials_cannot_exceed_raw_trials() -> None:
    """The correction can only shrink the count. The other direction would understate
    the deflation, which is the anti-conservative error."""
    with pytest.raises(ValueError, match="exceeds n_trials_raw"):
        report(n_trials_raw=10, n_trials_effective=11.0)


def test_an_inverted_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="inverted"):
        report(sharpe_ci95=(2.48, 0.31))


def test_a_point_estimate_outside_its_own_interval_is_refused() -> None:
    """This caught a nonsense fixture in this very file: a Sharpe of -0.119 carrying an
    interval of [0.31, 2.48], which `ships` then approved because it reads the lower
    bound and the lower bound was positive. The pair cannot arise from one bootstrap."""
    with pytest.raises(ValueError, match="outside its own interval"):
        report(sharpe_annual=-0.119, sharpe_ci95=(0.31, 2.48))


def test_a_report_with_no_trials_is_refused() -> None:
    with pytest.raises(ValueError, match="n_trials_raw"):
        report(n_trials_raw=0)


# --------------------------------------------------------------------------------------
# Serialisation.
# --------------------------------------------------------------------------------------


def test_non_finite_values_become_null_rather_than_invalid_json(tmp_path: Path) -> None:
    """`Infinity` and `NaN` are JavaScript literals; RFC 8259 has neither. `null` is
    chosen over a sentinel number because an infinite required-history means no sample
    length would do, and 1e308 would invite arithmetic on something that is not a
    length."""
    path = tmp_path / "m.json"
    write_metrics(
        path,
        report(
            sharpe_annual=-1.0,
            sharpe_ci95=(-2.0, 0.4),
            min_backtest_length_years=float("inf"),
        ),
    )
    raw = path.read_text(encoding="utf-8")
    assert '"min_backtest_length_years": null' in raw
    assert "Infinity" not in raw


def test_the_json_preserves_part_ds_field_order(tmp_path: Path) -> None:
    """The file should read like the spec it implements, so a reader can hold them side
    by side."""
    path = tmp_path / "m.json"
    write_metrics(path, report())
    keys = [
        line.split('"')[1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith('"')
    ]
    order = [k for k in keys if k in {"sharpe_annual", "n_trials_raw", "pbo", "newey_west_t"}]
    assert order == ["sharpe_annual", "n_trials_raw", "pbo", "newey_west_t"]


def test_the_file_ends_with_a_newline(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    write_metrics(path, report())
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_derived_verdicts_are_serialised_alongside_the_numbers(tmp_path: Path) -> None:
    """`interpretable` and `ships` are the two fields a reader acts on, so they belong
    in the machine-readable record rather than only in a printed headline."""
    payload = report().to_json()
    assert payload["interpretable"] is True
    assert payload["ships"] is True
    assert math.isclose(payload["sharpe_annual"], 1.42)
    assert payload["sharpe_ci95"] == [0.31, 2.48]


def test_the_headline_states_the_verdict_in_words() -> None:
    assert "ships" in report().headline()
    assert "does not ship" in report(pbo=0.8).headline()


def test_np_float_inputs_are_accepted() -> None:
    """`np.float64` is a `float` subclass, so the type guard must not reject the values
    the engine actually produces."""
    r = report(sharpe_annual=np.float64(1.42), pbo=np.float64(0.34))
    assert r.ships
