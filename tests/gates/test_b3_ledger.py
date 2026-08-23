"""B3 -- the trials ledger, enforced.

    B3: "The trials ledger is append-only and machine-written. Never hand-set `N`.
         Never delete a row. A bug fix marks `superseded_by`."

Not a new gate row. 03 Part G is explicit about scope discipline and B3 is already a
named hard invariant; making a stated constraint enforceable does not require minting
G11. The design and its two resolved conflicts are in
`docs/superpowers/specs/2026-08-21-trials-ledger-design.md`.

The two tests that carry the weight are the trap and the idempotency check. The rest
guard the properties those two depend on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST, CostModel
from falsify.evaluation import build_grid
from falsify.ledger import Ledger, Recording, Scope
from falsify.strategies.simple import BuyAndHold, CausalZScore, MACrossover
from falsify.synthetic import bars_from_close, gbm

CAPITAL = 10_000.0


@pytest.fixture(scope="module")
def bars() -> object:
    return bars_from_close(gbm(mu=0.08, sigma=0.20, n_bars=800, rng=np.random.default_rng(3)))


# --------------------------------------------------------------------------------------
# The trap. F7: a gate that has never failed is not a gate.
# --------------------------------------------------------------------------------------


def test_b3_the_ledger_counts_the_search_not_the_winner(bars: object, tmp_path: Path) -> None:
    """A researcher evaluates 50 configurations and reports the best one.

    The ledger must say 50. Reporting the winner and forgetting the search is the entire
    failure this project exists to measure -- it is what makes a deflated Sharpe
    meaningless, because the deflation is by `N` and `N` is the size of the search. This
    is the `N`-side counterpart to G7's leakage trap.

    Human memory is explicitly not an acceptable source for `N` (01 Part C), so the
    ledger is written by the engine rather than by the person, and the person cannot
    forget what they did not record.
    """
    ledger = Ledger(path=tmp_path / "trials.jsonl", recording=Recording.TRIALS)
    strategies = [MACrossover(fast, slow) for fast in range(5, 30, 5) for slow in range(35, 85, 5)]
    assert len(strategies) == 50, "the premise: fifty configurations were evaluated"

    grid = build_grid(bars, strategies, ZERO_COST, ledger=ledger)  # type: ignore[arg-type]
    best = int(np.argmax([np.mean(grid.returns[:, j]) for j in range(grid.n_configs)]))
    print(f"reported: {grid.names[best]}   ledger says N = {ledger.n_trials()}")

    assert ledger.n_trials() == 50, (
        f"the ledger recorded {ledger.n_trials()} trials for a search over 50 "
        "configurations. Deflating by that number would understate the search, which is "
        "the exact failure B3 exists to prevent."
    )


def test_b3_the_trap_can_fail(bars: object, tmp_path: Path) -> None:
    """F7 again: shown failing, not merely passing.

    Without the ledger there is nothing to count but the one result a person chose to
    keep, and `N = 1` is what a search of fifty looks like when nobody wrote it down.
    That is what the test above is defending against, so it is demonstrated here rather
    than asserted to be possible.
    """
    ledger = Ledger(path=tmp_path / "one.jsonl", recording=Recording.TRIALS)
    strategies = [MACrossover(fast, slow) for fast in range(5, 30, 5) for slow in range(35, 85, 5)]
    grid = build_grid(bars, strategies, ZERO_COST, ledger=Ledger.memory())  # type: ignore[arg-type]

    # The "researcher" runs only their winner through the recording ledger.
    best = int(np.argmax([np.mean(grid.returns[:, j]) for j in range(grid.n_configs)]))
    run_vectorized(bars, strategies[best], ZERO_COST, CAPITAL, "next_open", ledger=ledger)  # type: ignore[arg-type]

    assert ledger.n_trials() == 1, "the failure mode: fifty evaluated, one recorded"
    assert len(strategies) == 50


# --------------------------------------------------------------------------------------
# Idempotency. The precondition for G10, and Conflict 2 of the design.
# --------------------------------------------------------------------------------------


def test_b3_rerunning_the_same_search_does_not_grow_n(bars: object, tmp_path: Path) -> None:
    """Run the same sweep twice; `N` must not move.

    01 Part C writes `trial_id` as a `uuid4`, which would fail this: every run would
    append fresh rows, `N` would grow, and `metrics.json` -- which reads `n_trials_raw`
    from here -- would differ between runs, breaking G10 by construction. The id is a
    content hash instead, so a rerun of the same configurations on the same data at the
    same commit is the same set of trials, because it is.
    """
    path = tmp_path / "trials.jsonl"
    strategies = [MACrossover(f, s) for f in (5, 10, 20) for s in (40, 60, 90)]

    first = Ledger(path=path, recording=Recording.TRIALS)
    build_grid(bars, strategies, ZERO_COST, ledger=first)  # type: ignore[arg-type]
    after_one = first.n_trials()

    second = Ledger(path=path, recording=Recording.TRIALS)
    build_grid(bars, strategies, ZERO_COST, ledger=second)  # type: ignore[arg-type]
    after_two = second.n_trials()

    print(f"N after one run: {after_one}, after two: {after_two}")
    assert after_one == len(strategies) == 9
    assert after_two == after_one, (
        f"N grew from {after_one} to {after_two} on a rerun of the same search. That is "
        "the uuid4 failure the content-addressed id exists to avoid, and it would take "
        "G10 red."
    )


def test_b3_repeated_evaluation_of_one_configuration_is_one_trial(bars: object) -> None:
    """Conflict 1 of the design, resolved without an exemption.

    Part C rule 1 says EVERY engine invocation writes a row, and the gate suite invokes
    the engines tens of thousands of times -- G1's tau-test alone recomputes one
    configuration hundreds of times per strategy. Content-addressing means those are one
    distinct trial, which is what they are: one configuration, evaluated. Rule 1 holds
    literally and `N` counts distinct ids.
    """
    ledger = Ledger.memory()
    for _ in range(200):
        run_vectorized(bars, MACrossover(20, 50), ZERO_COST, CAPITAL, "next_open", ledger=ledger)  # type: ignore[arg-type]
    assert ledger.n_trials() == 1, (
        f"200 evaluations of one configuration counted as {ledger.n_trials()} trials. "
        "Rule 1 would then make N a count of engine calls rather than of the search."
    )


# --------------------------------------------------------------------------------------
# What distinguishes one trial from another.
# --------------------------------------------------------------------------------------


def test_b3_different_configurations_are_different_trials(bars: object) -> None:
    ledger = Ledger.memory()
    for strategy in (MACrossover(20, 50), MACrossover(21, 50), CausalZScore(20), BuyAndHold()):
        run_vectorized(bars, strategy, ZERO_COST, CAPITAL, "next_open", ledger=ledger)  # type: ignore[arg-type]
    assert ledger.n_trials() == 4


def test_b3_the_same_strategy_at_a_different_cost_is_a_different_trial(bars: object) -> None:
    """Cost is part of the configuration. A strategy that only survives at zero cost and
    the same strategy at 5 bps are two claims, and a search over both is a search of two."""
    ledger = Ledger.memory()
    for costs in (ZERO_COST, CostModel(commission_bps=5.0), CostModel(commission_bps=10.0)):
        run_vectorized(bars, MACrossover(20, 50), costs, CAPITAL, "next_open", ledger=ledger)  # type: ignore[arg-type]
    assert ledger.n_trials() == 3


def test_b3_the_same_strategy_on_different_data_is_a_different_trial() -> None:
    """`series_digest` carries this. `Bars` has no ticker and the manifest hash covers the
    whole manifest, so without a digest of the prices themselves two instruments over the
    same dates would collide -- and a search across two markets would count as one."""
    ledger = Ledger.memory()
    for seed in (1, 2, 3):
        bars = bars_from_close(gbm(mu=0.08, sigma=0.2, n_bars=400, rng=np.random.default_rng(seed)))
        run_vectorized(bars, MACrossover(20, 50), ZERO_COST, CAPITAL, "next_open", ledger=ledger)
    assert ledger.n_trials() == 3


# --------------------------------------------------------------------------------------
# Append-only. B3 rule 2.
# --------------------------------------------------------------------------------------


def test_b3_supersession_appends_and_never_deletes(bars: object, tmp_path: Path) -> None:
    """A bug fix marks `superseded_by`; it does not remove the row.

    What the project used to believe stays readable. That is the difference between a
    corrected record and a rewritten one, and it is why `records()` and `live()` are
    separate views rather than one filtered accessor.
    """
    path = tmp_path / "trials.jsonl"
    ledger = Ledger(path=path, recording=Recording.TRIALS)
    run_vectorized(bars, MACrossover(20, 50), ZERO_COST, CAPITAL, "next_open", ledger=ledger)  # type: ignore[arg-type]
    original = ledger.live()[0].trial_id

    ledger.supersede(original, by="fixed-in-abc123")

    assert len(ledger.records()) == 2, "supersession must append, not edit"
    assert ledger.records()[0].superseded_by is None, "the original row is untouched"
    assert ledger.records()[1].superseded_by == "fixed-in-abc123"
    assert ledger.n_trials() == 0, "a superseded trial no longer counts toward N"
    assert ledger.live() == (), "and does not appear in the live view"


def test_b3_the_file_only_ever_grows(bars: object, tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    sizes = []
    for strategy in (MACrossover(5, 20), MACrossover(10, 30), CausalZScore(15)):
        ledger = Ledger(path=path, recording=Recording.TRIALS)
        run_vectorized(bars, strategy, ZERO_COST, CAPITAL, "next_open", ledger=ledger)  # type: ignore[arg-type]
        sizes.append(path.stat().st_size)
    assert sizes == sorted(sizes) and len(set(sizes)) == 3


# --------------------------------------------------------------------------------------
# Scope. Part C rule 3's "matching the reporting scope".
# --------------------------------------------------------------------------------------


def test_b3_scope_separates_a_null_study_from_the_search_being_reported(
    bars: object, tmp_path: Path
) -> None:
    """G6 pushes 1,000 random-sign strategies through the pipeline. Those are genuinely
    1,000 distinct trials and belong in the ledger; they are simply not part of the search
    that produced the strategy being reported.

    Scope is what keeps both true at once -- one file holding every trial the project ever
    ran, and a reported `N` covering only the search that produced the reported result.
    Without it the choice would be between an incomplete ledger and an inflated `N`, and
    both are ways of getting the deflation wrong.
    """
    ledger = Ledger(path=tmp_path / "trials.jsonl", recording=Recording.TRIALS)
    for f, s in ((5, 20), (10, 30), (20, 50)):
        run_vectorized(bars, MACrossover(f, s), ZERO_COST, CAPITAL, "next_open", ledger=ledger)  # type: ignore[arg-type]
    for lookback in (10, 20, 30, 40):
        run_vectorized(bars, CausalZScore(lookback), ZERO_COST, CAPITAL, "next_open", ledger=ledger)  # type: ignore[arg-type]

    assert ledger.n_trials() == 7, "every trial is in the file"
    assert ledger.n_trials(Scope(strategies=("MACrossover",))) == 3
    assert ledger.n_trials(Scope(strategies=("CausalZScore",))) == 4
    assert ledger.n_trials(Scope(strategies=("MACrossover", "CausalZScore"))) == 7
    assert ledger.n_trials(Scope(strategies=("BuyAndHold",))) == 0
