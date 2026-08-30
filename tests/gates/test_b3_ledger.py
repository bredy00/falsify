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

import subprocess
from pathlib import Path

import numpy as np
import pytest

from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST, CostModel
from falsify.evaluation import build_grid
from falsify.ledger import Ledger, LedgerError, Recording, Scope, git_sha
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


# --------------------------------------------------------------------------------------
# `N` reaches the report from the ledger, and survives the process that produced it.
# --------------------------------------------------------------------------------------


def test_b3_n_is_cumulative_across_sessions(bars: object, tmp_path: Path) -> None:
    """Two processes, one file. The second must see the first one's search.

    This is the property that makes a file-backed ledger worth having. A researcher who
    sweeps twelve configurations on Monday and twelve more on Tuesday has searched
    twenty-four times, and a Tuesday report that deflates by twelve is describing a
    search that did not happen. Each `Ledger` here stands for a separate run: they share
    only the path, exactly as two invocations of `scripts/report.py` would.
    """
    path = tmp_path / "trials.jsonl"

    monday = Ledger(path=path, recording=Recording.TRIALS)
    build_grid(bars, [MACrossover(f, 60) for f in (5, 10, 15)], ZERO_COST, ledger=monday)  # type: ignore[arg-type]
    assert monday.n_trials() == 3

    tuesday = Ledger(path=path, recording=Recording.TRIALS)
    build_grid(bars, [MACrossover(f, 90) for f in (5, 10, 15)], ZERO_COST, ledger=tuesday)  # type: ignore[arg-type]

    assert tuesday.n_trials() == 6, (
        "a second run read back only its own trials. `N` that resets per process is `N` "
        "that shrinks every time you restart, and shrinking `N` inflates the deflated "
        "Sharpe -- the exact direction that flatters a result."
    )

    # And a third run that repeats Monday's sweep adds nothing: same code, same data,
    # same params, so the same content addresses.
    wednesday = Ledger(path=path, recording=Recording.TRIALS)
    build_grid(bars, [MACrossover(f, 60) for f in (5, 10, 15)], ZERO_COST, ledger=wednesday)  # type: ignore[arg-type]
    assert wednesday.n_trials() == 6, "re-running a finished search counted it twice"


def test_b3_recording_a_trial_does_not_change_the_code_identity(tmp_path: Path) -> None:
    """The ledger is tracked, so appending to it marks the tree dirty -- and `git_sha`
    must not see that.

    Left alone this is a feedback loop with teeth: a write dirties the tree, the dirty
    tree changes the SHA, the changed SHA is part of `trial_id`, so the next run mints
    fresh ids for the identical configuration. `N` would climb on every run and two runs
    would disagree, taking G10 with them. `git_sha` identifies the state of the code; the
    ledger is output the code produced.
    """
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "code.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "data" / "trials.jsonl").write_text("", encoding="utf-8")

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("add", "-A")
    git("commit", "-qm", "initial")

    git_sha.cache_clear()
    clean = git_sha(repo)
    assert not clean.endswith("-dirty"), "fixture repo did not start clean"

    # A trial lands in the ledger.
    (repo / "data" / "trials.jsonl").write_text('{"trial_id":"x"}\n', encoding="utf-8")
    git_sha.cache_clear()
    assert git_sha(repo) == clean, (
        "recording a trial changed the code identity. Every subsequent run would mint "
        "new trial ids for configurations it had already evaluated."
    )

    # An actual source change still must.
    (repo / "code.py").write_text("x = 2\n", encoding="utf-8")
    git_sha.cache_clear()
    assert git_sha(repo).endswith("-dirty"), (
        "a real source edit no longer marks the tree dirty; the exclusion is too wide"
    )
    git_sha.cache_clear()


def test_b3_a_scoped_count_on_a_none_ledger_refuses_rather_than_answering_zero() -> None:
    """`NONE` keeps no rows, so it cannot filter them -- and 0 is the dangerous answer.

    Returning 0 would under-report `N`, and under-reporting `N` deflates less, which
    makes a result look more significant than the search behind it justifies. Silence in
    the flattering direction is the one failure this project must not have.
    """
    ledger = Ledger.memory(Recording.NONE)
    with pytest.raises(LedgerError, match="retains no rows"):
        ledger.n_trials(Scope(strategies=("MACrossover",)))
    assert ledger.n_trials() == 0  # unscoped still answers


def test_b3_scope_separates_a_cost_sweep_from_the_search(bars: object, tmp_path: Path) -> None:
    """Eight cost levels on one configuration are eight trials and one candidate.

    They belong in the ledger -- they were evaluated. They do not belong in the `N` a
    deflated Sharpe is told, which wants the width of the choice rather than the count of
    the runs.
    """
    ledger = Ledger(path=tmp_path / "trials.jsonl", recording=Recording.TRIALS)
    for bps in (0.0, 1.0, 2.0, 5.0):
        run_vectorized(
            bars,  # type: ignore[arg-type]
            MACrossover(10, 60),
            CostModel(commission_bps=bps),
            CAPITAL,
            "next_open",
            ledger=ledger,
        )
    assert ledger.n_trials() == 4, "the sweep should record every level it evaluated"
    assert ledger.n_trials(Scope(cost_bps=0.0)) == 1, (
        "scoping to the reported cost should leave one candidate, not the whole sweep"
    )
