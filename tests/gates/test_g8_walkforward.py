"""G8 -- walk-forward integrity, and the integration layer built on it.

PLAYBOOK's condition is blunt: *zero index overlap train/test, assertion in code, not
in docs*. `Split` enforces that at construction, so the tests here spend their effort
on the things that are easy to get wrong once overlap is impossible -- the purge
arithmetic at block edges, the embargo direction, and whether the folds actually
tile the period they claim to.

The second half of the file is the integration the gate exists to enable: a grid of
configurations pushed through the certified engine, split by a purged walk-forward,
selected on each training block by a `SelectionRule`, scored only out of sample. That
is the machinery G9 needs, and building it here means CSCV assembles certified parts
rather than inventing them next to its own rank bookkeeping.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from falsify.core.types import Bars
from falsify.costs import ZERO_COST, CostModel
from falsify.evaluation import (
    StrategyGrid,
    build_grid,
    selection_degradation_slope,
    walk_forward_select,
)
from falsify.ledger import Ledger
from falsify.selection import ArgMax, EqualWeight, SelectionRule, Softmax, TopK
from falsify.strategies.base import Strategy
from falsify.strategies.overlays import TurnoverBuffer, VolTarget
from falsify.strategies.simple import CausalZScore, MACrossover
from falsify.synthetic import ar1, bars_from_close, gbm
from falsify.walkforward import (
    ExpandingWindow,
    InsufficientData,
    PurgedKFold,
    RollingWindow,
    Split,
    WalkForwardSplitter,
    purge_and_embargo,
)

# B3: the engines take a ledger, always. In-memory and non-persisting here --
# every invocation is still counted, which is what lets a test assert its own
# search size, but the gate suite does not write to the shipped ledger.
LEDGER = Ledger.memory()

N_OBS = 600
SEED = 8_080

SPLITTERS: tuple[WalkForwardSplitter, ...] = (
    ExpandingWindow(n_splits=5, test_size=60, min_train=120, purge=10),
    RollingWindow(n_splits=5, train_size=200, test_size=60, purge=10),
    PurgedKFold(n_splits=5, purge=10, embargo=0.01),
)

RULES: tuple[SelectionRule, ...] = (ArgMax(), TopK(3), Softmax(1.0), EqualWeight())


# ------------------------------------------------------------------- the gate


@pytest.mark.parametrize("splitter", SPLITTERS, ids=lambda s: s.name)
def test_g8_zero_index_overlap(splitter: WalkForwardSplitter) -> None:
    """The gate's condition, asserted directly on every fold of every splitter."""
    for k, split in enumerate(splitter.split(N_OBS)):
        overlap = np.intersect1d(split.train, split.test)
        assert overlap.size == 0, f"{splitter.name} fold {k} overlaps at {overlap[:5].tolist()}"
        assert split.n_train > 0 and split.n_test > 0
        assert np.all(split.train < N_OBS) and np.all(split.test < N_OBS)
        assert np.all(split.train >= 0) and np.all(split.test >= 0)


@pytest.mark.parametrize("splitter", SPLITTERS, ids=lambda s: s.name)
def test_g8_purge_opens_the_gap_it_promises(splitter: WalkForwardSplitter) -> None:
    """Purge must remove exactly the bars whose labels reach into the test block.

    Checked as an observable gap rather than by re-deriving the arithmetic the
    splitter used, which would only assert that the code equals itself.
    """
    for k, split in enumerate(splitter.split(N_OBS)):
        gap = split.gap_before_test()
        if split.test[0] == 0:
            continue  # no training data precedes the first k-fold block
        assert gap >= splitter.purge, (
            f"{splitter.name} fold {k}: only {gap} bars between the last training bar "
            f"and the test block, but purge={splitter.purge} was requested. Training "
            "labels still overlap the test period."
        )


def test_g8_embargo_removes_bars_after_the_test_block() -> None:
    """The embargo direction, isolated.

    Only `PurgedKFold` trains after a test block, so it is the only splitter where
    this can be observed at all -- and getting the direction backwards there would
    purge the wrong side while still producing non-overlapping, plausible folds.
    """
    without = PurgedKFold(n_splits=5, purge=0, embargo=0.0).split(N_OBS)
    with_embargo = PurgedKFold(n_splits=5, purge=0, embargo=0.05).split(N_OBS)
    n_embargo = int(np.ceil(0.05 * N_OBS))

    for k, (bare, guarded) in enumerate(zip(without, with_embargo, strict=True)):
        hi = int(bare.test[-1])
        after_bare = bare.train[bare.train > hi]
        after_guarded = guarded.train[guarded.train > hi]
        removed = after_bare.size - after_guarded.size
        if after_bare.size == 0:
            continue  # last fold has nothing after it
        assert removed == min(n_embargo, after_bare.size), (
            f"fold {k}: embargo removed {removed} bars after the test block, expected "
            f"{min(n_embargo, after_bare.size)}"
        )
        # And it must not have touched the bars before the block.
        assert np.array_equal(
            bare.train[bare.train < bare.test[0]], guarded.train[guarded.train < guarded.test[0]]
        ), (
            "the embargo removed bars before the test block; that is the purge's job "
            "and the two have been transposed"
        )


@pytest.mark.parametrize("splitter", SPLITTERS, ids=lambda s: s.name)
def test_g8_test_blocks_are_disjoint_and_ordered(splitter: WalkForwardSplitter) -> None:
    """Folds must tile the period without overlapping each other.

    Overlapping test blocks would score the same bar twice and make the stitched
    out-of-sample series double-count it.
    """
    splits = splitter.split(N_OBS)
    seen: list[int] = []
    for split in splits:
        seen.extend(split.test.tolist())
    assert len(seen) == len(set(seen)), "test blocks overlap; a bar is scored twice"
    starts = [int(s.test[0]) for s in splits]
    assert starts == sorted(starts), "folds are not in chronological order"


@pytest.mark.parametrize(
    "splitter",
    [
        ExpandingWindow(n_splits=5, test_size=60, min_train=120, purge=10),
        RollingWindow(n_splits=5, train_size=200, test_size=60, purge=10),
    ],
    ids=lambda s: s.name,
)
def test_g8_sequential_splitters_never_train_on_the_future(
    splitter: WalkForwardSplitter,
) -> None:
    """For a walk-forward, every training bar must precede its test block.

    `PurgedKFold` is deliberately excluded: it trains on both sides by design, which
    is why it is not a walk-forward and must not be read as a simulation of live
    trading.
    """
    for k, split in enumerate(splitter.split(N_OBS)):
        assert int(split.train.max()) < int(split.test[0]), (
            f"{splitter.name} fold {k} trains on bar {split.train.max()} which is at or "
            f"after the test block starting at {split.test[0]}"
        )


def test_g8_expanding_grows_and_rolling_does_not() -> None:
    """The two geometries must actually differ, or one of them is mislabelled."""
    expanding = ExpandingWindow(n_splits=5, test_size=60, min_train=120, purge=10).split(N_OBS)
    rolling = RollingWindow(n_splits=5, train_size=200, test_size=60, purge=10).split(N_OBS)

    exp_sizes = [s.n_train for s in expanding]
    roll_sizes = [s.n_train for s in rolling]
    print(f"expanding train sizes: {exp_sizes}\nrolling train sizes:   {roll_sizes}")
    assert exp_sizes == sorted(exp_sizes) and exp_sizes[-1] > exp_sizes[0], (
        "an expanding window must grow"
    )
    assert len(set(roll_sizes)) == 1, f"a rolling window must not change size: {roll_sizes}"


# ------------------------------------------------------ the gate can fail (F7)


class LeakySplitter(WalkForwardSplitter):
    """A splitter that forgets to purge -- the mistake G8 exists to catch.

    It produces folds that are non-overlapping and entirely plausible. Nothing about
    the resulting equity curve looks wrong; it is simply a little too good, because
    training labels ran into the test block.
    """

    def __init__(self, n_splits: int = 5, test_size: int = 60) -> None:
        super().__init__(purge=0, embargo=0.0)
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, n_obs: int) -> list[Split]:
        start = n_obs - self.n_splits * self.test_size
        out = []
        for k in range(self.n_splits):
            lo = start + k * self.test_size
            test = np.arange(lo, lo + self.test_size, dtype=np.int64)
            out.append(Split(train=np.arange(0, lo, dtype=np.int64), test=test))
        return out


def test_g8_fires_on_a_splitter_that_forgets_to_purge() -> None:
    """A gate that has never failed is not a gate (03 Part F, F7).

    The purge check must reject a splitter whose folds are individually legal. This
    is the failure mode that matters, because an overlapping split is caught by
    `Split` itself while a missing purge is not.
    """
    leaky = LeakySplitter()
    gaps = [s.gap_before_test() for s in leaky.split(N_OBS)]
    print(f"unpurged gaps: {gaps}")
    assert all(gap == 0 for gap in gaps), "the trap is not actually unpurged"

    required = 10
    offenders = [k for k, gap in enumerate(gaps) if gap < required]
    assert offenders, "the purge check would not have flagged the unpurged splitter"


def test_g8_split_refuses_to_construct_when_indices_overlap() -> None:
    """The structural half of the gate: a leaking partition is unconstructable."""
    with pytest.raises(ValueError, match="overlap"):
        Split(train=np.arange(0, 50, dtype=np.int64), test=np.arange(40, 60, dtype=np.int64))
    with pytest.raises(ValueError, match="no training data"):
        Split(train=np.array([], dtype=np.int64), test=np.arange(3, dtype=np.int64))
    with pytest.raises(ValueError, match="strictly increasing"):
        Split(train=np.array([5, 3, 1], dtype=np.int64), test=np.arange(10, 13, dtype=np.int64))


def test_g8_purge_helper_removes_exactly_the_overlapping_window() -> None:
    """The arithmetic, checked directly on a hand-computable case."""
    candidate = np.arange(0, 100, dtype=np.int64)
    test = np.arange(50, 60, dtype=np.int64)
    kept = purge_and_embargo(candidate, test, 100, purge=5, embargo=0.0)
    assert 44 in kept and 45 not in kept, "the purge boundary is off by one"
    assert not set(range(45, 60)).intersection(kept.tolist())
    assert 60 in kept, "purge must not remove bars after the test block; that is the embargo"


# --------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: ExpandingWindow(0, 10, 10), "n_splits"),
        (lambda: ExpandingWindow(2, 0, 10), "test_size"),
        (lambda: RollingWindow(2, 0, 10), "train_size"),
        (lambda: PurgedKFold(1), "n_splits"),
        (lambda: PurgedKFold(3, purge=-1), "purge"),
        (lambda: PurgedKFold(3, embargo=1.5), "embargo"),
    ],
)
def test_g8_malformed_splitters_are_rejected(factory: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("splitter", "n_obs"),
    [
        (ExpandingWindow(n_splits=5, test_size=60, min_train=120, purge=10), 20),
        (RollingWindow(n_splits=5, train_size=200, test_size=60, purge=10), 20),
        # k-fold only becomes impossible once a fold cannot leave a trainable
        # remainder, which is a far smaller n than the sequential splitters need --
        # 20 observations is a perfectly legal 5-fold split.
        (PurgedKFold(n_splits=5, purge=10, embargo=0.01), 8),
    ],
    ids=lambda x: getattr(x, "name", str(x)),
)
def test_g8_too_few_observations_raises_rather_than_shrinking_the_ensemble(
    splitter: WalkForwardSplitter, n_obs: int
) -> None:
    """Silently returning three folds when ten were asked for would make every
    downstream statistic describe a different ensemble than the caller believes."""
    with pytest.raises(InsufficientData):
        splitter.split(n_obs)


# --------------------------------------------------- the integration layer


@pytest.fixture(scope="module")
def bars() -> Bars:
    """AR(1), so the mean-reversion grid has real structure to select among."""
    return bars_from_close(ar1(0.95, 0.02, 1200, np.random.default_rng(SEED)))


@pytest.fixture(scope="module")
def grid(bars: Bars) -> StrategyGrid:
    """A realistic grid: two families, overlays, and a range of speeds.

    Deliberately heterogeneous. A grid of one family at twenty lookbacks is almost
    perfectly correlated and makes selection look easier than it is; mixing families
    and overlays is closer to what a search actually ranges over.
    """
    base = CausalZScore(20)
    strategies: list[Strategy] = [
        CausalZScore(10),
        CausalZScore(20),
        CausalZScore(40),
        CausalZScore(80),
        MACrossover(5, 20),
        MACrossover(10, 50),
        TurnoverBuffer(base, 0.25),
        VolTarget(base, 0.15, 60),
        TurnoverBuffer(VolTarget(base, 0.15, 60), 0.25),
    ]
    return build_grid(bars, strategies, CostModel(commission_bps=10.0), ledger=LEDGER)


def test_g8_grid_is_aligned_and_finite(grid: StrategyGrid) -> None:
    """Every column must describe the same bars, or cross-sectional selection is
    comparing configurations over different periods."""
    print(f"grid: {grid.n_obs} bars x {grid.n_configs} configurations")
    assert grid.n_configs == 9
    assert grid.n_obs > 500
    assert np.all(np.isfinite(grid.returns))
    assert len(grid.names) == grid.n_configs
    assert len(set(grid.names)) == grid.n_configs, "configurations must be distinguishable"


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.name)
def test_g8_walk_forward_selection_runs_end_to_end(rule: SelectionRule, grid: StrategyGrid) -> None:
    """Grid -> engine -> purged split -> rule -> out-of-sample score."""
    splitter = ExpandingWindow(n_splits=6, test_size=80, min_train=200, purge=10)
    result = walk_forward_select(grid, splitter, rule)

    print(
        f"{rule.name:<16} IS {result.is_sharpe.mean():+.4f}  OOS {result.oos_sharpe.mean():+.4f}  "
        f"stitched {result.stitched_sharpe():+.4f}  degradation {result.degradation():+.4f}"
    )
    assert result.n_splits == 6
    assert result.weights.shape == (6, grid.n_configs)
    assert np.allclose(result.weights.sum(axis=1), 1.0, atol=1e-12), "weights must sum to 1"
    assert np.all(result.weights >= 0.0)
    assert np.isfinite(result.stitched_sharpe())
    assert result.oos_returns.size == 6 * 80


def test_g8_the_rule_only_ever_sees_training_data(grid: StrategyGrid) -> None:
    """Structural, not conventional: the rule is handed a block that cannot contain a
    bar it will be scored on, because `Split` refuses to construct otherwise."""
    splitter = ExpandingWindow(n_splits=4, test_size=80, min_train=200, purge=10)
    splits = splitter.split(grid.n_obs)

    seen: list[int] = []

    class Spy(ArgMax):
        """Records the block sizes it is handed, so the claim is observed rather than
        taken on trust."""

        def weights(self, is_returns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            seen.append(len(is_returns))
            return super().weights(is_returns)

    walk_forward_select(grid, splitter, Spy())
    assert seen == [s.n_train for s in splits], (
        "the rule was given blocks that do not match the splitter's training indices"
    )


def test_g8_argmax_selects_the_in_sample_maximum(grid: StrategyGrid) -> None:
    """The one claim about selection that is exact rather than statistical.

    ArgMax puts all weight on the best training configuration, so the rule's
    in-sample Sharpe must equal the maximum column in-sample Sharpe on every fold.
    Verified across 20 seeds while these tests were being written: 20/20, exactly.

    This is asserted where `degradation() > 0` is not, and the difference is worth
    stating. An 80-bar out-of-sample Sharpe carries a standard error of about
    sqrt(252/80) = 1.77, so a six-fold mean carries roughly 0.72 -- and measured over
    20 seeds ArgMax's degradation came out at -0.041 +/- 0.125, indistinguishable
    from zero and negative in 10 of them. Asserting it is positive on a single path
    would be asserting on noise, which is the error this project exists to avoid.
    """
    splitter = ExpandingWindow(n_splits=6, test_size=80, min_train=200, purge=10)
    result = walk_forward_select(grid, splitter, ArgMax())
    best = result.config_is_sharpe.max(axis=1)
    assert np.allclose(result.is_sharpe, best, atol=1e-9), (
        "ArgMax's in-sample Sharpe is not the grid maximum, so it is not selecting the "
        "in-sample winner"
    )
    assert np.all(np.count_nonzero(result.weights, axis=1) == 1), "ArgMax must pick one"


def test_g8_selection_degradation_is_measured_and_reported(grid: StrategyGrid) -> None:
    """Report the price of selection; do not gate on it at this sample size.

    Degradation is the quantity DSR and PBO exist to correct, and it is the natural
    thing to want a gate on. It cannot carry one here: over 20 seeds the paired
    difference between ArgMax and EqualWeight came out at +0.102 +/- 0.136, positive
    in only 11 of 20. A gate on that would fail about half the time for reasons having
    nothing to do with the code.

    So it is measured, printed, and left to G9 -- where CSCV pools 12,870 splits and
    can say what six folds cannot.
    """
    splitter = ExpandingWindow(n_splits=6, test_size=80, min_train=200, purge=10)
    for rule in (ArgMax(), EqualWeight()):
        result = walk_forward_select(grid, splitter, rule)
        print(
            f"{rule.name:<12} IS {result.is_sharpe.mean():+.4f}  "
            f"OOS {result.oos_sharpe.mean():+.4f}  degradation {result.degradation():+.4f}  "
            f"stitched {result.stitched_sharpe():+.4f}"
        )
        assert np.isfinite(result.degradation())


def test_g8_ranking_informs_only_where_a_real_edge_exists(grid: StrategyGrid) -> None:
    """The honest version of the companion's G8 mapping.

    The companion suggests G8 should show in-sample and out-of-sample Sharpes
    anticorrelated across the grid, reproducing the source paper's central finding.
    Measured here that is not what happens on AR(1) -- and it should not be.

    A negative slope is the signature of selecting among configurations with **no true
    differential merit**, where in-sample ranking is ranking noise. This grid runs on a
    stationary mean-reverting process where the configurations genuinely differ: the
    slow z-scores earn a real edge and the trend-followers genuinely lose on it. Ranking
    therefore carries real information, and over 20 seeds the slope came out at
    +0.979 +/- 0.085, positive in 20 of 20.

    The compensation effect belongs to the memoryless case, which
    `test_g8_holds_on_a_memoryless_process` covers: there the slope is +0.215 +/- 0.265
    over 20 seeds, indistinguishable from zero and negative in 8 of them.

    So the slope is reported rather than gated -- one path puts it anywhere from -1.78
    to +2.26 on GBM -- and what is asserted is the claim that survives: a real edge
    survives the walk-forward.
    """
    splitter = ExpandingWindow(n_splits=8, test_size=80, min_train=200, purge=10)
    result = walk_forward_select(grid, splitter, ArgMax())
    slope, stderr = selection_degradation_slope(result)
    stitched = result.stitched_sharpe()
    print(
        f"AR(1): IS->OOS slope {slope:+.4f} +/- {stderr:.4f}  stitched OOS Sharpe {stitched:+.4f}"
    )
    assert np.isfinite(slope) and np.isfinite(stderr)
    assert stitched > 0.0, (
        f"walk-forward selection on a genuinely mean-reverting series produced a stitched "
        f"out-of-sample Sharpe of {stitched:+.4f}. Over 20 seeds the minimum was +0.314, "
        "so a non-positive value here means the edge is not surviving the split at all."
    )


def test_g8_softmax_temperature_moves_between_argmax_and_equal_weight(
    grid: StrategyGrid,
) -> None:
    """The dial `01` Part E2 describes, verified on walk-forward output.

    At low temperature the blend must resemble ArgMax; at high temperature,
    EqualWeight. This is the axis G9's headline figure is plotted against, so it has
    to behave before that figure means anything.
    """
    splitter = ExpandingWindow(n_splits=6, test_size=80, min_train=200, purge=10)
    argmax_w = walk_forward_select(grid, splitter, ArgMax()).weights
    equal_w = walk_forward_select(grid, splitter, EqualWeight()).weights

    cold = walk_forward_select(grid, splitter, Softmax(0.02)).weights
    hot = walk_forward_select(grid, splitter, Softmax(60.0)).weights

    cold_gap = float(np.abs(cold - argmax_w).max())
    hot_gap = float(np.abs(hot - equal_w).max())
    print(f"max |softmax(0.02) - argmax| = {cold_gap:.4f}   |softmax(60) - equal| = {hot_gap:.4f}")
    assert cold_gap < 0.05, "a cold softmax should converge on argmax"
    assert hot_gap < 0.05, "a hot softmax should converge on equal weight"


def test_g8_grid_rejects_malformed_input(bars: Bars) -> None:
    with pytest.raises(ValueError, match="at least one configuration"):
        build_grid(bars, [], ZERO_COST, ledger=LEDGER)
    with pytest.raises(ValueError, match="expected a"):
        StrategyGrid(returns=np.zeros(5), names=("a",), first_bar=0)
    with pytest.raises(ValueError, match="columns but"):
        StrategyGrid(returns=np.zeros((5, 2)), names=("a",), first_bar=0)
    with pytest.raises(ValueError, match="non-finite"):
        StrategyGrid(returns=np.full((5, 1), np.nan), names=("a",), first_bar=0)


def test_g8_walk_forward_is_deterministic(grid: StrategyGrid) -> None:
    """No RNG anywhere in the path (B9), so two runs must agree bitwise."""
    splitter = RollingWindow(n_splits=5, train_size=300, test_size=80, purge=10)
    first = walk_forward_select(grid, splitter, Softmax(1.0))
    second = walk_forward_select(grid, splitter, Softmax(1.0))
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.oos_returns, second.oos_returns)


def test_g8_holds_on_a_memoryless_process() -> None:
    """On GBM there is nothing to select, so selection must not appear to work.

    The complement of the AR(1) case: a walk-forward that reports a real edge where
    none exists is leaking, and this is the cheapest place to notice.
    """
    bars = bars_from_close(gbm(0.0, 0.20, 1200, np.random.default_rng(SEED + 1)))
    strategies: list[Strategy] = [CausalZScore(w) for w in (10, 20, 40, 80)]
    grid = build_grid(bars, strategies, CostModel(commission_bps=10.0), ledger=LEDGER)
    splitter = ExpandingWindow(n_splits=6, test_size=80, min_train=200, purge=10)
    result = walk_forward_select(grid, splitter, ArgMax())

    stitched = result.stitched_sharpe()
    print(f"GBM walk-forward stitched OOS Sharpe: {stitched:+.4f}")
    assert abs(stitched) < 2.0, (
        f"selection found an annualised Sharpe of {stitched:+.4f} on a driftless random "
        "walk, where no edge exists to find"
    )
