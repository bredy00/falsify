"""G8 -- purged, embargoed walk-forward. Specified by PLAYBOOK 5d, AFML ch. 7.

Walk-forward is the honest way to manufacture a pseudo-ensemble out of one price
path: slice it into quasi-independent windows, fit on each, and only ever score on
data the fit never saw. The two refinements that make it trustworthy rather than
merely conventional are purging and embargo, and both exist because a train/test
split on a time series is not a split on rows.

**Purge.** A decision made at bar `i` is not resolved at bar `i`. If it takes `h`
bars to play out, its label spans `[i, i+h]`, so any training bar within `h` of the
test block has a label overlapping the test period. Keeping it leaks the test set
into the fit through the *label* rather than through the feature -- which no amount
of lagging the signal will catch, because the feature really is causal. Purging drops
exactly those bars.

**Embargo.** Serial correlation runs the other way too. Training bars immediately
after a test block are contaminated by it, because the test block's own labels extend
forward into them. This only matters when training data follows a test block, which
for the sequential splitters here never happens and for `PurgedKFold` always does --
which is why the k-fold variant needs it and the walk-forward variants keep it
available but usually inert.

The gate's condition is index overlap asserted **in code, not in prose** (PLAYBOOK
G8). `Split` refuses to construct when train and test intersect, so a splitter cannot
return a leaking partition even by accident: the failure happens at construction
rather than silently downstream inside a Sharpe.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Index = NDArray[np.int64]


class InsufficientData(ValueError):
    """Not enough observations for the requested split geometry.

    Raised rather than quietly returning fewer folds than asked for: a caller that
    requested ten and silently received three would compute statistics over a
    different ensemble than the one it believes it has.
    """


@dataclass(frozen=True, slots=True)
class Split:
    """One train/test partition. Frozen (B7), validated on construction.

    The no-overlap check lives here rather than in a test because it is the gate's
    actual requirement. A splitter with an off-by-one at a block edge produces a
    result that looks entirely normal -- slightly too good -- and nothing raises.
    This makes that outcome unconstructable.
    """

    train: Index
    test: Index

    def __post_init__(self) -> None:
        if self.train.size == 0:
            raise ValueError("a split with no training data is not a split")
        if self.test.size == 0:
            raise ValueError("a split with no test data is not a split")
        overlap = np.intersect1d(self.train, self.test, assume_unique=False)
        if overlap.size:
            raise ValueError(
                f"train and test overlap at {overlap.size} indices "
                f"(first {overlap[:5].tolist()}); the fit would be scored on data it saw"
            )
        if np.any(np.diff(self.train) <= 0):
            raise ValueError("train indices must be strictly increasing and unique")
        if np.any(np.diff(self.test) <= 0):
            raise ValueError("test indices must be strictly increasing and unique")

    @property
    def n_train(self) -> int:
        return int(self.train.size)

    @property
    def n_test(self) -> int:
        return int(self.test.size)

    def gap_before_test(self) -> int:
        """Bars dropped between the last training bar and the first test bar.

        Zero means no purge was applied, which for a sequential splitter with a
        non-zero label horizon is a leak. Exposed so the purge can be verified from
        outside the splitter rather than trusted.
        """
        earlier = self.train[self.train < self.test[0]]
        if earlier.size == 0:
            return 0
        return int(self.test[0]) - int(earlier[-1]) - 1


def purge_and_embargo(
    candidate: Index, test: Index, n_obs: int, purge: int, embargo: float
) -> Index:
    """Drop training bars whose labels touch the test block, in either direction."""
    if candidate.size == 0:
        return candidate
    lo, hi = int(test[0]), int(test[-1])

    keep = np.ones(candidate.size, dtype=bool)
    if purge > 0:
        # A label at i spans [i, i+purge], so it reaches the test block whenever
        # i + purge >= lo. Those training bars are removed.
        keep &= ~((candidate >= lo - purge) & (candidate < lo))
    if embargo > 0.0:
        n_embargo = math.ceil(embargo * n_obs)
        keep &= ~((candidate > hi) & (candidate <= hi + n_embargo))
    # Anything inside the test block is never trainable regardless.
    keep &= ~((candidate >= lo) & (candidate <= hi))
    return np.asarray(candidate[keep], dtype=np.int64)


class WalkForwardSplitter(ABC):
    """Produces ordered train/test partitions of `range(n_obs)`."""

    def __init__(self, purge: int = 0, embargo: float = 0.0) -> None:
        if purge < 0:
            raise ValueError(f"purge must be non-negative, got {purge}")
        if not 0.0 <= embargo < 1.0:
            raise ValueError(f"embargo must be in [0, 1), got {embargo}")
        self.purge = purge
        self.embargo = embargo

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def split(self, n_obs: int) -> list[Split]:
        """Partitions, in chronological order of their test blocks."""

    def __repr__(self) -> str:
        return f"{self.name}(purge={self.purge}, embargo={self.embargo})"


class ExpandingWindow(WalkForwardSplitter):
    """Anchored walk-forward: training starts at bar 0 and grows.

    The closest analogue to how a strategy is actually run, since you never throw
    away history you already have. Later folds are therefore fitted on more
    information than earlier ones, which is realistic and does mean the folds are not
    exchangeable -- worth remembering before averaging across them.
    """

    def __init__(
        self,
        n_splits: int,
        test_size: int,
        min_train: int,
        purge: int = 0,
        embargo: float = 0.0,
    ) -> None:
        super().__init__(purge, embargo)
        if n_splits < 1:
            raise ValueError(f"n_splits must be at least 1, got {n_splits}")
        if test_size < 1:
            raise ValueError(f"test_size must be at least 1, got {test_size}")
        if min_train < 1:
            raise ValueError(f"min_train must be at least 1, got {min_train}")
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train = min_train

    def split(self, n_obs: int) -> list[Split]:
        needed = self.min_train + self.purge + self.n_splits * self.test_size
        if n_obs < needed:
            raise InsufficientData(
                f"{n_obs} observations cannot yield {self.n_splits} folds of "
                f"{self.test_size} after {self.min_train} training bars and a purge of "
                f"{self.purge}: {needed} are required"
            )
        splits = []
        start = n_obs - self.n_splits * self.test_size
        for k in range(self.n_splits):
            lo = start + k * self.test_size
            test = np.arange(lo, lo + self.test_size, dtype=np.int64)
            train = purge_and_embargo(
                np.arange(0, lo, dtype=np.int64), test, n_obs, self.purge, self.embargo
            )
            splits.append(Split(train=train, test=test))
        return splits


class RollingWindow(WalkForwardSplitter):
    """Sliding walk-forward: a fixed-length training window moves forward.

    Every fold is fitted on the same amount of data, which makes folds comparable in a
    way expanding ones are not, and it deliberately forgets the distant past -- the
    right choice when the process is not stationary, which is most of the time.
    """

    def __init__(
        self,
        n_splits: int,
        train_size: int,
        test_size: int,
        purge: int = 0,
        embargo: float = 0.0,
    ) -> None:
        super().__init__(purge, embargo)
        if n_splits < 1:
            raise ValueError(f"n_splits must be at least 1, got {n_splits}")
        if train_size < 1:
            raise ValueError(f"train_size must be at least 1, got {train_size}")
        if test_size < 1:
            raise ValueError(f"test_size must be at least 1, got {test_size}")
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size

    def split(self, n_obs: int) -> list[Split]:
        needed = self.train_size + self.purge + self.n_splits * self.test_size
        if n_obs < needed:
            raise InsufficientData(
                f"{n_obs} observations cannot yield {self.n_splits} folds of "
                f"{self.test_size} with a {self.train_size}-bar training window and a "
                f"purge of {self.purge}: {needed} are required"
            )
        splits = []
        start = n_obs - self.n_splits * self.test_size
        for k in range(self.n_splits):
            lo = start + k * self.test_size
            test = np.arange(lo, lo + self.test_size, dtype=np.int64)
            window_lo = max(0, lo - self.purge - self.train_size)
            train = purge_and_embargo(
                np.arange(window_lo, lo, dtype=np.int64), test, n_obs, self.purge, self.embargo
            )
            splits.append(Split(train=train, test=test))
        return splits


class PurgedKFold(WalkForwardSplitter):
    """Contiguous k-fold with purge and embargo, training on both sides of the block.

    Not a walk-forward, and it should not be read as one: training data follows the
    test block, so it uses information from after the test period and cannot be
    interpreted as a simulation of live trading. It earns its place because it is the
    geometry CSCV needs at G9, and because it is where the embargo actually does work
    -- the test block's own labels run forward into the training bars that follow it.
    """

    def __init__(self, n_splits: int = 5, purge: int = 0, embargo: float = 0.0) -> None:
        super().__init__(purge, embargo)
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits

    def split(self, n_obs: int) -> list[Split]:
        if n_obs < self.n_splits * 2:
            raise InsufficientData(
                f"{n_obs} observations cannot yield {self.n_splits} folds with a "
                "trainable remainder"
            )
        bounds = np.linspace(0, n_obs, self.n_splits + 1).astype(int)
        splits = []
        for k in range(self.n_splits):
            test = np.arange(bounds[k], bounds[k + 1], dtype=np.int64)
            candidate = np.setdiff1d(np.arange(n_obs, dtype=np.int64), test, assume_unique=True)
            train = purge_and_embargo(candidate, test, n_obs, self.purge, self.embargo)
            if train.size == 0:
                raise InsufficientData(
                    f"fold {k} has no training data left after purge and embargo"
                )
            splits.append(Split(train=train, test=test))
        return splits


__all__ = [
    "ExpandingWindow",
    "Index",
    "InsufficientData",
    "PurgedKFold",
    "RollingWindow",
    "Split",
    "WalkForwardSplitter",
    "purge_and_embargo",
]
