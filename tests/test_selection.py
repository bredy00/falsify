"""Unit tests for the SelectionRule interface. Session 3, per 03 Part I.

Unit tests only -- no engine, no integration. The point of building this before
G9 is that CSCV's rank bookkeeping should be written against a contract that is
already pinned down, so these tests are the contract.

The contract properties are quantified over arbitrary inputs with hypothesis
rather than over the three matrices I would have thought of, because "returns a
non-negative vector summing to one" is a claim about all inputs or it is not
worth making.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from falsify.selection import (
    SUM_TOL,
    ArgMax,
    DegenerateTrial,
    EqualWeight,
    SelectionRule,
    Softmax,
    TopK,
    in_sample_sharpe,
    noise_floor,
)

Matrix = NDArray[np.float64]

# One instance of every rule, so the contract tests cover the whole family and a
# new rule cannot be added without being held to it.
ALL_RULES: tuple[SelectionRule, ...] = (
    ArgMax(),
    EqualWeight(),
    TopK(1),
    TopK(3),
    Softmax(0.25),
    Softmax(1.0),
    Softmax(4.0),
)

PROPERTY_SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Minimum gap between sorted in-sample Sharpes before the order-dependent rules
# are held to rescaling and permutation invariance. Exact ties are broken
# positionally and so are legitimately not invariant to either -- documented on
# SelectionRule and pinned by test_ties_are_broken_positionally. Anything above
# the rounding floor is fair game.
MIN_SHARPE_GAP = 1e-6


def sharpes_are_separated(matrix: Matrix, min_gap: float = MIN_SHARPE_GAP) -> bool:
    """True when no two configurations are within `min_gap` on in-sample Sharpe."""
    ordered = np.sort(in_sample_sharpe(matrix))
    return bool(ordered.size < 2 or np.all(np.diff(ordered) > min_gap))


@st.composite
def separated_return_matrices(draw: st.DrawFn, min_n: int = 2, max_n: int = 12) -> Matrix:
    """Return blocks whose columns have well-separated in-sample Sharpes.

    Adding `k * step * sd_k` to column k shifts that column's Sharpe by exactly
    `k * step`, because the shift is measured in units of that column's own
    volatility. With `step >= 2` the induced gaps dominate: the carrier in
    `return_matrices` holds every unshifted |Sharpe| well below 1, so adjacent
    columns end up separated by roughly `step` rather than by luck.

    Separation is engineered rather than filtered for the same reason the base
    generator is: `assume(sharpes_are_separated(...))` on raw draws left 7
    surviving inputs out of 57 locally and tripped filter_too_much outright in
    CI. A property test exploring 7 cases is barely a test.
    """
    m = draw(return_matrices(min_n=min_n, max_n=max_n))
    n = m.shape[1]
    if n > 1:
        step = draw(st.floats(2.0, 6.0))
        m = m + np.arange(n) * step * m.std(axis=0, ddof=1)
    assume(sharpes_are_separated(m))  # backstop; should effectively never fire
    return m


@st.composite
def return_matrices(
    draw: st.DrawFn, min_n: int = 1, max_n: int = 12, min_t: int = 3, max_t: int = 40
) -> Matrix:
    """A (T_is, N) block of plausible per-bar returns, non-degenerate by
    construction rather than by filtering.

    Every column is an alternating +/-2 carrier plus noise bounded to [-1, 1],
    the whole thing scaled by a positive per-column volatility. Because the
    carrier's swing strictly exceeds the noise range, no column can come out
    constant -- so `in_sample_sharpe` never raises here and nothing needs
    discarding.

    Construction matters more than it looks. Filtering with
    `assume(sd > 1e-6)` on raw draws tripped hypothesis's filter_too_much health
    check, and it did so only under the `ci` profile, whose derandomize=True
    draws a different sequence than a local run. Constructed validity removes
    both the health check and that class of local-passes-CI-fails.

    Degenerate columns are covered separately, by
    test_constant_column_is_caught_despite_nonzero_computed_std.
    """
    t = draw(st.integers(min_t, max_t))
    n = draw(st.integers(min_n, max_n))

    carrier = np.where(np.arange(t) % 2 == 0, 2.0, -2.0)[:, None]
    noise: Matrix = draw(
        arrays(
            np.float64,
            (t, n),
            elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False, width=64),
        )
    )
    vols: Matrix = draw(
        arrays(
            np.float64,
            (n,),
            elements=st.floats(0.005, 0.05, allow_nan=False, allow_infinity=False, width=64),
        )
    )
    return np.asarray((noise + carrier) * vols, dtype=np.float64)


def matrix_for(rule: SelectionRule, data: st.DataObject, **kwargs: int) -> Matrix:
    """Draw a return block guaranteed wide enough for `rule`.

    `TopK(k)` is undefined for a grid narrower than k, and these property tests used
    to skip whenever hypothesis happened to draw one. Four skipped tests is four
    reporting neither pass nor fail -- and a skip conditioned on a random draw is
    worse than that, because how much of the property actually got checked then
    varies from run to run. Drawing N >= k instead makes the input valid by
    construction, so every example exercises the rule. The k > N error path keeps its
    own dedicated test.
    """
    floor = rule.k if isinstance(rule, TopK) else 1
    min_n = max(floor, kwargs.pop("min_n", 1))
    return data.draw(return_matrices(min_n=min_n, **kwargs))


# ------------------------------------------------------------------- contract


@pytest.mark.parametrize("rule", ALL_RULES, ids=lambda r: r.name)
@PROPERTY_SETTINGS
@given(data=st.data())
def test_contract_holds_for_every_rule(rule: SelectionRule, data: st.DataObject) -> None:
    """Non-negative, sums to 1 within SUM_TOL, one weight per configuration."""
    matrix = matrix_for(rule, data)
    n = matrix.shape[1]

    w = rule.weights(matrix)

    assert w.shape == (n,), f"{rule.name}: expected shape ({n},), got {w.shape}"
    assert np.all(np.isfinite(w)), f"{rule.name}: non-finite weight in {w}"
    assert np.all(w >= 0.0), f"{rule.name}: negative weight in {w}"
    assert abs(w.sum() - 1.0) < SUM_TOL, f"{rule.name}: weights sum to {w.sum()!r}"


@pytest.mark.parametrize("rule", ALL_RULES, ids=lambda r: r.name)
@PROPERTY_SETTINGS
@given(data=st.data())
def test_rules_are_deterministic_and_stateless(rule: SelectionRule, data: st.DataObject) -> None:
    """Same input, same output, bitwise -- and no state carried between calls.

    The second half matters as much as the first: a rule that accumulated
    anything across calls would make G9's 12,870 splits order-dependent, and
    nothing would raise (B9).
    """
    matrix = matrix_for(rule, data)
    first = rule.weights(matrix)
    # Interleave a different call; a stateful rule would drift here. Scaling and
    # row-reversal keep the input non-degenerate -- `np.abs` would not, since the
    # generator's carrier is a symmetric alternation and taking its absolute
    # value collapses a column to a constant.
    rule.weights(matrix[::-1] * 3.0)
    second = rule.weights(matrix)

    assert np.array_equal(first, second), f"{rule.name} is not deterministic across calls"


@pytest.mark.parametrize("rule", ALL_RULES, ids=lambda r: r.name)
@PROPERTY_SETTINGS
@given(data=st.data(), scale=st.floats(0.05, 20.0))
def test_rules_are_scale_invariant(rule: SelectionRule, data: st.DataObject, scale: float) -> None:
    """Scaling every return leaves the weights alone.

    Sharpe is scale-invariant, so any rule built on it must be too. A rule that
    failed this would silently depend on whether returns were expressed as
    fractions or percent.
    """
    floor = rule.k if isinstance(rule, TopK) else 2
    matrix = data.draw(separated_return_matrices(min_n=max(floor, 2)))
    base = rule.weights(matrix)
    scaled = rule.weights(matrix * scale)
    assert np.allclose(base, scaled, atol=1e-9), f"{rule.name} is not scale-invariant"


@pytest.mark.parametrize("rule", ALL_RULES, ids=lambda r: r.name)
@PROPERTY_SETTINGS
@given(data=st.data(), seed=st.integers(0, 2**32 - 1))
def test_rules_are_permutation_equivariant(
    rule: SelectionRule, data: st.DataObject, seed: int
) -> None:
    """Relabelling the configurations relabels the weights, nothing more.

    This is the property that makes a rule a function of the evidence rather than
    of column order -- and column order is exactly what changes when CSCV
    reshuffles blocks.
    """
    floor = rule.k if isinstance(rule, TopK) else 2
    matrix = data.draw(separated_return_matrices(min_n=max(floor, 2)))
    # Separation comes from the generator. Ties would make this ambiguous for the
    # discrete rules: permuting columns changes float summation order, so Sharpes
    # closer together than the rounding floor can swap places legitimately.
    order = np.random.default_rng(seed).permutation(matrix.shape[1])
    direct = rule.weights(matrix)[order]
    permuted = rule.weights(matrix[:, order])
    assert np.allclose(direct, permuted, atol=1e-12), f"{rule.name} depends on column order"


# ------------------------------------------------------- rule-specific behaviour


def test_argmax_selects_the_best_in_sample_sharpe() -> None:
    rng = np.random.default_rng(7)
    m = rng.normal(0.0, 0.01, (200, 6))
    m[:, 4] += 0.004  # a clear winner
    w = ArgMax().weights(m)
    assert w[int(np.argmax(in_sample_sharpe(m)))] == 1.0
    assert w.sum() == pytest.approx(1.0, abs=SUM_TOL)
    assert np.count_nonzero(w) == 1


def test_equal_weight_is_exactly_one_over_n() -> None:
    rng = np.random.default_rng(8)
    m = rng.normal(0.0, 0.01, (50, 7))
    assert np.allclose(EqualWeight().weights(m), 1.0 / 7, atol=0.0, rtol=0.0)


def test_topk_spreads_across_the_best_k() -> None:
    rng = np.random.default_rng(9)
    m = rng.normal(0.0, 0.01, (300, 8))
    m[:, [2, 5]] += 0.005  # two clear winners
    w = TopK(2).weights(m)
    assert np.count_nonzero(w) == 2
    assert sorted(np.flatnonzero(w).tolist()) == [2, 5]
    assert np.allclose(w[[2, 5]], 0.5)


def test_topk_endpoints_recover_argmax_and_equal_weight() -> None:
    """k=1 is ArgMax and k=N is EqualWeight, so TopK spans the same interval the
    temperature does -- just discretely."""
    rng = np.random.default_rng(10)
    m = rng.normal(0.0, 0.01, (200, 5))
    assert np.array_equal(TopK(1).weights(m), ArgMax().weights(m))
    assert np.allclose(TopK(5).weights(m), EqualWeight().weights(m))


# ------------------------------------------------------------ the temperature


def test_softmax_recovers_argmax_as_temperature_falls() -> None:
    """tau -> 0 is the argmax end of the dial, and it must get there without
    overflowing -- which is what subtracting max(z) before exp() buys."""
    rng = np.random.default_rng(11)
    m = rng.normal(0.0, 0.01, (250, 9))
    m[:, 3] += 0.006
    target = ArgMax().weights(m)
    for tau in (1e-2, 1e-4, 1e-8):
        w = Softmax(tau).weights(m)
        assert np.all(np.isfinite(w)), f"tau={tau} produced {w}"
        assert np.allclose(w, target, atol=1e-9), f"tau={tau} did not converge to argmax"


def test_softmax_recovers_equal_weight_as_temperature_rises() -> None:
    """tau -> infinity is the no-selection-bias asymptote."""
    rng = np.random.default_rng(12)
    m = rng.normal(0.0, 0.01, (250, 9))
    m[:, 3] += 0.006
    target = EqualWeight().weights(m)
    assert np.allclose(Softmax(1e6).weights(m), target, atol=1e-5)
    far = np.abs(Softmax(1e3).weights(m) - target).max()
    near = np.abs(Softmax(1.0).weights(m) - target).max()
    assert far < near, "raising tau must move the blend toward equal weight"


def test_softmax_weight_is_monotone_in_in_sample_sharpe() -> None:
    """A better in-sample Sharpe never earns a smaller weight. Without this the
    temperature sweep would not be interpretable as a shrinkage path."""
    rng = np.random.default_rng(13)
    m = rng.normal(0.0, 0.01, (400, 10))
    m += np.linspace(0.0, 0.004, 10)  # monotone edge across columns
    sharpe = in_sample_sharpe(m)
    w = Softmax(1.0).weights(m)
    order = np.argsort(sharpe)
    assert np.all(np.diff(w[order]) >= -1e-15), f"weights not monotone in Sharpe: {w[order]}"


def test_softmax_falls_back_to_equal_weight_when_nothing_distinguishes_columns() -> None:
    """Identical columns give zero cross-sectional spread. Equal weight is the
    honest answer; 0/0 is not."""
    col = np.random.default_rng(14).normal(0.0, 0.01, (100, 1))
    m = np.repeat(col, 5, axis=1)
    w = Softmax(1.0).weights(m)
    assert np.allclose(w, 0.2)
    assert abs(w.sum() - 1.0) < SUM_TOL


def test_softmax_single_configuration_is_the_whole_portfolio() -> None:
    m = np.random.default_rng(15).normal(0.0, 0.01, (40, 1))
    for rule in (Softmax(1.0), ArgMax(), EqualWeight(), TopK(1)):
        assert rule.weights(m) == pytest.approx([1.0], abs=SUM_TOL), rule.name


# ------------------------------------------------------------------ validation


def test_degenerate_column_raises_rather_than_scoring_zero() -> None:
    """Gate 0.4: an undefined Sharpe and a zero Sharpe are different claims, and
    only one of them is true. A constant column must not take a weight."""
    m = np.random.default_rng(16).normal(0.0, 0.01, (60, 4))
    m[:, 2] = 0.001  # constant: zero volatility
    for rule in ALL_RULES:
        if isinstance(rule, TopK) and rule.k > m.shape[1]:
            continue
        with pytest.raises(DegenerateTrial):
            rule.weights(m)


def test_constant_column_is_caught_despite_nonzero_computed_std() -> None:
    """Regression, found by the property tests.

    A column holding one repeated value does not produce std exactly 0.0: for
    0.001 across 60 rows the computed std is around 2e-19, because sum/n does not
    round-trip to the original float. An `sd == 0.0` check passes it through and
    the Sharpe lands near 1e16 -- finite, enormous and meaningless, and enough to
    take the entire portfolio under ArgMax or Softmax.
    """
    m = np.random.default_rng(16).normal(0.0, 0.01, (60, 4))
    m[:, 2] = 0.001

    raw_sd = m.std(axis=0, ddof=1)[2]
    assert raw_sd != 0.0, "premise of this regression no longer holds"
    assert abs(0.001 / raw_sd) > 1e12, "the undetected Sharpe should be absurd"

    with pytest.raises(DegenerateTrial, match="zero or non-finite"):
        in_sample_sharpe(m)


def test_softmax_does_not_amplify_rounding_dust_into_a_decision() -> None:
    """Regression, found by the property tests.

    Two columns where one is a positive rescaling of the other have *identical*
    true Sharpes, so the cross-sectional spread is exactly zero in exact
    arithmetic and around 1e-17 in floating point. Dividing z-scores by that dust
    drives them to +/-1, and at low temperature softmax collapses into an
    arbitrary argmax: a 0.9997/0.0003 split decided by rounding.

    CSCV in-sample blocks produce near-identical Sharpes routinely, so the
    temperature sweep at G9 would be noise wherever the grid agrees.
    """
    col = np.random.default_rng(18).normal(0.0, 0.01, (80, 1))
    m = np.hstack([col, col * 1.7])  # same Sharpe by construction

    sharpe = in_sample_sharpe(m)
    assert abs(sharpe[0] - sharpe[1]) < 1e-12, "premise: the Sharpes must tie"

    for tau in (0.25, 1.0, 4.0):
        w = Softmax(tau).weights(m)
        assert np.allclose(w, 0.5, atol=1e-12), f"tau={tau} split tied columns as {w}"


def test_ties_leave_the_discrete_winner_arbitrary_but_stable() -> None:
    """The documented limit, pinned so it cannot drift.

    Three columns that are positive rescalings of one another tie exactly on
    Sharpe. ArgMax still returns one selection and returns the same one every
    time, but *which* column it is comes from rounding, not from evidence -- on
    this input it is column 2, not column 0. So the guarantee is "exactly one,
    deterministically", never "the lowest index", and G9 must not read meaning
    into the identity.
    """
    col = np.random.default_rng(19).normal(0.0, 0.01, (80, 1))
    m = np.hstack([col, col * 2.5, col * 0.4])
    assert abs(np.ptp(in_sample_sharpe(m))) < 1e-12, "premise: the Sharpes must tie"

    w = ArgMax().weights(m)
    assert np.count_nonzero(w) == 1, f"argmax must still pick exactly one, got {w}"
    assert np.array_equal(w, ArgMax().weights(m)), "the pick must be stable per input"

    top2 = TopK(2).weights(m)
    assert np.count_nonzero(top2) == 2
    assert np.allclose(top2[top2 > 0], 0.5)

    # The continuous rule has no such ambiguity: it splits the tie evenly.
    assert np.allclose(Softmax(1.0).weights(m), 1.0 / 3.0)


@pytest.mark.parametrize(
    ("matrix", "match"),
    [
        (np.zeros((5,)), "expected a"),
        (np.zeros((2, 3, 4)), "expected a"),
        (np.zeros((1, 3)), "at least 2"),
    ],
)
def test_malformed_input_is_rejected(matrix: Matrix, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        in_sample_sharpe(matrix)


@pytest.mark.parametrize("k", [0, -1])
def test_topk_rejects_nonpositive_k(k: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        TopK(k)


def test_topk_rejects_k_above_n() -> None:
    m = np.random.default_rng(17).normal(0.0, 0.01, (30, 3))
    with pytest.raises(ValueError, match="exceeds the number of configurations"):
        TopK(4).weights(m)


@pytest.mark.parametrize("tau", [0.0, -1.0, np.nan, np.inf])
def test_softmax_rejects_invalid_temperature(tau: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        Softmax(tau)


def test_rule_names_are_distinct_and_record_their_parameters() -> None:
    """The name reaches the trials ledger, so Softmax(0.5) and Softmax(2.0) must
    not both appear as 'Softmax' -- tau is a trial (01 Part E2)."""
    names = [r.name for r in ALL_RULES]
    assert len(names) == len(set(names)), f"duplicate rule names: {names}"
    assert Softmax(0.5).name != Softmax(2.0).name
    assert TopK(1).name != TopK(3).name


@given(
    arrays(
        np.float64,
        st.tuples(st.integers(2, 40), st.integers(1, 8)),
        elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False, width=64),
    )
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_the_vectorised_floor_is_the_documented_scalar_floor(matrix: Matrix) -> None:
    """`in_sample_sharpe` computes every column's floor in one pass; `noise_floor`
    is the scalar definition that the docstring and Gate 0.4 both refer to.

    They are two expressions of one rule, and nothing but this test stops them
    drifting apart. The vectorised form exists because the per-column comprehension
    it replaced cost 0.68s +/- 0.145 of a 4.75s C(16,8) sweep (paired, n=18,
    t=4.7) -- a real saving, but worthless if it quietly changed the threshold at
    which a dust column is caught.
    """
    t_is = matrix.shape[0]
    scalar = np.array(
        [noise_floor(float(np.max(np.abs(col))), t_is) for col in matrix.T],
        dtype=np.float64,
    )
    vectorised = (
        np.finfo(np.float64).eps
        * np.maximum(np.max(np.abs(matrix), axis=0), 0.0)
        * max(t_is, 1)
        * 4.0
    )
    assert np.array_equal(scalar, vectorised), (
        "the vectorised floor no longer matches noise_floor(). One of them moved, "
        "and the dust threshold Gate 0.4 relies on is now ambiguous."
    )
