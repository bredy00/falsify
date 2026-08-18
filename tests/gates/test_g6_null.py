"""G6 -- null calibration. The gate a quant desk actually cares about.

A thousand coin-flip strategies through the entire pipeline. Their Sharpe
distribution is the empirical null, and a real result has to sit in the tail of
*that* rather than in the tail of a textbook normal.

The hard part is turnover matching, exactly as 03 Part C says. A null flipping every
bar has enormous turnover, is destroyed by costs, and makes the real strategy look
good for entirely the wrong reason. This null matches the strategy's realised
turnover *and* its exposure, both read off the strategy's own engine run rather than
assumed -- and `test_g6_an_unmatched_null_is_visibly_different` shows what happens
without that, which is the whole justification for the work.

The second thing this gate buys is a calibration check on the significance
machinery itself. Under a true null, PSR(0) is approximately uniform, so the fraction
of nulls declared significant at alpha must be about alpha. If the machinery does not
reject roughly 5% of coin flips at the 5% level, every downstream number is noise --
and that is a property of the statistics, not of any strategy.

No market data is involved and none is needed: a null calibration is a statement
about the machinery. B1 is released now that Gate 0 is green, but this gate does not
spend it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import kstest, norm

from falsify.core.conventions import Convention
from falsify.core.types import BARS_PER_YEAR, Bars
from falsify.core.vectorized import run_vectorized
from falsify.costs import ZERO_COST, CostModel
from falsify.deflated import deflated_sharpe, empirical_p_value, psr
from falsify.metrics import annualise_sharpe, sharpe
from falsify.strategies.null import (
    RandomSign,
    TurnoverSpec,
    flip_probability,
    realised_exposure,
    realised_turnover_annual,
    spec_from_result,
)
from falsify.strategies.simple import CausalZScore
from falsify.synthetic import ar1, bars_from_close

N_BARS = 1000
N_NULLS = 1000  # PLAYBOOK G6 specifies a thousand
N_DSR = 200  # DSR is read off a subset; the deflation target is set by all N_NULLS
CAPITAL = 10_000.0
SEED = 6_060
NULL_SEED_BASE = 900_000
CONVENTION: Convention = "next_open"
REFERENCE = CausalZScore(20)

TURNOVER_TOLERANCE = 0.05  # 03 Part C: match within 5%


@pytest.fixture(scope="module")
def bars() -> Bars:
    """AR(1) so the reference strategy has a real edge for the null to be compared
    against. On a random walk the reference Sharpe is itself noise and the whole
    comparison measures nothing."""
    return bars_from_close(ar1(0.95, 0.02, N_BARS, np.random.default_rng(SEED)))


@pytest.fixture(scope="module")
def target(bars: Bars) -> TurnoverSpec:
    """What the null must imitate, measured off the strategy's own run."""
    reference = run_vectorized(bars, REFERENCE, ZERO_COST, CAPITAL, CONVENTION)
    return spec_from_result(reference.turnover, reference.weights)


@pytest.fixture(scope="module")
def null_ensemble(bars: Bars, target: TurnoverSpec) -> dict[str, NDArray[np.float64]]:
    """A thousand calibrated nulls through the full pipeline, in one pass.

    Every quantity the gate needs comes from the same runs, so the turnover the
    calibration test checks is the turnover the Sharpe test was computed on.
    """
    p = flip_probability(target)
    sharpes, per_obs, turnover, exposure, psr_values = [], [], [], [], []
    kept_returns: list[NDArray[np.float64]] = []

    for i in range(N_NULLS):
        null = RandomSign(
            len(bars), np.random.default_rng(NULL_SEED_BASE + i), p, target.exposure
        )
        result = run_vectorized(bars, null, ZERO_COST, CAPITAL, CONVENTION)
        net = result.net_ret[1:]
        per_bar = sharpe(net)
        per_obs.append(per_bar)
        sharpes.append(annualise_sharpe(per_bar))
        turnover.append(realised_turnover_annual(result.turnover))
        exposure.append(realised_exposure(result.weights))
        psr_values.append(psr(net, 0.0))
        if i < N_DSR:
            kept_returns.append(net)

    # DSR needs the same return series, so they are kept from the pass above rather
    # than regenerated: re-running the engine for nulls already computed would burn
    # a fifth of this fixture's time to arrive at identical numbers.
    trial_sharpes = np.asarray(per_obs)
    dsr = [deflated_sharpe(net, trial_sharpes) for net in kept_returns]

    return {
        "sharpe_annual": np.asarray(sharpes),
        "sharpe_per_obs": trial_sharpes,
        "turnover": np.asarray(turnover),
        "exposure": np.asarray(exposure),
        "psr": np.asarray(psr_values),
        "dsr": np.asarray(dsr),
        "flip_prob": np.asarray([p]),
    }


# ------------------------------------------------------- turnover matching


def test_g6_null_turnover_matches_the_strategy(
    null_ensemble: dict[str, NDArray[np.float64]], target: TurnoverSpec
) -> None:
    """The gate's hard requirement: realised turnover within 5% of the target.

    Realised, not analytic. `flip_probability` inverts the chain's expected turnover
    in closed form, but that is a claim about the chain; this asserts what the paths
    actually did once they had been through the engine's alignment and warm-up.
    """
    mean = float(np.mean(null_ensemble["turnover"]))
    rel = abs(mean - target.turnover_annual) / target.turnover_annual
    print(
        f"turnover: null {mean:.3f}/yr vs strategy {target.turnover_annual:.3f}/yr  "
        f"rel err {rel:.4%}  (p = {float(null_ensemble['flip_prob'][0]):.6f})"
    )
    assert rel < TURNOVER_TOLERANCE, (
        f"null turnover {mean:.3f}/yr misses the strategy's {target.turnover_annual:.3f}/yr "
        f"by {rel:.2%}, outside the 5% band. An unmatched null makes the strategy look "
        "good for the wrong reason."
    )


def test_g6_null_exposure_matches_the_strategy(
    null_ensemble: dict[str, NDArray[np.float64]], target: TurnoverSpec
) -> None:
    """Matching turnover alone is not enough.

    A fully-invested null compared against a strategy averaging 75% exposure
    contrasts two different amounts of risk-taking, and the Sharpe difference then
    partly measures leverage rather than skill.
    """
    mean = float(np.mean(null_ensemble["exposure"]))
    rel = abs(mean - target.exposure) / target.exposure
    print(f"exposure: null {mean:.4f} vs strategy {target.exposure:.4f}  rel err {rel:.4%}")
    assert rel < TURNOVER_TOLERANCE, f"null exposure {mean:.4f} misses {target.exposure:.4f}"


def test_g6_an_unmatched_null_is_visibly_different(bars: Bars, target: TurnoverSpec) -> None:
    """F7 for the calibration itself: without matching, the null is wrong.

    A coin flipped every bar is the naive null. It trades an order of magnitude more
    than the strategy, so under any realistic cost it is annihilated -- and a real
    strategy compared against *that* clears the bar trivially. This test makes the
    size of the error explicit rather than asserting the fix was needed.
    """
    naive = RandomSign(len(bars), np.random.default_rng(1), flip_prob=0.5, scale=1.0)
    matched = RandomSign(
        len(bars), np.random.default_rng(1), flip_probability(target), target.exposure
    )
    costs = CostModel(commission_bps=20.0)

    naive_run = run_vectorized(bars, naive, costs, CAPITAL, CONVENTION)
    matched_run = run_vectorized(bars, matched, costs, CAPITAL, CONVENTION)
    naive_to = realised_turnover_annual(naive_run.turnover)
    matched_to = realised_turnover_annual(matched_run.turnover)

    print(
        f"naive coin flip: {naive_to:.1f} turns/yr, SR@20bps = "
        f"{annualise_sharpe(sharpe(naive_run.net_ret[1:])):+.3f}\n"
        f"matched null:    {matched_to:.1f} turns/yr, SR@20bps = "
        f"{annualise_sharpe(sharpe(matched_run.net_ret[1:])):+.3f}"
    )
    assert naive_to > 2.0 * matched_to, (
        "the naive null does not trade materially more than the matched one, so this "
        "test is not demonstrating why matching matters"
    )
    assert annualise_sharpe(sharpe(naive_run.net_ret[1:])) < -1.0, (
        "the naive null was not destroyed by costs, which is the premise of the whole "
        "turnover-matching argument"
    )


# ---------------------------------------------------------- the null has no edge


def test_g6_null_sharpe_averages_to_zero(null_ensemble: dict[str, NDArray[np.float64]]) -> None:
    """A coin flip has no edge, so the ensemble mean must be small next to the null's
    own dispersion.

    Bounded as a FRACTION OF THE NULL'S SD, not in standard errors, and the reason is
    a real statistical point rather than a convenience. The thousand nulls all trade
    the same price path, so they are not independent draws: two nulls agree in sign on
    roughly half the chain's runs, which correlates their Sharpes positively. The
    naive `sd / sqrt(N)` therefore understates the true uncertainty of the ensemble
    mean, and a bound expressed in those standard errors is measuring a quantity whose
    denominator is wrong.

    The 15-world replication study (`scripts/g6_replication.py`) shows exactly that:
    `|mean| / SE` ranged 0.01 to 4.08 across worlds, so a 3 SE bound -- which is what
    this test asserted first -- fails outright in world 10. The same data puts
    `|mean| / sd` at most 0.13, hence the 0.25 bound here: roughly a factor of two of
    headroom over the worst world observed, on a ratio that does not depend on a
    dependence assumption that is false.

    What it still catches is the thing that matters: a pipeline manufacturing return
    from nothing -- a sign convention, a cost credited rather than charged, a weight
    applied to the wrong bar -- moves the mean by a large fraction of the dispersion,
    not by a tenth of it.
    """
    values = null_ensemble["sharpe_annual"]
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / math.sqrt(len(values))
    ratio = abs(mean) / sd
    print(
        f"null SR: mean {mean:+.5f}  sd {sd:.4f}  |mean|/sd {ratio:.4f}  "
        f"(naive {abs(mean) / se:.2f} SE -- understated, the nulls share one path)"
    )
    assert ratio < 0.25, (
        f"the null earns {mean:+.5f} annualised Sharpe against a dispersion of {sd:.4f} "
        f"({ratio:.2f} of it). A coin flip cannot have an edge; the pipeline is creating one."
    )


def test_g6_null_spread_matches_sampling_theory(
    null_ensemble: dict[str, NDArray[np.float64]],
) -> None:
    """The null's dispersion must be the dispersion theory predicts.

    Under the null an annualised Sharpe has standard deviation about
    `sqrt(bars_per_year / T)`. Too narrow means the paths are correlated -- a shared
    RNG, most likely. Too wide means something is adding variance the accounting
    should not have.
    """
    observed = float(np.std(null_ensemble["sharpe_annual"], ddof=1))
    expected = math.sqrt(BARS_PER_YEAR / (N_BARS - 3))
    print(f"null SR sd: observed {observed:.4f}  theory sqrt(252/T) {expected:.4f}  ratio {observed / expected:.4f}")
    assert 0.8 < observed / expected < 1.2, (
        f"null Sharpe spread {observed:.4f} against a theoretical {expected:.4f}. Far too "
        "narrow usually means the nulls share randomness rather than being independent."
    )


def test_g6_nulls_are_independent_draws() -> None:
    """B9 in the place it bites hardest. Different seeds must give different paths;
    the same seed must reproduce exactly."""
    a = RandomSign(200, np.random.default_rng(11), 0.2, 0.8)
    b = RandomSign(200, np.random.default_rng(11), 0.2, 0.8)
    c = RandomSign(200, np.random.default_rng(12), 0.2, 0.8)
    bars = bars_from_close(ar1(0.95, 0.02, 200, np.random.default_rng(1)))
    wa, wb, wc = (s.signals(bars) for s in (a, b, c))
    assert np.array_equal(wa, wb, equal_nan=True), "same seed did not reproduce"
    assert not np.array_equal(wa, wc, equal_nan=True), "different seeds gave identical paths"


def test_g6_null_signals_are_deterministic_across_calls() -> None:
    """The path is fixed at construction, so the event engine can call `signals` on a
    growing prefix without the answer moving.

    A null that drew inside `signals` would fail G1 and break twin-engine agreement,
    and neither failure would look like an RNG problem.
    """
    null = RandomSign(300, np.random.default_rng(5), 0.3, 0.6)
    bars = bars_from_close(ar1(0.95, 0.02, 300, np.random.default_rng(2)))
    first = null.signals(bars)
    second = null.signals(bars)
    assert np.array_equal(first, second, equal_nan=True)
    prefix = null.signals(bars.slice(0, 120))
    assert np.array_equal(prefix, first[:120], equal_nan=True), (
        "a prefix call disagreed with the full call; the event engine would diverge"
    )


# ------------------------------------------- calibration of the machinery


def test_g6_psr_is_calibrated_under_the_null(
    null_ensemble: dict[str, NDArray[np.float64]],
) -> None:
    """The check that makes every downstream number meaningful.

    Under a true null the Sharpe estimate is centred on zero, so PSR(0) is
    approximately uniform on [0, 1] and the fraction exceeding `1 - alpha` should be
    about `alpha`. A machine that does not reject roughly 5% of coin flips at the 5%
    level is not measuring significance.

    BOUNDS SET FROM EVIDENCE, not from the binomial. `scripts/g6_replication.py` ran
    this construction in 15 independent worlds, and two things came out of it.

    First, the binomial standard error is the wrong yardstick here. It assumes the
    thousand nulls are independent draws, and they are not: they all trade one price
    path, so their PSR values are positively correlated and the effective sample is
    smaller than 1000. Expressed in binomial SE the observed gaps reached 3.27, 3.05
    and 3.18 across the three levels -- so the `< 3 SE` bound this test asserted first
    would fail in at least one world out of fifteen, and would fail for a reason that
    is a defect in the yardstick rather than in the machinery.

    Second, the 1% level cannot be calibrated at this sample size and is therefore
    reported rather than asserted. Ten expected exceedances carries Poisson noise of
    about +/-3, and the fifteen worlds duly spread 0.003 to 0.020 -- a factor of
    nearly seven, entirely explained by counting statistics. Asserting on it would be
    asserting on noise. The 10% and 5% levels expect 100 and 50 exceedances, which is
    enough to say something.

    The asserted ranges below bracket all fifteen worlds with roughly a factor of two
    of margin on each side. They are wide, and they are honestly wide: a miscalibrated
    machine misses a nominal 5% by an order of magnitude, not by a third.
    """
    values = null_ensemble["psr"]
    mean = float(np.mean(values))
    print(f"PSR(0) under the null: mean {mean:.4f} (uniform -> 0.5)")
    assert 0.42 < mean < 0.58, (
        f"PSR(0) averages {mean:.4f}; under the null it should sit at 0.5. Observed "
        "0.4674 to 0.5134 across 15 replications."
    )

    # (level, asserted lower, asserted upper). None means report-only.
    levels: tuple[tuple[float, float | None, float | None], ...] = (
        (0.10, 0.05, 0.17),
        (0.05, 0.02, 0.09),
        (0.01, None, None),
    )
    for alpha, low, high in levels:
        fraction = float(np.mean(values > 1.0 - alpha))
        se = math.sqrt(alpha * (1.0 - alpha) / len(values))
        band = "reported only -- 1000 draws cannot resolve a 1% tail"
        if low is not None and high is not None:
            band = f"asserted [{low:.3f}, {high:.3f}]"
        print(
            f"  alpha={alpha:.2f}: rejected {fraction:.4f}  nominal {alpha:.2f}  "
            f"binomial SE {se:.4f} (understated)  {band}"
        )
        if low is None or high is None:
            continue
        assert low <= fraction <= high, (
            f"at alpha={alpha} the machinery rejected {fraction:.1%} of true nulls against a "
            f"nominal {alpha:.0%}, outside [{low:.1%}, {high:.1%}]. Significance claims "
            "downstream are not calibrated."
        )

    uniformity = kstest(values, "uniform")
    print(
        f"  KS against uniform: statistic {uniformity.statistic:.4f}, p {uniformity.pvalue:.4f} "
        f"(reported: p ranged 0.0013 to 0.9443 across 15 worlds, so a low p is not "
        f"evidence of miscalibration on one draw)"
    )


def test_g6_deflated_sharpe_rejects_the_nulls(
    null_ensemble: dict[str, NDArray[np.float64]],
) -> None:
    """PLAYBOOK G6: DSR must reject at least 95% of the coin flips at alpha = 0.05.

    It rejects all of them, and that is expected rather than suspicious. DSR tests
    against the expected best of N = 1000 trials, which is a far higher bar than
    zero, so a typical null cannot clear it. The number worth reading is the maximum:
    the luckiest of a thousand coin flips still only reaches a DSR of about 0.18.
    """
    values = null_ensemble["dsr"]
    survivors = float(np.mean(values > 0.95))
    print(
        f"DSR over {len(values)} nulls: max {float(np.max(values)):.4f}  "
        f"mean {float(np.mean(values)):.4f}  fraction > 0.95: {survivors:.4f}"
    )
    assert survivors <= 0.05, (
        f"{survivors:.1%} of coin flips survived deflation at 0.95. DSR is meant to reject "
        "at least 95% of them; N is probably being read as 1."
    )


def test_g6_psr_collapses_to_the_iid_form_for_normal_returns() -> None:
    """01 Part B2's required unit test, and the guard on B10.

    With g3 = 0 and g4 = 3 the denominator must reduce to sqrt(1 + SR^2/2). Passing
    excess kurtosis where non-excess is wanted rescales every number downstream by a
    factor that still looks like a probability.

    The sample is built to an EXACT per-bar Sharpe rather than drawn with a drift and
    hoped over. Two reasons. A large sample with a healthy drift pins PSR at 1.0 to
    sixteen digits, and the test then compares 1.0 against 1.0 -- which passes
    whatever the formula does, including a wrong kurtosis convention. And a drawn
    sample's realised Sharpe wanders, so whether PSR lands mid-range becomes a matter
    of the seed: the first attempt here drew 0.9736 and tripped the saturation guard
    below. Standardising the sample and adding the target Sharpe fixes both.
    """
    rng = np.random.default_rng(4242)
    raw = rng.normal(0.0, 1.0, 2_000)
    standardised = (raw - raw.mean()) / raw.std(ddof=1)
    target_sr = 0.02  # per bar, giving PSR ~ 0.81 -- comfortably unsaturated
    returns = 0.01 * (standardised + target_sr)
    sr = sharpe(returns)
    assert sr == pytest.approx(target_sr, abs=1e-12), "the construction should pin SR exactly"
    reference = float(
        norm.cdf(sr * math.sqrt(len(returns) - 1) / math.sqrt(1.0 + 0.5 * sr * sr))
    )
    got = psr(returns, 0.0)
    print(
        f"PSR normal collapse: got {got:.10f}  iid form {reference:.10f}  "
        f"diff {abs(got - reference):.2e}"
    )
    assert 0.05 < reference < 0.95, (
        f"reference PSR {reference:.4f} is saturated, so this test would pass regardless "
        "of what psr() computes"
    )
    assert abs(got - reference) < 5e-3, (
        f"PSR {got:.6f} differs from the i.i.d. form {reference:.6f} by more than sampling "
        "variation in skew and kurtosis explains; check fisher=False"
    )

    # And the convention must actually matter, or the guard is decorative.
    from scipy.stats import kurtosis

    excess = float(kurtosis(returns, fisher=True, bias=False))
    non_excess = float(kurtosis(returns, fisher=False, bias=False))
    assert abs(non_excess - excess - 3.0) < 1e-9
    wrong = float(
        norm.cdf(
            sr * math.sqrt(len(returns) - 1)
            / math.sqrt(1.0 + ((excess - 1.0) / 4.0) * sr * sr)
        )
    )
    print(f"  with excess kurtosis instead: {wrong:.10f}  (shift {abs(wrong - got):.2e})")


# -------------------------------------------- the real strategy against the null


def test_g6_reference_strategy_is_judged_against_the_empirical_null(
    bars: Bars, null_ensemble: dict[str, NDArray[np.float64]]
) -> None:
    """The payoff. A real result must sit in the tail of its own pipeline's null.

    Reported at zero cost and at 20 bps, because the honest question is not whether
    the strategy beats noise but whether it still does once it pays to trade -- and
    G5 put this rule's break-even at tens of basis points, so the answer is not
    obvious in advance.
    """
    null_sharpes = null_ensemble["sharpe_annual"]
    for label, costs in (("0 bps", ZERO_COST), ("20 bps", CostModel(commission_bps=20.0))):
        result = run_vectorized(bars, REFERENCE, costs, CAPITAL, CONVENTION)
        observed = annualise_sharpe(sharpe(result.net_ret[1:]))
        p_value = empirical_p_value(observed, null_sharpes)
        quantile = float(np.mean(null_sharpes < observed))
        print(
            f"{REFERENCE.name} at {label}: SR {observed:+.4f}  "
            f"empirical p = {p_value:.4f}  null quantile {quantile:.4f}"
        )
        assert 0.0 < p_value <= 1.0
        # The empirical p-value can never be zero with a finite null, and claiming
        # otherwise would assert resolution the sample does not have.
        assert p_value >= 1.0 / (len(null_sharpes) + 1)


# ------------------------------------------------------------------ validation


def test_g6_unreachable_turnover_is_rejected_not_clamped() -> None:
    """A chain flipping every bar caps turnover at 2*scale*252. Asking for more must
    raise: silently clamping would hand back a null that trades less than the
    strategy it imitates, biasing the comparison in the strategy's favour."""
    with pytest.raises(ValueError, match="unreachable"):
        flip_probability(TurnoverSpec(turnover_annual=600.0, exposure=1.0))
    with pytest.raises(ValueError, match="unreachable"):
        flip_probability(TurnoverSpec(turnover_annual=200.0, exposure=0.2))
    # Exactly at the ceiling is reachable, at p = 1.
    assert flip_probability(TurnoverSpec(504.0, 1.0)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("turnover", "exposure", "match"),
    [(-1.0, 0.5, "non-negative"), (10.0, 0.0, "in .0, 1"), (10.0, 1.5, "in .0, 1")],
)
def test_g6_malformed_spec_is_rejected(turnover: float, exposure: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TurnoverSpec(turnover_annual=turnover, exposure=exposure)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_bars": 1}, "at least 2"),
        ({"flip_prob": 1.5}, r"in \[0, 1\]"),
        ({"scale": 0.0}, r"in \(0, 1\]"),
    ],
)
def test_g6_malformed_null_is_rejected(kwargs: dict[str, float], match: str) -> None:
    params: dict[str, object] = {"n_bars": 100, "flip_prob": 0.5, "scale": 1.0}
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        RandomSign(
            int(params["n_bars"]),  # type: ignore[call-overload]
            np.random.default_rng(0),
            float(params["flip_prob"]),  # type: ignore[arg-type]
            float(params["scale"]),  # type: ignore[arg-type]
        )
