"""Phase 8 status report as a PDF.

    uv run --group docs python scripts/phase8_report.py

Styling, palette and page furniture are imported from `status_report.py` rather than
copied, so the two reports stay one visual system and a change to the house style lands
in both. What lives here is the content: where the project stands after Phase 8, the
completion checklists against each specification document, and the measurements this
phase produced.

Numbers are read at run time where they can be -- git SHA, test count, gate files -- so
the report cannot claim to describe a tree it does not. The measured statistics are dated
constants, because rerunning a 16-path sweep to render a PDF would be its own kind of
silly, and each is traceable to the test that asserts it.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer

from scripts.status_report import S, git, on_page, p, status_chip, table

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DATE = date(2026, 8, 24)
OUT_PATH = REPO_ROOT / "docs" / f"status-{REPORT_DATE.isoformat()}.pdf"

FULL = A4[0] - 36 * mm


@dataclass(frozen=True, slots=True)
class Row:
    """One checklist line: what a document asked for, and what exists."""

    item: str
    state: str  # "green", "partial", "open"
    note: str


def collected() -> str:
    """Read the suite size from the tree rather than quoting it from memory."""
    try:
        out = subprocess.run(
            ["uv", "run", "pytest", "tests", "-q", "-m", "not live", "--collect-only"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        ).stdout
        for line in reversed(out.splitlines()):
            if "test" in line and "collected" in line:
                return line.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return "unavailable"


def checklist(title: str, rows: tuple[Row, ...]) -> list[Any]:
    body: list[list[Any]] = [["item", "state", "note"]]
    body.extend([[p(r.item, S.cell), status_chip(r.state), p(r.note, S.cell)] for r in rows])
    return [
        p(title, S.h3),
        Spacer(1, 2 * mm),
        table(body, [FULL * 0.30, FULL * 0.13, FULL * 0.57]),
        Spacer(1, 5 * mm),
    ]


GATES = (
    Row("G1 causality", "green", "500 taus x 20 seeds, both cut modes, four zoo strategies"),
    Row(
        "G2 twin engines", "green", "max relative deviation 0.000e+00; fixture widened to 400 bars"
    ),
    Row("G3 analytic recovery", "green", "simple and log Sharpe; beta_simple - beta_log = sigma/2"),
    Row("G4 zero-cost identity", "green", "exact float equality"),
    Row("G5 cost monotonicity", "green", "break-even cost finite and reported"),
    Row("G6 null calibration", "green", "bounds set from a 15-world replication study"),
    Row("G7 leakage trap", "green", "G1 flags the injected leak"),
    Row("G8 walk-forward", "green", "purge and embargo; zero index overlap asserted"),
    Row("G9 overfit probability", "green", "12,870 splits; null calibrates to 0.5 at every S"),
    Row("G10 reproducibility", "green", "two runs byte-identical: three figures and metrics.json"),
)

INVARIANTS = (
    Row("B1 no network before Gate 0", "green", "phases 0-4 offline; one deliberate fetch script"),
    Row("B2 every number has an error bar", "green", "structural in Performance and MetricsReport"),
    Row("B3 append-only trials ledger", "green", "ledger required on both engines; N never typed"),
    Row("B4 strategies emit weights", "green", "sizing and execution belong to the engine"),
    Row("B5 both engines identical", "green", "changed together; G2 re-run in the same commit"),
    Row("B6 no bfill", "green", "forward-fill only, declared"),
    Row("B7 frozen dataclasses", "green", "no in-place mutation of Bars or Result"),
    Row("B8 per-observation internally", "green", "annualised only at the reporting boundary"),
    Row("B9 seeds threaded explicitly", "green", "no global RNG anywhere; G10 depends on it"),
    Row("B10 non-excess kurtosis", "green", "fisher=False; asserted against the Lo collapse"),
)

DOCS = (
    Row("00 Gate 0.0 A/B/C", "green", "exact quadrature, diff 0.00e+00; the compensation figure"),
    Row("00 Gate 0.1-0.5", "green", "recovery, convergence, signal, degeneracy, determinism"),
    Row("00 phase ordering", "green", "phases 0 through 7 complete; this is Phase 8+"),
    Row("01 B1 Sharpe and SE", "green", "Lo (2002) non-normal, plus the Newey-West HAC t"),
    Row("01 B2 PSR", "green", "deflated.psr"),
    Row("01 B3 deflated Sharpe", "green", "now fed an N read from the ledger"),
    Row("01 B4 effective trials", "green", "participation ratio primary, clustering secondary"),
    Row("01 B5 stationary bootstrap", "green", "twin implementations, bitwise identical"),
    Row("01 B6 PBO via CSCV", "green", "G9"),
    Row("01 Part C trials ledger", "green", "implemented; the trap counts 50, not 1"),
    Row("01 Part D reporting contract", "green", "ten fields, written to metrics.json"),
    Row("01 Part E selection rules", "green", "four rules; the PBO-vs-temperature figure"),
    Row("02 Parts A-H", "green", "A4 ruled and verified live; data contract with manifest"),
    Row("03 Parts B, G, H", "green", "ten invariants held; Part H decisions respected"),
    Row(
        "PLAYBOOK Phase 6 zoo", "partial", "TSMOM added; MA, mean reversion, vol target, null done"
    ),
    Row("PLAYBOOK Phase 7 cross-sectional", "open", "N-asset universe and factor attribution"),
    Row("PLAYBOOK Phase 8 reporting", "partial", "metrics.json done; tearsheet and surfaces open"),
    Row("PLAYBOOK Phase 9 writeup", "open", "docs/research-note.md"),
)

SPY_ROWS = (
    ("TSMomentum(12m,1m)", "+0.606", "0.340", "1.56", "+1.93"),
    ("MOP form: VolTarget(0.40, cap 2)", "+0.673", "0.340", "3.27", "+2.10"),
    ("TSMomentum(12m,3m)", "+0.203", "0.336", "1.56", "+0.65"),
    ("TSMomentum(6m,1m)", "+0.166", "0.327", "3.38", "+0.55"),
    ("BuyAndHold", "+0.830", "0.325", "0.00", "+2.80"),
    ("MACrossover(20,50)", "+0.217", "0.319", "9.41", "+0.72"),
)

PROCESS_ROWS = (
    ("GBM, mu = 0.08 (the null)", "-0.141", "0.145", "-1.0", "no edge, as it must be"),
    ("GBM, mu = 0.00 (the null)", "-0.134", "0.099", "-1.4", "no edge"),
    ("persistent drift, psi = 0.98", "+0.804", "0.203", "+4.0", "power"),
    ("persistent drift, psi = 0.99", "+1.659", "0.262", "+6.3", "power"),
    ("stationary AR(1)", "-1.041", "0.102", "-10.2", "loses to its adversary"),
)

GENERATOR_ROWS = (
    ("GBM, no autocorrelation", "+0.092", "0.092"),
    ("AR(1) RETURNS, phi = 0.10", "+0.041", "0.086"),
    ("AR(1) RETURNS, phi = 0.30", "+0.025", "0.109"),
    ("persistent DRIFT, psi = 0.95", "+0.516", "0.169"),
    ("persistent DRIFT, psi = 0.99", "+1.866", "0.225"),
    ("persistent DRIFT, psi = 0.995", "+2.572", "0.257"),
)


def story() -> list[Any]:
    sha = git("rev-parse", "--short", "HEAD") or "unknown"
    flow: list[Any] = [
        p("falsify — Phase 8", S.h1),
        p(
            "Time-series momentum, measured against Moskowitz-Ooi-Pedersen (2012), and the "
            "state of every specification document the build was given.",
            S.lede,
        ),
        Spacer(1, 4 * mm),
        p(f"Commit {sha} · {REPORT_DATE.isoformat()} · {collected()}", S.small),
        Spacer(1, 7 * mm),
        p("What Phase 8 was for", S.h2),
        p(
            "PLAYBOOK names time-series momentum as <b>“a free calibration check against the "
            "literature”</b>: published Sharpe near 0.8, and <b>“if yours comes out at 3.0 on "
            "SPY, you have a bug.”</b> The value of the check is that it is falsifiable "
            "before the code is written — the number to beat was fixed by someone else, "
            "years ago, and cannot be adjusted after the fact.",
            S.body,
        ),
        Spacer(1, 3 * mm),
        p(
            "It is an upper bound, not a target, and the distinction matters. The published "
            "0.8 is a diversified basket of 58 futures across four asset classes, each "
            "volatility-scaled and equally weighted. Most of that Sharpe is diversification. "
            "A single equity index cannot reproduce it, and a single equity index that "
            "appeared to would be reporting luck.",
            S.body,
        ),
        Spacer(1, 6 * mm),
        p("On real SPY, 2015-2024, zero cost", S.h2),
        table(
            [
                ["strategy", "SR", "±SE", "turns/yr", "HAC t"],
                *[
                    [
                        p(a, S.cell),
                        p(b, S.cell_mono),
                        p(c, S.cell_mono),
                        p(d, S.cell_mono),
                        p(e, S.cell_mono),
                    ]
                    for a, b, c, d, e in SPY_ROWS
                ],
            ],
            [FULL * 0.40, FULL * 0.14, FULL * 0.14, FULL * 0.16, FULL * 0.16],
        ),
        Spacer(1, 4 * mm),
        p(
            "0.606 against a published 0.8 looks like a hit and is not one. The standard "
            "error is 0.340 — wide enough to contain 0.8, 0.606 and zero at once. What the "
            "check licenses is the negative: nothing is near 3.0, so nothing indicates a bug.",
            S.body,
        ),
        Spacer(1, 3 * mm),
        p(
            "<b>Buy-and-hold beat every configuration</b>, at +0.830, and is the only strategy "
            "in the zoo whose Newey-West t-statistic clears 2. That is stated here rather "
            "than buried, and it is the same answer the A4 live test reached from the other "
            "direction.",
            S.body,
        ),
        Spacer(1, 6 * mm),
        p("The synthetic triple: null, power, adversary", S.h2),
        p("16 paths of 1,600 bars each, TimeSeriesMomentum(12m, 1m), zero cost.", S.small),
        Spacer(1, 2 * mm),
        table(
            [
                ["process", "SR", "±SE", "SE from 0", "verdict"],
                *[
                    [
                        p(a, S.cell),
                        p(b, S.cell_mono),
                        p(c, S.cell_mono),
                        p(d, S.cell_mono),
                        p(e, S.cell),
                    ]
                    for a, b, c, d, e in PROCESS_ROWS
                ],
            ],
            [FULL * 0.30, FULL * 0.12, FULL * 0.12, FULL * 0.16, FULL * 0.30],
        ),
        Spacer(1, 4 * mm),
        p(
            "The last row is the one worth having. A trend follower loses at ten standard "
            "errors on the exact stationary process <font face='Courier'>CausalZScore</font> "
            "earns on — same engine, same costs, same measurement. An edge is a property of "
            "a process, not of a strategy.",
            S.body,
        ),
        PageBreak(),
        p("The generator that had to be replaced", S.h2),
        p(
            "A power test needs a process with a trend in it, and the obvious construction — "
            "autocorrelate the returns — does not work. This was measured rather than "
            "reasoned about, after the first version was already written:",
            S.body,
        ),
        Spacer(1, 3 * mm),
        table(
            [
                ["process", "12-month TSMOM SR", "±SE"],
                *[
                    [p(a, S.cell), p(b, S.cell_mono), p(c, S.cell_mono)]
                    for a, b, c in GENERATOR_ROWS
                ],
            ],
            [FULL * 0.46, FULL * 0.27, FULL * 0.27],
        ),
        Spacer(1, 4 * mm),
        p(
            "Nothing, at a daily φ of 0.30 that no real market approaches. The reason is "
            "arithmetic: an AR(1) has autocorrelation φ<super>k</super>, and 0.30<super>252"
            "</super> is zero. A signal that integrates a year of returns cannot read a "
            "structure that has decayed within a fortnight. <b>Momentum is not lag-1 "
            "autocorrelation.</b>",
            S.body,
        ),
        Spacer(1, 3 * mm),
        p(
            "Had the power test shipped on that generator it would have failed the strategy "
            "while the strategy was correct — the worst kind of test, one that is wrong in "
            "the direction of looking rigorous. The last row also explains why 3.0 is a sound "
            "bug threshold: even a process far more obliging than any market only reaches 2.57.",
            S.body,
        ),
        Spacer(1, 6 * mm),
        p("Three defects found by running it", S.h2),
        table(
            [
                ["what", "how it surfaced"],
                [
                    p(
                        "<font face='Courier'>lookback</font> must be "
                        "<font face='Courier'>window + 1</font>",
                        S.cell,
                    ),
                    p(
                        "The engine slices <font face='Courier'>signals[start - lag:]</font> "
                        "and reads from exactly "
                        "<font face='Courier'>lookback</font>. Declaring the window handed it "
                        "one NaN and the run was refused.",
                        S.cell,
                    ),
                ],
                [
                    p("G2's fixture was too short", S.cell),
                    p(
                        "TSMOM joined ZOO so G1 and G2 certify it. G2's 220 bars left zero "
                        "reported bars against a "
                        "253-bar lookback — the fixture was wrong, not the gate. It is 400 now.",
                        S.cell,
                    ),
                ],
                [
                    p("Vol targeting is not a no-op", S.cell),
                    p(
                        "Synthetic constant-vol data said the 40% cap binds everywhere. On "
                        "SPY it changes the weight at "
                        "exactly 65 of 2,261 bars — exactly the 65 above 40% vol — from "
                        "2020-03-18 to 2020-06-18.",
                        S.cell,
                    ),
                ],
            ],
            [FULL * 0.28, FULL * 0.72],
        ),
        Spacer(1, 3 * mm),
        p(
            "That third one is the argument for live tests in one line: an overlay exercised "
            "only on synthetic data looks like dead code, because synthetic data has no "
            "volatility clustering for it to respond to.",
            S.body,
        ),
        PageBreak(),
        p("Checklists", S.h2),
        p(
            "Every gate and every invariant the build was given, against what exists in the "
            "tree at this commit.",
            S.body,
        ),
        Spacer(1, 5 * mm),
        *checklist("Gate set — PLAYBOOK Part 2", GATES),
        *checklist("Invariants — 03 Part B", INVARIANTS),
        PageBreak(),
        *checklist("Specification documents", DOCS),
        p("What is next", S.h2),
        table(
            [
                ["item", "why"],
                [
                    p("Phase 7 — cross-sectional and attribution", S.cell),
                    p(
                        "N-asset universe, decile long/short, Fama-French plus momentum "
                        "with Newey-West standard "
                        "errors. This is the revisit condition Part H decision 1 named for "
                        "the SPY-only universe.",
                        S.cell,
                    ),
                ],
                [
                    p("Phase 8 — tearsheet and parameter surfaces", S.cell),
                    p(
                        "The in-sample against out-of-sample heatmap PLAYBOOK calls the "
                        "best figure in the repo.",
                        S.cell,
                    ),
                ],
                [
                    p("Phase 9 — research note", S.cell),
                    p(
                        "docs/research-note.md, structured as a paper, with the limitations "
                        "section the data biases "
                        "already require.",
                        S.cell,
                    ),
                ],
            ],
            [FULL * 0.34, FULL * 0.66],
        ),
        Spacer(1, 6 * mm),
        p(
            f"Generated {REPORT_DATE.isoformat()} from {sha}. Every statistic in this report is "
            "asserted by a test in the tree and carries the standard error it was measured "
            "with; none was rounded toward a nicer number.",
            S.small,
        ),
    ]
    return flow


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title=f"falsify — Phase 8 status {REPORT_DATE.isoformat()}",
        author="falsify",
    )
    doc.build(story(), onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
