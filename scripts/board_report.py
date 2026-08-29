"""The project board as a PDF.

    uv run --group docs python scripts/board_report.py

Styling, palette and page furniture are imported from `status_report.py` rather than
copied, so every report the project produces reads as one system. What lives here is the
content: where the board stands, what each specification document asked for against what
exists, and the measurements the current phase produced.

Numbers that can be read from the tree are read at run time -- git SHA, suite size -- so
the report cannot claim to describe a tree it does not. Measured statistics are dated
constants, each traceable to the test that asserts it, because rerunning a factor
regression to render a PDF would be its own kind of silly.
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
REPORT_DATE = date(2026, 8, 28)
OUT_PATH = REPO_ROOT / "docs" / f"status-{REPORT_DATE.isoformat()}.pdf"

FULL = A4[0] - 36 * mm


@dataclass(frozen=True, slots=True)
class Row:
    """One checklist line: what was asked for, and what exists."""

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


def cells(rows: tuple[tuple[str, ...], ...], mono_from: int = 1) -> list[list[Any]]:
    out = []
    for row in rows:
        out.append([p(v, S.cell if i < mono_from else S.cell_mono) for i, v in enumerate(row)])
    return out


GATES = (
    Row("G1 causality", "green", "500 taus x 20 seeds, both cut modes, four zoo strategies"),
    Row("G2 twin engines", "green", "worst 0.000e+00 across 1,890 combinations"),
    Row("G3 analytic recovery", "green", "worst level gap 0.90 SE, paired gap 0.00 SE"),
    Row("G4 zero-cost identity", "green", "288/288 bitwise identical"),
    Row("G5 cost monotonicity", "green", "12/12 sweeps monotone, break-even reported"),
    Row("G6 null calibration", "green", "bounds set from a 15-world replication study"),
    Row("G7 leakage trap", "green", "G1 flags the injected leak"),
    Row("G8 walk-forward", "green", "purge and embargo; zero index overlap asserted"),
    Row("G9 overfit probability", "green", "12,870 splits; null calibrates to 0.5 at every S"),
    Row("G10 reproducibility", "green", "three figures and metrics.json byte-identical"),
)

INVARIANTS = (
    Row("B1 no network before Gate 0", "green", "phases 0-4 offline; one deliberate fetch script"),
    Row("B2 every number has an error bar", "green", "structural in Performance and MetricsReport"),
    Row("B3 append-only trials ledger", "green", "required on both engines; N read, never typed"),
    Row("B4 strategies emit weights", "green", "sizing and execution belong to the engine"),
    Row("B5 both engines identical", "green", "changed together; G2 re-run in the same commit"),
    Row("B6 no bfill", "green", "panel aligns on intersection rather than filling"),
    Row("B7 frozen dataclasses", "green", "checked by health_check"),
    Row("B8 per-observation internally", "green", "annualised only at the reporting boundary"),
    Row("B9 seeds threaded explicitly", "green", "no global RNG; G10 depends on it"),
    Row("B10 non-excess kurtosis", "green", "fisher=False, asserted against the Lo collapse"),
)

DOCS = (
    Row("00 Gate 0.0 A/B/C", "green", "exact quadrature, diff 0.00e+00; the compensation figure"),
    Row("00 Gate 0.1-0.5", "green", "recovery, convergence, signal, degeneracy, determinism"),
    Row("01 B1 Sharpe, SE and HAC t", "green", "Lo (2002) non-normal plus Newey-West"),
    Row("01 B2 / B3 PSR and DSR", "green", "now fed an N read from the ledger"),
    Row("01 B4 effective trials", "green", "participation ratio primary, clustering secondary"),
    Row("01 B5 stationary bootstrap", "green", "twin implementations, bitwise identical"),
    Row("01 B6 PBO via CSCV", "green", "G9"),
    Row("01 Part C trials ledger", "green", "the trap counts 50, not 1"),
    Row("01 Part D reporting contract", "green", "ten fields, written to metrics.json"),
    Row("01 Part E selection rules", "green", "four rules; the PBO-vs-temperature figure"),
    Row("02 Parts A-H", "green", "A4 ruled and verified live; manifest data contract"),
    Row("03 Parts B, G, H", "green", "ten invariants held; Part H revisited on its own terms"),
    Row("PLAYBOOK Phase 6 zoo", "green", "MA, mean reversion, vol target, null, TSMOM"),
    Row(
        "PLAYBOOK Phase 7 cross-sectional",
        "green",
        "N-asset panel, long/short, attribution",
    ),
    Row("PLAYBOOK Phase 8 reporting", "green", "metrics.json, tearsheet, IS/OOS parameter surface"),
    Row("PLAYBOOK Phase 9 writeup", "green", "docs/research-note.md; README leads with the result"),
)

ATTRIBUTION = (
    ("strategy", "alpha/yr", "HAC t", "R2", "Mkt-RF", "SMB", "HML", "UMD"),
    ("BuyAndHold", "+0.12%", "+0.37", "0.995", "+0.975", "-0.126", "+0.015", "-0.007"),
    ("TSMomentum(12m,1m)", "-0.02%", "-0.01", "0.408", "+0.618", "-0.064", "+0.182", "+0.244"),
    ("TSMomentum(12m,3m)", "-6.43%", "-1.47", "0.370", "+0.575", "-0.089", "+0.094", "+0.284"),
    ("XS momentum 12m", "-2.51%", "-1.29", "0.520", "+0.006", "+0.032", "-0.020", "+0.323"),
    ("XS momentum 6m", "-0.60%", "-0.30", "0.366", "-0.026", "+0.010", "+0.017", "+0.259"),
    ("XS momentum 1m", "-5.50%", "-2.49", "0.056", "-0.048", "-0.015", "-0.023", "+0.065"),
)

CROSS_SECTIONAL = (
    ("construction", "SR", "+/-SE", "turns/yr", "HAC t"),
    ("XS momentum 12m, monthly", "-0.065", "0.334", "4.79", "-0.20"),
    ("XS momentum 12m, daily", "-0.023", "0.334", "23.03", "-0.07"),
    ("XS momentum 6m, monthly", "+0.157", "0.325", "6.58", "+0.51"),
    ("XS momentum 1m, monthly", "-0.432", "0.318", "16.31", "-1.37"),
)

CHAIN = (
    ("layer", "measured", "verdict"),
    ("B3 trials ledger", "24 trials for 24 configurations, idempotent on rerun", "counted"),
    ("B4 effective N", "N = 24 -> N_eff 2.21, compression 0.09", "the search was narrow"),
    ("B5 bootstrap CI", "+0.575  [-0.039, +1.276]  95%", "contains zero"),
    ("B1 HAC t-statistic", "+1.922", "below 2"),
    ("G9 PBO", "0.8214 over 24 configurations", "above the 0.5 line"),
    ("Phase 7 alpha", "-0.02%/yr at t = -0.01", "no alpha"),
    ("01 Part D contract", "interpretable = False, ships = False", "does not ship"),
)


def story() -> list[Any]:
    sha = git("rev-parse", "--short", "HEAD") or "unknown"
    flow: list[Any] = [
        p("falsify — the board", S.h1),
        p(
            "Every phase complete. Ten gates green, ten invariants held, and a strategy the "
            "engine declines to trade -- which is the machinery working, not failing.",
            S.lede,
        ),
        Spacer(1, 4 * mm),
        p(f"Commit {sha} · {REPORT_DATE.isoformat()} · {collected()}", S.small),
        Spacer(1, 7 * mm),
        p("The finding", S.h2),
        p(
            "PLAYBOOK asks for one thing above the rest: <i>“If your alpha t-stat drops "
            "below 2 after controlling for momentum, say so in the README. That single act "
            "of intellectual honesty is worth more to a reader than a 2.5 Sharpe.”</i>",
            S.body,
        ),
        Spacer(1, 3 * mm),
        p(
            "It does not drop below 2. It drops to <b>zero</b> — t = -0.01. The +0.606 "
            "annualised Sharpe <font face='Courier'>TimeSeriesMomentum(12m,1m)</font> earns "
            "on SPY over 2015-2024 is fully accounted for by two exposures the regression "
            "names: <b>+0.618 on the market</b>, because a trend follower is long a rising "
            "index most of the decade, and <b>+0.244 on UMD</b>, because it is a momentum "
            "strategy and UMD is the momentum factor. Price both and nothing remains.",
            S.body,
        ),
        Spacer(1, 6 * mm),
        p("Carhart four-factor attribution, HAC standard errors, close-to-close", S.h2),
        table(
            cells(ATTRIBUTION),
            [FULL * 0.24, *[FULL * 0.109] * 7],
        ),
        Spacer(1, 4 * mm),
        p(
            "The buy-and-hold row is the calibration that makes the rest readable: SPY "
            "loads 0.975 on the market at R² 0.995 with no alpha, which is exactly what an "
            "index fund should show. Run against the <font face='Courier'>next_open</font> "
            "convention that same check gave β = 0.367 and R² = 0.159 — because "
            "<font face='Courier'>next_open</font> measures open-to-open while the factors "
            "are close-to-close, and the two correlate 0.40 daily. Every β in the first "
            "table was wrong and nothing else would have flagged it.",
            S.body,
        ),
        PageBreak(),
        p("Every layer agrees, and they were built independently", S.h2),
        p(
            "The strongest check available is not any one number but the fact that seven "
            "independent measurements of the same strategy reach the same verdict. Run on "
            "real SPY through the whole chain:",
            S.body,
        ),
        Spacer(1, 3 * mm),
        table(cells(CHAIN, mono_from=3), [FULL * 0.22, FULL * 0.48, FULL * 0.30]),
        Spacer(1, 4 * mm),
        p(
            "None of these was tuned to agree with the others. The bootstrap knows nothing "
            "about the factor model; PBO knows nothing about the sample length; the ledger "
            "counts configurations without looking at their returns. That they converge is "
            "the argument.",
            S.body,
        ),
        Spacer(1, 6 * mm),
        p("Phase 7 — the long/short spread", S.h2),
        p(
            "Nine SPDR sector funds, dollar-neutral tertile long/short, 2015-2024, zero cost:",
            S.body,
        ),
        Spacer(1, 3 * mm),
        table(cells(CROSS_SECTIONAL), [FULL * 0.34, *[FULL * 0.165] * 4]),
        Spacer(1, 4 * mm),
        p(
            "No edge, and that is the result — nothing approaches a HAC t of 2. The gap "
            "from the time-series case is the content: the same signal earns +0.606 on SPY "
            "and nothing at all when held dollar-neutral. That difference is the market's "
            "drift, which a long-biased strategy collects and a neutral book cannot.",
            S.body,
        ),
        Spacer(1, 3 * mm),
        p(
            "The panel engine is Part E written a second time, so it is held to G2's "
            "standard: at N = 1 it reproduces <font face='Courier'>run_vectorized</font> "
            "bitwise, max |diff| = 0.000e+00. That check caught a real bug on its first "
            "run — the warm-up was <font face='Courier'>max(first_nonzero, lag)</font> "
            "instead of <font face='Courier'>first_nonzero + lag</font>, wrong by 1,313 on "
            "a 10,000 account.",
            S.body,
        ),
        PageBreak(),
        p("Checklists", S.h2),
        p(
            "Every gate, invariant and specification document against what exists at this commit.",
            S.body,
        ),
        Spacer(1, 5 * mm),
        *checklist("Gate set — PLAYBOOK Part 2", GATES),
        *checklist("Invariants — 03 Part B", INVARIANTS),
        PageBreak(),
        *checklist("Specification documents", DOCS),
        p("What is left", S.h2),
        table(
            [
                ["item", "why"],
                [
                    p("Nothing blocking", S.cell),
                    p(
                        "Phases 0 through 9 are complete. The research note is written, the "
                        "README leads with the result, and every figure the specification "
                        "asked for exists.",
                        S.cell,
                    ),
                ],
                [
                    p("If the project continues", S.cell),
                    p(
                        "A longer window is the only thing that would change the verdict: "
                        "10.7 years are needed to distinguish this Sharpe from selection luck "
                        "and 10.0 exist. Evaluating more configurations widens that gap "
                        "rather than closing it.",
                        S.cell,
                    ),
                ],
            ],
            [FULL * 0.34, FULL * 0.66],
        ),
        Spacer(1, 6 * mm),
        p(
            f"Generated {REPORT_DATE.isoformat()} from {sha}. Every statistic here is asserted "
            "by a test in the tree and carries the standard error it was measured with; none "
            "was rounded toward a nicer number.",
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
        title=f"falsify — board {REPORT_DATE.isoformat()}",
        author="falsify",
    )
    doc.build(story(), onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
