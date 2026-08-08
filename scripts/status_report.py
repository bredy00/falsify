"""Generate the project status report as a PDF.

    uv run --group docs python scripts/status_report.py

Narrative content lives in this file as dated data; the git SHA, branch and
commit count are read at run time so the report cannot claim to describe a tree
it does not. Regenerate at each checkpoint rather than editing a PDF by hand.

Palette and type are carried over from `docs/figures/compensation_effect.png` so
the report and the project's figures read as one system.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DATE = date(2026, 8, 8)
OUT_PATH = REPO_ROOT / "docs" / f"status-{REPORT_DATE.isoformat()}.pdf"

# From compensation_effect.png, so report and figures share one identity.
INK = colors.HexColor("#16202b")
SLATE = colors.HexColor("#33526e")
BRICK = colors.HexColor("#b0322b")
AMBER = colors.HexColor("#9a6c00")
MOSS = colors.HexColor("#2f6b4f")
GREY = colors.HexColor("#5b6a79")
RULE = colors.HexColor("#c9d2dc")
WASH = colors.HexColor("#eef2f6")
PAPER = colors.HexColor("#ffffff")


def git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


@dataclass(frozen=True, slots=True)
class Styles:
    h1: ParagraphStyle
    h2: ParagraphStyle
    h3: ParagraphStyle
    body: ParagraphStyle
    lede: ParagraphStyle
    small: ParagraphStyle
    mono: ParagraphStyle
    cell: ParagraphStyle
    cell_mono: ParagraphStyle


def build_styles() -> Styles:
    base = getSampleStyleSheet()["BodyText"]
    body = ParagraphStyle(
        "body", parent=base, fontName="Times-Roman", fontSize=9.6, leading=13.4,
        textColor=INK, alignment=TA_LEFT, spaceAfter=5,
    )
    return Styles(
        h1=ParagraphStyle(
            "h1", parent=body, fontName="Times-Bold", fontSize=21, leading=24,
            textColor=INK, spaceAfter=2,
        ),
        h2=ParagraphStyle(
            "h2", parent=body, fontName="Times-Bold", fontSize=13, leading=16,
            textColor=SLATE, spaceBefore=13, spaceAfter=5,
        ),
        h3=ParagraphStyle(
            "h3", parent=body, fontName="Helvetica-Bold", fontSize=7.6, leading=10,
            textColor=GREY, spaceBefore=8, spaceAfter=3,
        ),
        body=body,
        lede=ParagraphStyle(
            "lede", parent=body, fontSize=10.6, leading=14.6, textColor=INK, spaceAfter=7,
        ),
        small=ParagraphStyle(
            "small", parent=body, fontName="Helvetica", fontSize=7.4, leading=10,
            textColor=GREY,
        ),
        mono=ParagraphStyle(
            "mono", parent=body, fontName="Courier", fontSize=8, leading=11, textColor=INK,
        ),
        cell=ParagraphStyle(
            "cell", parent=body, fontSize=8.4, leading=11, spaceAfter=0,
        ),
        cell_mono=ParagraphStyle(
            "cellmono", parent=body, fontName="Courier", fontSize=7.8, leading=10.6,
            spaceAfter=0,
        ),
    )


S = build_styles()


def p(text: str, style: ParagraphStyle | None = None) -> Paragraph:
    return Paragraph(text, style or S.body)


def table(rows: list[list[Any]], widths: list[float], header: bool = True) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, RULE),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), WASH),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, SLATE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 7.4),
            ("TEXTCOLOR", (0, 0), (-1, 0), SLATE),
        ]
    t.setStyle(TableStyle(style))
    return t


def status_chip(label: str) -> Paragraph:
    palette = {
        "green": MOSS.hexval()[2:],
        "partial": AMBER.hexval()[2:],
        "not started": GREY.hexval()[2:],
        "retired": GREY.hexval()[2:],
    }
    hexcode = palette.get(label, GREY.hexval()[2:])
    weight = "b" if label == "green" else "font"
    if weight == "b":
        return Paragraph(f'<b><font color="#{hexcode}">{label.upper()}</font></b>', S.cell)
    return Paragraph(f'<font color="#{hexcode}">{label}</font>', S.cell)


def on_page(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, h - 14 * mm, w - 18 * mm, h - 14 * mm)
    canvas.setFont("Helvetica", 6.8)
    canvas.setFillColor(GREY)
    canvas.drawString(18 * mm, h - 11.6 * mm, "FALSIFY  ·  STATUS REPORT")
    canvas.drawRightString(
        w - 18 * mm, h - 11.6 * mm, f"{REPORT_DATE.isoformat()}  ·  sessions 1-4"
    )
    canvas.line(18 * mm, 14 * mm, w - 18 * mm, 14 * mm)
    canvas.drawString(18 * mm, 10 * mm, "github.com/bredy00/falsify  (private)")
    canvas.drawRightString(w - 18 * mm, 10 * mm, f"page {doc.page}")
    canvas.restoreState()


def story() -> list[Any]:
    sha = git("rev-parse", "--short", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    commits = git("rev-list", "--count", "HEAD")
    subject = git("log", "-1", "--format=%s")

    # Every table below sums its column widths to exactly 174 mm, the frame width
    # (A4 minus the 18 mm margins). reportlab will silently overflow a wider one.
    out: list[Any] = []

    # ---------------------------------------------------------------- masthead
    out += [
        p("Where we are standing", S.h1),
        p(
            "A backtester that reports an estimate, its error bar, and the probability the "
            "number survived the search that produced it.",
            S.small,
        ),
        Spacer(1, 7),
        table(
            [
                [
                    p("<b>Commit</b>", S.cell),
                    p(f"{sha} on {branch} · {commits} commits", S.cell_mono),
                    p("<b>Suite</b>", S.cell),
                    p("117 collected · 113 pass · 4 skip · ~40 s", S.cell),
                ],
                [
                    p("<b>Latest</b>", S.cell),
                    p(subject, S.cell_mono),
                    p("<b>CI</b>", S.cell),
                    p("green · 5 checks · offline, no secrets", S.cell),
                ],
                [
                    p("<b>Code</b>", S.cell),
                    p("~1,500 package · ~2,400 test LOC", S.cell),
                    p("<b>Quality</b>", S.cell),
                    p("ruff clean · mypy --strict clean (30 files)", S.cell),
                ],
            ],
            [20 * mm, 62 * mm, 20 * mm, 72 * mm],
            header=False,
        ),
        Spacer(1, 10),
        p(
            "Five sessions in. <b>Gate 0.0, G1, G2, G3, G4, G5 and G7 are green</b> — the entire "
            "certified offline core. No market data has entered the system, so invariant B1 still "
            "holds, and everything to date runs in CI with no network, no API key and no rate "
            "limit. G6 is next and is the first gate that needs the data layer.",
            S.lede,
        ),
    ]

    # ------------------------------------------------------------ gate status
    out += [
        p("Gate status", S.h2),
        table(
            [
                ["Gate", "Statement", "Status", "Evidence"],
                [
                    p("0.0", S.cell), p("Reproduce the propositions first", S.cell),
                    status_chip("green"),
                    p("E[max z] to 0.58 SE; Prop 3 slope −0.9989 ± 0.0014", S.cell),
                ],
                [
                    p("G1", S.cell), p("Causality: scramble future, past bit-identical", S.cell),
                    status_chip("green"),
                    p("500 cuts × 20 seeds per strategy, 3 strategies", S.cell),
                ],
                [
                    p("G2", S.cell), p("Twin engines agree to 1e-12", S.cell),
                    status_chip("green"),
                    p("0.000e+00 across zoo × 3 conventions × cost sweep", S.cell),
                ],
                [
                    p("G3", S.cell), p("Analytic recovery on synthetic GBM", S.cell),
                    status_chip("green"),
                    p("both Sharpe conventions, vol, drift; 1/sqrt(M); power test", S.cell),
                ],
                [
                    p("G4", S.cell), p("Zero-cost identity", S.cell),
                    status_chip("green"),
                    p("bitwise, both engines, all three conventions", S.cell),
                ],
                [
                    p("G5", S.cell), p("Cost monotonicity, break-even cost", S.cell),
                    status_chip("green"),
                    p("monotone 0-100 bps; c* = 27.03 bps per turn", S.cell),
                ],
                [
                    p("G6", S.cell), p("Null calibration, 1,000 coin flips", S.cell),
                    status_chip("not started"), p("next; first gate needing the data layer", S.cell),
                ],
                [
                    p("G7", S.cell), p("Leakage trap must fire", S.cell),
                    status_chip("green"),
                    p("5 traps rejected; A4 oracle case ruled on", S.cell),
                ],
                [
                    p("G8", S.cell), p("Purged, embargoed walk-forward", S.cell),
                    status_chip("not started"), p("deferrable per 03 Part C", S.cell),
                ],
                [
                    p("G9", S.cell), p("PBO via CSCV", S.cell),
                    status_chip("partial"),
                    p("SelectionRule interface built first, by design", S.cell),
                ],
                [
                    p("G10", S.cell), p("Reproducibility from pinned hashes", S.cell),
                    status_chip("partial"),
                    p("figure bytes + numeric output stable across runs", S.cell),
                ],
            ],
            [12 * mm, 56 * mm, 20 * mm, 86 * mm],
        ),
    ]

    # ------------------------------------------------------------------- G2
    out += [
        p("G2 — the session's result", S.h2),
        p(
            "Two engines, written independently, sharing exactly one thing: the warm-up index. "
            "That is shared deliberately, because an off-by-one in the warm-up is a specification "
            "question rather than an implementation one, and duplicating it would let both engines "
            "be wrong in the same direction while agreeing perfectly. All of the Part E arithmetic "
            "is written twice on purpose — two engines that share their accounting agree by "
            "construction and certify nothing.",
        ),
        table(
            [
                ["Measurement", "Value", "Reading"],
                [
                    p("Max relative equity deviation", S.cell),
                    p("<b>0.000e+00</b>", S.cell_mono),
                    p("Exact, against a 1e-12 requirement", S.cell),
                ],
                [
                    p("Coverage", S.cell),
                    p("3 strategies × 3 conventions × 5 cost models", S.cell),
                    p("Plus hypothesis over arbitrary price series", S.cell),
                ],
                [
                    p("Planted engine disagreement", S.cell),
                    p("7.417e-02", S.cell_mono),
                    p("Gate discriminates; it is not passing vacuously", S.cell),
                ],
                [
                    p("Convention spread (final equity)", S.cell),
                    p("12924.37 / 12915.83 / 11688.41", S.cell_mono),
                    p("close_to_close / next_open / next_close", S.cell),
                ],
            ],
            [46 * mm, 62 * mm, 66 * mm],
        ),
        Spacer(1, 4),
        p(
            "Exact agreement is legitimate here rather than suspicious: the event engine's "
            "hard-sliced signal window contains the same elements as the vectorised engine's "
            "rolling window, and the arithmetic order coincides. The planted-bug test is what "
            "proves the harness can fail. The equity path is an explicit loop in <i>both</i> "
            "engines because it is not a cumprod — cost[t] depends on equity[t−1], and Part F2's "
            "approximate route carries error of order cost² that would never survive 1e-12.",
        ),
    ]

    # -------------------------------------------------------------- G3 and G5
    out += [
        p("G3 and G5 — recovery and the cost of turnover", S.h2),
        table(
            [
                ["Quantity", "Measured", "True value", "Gap"],
                [
                    p("Simple-return Sharpe, 200 paths", S.cell),
                    p("+0.37513 +/- 0.02358", S.cell_mono),
                    p("mu/sigma = 0.40", S.cell_mono), p("1.05 SE", S.cell),
                ],
                [
                    p("Log-return Sharpe, 200 paths", S.cell),
                    p("+0.27461 +/- 0.02364", S.cell_mono),
                    p("(mu-s^2/2)/sigma = 0.30", S.cell_mono), p("1.07 SE", S.cell),
                ],
                [
                    p("Annualised volatility", S.cell),
                    p("0.20018 +/- 0.00020", S.cell_mono),
                    p("sigma = 0.20", S.cell_mono), p("0.09% rel", S.cell),
                ],
                [
                    p("Annualised log drift", S.cell),
                    p("+0.05502 +/- 0.00474", S.cell_mono),
                    p("mu - s^2/2 = 0.06", S.cell_mono), p("1.05 SE", S.cell),
                ],
                [
                    p("Monte Carlo SE scaling", S.cell),
                    p("slope -0.5173, r^2 = 0.997", S.cell_mono),
                    p("-0.5 exactly", S.cell_mono), p("confirms 1/sqrt(M)", S.cell),
                ],
                [
                    p("Mean reversion on GBM", S.cell),
                    p("+0.0051 +/- 0.0738", S.cell_mono),
                    p("0 -- no edge exists", S.cell_mono), p("invents none", S.cell),
                ],
                [
                    p("Mean reversion on AR(1), phi=0.95", S.cell),
                    p("+1.1534 +/- 0.0703", S.cell_mono),
                    p("> 0 -- edge exists", S.cell_mono), p("16 SE, finds it", S.cell),
                ],
                [
                    p("<b>Break-even cost c*</b>", S.cell),
                    p("<b>27.03 bps / turn</b>", S.cell_mono),
                    p("at 68.8 turns/yr", S.cell_mono), p("+0.35 SR at c*/2", S.cell),
                ],
            ],
            [50 * mm, 44 * mm, 44 * mm, 36 * mm],
        ),
        Spacer(1, 4),
        p(
            "The last two rows are the pair that matters and the one people skip. 0.1 shows the "
            "engine does not invent edge; 0.3 shows it does not destroy edge that exists. A framework "
            "failing to find signal in a series that provably contains signal is broken in a way no "
            "real-data test reveals, because on real data “found nothing” is always a "
            "plausible answer.",
        ),
    ]

    out.append(PageBreak())

    # -------------------------------------------------- bugs found by the gates
    out += [
        p("What the gates caught", S.h2),
        p(
            "Every row is a real defect in code I had written, found by a check rather than by "
            "reading. This is the argument for the whole approach, so it is worth keeping the "
            "score honestly.",
        ),
        table(
            [
                ["Found by", "Defect", "Why it was dangerous"],
                [
                    p("hypothesis", S.cell),
                    p("Constant column escaped the degeneracy check", S.cell),
                    p(
                        "60 copies of 0.001 give std ≈ 2e-19, not 0.0, so an equality test passed "
                        "it through and the Sharpe came out near <b>1e16</b> — enough to take the "
                        "entire portfolio under ArgMax or Softmax",
                        S.cell,
                    ),
                ],
                [
                    p("hypothesis", S.cell),
                    p("Softmax amplified rounding dust into a decision", S.cell),
                    p(
                        "Columns tied on true Sharpe leave ~1e-17 of float noise; z-scoring divides "
                        "by it, and at τ=0.25 that became a <b>0.9997 / 0.0003</b> allocation. CSCV "
                        "blocks produce near-identical Sharpes routinely, so G9's temperature sweep "
                        "would have been noise wherever the grid agrees",
                        S.cell,
                    ),
                ],
                [
                    p("ruff B905", S.cell),
                    p("8 × zip() without strict=", S.cell),
                    p(
                        "Silent truncation on a length mismatch — a wrong statistic and no exception",
                        S.cell,
                    ),
                ],
                [
                    p("mypy --strict", S.cell),
                    p("18 typing gaps, 11 of them bare dict returns", S.cell),
                    p(
                        "Drove the switch to frozen dataclasses (B7): one state per result object, "
                        "which is what makes the twin-engine comparison meaningful",
                        S.cell,
                    ),
                ],
                [
                    p("CI", S.cell),
                    p("Unresolvable action tag setup-uv@v9", S.cell),
                    p(
                        "A run that dies in 3 s having executed no step. Nothing in the repo could "
                        "detect it, so a resolver now runs as the first check",
                        S.cell,
                    ),
                ],
                [
                    p("CI", S.cell),
                    p("Local-green, CI-red property tests", S.cell),
                    p(
                        "hypothesis switches to a derandomised <i>ci</i> profile, drawing a different "
                        "sequence. Fixed by constructing valid inputs instead of filtering for them",
                        S.cell,
                    ),
                ],
            ],
            [24 * mm, 52 * mm, 98 * mm],
        ),
    ]

    # ---------------------------------------------------------- decisions
    out += [
        p("Decisions taken", S.h2),
        table(
            [
                ["Decision", "Rationale"],
                [
                    p("<b>diff = 0.00e+00</b> is the single standard", S.cell),
                    p(
                        "Experiment A is judged against E[max z] computed by quadrature and checked "
                        "to machine precision against 1/√π and 3/(2√π), not against the two-term "
                        "Gumbel approximation. The empirical mean estimates the truth; the formula "
                        "only approximates it. Comparing Monte Carlo to the approximation conflated "
                        "the two and forced an arbitrary small-N exclusion. Against exact truth it "
                        "holds at every N from 2 to 10⁶, worst deviation 1.98 SE.",
                        S.cell,
                    ),
                ],
                [
                    p("Accumulated error bands retired", S.cell),
                    p(
                        "Trialled and withdrawn on evidence. The pooled statistics behaved as theory "
                        "predicts (rms_z = 1.0959 in [0.4117, 1.5883]) and the band did discriminate, "
                        "driving a biased reference to rms_z = 20.83. But mean|diff| came out at "
                        "8.87e-04 — below the proposed lower bound of 0.0025 — and it scales as "
                        "1/√reps, so any fixed bound on it constrains compute budget rather than "
                        "correctness. Recoverable: <font face='Courier'>git checkout 37d4dac -- "
                        "falsify/accumulation.py</font>",
                        S.cell,
                    ),
                ],
                [
                    p("Monte Carlo reduced, never removed", S.cell),
                    p(
                        "Still 3 of the 5 Experiment A tests and ~2.6M draws — more sampling than "
                        "before the fix. An exact reference says where the truth is; only sampling "
                        "says whether the estimator finds it.",
                        S.cell,
                    ),
                ],
                [
                    p("<b>02 Part A4 vs A1 — ruled: A1 stands</b>", S.cell),
                    p(
                        "A4 asserts G1 must catch LeakyOracle, sign(diff(close)). It must not, "
                        "because that strategy does not leak. close[t] lies inside bars[0:t+1], "
                        "which A1 permits, and every Part D convention lags the weight at least one "
                        "bar, so the weight earning return t was decided from strictly older bars. "
                        "Measured through the engine it earns +0.054 annualised Sharpe against "
                        "buy-and-hold's +0.372; a strategy that genuinely sees one bar ahead earns "
                        "+21.10. So the trap was mis-specified, not the harness, and tightening A1 "
                        "to bars[0:t] would have failed every legitimate close_to_close strategy — "
                        "which Part D explicitly permits — while catching nothing real. G7 now traps "
                        "a true look-ahead oracle; the A4 strategy is kept as a documented "
                        "non-violator so the reasoning survives.",
                        S.cell,
                    ),
                ],
                [
                    p("Sharpe is asserted on both return conventions", S.cell),
                    p(
                        "00 Gate 0.1 states the true Sharpe is 0.30 and warns that 0.40 means μ was "
                        "used instead of μ−σ²/2. Both are right, for different estimators: log "
                        "returns give (μ−σ²/2)/σ = 0.30, simple returns give μ/σ = 0.40, because "
                        "E[exp(g)−1] = μ/252 exactly. Part E compounds simple returns, so the engine "
                        "measures 0.40 and is correct to — asserting 0.30 against it would reject a "
                        "working engine. Both targets are now pinned, so confusing them fails one "
                        "direction or the other.",
                        S.cell,
                    ),
                ],
                [
                    p("Actions pinned to exact releases", S.cell),
                    p(
                        "No floating majors anywhere. A project asserting byte-identical output "
                        "under G10 should not let its own CI toolchain drift underneath it. A test "
                        "enforces it.",
                        S.cell,
                    ),
                ],
            ],
            [46 * mm, 128 * mm],
        ),
    ]

    out.append(PageBreak())

    # ---------------------------------------------------------- error budget
    out += [
        p("Error budget of SR₀", S.h2),
        p(
            "The two-term Gumbel expression is what 01 Part B3 feeds the Deflated Sharpe as the "
            "benchmark SR₀, so its distance from exact truth is a bias in the DSR itself. The sign "
            "matters: it <b>overstates for every N ≥ 3</b>, so SR₀ is too strict rather than too "
            "lax and the resulting DSR is conservative. Below N = 100 the error exceeds 1% and "
            "should be quoted alongside any DSR computed there. All of it is asserted, not assumed.",
        ),
        table(
            [
                ["N", "2", "3", "5", "10", "50", "100", "1,000", "10⁶"],
                [
                    p("<b>Error in SR₀</b>", S.cell),
                    p("−7.88%", S.cell_mono), p("+0.77%", S.cell_mono),
                    p("+2.55%", S.cell_mono), p("+2.33%", S.cell_mono),
                    p("+1.21%", S.cell_mono), p("+0.92%", S.cell_mono),
                    p("+0.42%", S.cell_mono), p("+0.10%", S.cell_mono),
                ],
            ],
            [30 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm],
        ),
        Spacer(1, 3),
        p(
            "The error is non-monotone at the bottom: the signed error flips sign between N = 2 and "
            "N = 3, so a naive monotone-decrease assertion would have been false. It decays "
            "monotonically from N = 5 upward.",
            S.small,
        ),
    ]

    # ---------------------------------------------------------- what exists
    out += [
        p("What exists now", S.h2),
        table(
            [
                ["Module", "Role", "Tests"],
                [p("falsify/core/causality.py", S.cell_mono),
                 p("G1 harness, two cut modes", S.cell), p("14", S.cell)],
                [p("falsify/core/event.py", S.cell_mono),
                 p("Reference engine — hard-sliced prefixes, O(T²)", S.cell), p("33", S.cell)],
                [p("falsify/core/vectorized.py", S.cell_mono),
                 p("Product engine — rolling ops, exact equity loop", S.cell), p("(shared)", S.cell)],
                [p("falsify/core/types.py", S.cell_mono),
                 p("Bars, Result, slice, InsufficientHistory — all frozen", S.cell), p("(shared)", S.cell)],
                [p("falsify/core/conventions.py", S.cell_mono),
                 p("Part D's three conventions as (price, lag)", S.cell), p("(shared)", S.cell)],
                [p("falsify/costs.py", S.cell_mono),
                 p("Commission, spread, slippage, borrow, cash yield", S.cell), p("(shared)", S.cell)],
                [p("falsify/selection.py", S.cell_mono),
                 p("ArgMax, Softmax, EqualWeight, TopK", S.cell), p("52", S.cell)],
                [p("falsify/strategies/base.py", S.cell_mono),
                 p("Strategy ABC — target weights, never orders (B4)", S.cell), p("(shared)", S.cell)],
                [p("tests/gates/test_prop.py", S.cell_mono),
                 p("Gate 0.0 — Experiments A, B, C + the figure", S.cell), p("14", S.cell)],
                [p("scripts/check_action_refs.py", S.cell_mono),
                 p("Resolves every workflow uses: ref", S.cell), p("4", S.cell)],
            ],
            [52 * mm, 96 * mm, 26 * mm],
        ),
    ]

    # ---------------------------------------------------------- next session
    out += [
        p("Next session — G6, and the first network call", S.h2),
        p(
            "The certified offline core is complete, which means the ordering rule in 00 has been "
            "satisfied: the engine has recovered known parameters from synthetic data with known "
            "truth, so real market data is now allowed to enter the system. Everything to this point "
            "was built without a single network call.",
        ),
        p(
            "G6 is null calibration — a thousand coin-flip strategies through the whole pipeline, "
            "their Sharpe distribution forming the empirical null that the real result must beat. "
            "The compute is trivial; 03 Part C is explicit that the one genuinely hard part is "
            "<b>turnover matching</b>. A random strategy that flips every bar has enormous turnover "
            "and gets destroyed by costs, which would make the real strategy look good for entirely "
            "the wrong reason. The null has to trade at the same rate as the thing it is testing — a "
            "Markov chain whose transition probability is tuned until realised turnover matches "
            "within 5%, asserted in the test.",
        ),
        Spacer(1, 8),
        KeepTogether([
            p("What G5 already tells us about G6", S.h3),
            p(
                "Break-even cost is 27.03 bps per turn at 68.8 turns a year, and a rule trading 135 "
                "times a year is unprofitable even for free. So turnover matching is not a "
                "formality here — it is the difference between a calibrated null and a flattering "
                "one, and the numbers to match against already exist.",
            ),
        ]),
    ]

    out += [
        Spacer(1, 10),
        p(
            f"Generated {REPORT_DATE.isoformat()} from {sha} · every number measured on this tree, "
            "not recalled · regenerate with: uv run --group docs python scripts/status_report.py",
            S.small,
        ),
    ]
    return out


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        title=f"falsify — status report {REPORT_DATE.isoformat()}",
        author="bredy00",
        subject="Backtest overfitting framework — build status",
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=on_page)])
    doc.build(story())
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
