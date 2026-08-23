"""The regression comparison, as a PDF.

    uv run --group docs python scripts/regression_report.py

Style helpers are imported from `status_report` rather than copied, so the two
documents cannot drift apart typographically. The numbers come from
`regression_comparison`, which is also what draws the figure -- so the table and the
picture are guaranteed to describe the same fit rather than two runs that happened to
agree.
"""

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, Image, PageTemplate, Spacer

from scripts.regression_comparison import (
    FIGURE_PATH,
    Panel,
    pipeline_panels,
    synthetic_panels,
)
from scripts.status_report import GREY, RULE, S, on_page, p, table

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DATE = date(2026, 8, 20)
OUT_PATH = REPO_ROOT / "docs" / f"regression-comparison-{REPORT_DATE.isoformat()}.pdf"


def git(*args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
    )
    return out.stdout.strip() if out.returncode == 0 else "unavailable"


def fit_rows(panels: list[Panel]) -> list[list[Any]]:
    rows: list[list[Any]] = [["Panel", "β yx", "β xy", "ρ", "ρ²", "|Δ| identity", "angle", "n"]]
    for panel in panels:
        f = panel.fit
        rows.append(
            [
                p(panel.title, S.cell),
                p(f"{f.beta_yx:+.4f}", S.cell_mono),
                p(f"{f.beta_xy:+.4f}", S.cell_mono),
                p(f"{f.rho:+.4f}", S.cell_mono),
                p(f"{f.rho**2:.4f}", S.cell_mono),
                p(f"{f.rho_squared_identity_error():.1e}", S.cell_mono),
                p(f"{f.angle_between_lines_degrees():.1f}°", S.cell_mono),
                p(f"{f.n:,}", S.cell_mono),
            ]
        )
    return rows


def story(synthetic: list[Panel], pipeline: list[Panel]) -> list[Any]:
    sha = git("rev-parse", "--short", "HEAD")
    out: list[Any] = [
        p("Two regressions, one scatter", S.h1),
        p(
            "Gate 0.0's figure, regenerated against everything built since — and the "
            "decomposition the single OLS line was hiding.",
            S.small,
        ),
        Spacer(1, 8),
        p(
            "The original figure drew one line per panel: the OLS regression of out-of-sample "
            "Sharpe on in-sample Sharpe. That line answers one question — <i>given x, predict "
            "y</i> — and a scatter supports three. This regenerates the same panels with all "
            "three, and adds a second row asking the same question of the certified engine, "
            "real transaction costs, purged walk-forward folds and the overlay stack.",
            S.lede,
        ),
        p("The three quantities, and the identity binding them", S.h2),
        table(
            [
                ["Quantity", "Definition", "What it answers"],
                [
                    p("<b>β<sub>yx</sub></b>", S.cell),
                    p("cov(X,Y) / var(X)", S.cell_mono),
                    p(
                        "Given x, predict y. Minimises <i>vertical</i> error. The original line.",
                        S.cell,
                    ),
                ],
                [
                    p("<b>β<sub>xy</sub></b>", S.cell),
                    p("cov(X,Y) / var(Y)", S.cell_mono),
                    p(
                        "Given y, predict x. Minimises <i>horizontal</i> error. A different line.",
                        S.cell,
                    ),
                ],
                [
                    p("<b>ρ</b>", S.cell),
                    p("cov(X,Y) / (sd<sub>X</sub> · sd<sub>Y</sub>)", S.cell_mono),
                    p("How tight the association is, in no units at all.", S.cell),
                ],
            ],
            [26 * mm, 54 * mm, 94 * mm],
        ),
        Spacer(1, 5),
        p(
            "They are bound exactly: <b>ρ² = β<sub>yx</sub> · β<sub>xy</sub></b>, and equally "
            "<b>β<sub>yx</sub> = ρ · sd<sub>Y</sub>/sd<sub>X</sub></b>. Both are asserted to "
            "machine precision over arbitrary inputs in <font face='Courier'>tests/"
            "test_regression.py</font>, and the residual |Δ| is printed in every panel below so "
            "a reader can check rather than trust. It holds only when all three share the same "
            "<font face='Courier'>ddof</font>; mixing sample and population conventions breaks "
            "it by a factor of n/(n−1), which is small, plausible-looking and wrong.",
        ),
        p(
            "<b>β<sub>yx</sub> is not 1/β<sub>xy</sub></b> unless |ρ| = 1. Each regression is "
            "pulled toward the axis it predicts, so the two lines open a scissor whose angle is "
            "a direct reading of how far the relationship is from deterministic: shut at ρ² = 1, "
            "a right angle at ρ = 0. They cross at the centroid, which is the one point neither "
            "regression can get wrong.",
        ),
    ]

    if FIGURE_PATH.exists():
        out += [
            Spacer(1, 6),
            Image(str(FIGURE_PATH), width=174 * mm, height=174 * mm * 1560 / 2475),
            Spacer(1, 4),
        ]

    out += [
        p("Measured", S.h2),
        table(
            fit_rows(synthetic + pipeline),
            [46 * mm, 19 * mm, 19 * mm, 17 * mm, 17 * mm, 24 * mm, 16 * mm, 16 * mm],
        ),
        Spacer(1, 5),
        p(
            "Top three rows are session 1's synthetic construction, unchanged. Bottom three are "
            "the same three regimes measured through the built pipeline.",
            S.small,
        ),
        p("What the second line reveals", S.h2),
        p(
            "<b>1. The identity holds everywhere.</b> |Δ| ranges from 0.0e+00 to 1.7e-16 across "
            "all six panels — machine precision. The decomposition is self-consistent, so the "
            "three numbers in each panel genuinely describe one scatter rather than three "
            "separate calculations that happen to sit near each other.",
        ),
        p(
            "<b>2. A single slope can badly overstate a relationship.</b> On <i>Engine on GBM</i> "
            "the original figure's line would have read β<sub>yx</sub> = −0.360 — a visible "
            "negative relationship, and exactly the compensation effect one goes looking for. "
            "The other slope is β<sub>xy</sub> = −0.028, thirteen times smaller, and ρ² = 0.0100. "
            "The relationship explains <b>one per cent</b> of the variance. The asymmetry is not "
            "an error in either slope; it is what a near-zero correlation looks like when the two "
            "variables have very different dispersions, and only the correlation says so. This is "
            "the strongest argument in the document for reporting all three.",
        ),
        p(
            "<b>3. The sign flips between the rows, and that is correct.</b> Synthetic AR(1) gives "
            "ρ = −0.4749: the in-sample winner reverts, which is Proposition 5. The pipeline on "
            "AR(1) gives ρ = <b>+0.7216</b>: in-sample ranking predicts out-of-sample. The two "
            "panels ask different questions. The synthetic panel ranks noise realisations of one "
            "process, where the ranking is pure luck and must revert. The pipeline panel ranks "
            "configurations with genuinely different merit — slow z-scores earn on a "
            "mean-reverting series, trend-followers lose on it — so ranking carries real "
            "information. A negative slope is the signature of selecting among configurations "
            "with no true differential edge, which is the GBM panel, not this one.",
        ),
        p(
            "<b>4. The angle is the readable summary.</b> 87.7° on the random walk (knowing x "
            "tells you nothing), 39.2° on synthetic AR(1), 0.1° under the common-mean constraint "
            "where the lines lie on top of each other and the relationship is deterministic. It "
            "carries the same content as ρ² but is legible at a glance from the picture.",
        ),
        p(
            "<b>5. Sample size differs sharply between the rows.</b> The synthetic panels carry "
            "n = 1,000 independent paths; the pipeline panels carry n = 60 to 90, being folds × "
            "configurations. The bottom row's estimates are correspondingly noisier, and that is "
            "a limitation of the comparison rather than a finding — worth remembering before "
            "reading much into the exact values there.",
        ),
        p("On numerical methods", S.h2),
        p(
            "Nothing here uses an iterative or approximate numerical method: every quantity is a "
            "closed-form moment computed in one pass, which is why the identity closes to 1e-16 "
            "rather than to a tolerance. Where the project does approximate — the two-term Gumbel "
            "expression for SR₀, whose error against exact quadrature is characterised at Gate "
            "0.0 — the approximation is measured rather than assumed. That is the pattern to "
            "carry into any numerical-methods work later: compute the exact reference first, then "
            "quantify what the method costs against it.",
        ),
        Spacer(1, 8),
        p(
            f"Generated {REPORT_DATE.isoformat()} from {sha} · figure and table computed in one "
            "pass so they cannot disagree · regenerate with "
            "uv run --group docs python scripts/regression_report.py",
            S.small,
        ),
    ]
    return out


def main() -> int:
    synthetic, pipeline = synthetic_panels(), pipeline_panels()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        title=f"falsify — regression comparison {REPORT_DATE.isoformat()}",
        author="bredy00",
        subject="Bivariate decomposition of the Gate 0.0 scatter",
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="body",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=on_page)])
    doc.build(story(synthetic, pipeline))
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    _ = (GREY, RULE)  # imported for palette parity with the status report
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
