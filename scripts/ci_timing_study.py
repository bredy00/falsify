"""How long does CI actually take? A distribution, not a single measurement.

    uv run python scripts/ci_timing_study.py [--local N]

A quoted "CI is green in 40 s" is one draw. This project asserts error bars on every
performance number it reports (B2) and then repeatedly quoted a bare wall-clock time
for its own build, which is the same mistake in a different costume: across this
repository's history that number has been reported as 39, 40, 47 and 55 seconds, all
truthfully, all from single runs.

So this pulls every successful workflow run from the GitHub API and summarises the
distribution, and optionally times the local suite repeatedly for comparison. The
useful output is the spread: if the standard deviation is a meaningful fraction of the
mean, then any single timing -- including a flattering one -- is not evidence about a
change, and a "speedup" smaller than the noise is not a speedup.

Uses the API rather than triggering builds: the history already exists and burning CI
minutes to measure CI would be its own kind of silly.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Run:
    """One completed workflow run. Frozen (B7)."""

    number: int
    seconds: float
    title: str


def _iso(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def fetch_runs(limit: int = 100) -> list[Run]:
    """Every successful run, newest first, with its wall-clock duration."""
    out = subprocess.run(
        [
            "gh", "run", "list", "--limit", str(limit),
            "--json", "databaseId,conclusion,createdAt,updatedAt,displayTitle,number",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        print(f"gh failed: {out.stderr.strip()}", file=sys.stderr)
        return []

    runs: list[Run] = []
    for row in json.loads(out.stdout):
        if row.get("conclusion") != "success":
            continue
        seconds = (_iso(row["updatedAt"]) - _iso(row["createdAt"])).total_seconds()
        if seconds <= 0:
            continue
        runs.append(
            Run(
                number=int(row.get("number", 0)),
                seconds=seconds,
                title=str(row.get("displayTitle", ""))[:52],
            )
        )
    return runs


def summarise(label: str, values: list[float]) -> None:
    if not values:
        print(f"{label}: no samples")
        return
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    se = sd / (n ** 0.5) if n > 1 else 0.0
    ordered = sorted(values)
    print(
        f"\n{label}  (n = {n})\n"
        f"  mean      {mean:7.1f} s  +/- {se:.1f} (SE)\n"
        f"  sd        {sd:7.1f} s   -> {sd / mean:.1%} of the mean\n"
        f"  min / max {ordered[0]:7.1f} / {ordered[-1]:.1f} s   "
        f"(spread {ordered[-1] - ordered[0]:.1f} s)\n"
        f"  median    {statistics.median(values):7.1f} s"
    )
    if n >= 4:
        lo = ordered[max(0, round(0.1 * (n - 1)))]
        hi = ordered[min(n - 1, round(0.9 * (n - 1)))]
        print(f"  p10 / p90 {lo:7.1f} / {hi:.1f} s")
    print(
        f"  --> a difference smaller than about {2 * sd:.0f} s between two single runs "
        "is not evidence of anything"
    )


def time_local(repeats: int) -> list[float]:
    """Time the local suite `repeats` times, to compare its spread against CI's."""
    times: list[float] = []
    for i in range(repeats):
        started = datetime.now()
        result = subprocess.run(
            ["uv", "run", "pytest", "tests", "-q", "-n", "auto", "--min-collected=275"],
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = (datetime.now() - started).total_seconds()
        status = "ok" if result.returncode == 0 else f"FAILED({result.returncode})"
        print(f"  local run {i + 1}/{repeats}: {elapsed:.1f} s  {status}", flush=True)
        times.append(elapsed)
    return times


def main() -> int:
    print("CI timing study -- the distribution behind the quoted number")
    print("=" * 68)

    runs = fetch_runs()
    if runs:
        print("\nevery successful workflow run (newest first):")
        for r in runs:
            print(f"  #{r.number:<4} {r.seconds:6.1f} s   {r.title}")
        summarise("GitHub Actions, total job wall clock", [r.seconds for r in runs])

    repeats = 0
    if "--local" in sys.argv:
        idx = sys.argv.index("--local")
        repeats = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 3
    if repeats:
        print(f"\ntiming the local suite {repeats} times:")
        summarise("local suite, wall clock", time_local(repeats))

    print(
        "\nReading it: the quoted figure is the mean, and any single run is a draw from "
        "this distribution. Report a range, and treat a 'speedup' smaller than the "
        "spread as unmeasured rather than real."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
