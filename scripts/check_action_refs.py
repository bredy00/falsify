"""Verify every `uses:` ref in every workflow actually exists.

The countermeasure for run 31202076622, which died in three seconds with
"Unable to resolve action astral-sh/setup-uv@v9, unable to find version v9" --
having executed no step, so no test in this repository could have detected it.

actionlint does not catch this. It validates workflow syntax, expressions and
embedded shell, but never resolves an action ref against the remote, so a
perfectly well-formed reference to a tag that was never published passes it.
This does resolve them.

Uses `git ls-remote`, which needs no API token and has no rate limit, so it
behaves the same locally and in CI. A ref that looks like a full 40-character
commit SHA is accepted without a lookup, since `ls-remote` does not list
arbitrary commits.

    python scripts/check_action_refs.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"

# `uses: owner/repo@ref` or `owner/repo/path@ref`. Local (./...) and docker
# (docker://...) refs are skipped: neither is resolved from a remote tag.
USES = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)", re.MULTILINE)
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def remote_has_ref(owner_repo: str, ref: str) -> bool:
    """True when `ref` exists as a tag or branch on the remote."""
    result = subprocess.run(
        [
            "git",
            "ls-remote",
            "--tags",
            "--heads",
            f"https://github.com/{owner_repo}.git",
            ref,
            f"refs/tags/{ref}",
            f"refs/heads/{ref}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        print(f"  ! git ls-remote failed for {owner_repo}: {result.stderr.strip()}")
        return False
    return bool(result.stdout.strip())


def main() -> int:
    if not WORKFLOW_DIR.is_dir():
        print(f"no workflow directory at {WORKFLOW_DIR}")
        return 0

    problems: list[str] = []
    checked = 0

    for workflow in sorted(WORKFLOW_DIR.glob("*.y*ml")):
        for raw in USES.findall(workflow.read_text(encoding="utf-8")):
            spec = raw.strip().strip("\"'")
            if spec.startswith(("./", ".\\", "docker://")):
                continue
            if "@" not in spec:
                problems.append(f"{workflow.name}: `{spec}` has no @ref (unpinned)")
                continue

            path, ref = spec.rsplit("@", 1)
            owner_repo = "/".join(path.split("/")[:2])
            checked += 1

            if FULL_SHA.match(ref):
                print(f"  ok  {spec}  (pinned to a full SHA)")
                continue
            if remote_has_ref(owner_repo, ref):
                print(f"  ok  {spec}")
            else:
                problems.append(
                    f"{workflow.name}: `{spec}` does not resolve -- "
                    f"{owner_repo} publishes no tag or branch named {ref}"
                )

    print(f"\nchecked {checked} action ref(s) across {WORKFLOW_DIR.name}/")
    if problems:
        print("\nUNRESOLVABLE ACTION REFS:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nThis is the failure that produces a run dying before any step executes.")
        return 1
    print("all action refs resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
