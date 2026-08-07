"""Offline tests for the workflow action-ref checker.

Only the parsing is tested here. Resolution goes over the network by design, so
asserting on it would make the suite depend on github.com being reachable -- and
a gate that fails because of someone else's outage teaches you to ignore it.

The resolver itself was verified by reintroducing `astral-sh/setup-uv@v9`, the
exact ref that killed run 31202076622, and confirming the checker exits 1 with
"publishes no tag or branch named v9".
"""

from __future__ import annotations

from scripts.check_action_refs import FULL_SHA, USES

SAMPLE = """
jobs:
  gates:
    steps:
      - uses: actions/checkout@v7.0.1
      - name: Install uv
        uses: astral-sh/setup-uv@v9.0.0
        with:
          enable-cache: true
      - uses: ./.github/actions/local-thing
      - uses: docker://alpine:3.20
      - uses: actions/cache@5a3ec84eff668545956fd18022155c47e93e2684
      - run: echo "uses: not-an-action@v1 appears in a script, not as a step"
"""


def test_parser_finds_every_uses_entry() -> None:
    found = [spec.strip() for spec in USES.findall(SAMPLE)]
    assert found == [
        "actions/checkout@v7.0.1",
        "astral-sh/setup-uv@v9.0.0",
        "./.github/actions/local-thing",
        "docker://alpine:3.20",
        "actions/cache@5a3ec84eff668545956fd18022155c47e93e2684",
    ], found


def test_local_and_docker_refs_are_distinguishable() -> None:
    """Neither resolves from a remote tag, so both must be skipped rather than
    reported as broken."""
    found = [spec.strip() for spec in USES.findall(SAMPLE)]
    skipped = [s for s in found if s.startswith(("./", ".\\", "docker://"))]
    assert len(skipped) == 2


def test_full_sha_pins_are_recognised() -> None:
    """A 40-hex pin is the strongest form and must not be sent to ls-remote,
    which cannot list arbitrary commits."""
    assert FULL_SHA.match("5a3ec84eff668545956fd18022155c47e93e2684")
    assert not FULL_SHA.match("v7.0.1")
    assert not FULL_SHA.match("5a3ec84")  # abbreviated, not a full SHA


def test_every_ref_in_the_real_workflows_is_pinned() -> None:
    """No floating majors in this repo: G10 asserts byte-identical output, and a
    toolchain that drifts underneath it undermines that before a figure exists."""
    from scripts.check_action_refs import WORKFLOW_DIR

    for workflow in sorted(WORKFLOW_DIR.glob("*.y*ml")):
        for raw in USES.findall(workflow.read_text(encoding="utf-8")):
            spec = raw.strip().strip("\"'")
            if spec.startswith(("./", ".\\", "docker://")):
                continue
            assert "@" in spec, f"{workflow.name}: {spec} is unpinned"
            ref = spec.rsplit("@", 1)[1]
            assert FULL_SHA.match(ref) or ref.count(".") >= 2, (
                f"{workflow.name}: {spec} pins a floating major; "
                "use an exact release tag or a full SHA"
            )
