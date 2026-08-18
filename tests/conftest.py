"""Suite-level guards.

A gate suite that runs zero tests reports success. That is a worse failure than a
red build, because red gets investigated and green does not -- and it is the same
class of problem as the CI run that died in three seconds before executing a
single step: the signal looked like information but carried none.

`--min-collected=N` makes the suite assert its own size. CI passes it, so a
collection error, a bad import, a renamed directory or an over-broad deselect can
no longer masquerade as a pass.
"""

from __future__ import annotations

import pytest

# Guards against catastrophic silent loss, not against ordinary churn: set below
# the current count so adding or removing a few tests needs no edit, but losing a
# whole file fails loudly. Raise it as the suite grows.
# 225 collected at G6 (Gate 0.0-0.4, G1-G7, selection, overlays, action refs).
DEFAULT_MIN_COLLECTED = 200


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--min-collected",
        action="store",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Fail the run unless at least N tests were collected. "
            f"CI uses {DEFAULT_MIN_COLLECTED}. A suite that silently shrank to nothing "
            "otherwise reports success."
        ),
    )


def pytest_collection_finish(session: pytest.Session) -> None:
    minimum = int(session.config.getoption("--min-collected"))
    if minimum <= 0:
        return

    collected = len(session.items)
    if collected < minimum:
        raise pytest.UsageError(
            f"collected {collected} tests, expected at least {minimum}. "
            "Either the suite lost tests (a bad import, a renamed path, an over-broad "
            "deselect), or it genuinely grew smaller and tests/conftest.py needs updating. "
            "Do not lower the floor to make this pass without checking which."
        )
