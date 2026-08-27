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
# 439 selected at Phase 8 / time-series momentum (live tests are deselected in CI).
DEFAULT_MIN_COLLECTED = 430


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


def _minimum(config: pytest.Config) -> int:
    return int(config.getoption("--min-collected"))


def _enforce(count: int, minimum: int) -> None:
    if minimum <= 0 or count >= minimum:
        return
    raise pytest.UsageError(
        f"collected {count} tests, expected at least {minimum}. "
        "Either the suite lost tests (a bad import, a renamed path, an over-broad "
        "deselect), or it genuinely grew smaller and tests/conftest.py needs updating. "
        "Do not lower the floor to make this pass without checking which."
    )


def _running_distributed(config: pytest.Config) -> bool:
    return getattr(config.option, "dist", "no") not in ("no", None)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Serial enforcement.

    Skipped entirely under xdist, and that is not an oversight. Workers each perform
    a full collection, so raising here fires inside a worker process, where pytest
    surfaces a UsageError as an INTERNALERROR with a traceback rather than as the
    clean message this check exists to print. The controller-side hook below does the
    job instead.
    """
    config = session.config
    if hasattr(config, "workerinput") or _running_distributed(config):
        return
    _enforce(len(session.items), _minimum(config))


def pytest_xdist_node_collection_finished(node: object, ids: list[str]) -> None:
    """Distributed enforcement, controller side.

    xdist hands the controller each worker's collected ids, and every worker collects
    the whole suite before the controller assigns it a subset -- so `ids` is the full
    count and this is the same assertion, made where it can be reported cleanly.

    Without this the floor silently stops guarding the moment the suite is run in
    parallel, which is precisely the kind of countermeasure that quietly evaporates
    and leaves a green build meaning nothing.
    """
    config = getattr(node, "config", None)
    if config is None:
        return
    _enforce(len(ids), _minimum(config))
