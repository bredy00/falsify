# falsify -- entry points. CI runs these same targets, so a green `make ci`
# locally means a green build (Phase 0, PLAYBOOK Part 4).

.PHONY: help install lint fmt typecheck test gates prop ci reproduce clean

RUN := uv run

help:
	@echo "install     sync the locked environment (uv.lock)"
	@echo "lint        ruff check"
	@echo "fmt         ruff format (writes)"
	@echo "typecheck   mypy --strict over falsify/ and tests/"
	@echo "test        full pytest suite"
	@echo "gates       the gate suite only, verbose"
	@echo "prop        Gate 0.0 with printed statistics and the figure"
	@echo "ci          lint + typecheck + gates, exactly as CI runs them"
	@echo "reproduce   G10: assert two runs are byte-identical"

install:
	uv sync --all-groups

lint:
	$(RUN) ruff check .

fmt:
	$(RUN) ruff format .

typecheck:
	$(RUN) mypy

test:
	$(RUN) pytest

gates:
	$(RUN) pytest tests -v --min-collected=75

prop:
	$(RUN) pytest tests/gates/test_prop.py -v -s

ci: check-actions lint typecheck gates

# Catches an unresolvable `uses:` ref before a push turns it into a run that dies
# before executing a step. actionlint does not check this.
check-actions:
	$(RUN) python scripts/check_action_refs.py

# The failure CI cannot self-report: a workflow that dies before running a step.
# Checks the newest run for HEAD actually completed and succeeded.
verify-ci:
	@sha=$$(git rev-parse HEAD); \
	 conclusion=$$(gh run list --limit 10 --json headSha,conclusion,status \
	   --jq "[.[] | select(.headSha==\"$$sha\")][0] | \"\(.status)/\(.conclusion)\""); \
	 echo "HEAD $$sha -> $$conclusion"; \
	 test "$$conclusion" = "completed/success" \
	   || (echo "verify-ci: HEAD has no successful completed run"; exit 1)

# G10. Two runs from the same seeds must produce byte-identical figures.
# Currently covers Gate 0.0's figure; extends to metrics.json when the
# reporting layer lands (Phase 8).
reproduce:
	@mkdir -p outputs
	@$(RUN) pytest tests/gates/test_prop.py -q
	@sha256sum docs/figures/*.png > outputs/reproduce-1.sha256
	@$(RUN) pytest tests/gates/test_prop.py -q
	@sha256sum docs/figures/*.png > outputs/reproduce-2.sha256
	@diff outputs/reproduce-1.sha256 outputs/reproduce-2.sha256 \
		&& echo "reproduce: byte-identical across two runs" \
		|| (echo "reproduce: FAILED -- output is not deterministic"; exit 1)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache outputs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
