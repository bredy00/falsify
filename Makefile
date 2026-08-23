# falsify -- entry points. CI runs these same targets, so a green `make ci`
# locally means a green build (Phase 0, PLAYBOOK Part 4).

.PHONY: help install lint fmt typecheck test gates prop ci reproduce clean g9-figure report

RUN := uv run

help:
	@echo "install     sync the locked environment (uv.lock)"
	@echo "lint        ruff check"
	@echo "fmt         ruff format (writes)"
	@echo "typecheck   mypy --strict over falsify/ and tests/"
	@echo "test        full pytest suite"
	@echo "gates       the gate suite only, verbose"
	@echo "prop        Gate 0.0 with printed statistics and the figure"
	@echo "g9-figure   G9 PBO-vs-temperature at the full 12,870 splits (minutes)"
	@echo "ci          lint + typecheck + gates, exactly as CI runs them"
	@echo "report      write outputs/metrics.json (01 Part D contract)"
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
	$(RUN) pytest tests -v -n auto -m "not live" --min-collected=390

live:
	$(RUN) pytest tests/live -v -s -m live

prop:
	$(RUN) pytest tests/gates/test_prop.py -v -s

# G9's headline figure, at the full C(16,8) = 12,870 splits. Minutes, not seconds --
# the CI gate runs C(10,5) on purpose.
g9-figure:
	$(RUN) python scripts/g9_temperature.py

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
