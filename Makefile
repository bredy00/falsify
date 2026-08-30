# falsify -- entry points. CI runs these same targets, so a green `make ci`
# locally means a green build (Phase 0, PLAYBOOK Part 4).

.PHONY: help install lint fmt fmt-check typecheck test gates prop ci reproduce clean g9-figure report report-pdf tearsheet surface search-cost

RUN := uv run

help:
	@echo "install     sync the locked environment (uv.lock)"
	@echo "lint        ruff check"
	@echo "fmt         ruff format (writes)"
	@echo "fmt-check   ruff format --check (CI runs this)"
	@echo "typecheck   mypy --strict over falsify/ and tests/"
	@echo "test        full pytest suite"
	@echo "gates       the gate suite only, verbose"
	@echo "prop        Gate 0.0 with printed statistics and the figure"
	@echo "g9-figure   G9 PBO-vs-temperature at the full 12,870 splits (minutes)"
	@echo "report-pdf  the project board as a PDF"
	@echo "tearsheet   Phase 8 tearsheet (needs the cache)"
	@echo "surface     in-sample vs out-of-sample parameter surface (needs the cache)"
	@echo "search-cost what searching costs -- offline, seeded, in reproduce"
	@echo "ci          lint + typecheck + gates, exactly as CI runs them"
	@echo "report      write outputs/metrics.json (01 Part D contract)"
	@echo "reproduce   G10: assert two runs are byte-identical"

install:
	uv sync --all-groups

lint:
	$(RUN) ruff check .

fmt:
	$(RUN) ruff format .

fmt-check:
	$(RUN) ruff format --check .

typecheck:
	$(RUN) mypy

test:
	$(RUN) pytest

gates:
	$(RUN) pytest tests -v -n auto -m "not live" --min-collected=475

live:
	$(RUN) pytest tests/live -v -s -m live

prop:
	$(RUN) pytest tests/gates/test_prop.py -v -s

# outputs/metrics.json -- 01 Part D's reporting contract. Deterministic: fixed seeds,
# no wall-clock field, provenance from the git SHA and the manifest digest.
report:
	$(RUN) python scripts/report.py

# The project board: gate table, invariants, checklists, measurements.
report-pdf:
	$(RUN) --group docs python scripts/board_report.py

# Phase 8's two figures. Both need the populated cache, which is why neither is in
# `make reproduce` -- G10 has to pass in a clean checkout with no network.
tearsheet:
	$(RUN) python scripts/tearsheet.py

surface:
	$(RUN) python scripts/parameter_surface.py

# Offline and seeded, so unlike the two above it belongs to `make reproduce`.
search-cost:
	$(RUN) python scripts/search_cost.py

# G9's headline figure, at the full C(16,8) = 12,870 splits. Minutes, not seconds --
# the CI gate runs C(10,5) on purpose.
g9-figure:
	$(RUN) python scripts/g9_temperature.py

ci: check-actions lint fmt-check typecheck gates

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
#
# Hashes only the figures this target actually REGENERATES. It used to hash
# docs/figures/*.png, which looked like five figures of coverage and was one:
# the other four are built from the SPY cache, were never rebuilt here, and so
# matched themselves trivially. A gate that passes by comparing a file to itself
# is worse than none, because it reports confidence it did not earn.
#
# The metrics half of G10 is in tests/gates/test_g10_reproducibility.py, which
# needs no figure and runs inside the gate suite.
REPRODUCIBLE := docs/figures/compensation_effect.png docs/figures/search_cost.png

reproduce:
	@mkdir -p outputs
	@$(RUN) pytest tests/gates/test_prop.py -q
	@$(RUN) python scripts/search_cost.py > /dev/null
	@sha256sum $(REPRODUCIBLE) > outputs/reproduce-1.sha256
	@$(RUN) pytest tests/gates/test_prop.py -q
	@$(RUN) python scripts/search_cost.py > /dev/null
	@sha256sum $(REPRODUCIBLE) > outputs/reproduce-2.sha256
	@diff outputs/reproduce-1.sha256 outputs/reproduce-2.sha256 \n		&& echo "reproduce: byte-identical across two runs" \n		|| (echo "reproduce: FAILED -- output is not deterministic"; exit 1)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache outputs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
