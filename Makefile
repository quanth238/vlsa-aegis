.PHONY: init test audit synthetic manifest-smoke clean-check remote-preflight

PYTHON ?= python3
PYTHONPATH := src

init:
	./init.sh

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

audit:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/audit_harness.py

manifest-smoke:
	mkdir -p .harness/tmp
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m crfs_harness.cli build-manifest --output .harness/tmp/smoke.jsonl --episodes 0

synthetic: manifest-smoke
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m crfs_harness.cli synthetic --manifest .harness/tmp/smoke.jsonl --output-root .harness/tmp/synthetic --run-id local-fixture

clean-check:
	git diff --check
	git status --short

remote-preflight:
	scripts/hpc/preflight.sh
