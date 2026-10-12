.PHONY: install test smoke clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e '.[dev]'
test:
	$(PYTHON) -m pytest -q
smoke:
	$(PYTHON) -m pytest -q tests/test_public_release_smoke.py
clean:
	rm -rf .debug_assistant runs .pytest_cache .coverage *.egg-info build dist
