.PHONY: install test smoke clean
install:
	pip install -e '.[dev,eval]'
test:
	pytest
smoke:
	DEBUG_AGENT_PROVIDER=mock debug-assistant diagnose --issue examples/issues/example_issue.md --repo examples/fixture_repo --output runs/smoke
clean:
	rm -rf .debug_assistant runs .pytest_cache .coverage
