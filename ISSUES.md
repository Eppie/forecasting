# Development Issues

This file records issues encountered while updating the project and their solutions.

## Missing `pytest-cov`
- Problem: While modifying `scripts/run_checks.sh` to generate coverage information, running
`pytest --cov` failed because the `pytest-cov` plugin was not installed.
- Solution: added `pytest-cov>=4.1.0` to `pyproject.toml` dependencies.

## Updating Workflow Sequence

- Problem: Changing `run_workflow` to execute all workflow steps introduced failing tests
because stubs raise `NotImplementedError`.
- Solution: The tests were updated to mock these
functions so the workflow can be exercised without executing the stubs.

## LLM Calls in Tests

- Problem: After implementing the previously stubbed workflow functions, running the tests
attempted real network calls to the Ollama API. This caused failures when
Ollama was not available.
- Solution: Update the unit tests to mock the ``ollama.chat`` calls and verify the
new logic without requiring network access.

## Ruff line length errors
- Problem: When updating docstrings in ``workflow.py`` several lines exceeded the 120-character limit enforced by Ruff (E501).
- Solution: The lines were wrapped to stay within the limit.
