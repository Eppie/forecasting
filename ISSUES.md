# Development Issues

This file records issues encountered while updating the project and their solutions.

## Missing `pytest-cov`

While modifying `scripts/run_checks.sh` to generate coverage information, running
`pytest --cov` failed because the `pytest-cov` plugin was not installed.
Solution: added `pytest-cov>=4.1.0` to `pyproject.toml` dependencies.

## Updating Workflow Sequence

Changing `run_workflow` to execute all workflow steps introduced failing tests
because stubs raise `NotImplementedError`. The tests were updated to mock these
functions so the workflow can be exercised without executing the stubs.
