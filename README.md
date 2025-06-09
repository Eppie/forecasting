# Forecasting System

This project offers a command line interface for asking forecasting questions.

## Usage

Install the project dependencies and then run:

```bash
forecasting forecast "Will AGI arrive by 2030?"
```

Add `--verbose` or `-v` to get extra output:

```bash
forecasting forecast "Will we colonize Mars soon?" --verbose
```

The underlying forecasting implementation is still a stub.

## Development

Install `pre-commit` and set up the hooks:

```bash
pip install pre-commit
pre-commit install
```

Run all checks (formatting, linting, type checking, and tests):

```bash
./scripts/run_checks.sh
```
