#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate && uv pip install -e . && uv run pytest

uv run ruff format .
uv run ruff check --fix .

uv run mypy .
