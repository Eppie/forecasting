1. Analyze existing repository and root AGENTS.md instructions. (done)
2. Create PLAN.md summarizing tasks.
3. Add new module `src/workflow.py` with stubs for forecast workflow steps.
   - Each function will have type hints and Google-style docstrings.
   - Functions: `clarify_question`, `set_base_rate`, `decompose_problem`, `gather_evidence`, `update_prior`, `produce_forecast`, `sanity_checks`, `cross_validate`, `record_forecast`.
   - For now, each function raises `NotImplementedError`.
   - Add dataclasses for `Question`, `BaseRate`, etc. if needed.
4. Update `src/__init__.py` to export workflow functions.
5. Create tests in `tests/test_workflow.py` verifying that functions are defined and raise `NotImplementedError` when called.
6. Run formatting and linting (`ruff format`, `ruff check --fix`), type checking (`mypy .`), and tests (`pytest`).
7. Commit changes.
