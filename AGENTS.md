## 1. Project Overview

This is a Python project that uses `ruff` for linting and formatting, `mypy` for static type checking, and `pytest` for testing. All code should be written for Python 3.13 and adhere to modern Python conventions. The goal is to produce clean, efficient, and maintainable code.

## 2. General Instructions

*   **Python Version:** All code must be compatible with Python 3.13.
*   **Dependencies:** Manage project dependencies using a `pyproject.toml` and a virtual environment.
*   **File Encoding:** All Python files must be UTF-8 encoded.
*   **Clarity and Simplicity:** Write code that is easy to read and understand. Add comments for complex logic.
*   **Naming Conventions:** Use descriptive and meaningful names for variables, functions, and classes. Follow PEP 8 naming conventions (e.g., `snake_case` for functions and variables, `PascalCase` for classes).

## 3. Tooling and Commands

The following tools are used in this project. Adhere to their configurations and use the specified commands for linting, formatting, type checking, and testing.

### 3.1 Ruff (Linting and Formatting)

`Ruff` is used for both code linting and formatting to ensure a consistent code style and to catch common errors. The configuration is located in the `pyproject.toml` file.

**Instructions for Codex:**

*   Before committing any code changes, run `uv run ruff format .` to format the code.
*   After formatting, run `uv run ruff check --fix .` to lint the code and automatically fix any issues.
*   Ensure all new code passes `uv run ruff check .` without any errors.

### 3.2 Mypy (Static Type Checking)

`Mypy` is used to enforce static typing, which helps in catching type-related errors early. The configuration is located in the `pyproject.toml` file.

**Instructions for Codex:**

*   All new functions and methods must have type hints for all arguments and return values.
*   Avoid using `Any` where a more specific type can be used.
*   Run `uv run mypy .` to perform a static type check on the entire codebase. The command should pass without any errors.

### 3.3 Pytest (Testing)

`Pytest` is the testing framework for this project. All new features should be accompanied by corresponding tests.

**Best Practices:**

*   **Test Discovery:** Follow the standard `pytest` test discovery rules, such as naming test files `test_*.py` or `*_test.py` and test functions with a `test_` prefix.
*   **Arrange-Act-Assert:** Structure your tests using the "Arrange, Act, Assert" pattern for clarity.
*   **Fixtures:** Use `pytest` fixtures for setting up and tearing down test states. This is preferred over explicit setup and teardown functions.
*   **Parameterization:** Use the `@pytest.mark.parametrize` decorator to run the same test with different inputs, which helps in avoiding code duplication.
*   **Keep Tests Independent:** Each test should be independent and not rely on the state of other tests.
*   **Mocking:** Mock external dependencies and services to isolate the code being tested.

**Instructions for Codex:**

*   When adding new functionality, create a corresponding test file and add comprehensive unit tests.
*   Ensure all tests pass by running the `uv run pytest` command.
*   Aim for high test coverage for any new code that is generated.

### 3.4 Pre-commit

`pre-commit` manages the lint and type-check hooks for this project.

*   Install the hooks with `uv run pre-commit install`.
*   Run `pre-commit run --files <modified files>` before committing, or run
    `pre-commit run --all-files` for a full check.
*   You can also execute `./scripts/run_checks.sh` to run all checks manually.

## 4. Python 3.13 Best Practices

This project uses Python 3.13, and the code should reflect modern practices.

*   **Type Hinting:** Make full use of the typing features available in Python 3.13.
*   **Avoid Deprecated Features:** Do not use any functions or modules that are deprecated in Python 3.13.
*   **Documentation:** Use Google-style docstrings.