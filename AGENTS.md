# NeuroFlux Development Guidelines for Agentic Coding Agents

## Build/Test Commands

- All tests: `python test_runner.py`
- Pytest suite: `python -m pytest`
- Specific test file: `python -m pytest src/tests/test_specific_file.py`
- Single test method: `python -m pytest src/tests/test_specific_file.py::TestClass::test_method -v`
- With coverage: `python -m pytest --cov=src --cov-report=html`
- Integration tests: `python -m pytest -m integration`
- Async tests: `python -m pytest -m asyncio`

## Code Quality

- Format: `black .`
- Lint: `flake8 .`
- Type check: `mypy .`
- Full check: `black . && flake8 . && mypy .`

## Code Style Guidelines

- Imports: Standard first, third-party second, local last.
- Naming: snake_case functions/variables, PascalCase classes, UPPER_CASE constants, _private.
- Error Handling: try/except with specific exceptions, log errors, use cprint.
- Type Hints: Use typing module.
- Documentation: Docstrings with Args, Returns, Raises.
- Async: Use async/await with error handling.

Built with love by Nyros Veil 🚀