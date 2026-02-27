# Contributing to NASE

## Development Workflow

1. Create a feature branch.
2. Install development dependencies:
   ```bash
   python -m pip install -e .[dev]
   pre-commit install
   ```
3. Make focused, testable changes.
4. Run checks locally:
   ```bash
   ruff check .
   ruff format --check .
   pytest -q
   ```
5. Open a pull request with:
   - a concise motivation,
   - implementation notes,
   - validation evidence.

## Coding Guidelines

- Use explicit typing and dataclasses for structured config/state.
- Keep functions small and unit-testable.
- Preserve deterministic behaviour by threading seed values through data generation and experiments.
- Avoid unnecessary dense `O(n^2)` memory usage for large `n`; prefer kNN sparse paths when configured.
- Keep docs in en-GB spelling.

## Research and Attribution

- Write all documentation in original language.
- Do not copy text from papers.
- Include citations for ideas adapted from published work.
