# LLM testing (tests/)

**Full testing guide:** see [../TESTING.md](../TESTING.md) for project bearings, unit tests, value benchmark, and when to use Playwright.

## LLM-specific setup

- **Dependencies:** `pip install -e ".[dev,llm]"` (or `uv sync --extra dev`).
- **API keys:** Set at least one of `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, etc. LLM tests are skipped when no key is set.
- **Run LLM tests:** From repo root, `uv run pytest tests/test_llm_comprehensive.py -v -s` or `uv run python run_llm_tests.py`.

## Test problems (scenarios)

Defined in `tests/test_problems.py`; type `Scenario` (medium and hard).

- **Medium:** Sequential Project Setup, Parallel Component Development, Conditional Data Processing.
- **Hard:** Safe Production Deployment, Complex Backend Development, Quality Solution Under Constraints.

To add a scenario, append a `Scenario(...)` to `create_medium_problems()` or `create_hard_problems()` and extend `get_all_problems()`.
