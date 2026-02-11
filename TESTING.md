# Testing Guide – Bearings & How to Run Tests

This doc gives **bearings** for the repo and what’s needed so an agent (or you) can **run the program** and do **proper testing** via **CLI** or **Playwright**.

---

## 1. What This Repo Is

- **LisaSimpson** is a **Python library**: a *deliberative agent* that plans, verifies, and learns (GOAP-style planning, semantic verification, memory).
- A **web UI** in `web/` (FastAPI + single-page frontend) lets you run scenarios, view plans/execution/verification/lessons, and run the value benchmark. **CLI + pytest** remain the primary way to run tests; Playwright can be used to automate the web UI if desired.

---

## 2. Quick Bearings

| What | Where |
|------|--------|
| Core library | `deliberative_agent/` (agent, planning, goals, actions, execution, memory, swarm, LLM integration) |
| Unit tests | `tests/` (pytest; most don’t need API keys) |
| LLM tests | `tests/test_llm_comprehensive.py`, `tests/test_llm_providers.py` – need API keys |
| CLI entry points | `run_llm_tests.py`, `run_llm_benchmark.py`, `example_llm_testing.py` |
| Benchmark runner (alternate) | `deliberative_agent/benchmark_runner.py` (has its own `__main__`) |
| **Value benchmark** (Deliberative vs baseline) | `run_value_benchmark.py`, `deliberative_agent/value_benchmark.py` |
| **Tools comparison** (with vs without tools) | `tests/test_agent_tools_comparison.py`, `run_tools_comparison.py` |
| **Web UI** | `web/main.py`, `web/static/index.html` – run with `uv run uvicorn web.main:app --reload --port 8000` |
| Config | `pyproject.toml` (deps, pytest, ruff, mypy) |

---

## 3. What You Need for CLI Testing (So an Agent Can “Use the Program”)

### 3.1 Environment

- **Python 3.10+** (e.g. `uv` or `pip`).
- From repo root:
  - `uv sync` or `pip install -e ".[dev]"` for core + tests.
  - `pip install -e ".[dev,llm]"` if you will run LLM scripts/tests.

### 3.2 Commands the Agent Can Run

**Unit tests (no API keys):**

```bash
# All unit tests (exclude LLM tests so no keys needed)
uv run pytest tests/ -v --ignore=tests/test_llm_comprehensive.py --ignore=tests/test_llm_providers.py

# Single file
uv run pytest tests/test_core.py tests/test_planning.py -v

# With coverage
uv run pytest tests/ --ignore=tests/test_llm_comprehensive.py --ignore=tests/test_llm_providers.py --cov=deliberative_agent
```

**LLM tests (need at least one provider API key):**

```bash
# Set one key, e.g. OpenAI
export OPENAI_API_KEY=your_key

# Full LLM test suite (all providers you have keys for)
uv run python run_llm_tests.py

# One provider, one difficulty
uv run python run_llm_tests.py --provider openai --difficulty medium

# Benchmark suite (multiple providers/models)
uv run python run_llm_benchmark.py --difficulty medium --output benchmark_results.json
```

**Example script (good smoke test that the agent “uses” the library):**

```bash
export OPENAI_API_KEY=your_key   # or ANTHROPIC_API_KEY, etc.
uv run python example_llm_testing.py
```

**Value benchmark (do we add real value vs blind iteration?):**

```bash
# Fast: mock executor, no LLM. Compares planning vs random-order baseline.
uv run python run_value_benchmark.py

# More baseline runs for smoother stats
uv run python run_value_benchmark.py --baseline-runs 20

# With real LLM (OpenRouter/OpenAI) – slower, uses API
OPENROUTER_API_KEY=sk-... uv run python run_value_benchmark.py --llm

# Save JSON report
uv run python run_value_benchmark.py --output value_report.json
```

Scenarios: Safe Production Deployment, CI Pipeline, Backend with Dependencies, Bug Fix with Regression Test. Deliberative uses GOAP planning (optimal order); baseline picks random applicable actions until goal or max steps. Metrics: success rate, steps to goal, cost. Typically deliberative gets 100% success with minimal steps; baseline often gets stuck or uses many more steps.

**Tools comparison (with vs without tools):**

```bash
# Pytest: mock baseline always runs; with/without-tools comparison runs only when an LLM key is set
uv run pytest tests/test_agent_tools_comparison.py -v -s

# Script: run all scenarios (no-tools vs with-tools table) or a single scenario (needs key)
OPENROUTER_API_KEY=sk-... uv run python run_tools_comparison.py --all
OPENROUTER_API_KEY=sk-... uv run python run_tools_comparison.py --scenario "Web research (tools REQUIRED)"
```

Tool scenarios (Web research, Python version from web, Multi-step search then snippet, Snippet output) appear in the **web UI scenario dropdown** and in `run_tools_comparison.py --all`. The test reports **accuracy** (success rate), **efficiency** (avg steps and duration), and a per-scenario **winner**; it asserts that with-tools achieves strictly higher success rate. The mock baseline (`test_mock_executor_baseline_no_llm`) runs without any key. **SERPAPI_API_KEY** (storable via Web UI API Keys) is needed for web-search tool scenarios.

**API integration tests** (`tests/test_web_api.py`):

- Exercise the app via HTTP (GET /api/health, GET /api/scenarios, POST /api/run). No-LLM test runs without a key; LLM and tool-scenario tests are skipped when OPENROUTER_API_KEY/OPENAI_API_KEY are not set. Requires the `web` extra: `uv sync --extra web` (or install FastAPI). Run: `uv run pytest tests/test_web_api.py -v`.

**Failure feedback demo** (dependencies not met, exploration limit):

```bash
uv run python scripts/run_failure_feedback_demo.py
```

Runs two no_plan_found scenarios and prints `failure_reason` and `suggested_next_steps` so you can see how the agent gives feedback for further direction.

### 3.3 Notes

- **LLM tests** (`test_llm_client`, `test_medium_problems_all_providers`, etc.) are **skipped** when no API key is set; they pass or run when e.g. `OPENROUTER_API_KEY` or `OPENAI_API_KEY` is set. A `client` fixture in `tests/conftest.py` provides an LLM client for the first available provider.

---

## 4. What You Need for Playwright Testing

- This repo has **no browser UI**. Playwright is not required to test the current program.
- If you later add a **web dashboard** (e.g. for swarm/benchmark results):
  - Run that app (e.g. `npm run dev` or `uv run python -m backend.main`).
  - Use Playwright (or the Playwright skill’s CLI) to open that URL, snapshot, click, and assert.
  - Prefer `output/playwright/` for traces/screenshots if you follow the skill’s guardrails.

---

## 5. One-Line “Smoke Test” for an Agent

Without API keys:

```bash
uv run pytest tests/test_core.py tests/test_agent.py -v
```

With one LLM key (e.g. OpenAI):

```bash
export OPENAI_API_KEY=sk-...
uv run python run_llm_tests.py --provider openai --difficulty medium --quiet
```

---

## 6. Session Log (Between Commits)

See **`session_log.md`** in the repo root for a running log of changes and decisions between git commits (per project preference).
