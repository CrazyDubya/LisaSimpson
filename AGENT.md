# Agent guide (AI / Cursor / Codex)

Short reference for agents working in this repo. Human docs: [README.md](README.md), [TESTING.md](TESTING.md).

## What this repo is

- **Python library**: deliberative agent (GOAP planning, semantic verification, memory). Optional **web UI** in `web/`; use via Python, CLI, or browser.
- **Root**: run commands from repo root; `uv run` preferred for pytest and scripts.

## Layout

| Purpose | Path |
|--------|------|
| Core agent, planning, goals, actions, execution, memory | `deliberative_agent/agent.py`, `planning.py`, `goals.py`, `actions.py`, `execution.py`, `memory.py` |
| LLM: public API | `deliberative_agent/llm_integration.py` (`create_llm_client`, `LLMProvider`) |
| LLM: benchmark/config | `deliberative_agent/llm_providers.py` |
| Value benchmark (Deliberative vs baseline) | `deliberative_agent/value_benchmark.py`, `run_value_benchmark.py` |
| Input safety / prompt injection | `deliberative_agent/input_safety.py`, `llm_executor.py` |
| Unit tests | `tests/` (pytest) |
| Test scenarios (medium/hard) | `tests/test_problems.py` (`Scenario`, `get_all_problems`) |
| Failure-feedback demos | `scripts/run_failure_feedback_demo.py`, `run_real_world_failure_feedback.py`, `run_fail_then_succeed_with_feedback.py` |
| **Web UI** | `web/main.py`, `web/static/index.html` |
| Docs | `README.md`, `TESTING.md`, `SECURITY.md`, `NEXT_STEPS.md`, `docs/` |

## Commands (from repo root)

```bash
# Unit tests (no API keys). Omit test_llm_providers; test_llm_comprehensive skips when no key.
uv run pytest tests/ -v --ignore=tests/test_llm_providers.py -x

# Value benchmark (mock executor, fast)
uv run python run_value_benchmark.py --quiet

# Value benchmark + memory demo (run each scenario twice with shared memory)
uv run python run_value_benchmark.py --memory

# LLM tests (need OPENROUTER_API_KEY or OPENAI_API_KEY etc.)
uv run python run_llm_tests.py --provider openrouter --difficulty medium
uv run python example_llm_testing.py

# Web UI (install: uv sync --extra web). From repo root:
uv run uvicorn web.main:app --reload --host 0.0.0.0 --port 8000
# Then open http://localhost:8000/
```

## Conventions

- **Python**: 4-space indent, type hints preferred. Snake_case for modules/functions. Keep routers/entrypoints thin; put logic in `deliberative_agent/` modules.
- **Tests**: `Scenario` and `RunResult`/`RunSummary` in `tests/test_problems.py` and `test_llm_comprehensive.py` (not `Test*` to avoid pytest collection).
- **Secrets**: API keys from env only; never commit. See [SECURITY.md](SECURITY.md).
- **Session log**: [session_log.md](session_log.md) is used for between-commit notes; keep it updated for significant changes.

## Other docs

- [TESTING.md](TESTING.md) – Full testing guide, CLI commands, when Playwright applies.
- [SECURITY.md](SECURITY.md) – Keys, input safety, prompt-injection mitigation.
- [NEXT_STEPS.md](NEXT_STEPS.md) – Backlog.
- [docs/LLM_LAYERS.md](docs/LLM_LAYERS.md) – `llm_integration` vs `llm_providers`.
- [docs/archive/](docs/archive/) – Archived reviews and implementation summary.
