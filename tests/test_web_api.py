"""
API-level integration tests: exercise the app via HTTP.

Tests the full stack (routing, scenario resolution, executor choice, serialization).
Requires the web extra (FastAPI). LLM tests are skipped when no API key is set.
"""

import os
import pytest

# Skip entire module if web app or TestClient deps cannot be imported (e.g. web extra or httpx not installed)
pytest.importorskip("fastapi")
try:
    from fastapi.testclient import TestClient
    from web.main import app
except (ImportError, RuntimeError) as e:
    pytest.skip(f"Web app or TestClient not available: {e}", allow_module_level=True)


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    """GET /api/health returns ok."""
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_api_scenarios_list(client):
    """GET /api/scenarios returns a list including presets and tool scenarios."""
    r = client.get("/api/scenarios")
    assert r.status_code == 200
    data = r.json()
    scenarios = data.get("scenarios") or []
    assert len(scenarios) >= 1
    names = [s.get("name") for s in scenarios if s.get("name")]
    assert "Sequential Project Setup" in names
    assert "Web research (tools REQUIRED)" in names


def test_api_run_no_llm_preset(client):
    """POST /api/run with use_llm=false and a preset scenario returns 200 and success."""
    r = client.post(
        "/api/run",
        json={"scenario_name": "Sequential Project Setup", "use_llm": False},
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "success"
    assert "plan" in data
    assert "state" in data


def test_api_run_unknown_scenario_404(client):
    """POST /api/run with unknown scenario_name returns 404."""
    r = client.post(
        "/api/run",
        json={"scenario_name": "Nonexistent Scenario XYZ", "use_llm": False},
    )
    assert r.status_code == 404


def test_api_run_with_llm_preset(client):
    """POST /api/run with use_llm=true and preset scenario (requires OPENROUTER_API_KEY or OPENAI_API_KEY + openai package)."""
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Set OPENROUTER_API_KEY or OPENAI_API_KEY to run LLM API test")
    try:
        import openai  # noqa: F401
    except ImportError:
        pytest.skip("openai package required for LLM API test (uv sync --extra llm)")
    r = client.post(
        "/api/run",
        json={"scenario_name": "Sequential Project Setup", "use_llm": True},
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "success"


def test_api_run_with_llm_tool_scenario(client):
    """POST /api/run with use_llm=true and tool-required scenario (requires API keys + openai package)."""
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Set OPENROUTER_API_KEY or OPENAI_API_KEY to run tool scenario API test")
    try:
        import openai  # noqa: F401
    except ImportError:
        pytest.skip("openai package required for LLM API test (uv sync --extra llm)")
    r = client.post(
        "/api/run",
        json={"scenario_name": "Web research (tools REQUIRED)", "use_llm": True},
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "success"
