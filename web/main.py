"""
Web UI backend for the Deliberative Agent.

Run from repo root:
  uv run uvicorn web.main:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on path (for tests + deliberative_agent)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web.keys_store import load_into_env, get_status, save_key, remove_key, ALLOWED_KEY_NAMES
from web.serializers import (
    agent_result_to_dict,
    world_state_to_dict,
)

# Load persisted API keys into env at startup
load_into_env()

app = FastAPI(
    title="LisaSimpson Web UI",
    description="Interact with the Deliberative Agent and view plans, execution, and benchmarks.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (index.html, etc.) at /static
_STATIC = Path(__file__).resolve().parent / "static"
if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


# --- Request/response models ---

class RunRequest(BaseModel):
    scenario_name: str = ""
    use_llm: bool = False
    custom_scenario: dict | None = None  # name, description, initial_facts, goal_facts


class ValueBenchmarkRequest(BaseModel):
    baseline_runs: int = 5


class KeySetRequest(BaseModel):
    key_name: str
    value: str


class KeyRemoveRequest(BaseModel):
    key_name: str


class ToolRunRequest(BaseModel):
    tool: str  # browse_web | run_bash | read_file | write_file
    params: dict = {}


# --- API routes ---

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/keys/status")
def keys_status():
    """Return which API keys are configured (true/false). Never returns values."""
    return {"keys": get_status(), "allowed": list(sorted(ALLOWED_KEY_NAMES))}


@app.post("/api/keys")
def keys_set(req: KeySetRequest):
    """Store an API key securely. Key name must be in allowed list."""
    try:
        save_key(req.key_name.strip(), req.value)
        return {"ok": True, "message": f"{req.key_name} saved"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/keys/{key_name}")
def keys_remove(key_name: str):
    """Remove a stored API key (clears from file and process env)."""
    try:
        remove_key(key_name)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/tools/run")
def tools_run(req: ToolRunRequest):
    """Run a tool: browse_web, run_bash, read_file, write_file. Paths and cwd restricted to project root."""
    from web.tools import run_tool, TOOLS
    if req.tool not in TOOLS:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {req.tool}. Allowed: {list(TOOLS.keys())}")
    result = run_tool(req.tool, req.params or {})
    return result


@app.get("/api/tools/list")
def tools_list():
    """List available tools and their required params."""
    from web.tools import TOOLS
    return {"tools": [{"name": k, "params": v[1]} for k, v in TOOLS.items()]}


@app.get("/api/scenarios")
def list_scenarios():
    """Return available test scenarios (name, description, difficulty, initial facts)."""
    try:
        from tests.test_problems import get_scenarios_for_run
    except ImportError:
        raise HTTPException(status_code=500, detail="Could not import test scenarios (run from repo root)")
    problems = get_scenarios_for_run()
    out = []
    for p in problems:
        out.append({
            "name": p.name,
            "description": p.description,
            "difficulty": p.difficulty.value if hasattr(p.difficulty, "value") else str(p.difficulty),
            "initial_facts": world_state_to_dict(p.initial_state),
            "max_steps": p.max_steps,
        })
    return {"scenarios": out}


def _build_custom_scenario(custom: dict):
    """Build a Scenario-like object from custom_scenario payload (initial_facts, goal_facts)."""
    from deliberative_agent import (
        WorldState,
        Goal,
        Action,
        Fact,
        Confidence,
        ConfidenceSource,
        VerificationPlan,
    )
    initial_facts = custom.get("initial_facts") or []
    goal_facts = custom.get("goal_facts") or []
    if not goal_facts:
        raise ValueError("goal_facts cannot be empty")
    state = WorldState()
    for f in initial_facts:
        pred = f.get("predicate") or "unknown"
        args = tuple(f.get("args") or [])
        state.add_fact(Fact(pred, args, Confidence(1.0, ConfidenceSource.OBSERVATION)))
    state.add_fact(Fact("custom_task_started", (), Confidence(1.0, ConfidenceSource.OBSERVATION)))
    goal_list = [(g.get("predicate") or "unknown", tuple(g.get("args") or [])) for g in goal_facts]

    def predicate(s):
        return all(s.has_fact(p, *a) is not None for p, a in goal_list)

    goal = Goal(
        id="custom",
        description=custom.get("description") or "Custom task",
        predicate=predicate,
        verification=VerificationPlan([]),
    )
    actions = []
    for i, (pred, args) in enumerate(goal_list):
        if i == 0:
            precond = lambda s, p="custom_task_started", a=(): s.has_fact(p, *a) is not None
        else:
            prev_p, prev_a = goal_list[i - 1]
            precond = lambda s, p=prev_p, a=prev_a: s.has_fact(p, *a) is not None
        effect = Fact(pred, args, Confidence(0.9, ConfidenceSource.INFERENCE))
        actions.append(
            Action(
                name=f"achieve_{pred}_{i}",
                description=f"Achieve {pred}{args!r}",
                preconditions=[precond],
                effects=[effect],
                cost=1.0,
                reversible=True,
            )
        )
    return type("CustomScenario", (), {
        "name": custom.get("name") or "Custom task",
        "description": custom.get("description") or "Custom task",
        "initial_state": state,
        "goal": goal,
        "available_actions": actions,
    })()


@app.post("/api/run")
async def run_agent(req: RunRequest):
    """Run the agent on a preset scenario or a custom task. Returns status, plan, state, verification, lessons, failure info."""
    try:
        from tests.test_problems import get_all_problems
        from deliberative_agent import DeliberativeAgent
        from deliberative_agent.execution import ActionExecutor
        from deliberative_agent.value_benchmark import MockExecutor
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Import error: {e}")

    if req.custom_scenario:
        try:
            scenario = _build_custom_scenario(req.custom_scenario)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        if not req.scenario_name:
            raise HTTPException(status_code=400, detail="Provide scenario_name or custom_scenario")
        from tests.test_problems import get_scenarios_for_run
        problems = get_scenarios_for_run()
        scenario = next((p for p in problems if p.name == req.scenario_name), None)
        if not scenario:
            raise HTTPException(status_code=404, detail=f"Scenario not found: {req.scenario_name}")

    executor: ActionExecutor
    if req.use_llm:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="use_llm=true but no OPENROUTER_API_KEY or OPENAI_API_KEY set. Add keys in API Keys section.",
            )
        try:
            from deliberative_agent.llm_integration import create_llm_client, LLMProvider
            from web.tools import run_tool
            from web.agent_executor import ToolCapableExecutor
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"LLM/tools not available: {e}")
        load_into_env()
        provider = LLMProvider.OPENROUTER if os.getenv("OPENROUTER_API_KEY") else LLMProvider.OPENAI
        client = create_llm_client(provider)
        executor = ToolCapableExecutor(client, tool_runner=run_tool, verbose=False)
    else:
        executor = MockExecutor()

    agent = DeliberativeAgent(actions=scenario.available_actions, action_executor=executor)
    initial_state = scenario.initial_state.copy()
    result = await agent.achieve(scenario.goal, initial_state)
    return agent_result_to_dict(result)


@app.post("/api/value-benchmark")
async def value_benchmark(req: ValueBenchmarkRequest):
    """Run value benchmark (mock executor) and return report."""
    try:
        from deliberative_agent.value_benchmark import (
            run_value_benchmark,
            get_real_world_scenarios,
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Import error: {e}")
    report = await run_value_benchmark(
        baseline_runs_per_scenario=min(20, max(1, req.baseline_runs)),
        baseline_seed=42,
    )
    # Serialize report for JSON
    return {
        "scenarios": report.scenarios,
        "deliberative_results": [
            {
                "scenario_id": r.scenario_id,
                "success": r.success,
                "steps": r.steps,
                "total_cost": r.total_cost,
                "duration_seconds": r.duration_seconds,
                "failure_reason": r.failure_reason,
            }
            for r in report.deliberative_results
        ],
        "baseline_results": [
            {
                "scenario_id": r.scenario_id,
                "success": r.success,
                "steps": r.steps,
                "total_cost": r.total_cost,
                "duration_seconds": r.duration_seconds,
            }
            for r in report.baseline_results
        ],
        "summary": report.summary,
    }


# --- Serve SPA ---

@app.get("/")
def index():
    """Serve the single-page app."""
    index_path = _STATIC / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Static files not found (web/static/index.html)")
    return FileResponse(index_path)
