"""
Compare agent runs WITH tools vs WITHOUT tools (same scenarios, same LLM).

Reports clear metrics so we can see when tools are better:
- **Accuracy**: success rate (scenarios solved); accuracy scenarios verify *content* (e.g. version format, snippet output).
- **Efficiency**: avg steps to success (fewer = better), avg duration (tool runs may be slower due to APIs).
- **Per-scenario winner**: no-tools vs WITH TOOLS vs tie.

Scenarios include:
- Preset (no tools): Sequential Project Setup, Conditional Data Processing.
- Tool-required: Web research (must call web_search and add new_facts).
- Accuracy: Python version from web (verify version format); Snippet output (verify stdout contains 42).
- Multi-step: Search then snippet (two tool-required actions in sequence).

Run with OPENROUTER_API_KEY or OPENAI_API_KEY; SERPAPI_API_KEY for web search. Use -s to see the comparison table and verdict.
"""

import os
import time
import pytest

from deliberative_agent import DeliberativeAgent
from deliberative_agent.value_benchmark import MockExecutor


def _has_llm_key() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _has_llm_deps() -> bool:
    """True if openai (or equivalent) is available for LLM client."""
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


# Scenario names used by the tools comparison test (presets + tool scenarios from tests.tool_scenarios)
TOOLS_COMPARISON_SCENARIO_NAMES = [
    "Sequential Project Setup",
    "Conditional Data Processing",
    "Web research (tools REQUIRED)",
    "Python version from web (accuracy)",
    "Multi-step: search then snippet",
    "Snippet output (accuracy)",
]


@pytest.fixture(scope="module")
def scenario():
    """Default scenario (Sequential Project Setup) for baseline test."""
    from tests.test_problems import get_scenarios_for_run
    problems = get_scenarios_for_run()
    for p in problems:
        if p.name == "Sequential Project Setup":
            return p
    return problems[0] if problems else None


@pytest.mark.asyncio
async def test_tools_comparison_multiple_scenarios():
    """
    Run several scenarios with and without tools; print a comparison table.
    Includes a tool-required scenario that only passes when tools are used.
    """
    if not _has_llm_key():
        pytest.skip("Set OPENROUTER_API_KEY or OPENAI_API_KEY to run tools comparison")
    if not _has_llm_deps():
        pytest.skip("openai package required for tools comparison (uv sync --extra llm)")

    from tests.test_problems import get_scenarios_for_run
    from deliberative_agent.llm_integration import create_llm_client, LLMProvider
    from deliberative_agent.llm_executor import SimpleLLMExecutor

    try:
        from web.keys_store import load_into_env
        load_into_env()
    except Exception:
        pass
    from web.tools import run_tool
    from web.agent_executor import ToolCapableExecutor

    provider = LLMProvider.OPENROUTER if os.getenv("OPENROUTER_API_KEY") else LLMProvider.OPENAI
    client = create_llm_client(provider)

    all_scenarios = get_scenarios_for_run()
    name_to_sc = {p.name: p for p in all_scenarios}
    scenarios_to_run = [name_to_sc[n] for n in TOOLS_COMPARISON_SCENARIO_NAMES if n in name_to_sc]

    results_no = []   # (name, success, steps, status, duration_sec)
    results_with = []

    for sc in scenarios_to_run:
        # Without tools
        exec_no = SimpleLLMExecutor(client, verbose=False)
        agent_no = DeliberativeAgent(actions=sc.available_actions, action_executor=exec_no)
        t0 = time.perf_counter()
        res_no = await agent_no.achieve(sc.goal, sc.initial_state.copy())
        dur_no = time.perf_counter() - t0
        steps_no = len(res_no.plan.steps) if res_no.plan else 0
        results_no.append((sc.name, res_no.success, steps_no, res_no.status, dur_no))

        # With tools
        exec_with = ToolCapableExecutor(client, tool_runner=run_tool, verbose=False)
        agent_with = DeliberativeAgent(actions=sc.available_actions, action_executor=exec_with)
        t0 = time.perf_counter()
        res_with = await agent_with.achieve(sc.goal, sc.initial_state.copy())
        dur_with = time.perf_counter() - t0
        steps_with = len(res_with.plan.steps) if res_with.plan else 0
        results_with.append((sc.name, res_with.success, steps_with, res_with.status, dur_with))

    n = len(scenarios_to_run)
    success_no = sum(r[1] for r in results_no)
    success_with = sum(r[1] for r in results_with)
    steps_no_ok = [r[2] for r in results_no if r[1]]
    steps_with_ok = [r[2] for r in results_with if r[1]]
    avg_steps_no = sum(steps_no_ok) / len(steps_no_ok) if steps_no_ok else 0
    avg_steps_with = sum(steps_with_ok) / len(steps_with_ok) if steps_with_ok else 0
    avg_dur_no = sum(r[4] for r in results_no) / n
    avg_dur_with = sum(r[4] for r in results_with) / n

    # Per-scenario table
    print("\n--- Tools comparison (multiple scenarios) ---")
    print(f"{'Scenario':<38} {'No tools':<28} {'With tools':<28} {'Winner'}")
    print("-" * 100)
    for i, sc in enumerate(scenarios_to_run):
        name = sc.name[:36]
        no_ok, no_steps, no_status, no_dur = results_no[i][1], results_no[i][2], results_no[i][3], results_no[i][4]
        with_ok, with_steps, with_status, with_dur = results_with[i][1], results_with[i][2], results_with[i][3], results_with[i][4]
        no_str = f"ok={no_ok} steps={no_steps} {no_dur:.1f}s ({no_status})"
        with_str = f"ok={with_ok} steps={with_steps} {with_dur:.1f}s ({with_status})"
        if no_ok and not with_ok:
            winner = "no-tools"
        elif with_ok and not no_ok:
            winner = "WITH TOOLS"
        elif with_ok and no_ok:
            if with_steps < no_steps or (with_steps == no_steps and with_dur < no_dur):
                winner = "with-tools"
            elif no_steps < with_steps or (no_steps == with_steps and no_dur < with_dur):
                winner = "no-tools"
            else:
                winner = "tie"
        else:
            winner = "tie (both failed)"
        print(f"{name:<38} {no_str:<28} {with_str:<28} {winner}")
    print("-" * 100)

    # Aggregate metrics: accuracy and efficiency
    print("\n--- Aggregate metrics (tools vs no tools) ---")
    print(f"  Success rate:     No tools {success_no}/{n} ({100*success_no/n:.0f}%)  |  With tools {success_with}/{n} ({100*success_with/n:.0f}%)")
    print(f"  Avg steps (ok):   No tools {avg_steps_no:.1f}  |  With tools {avg_steps_with:.1f}  (fewer = more efficient)")
    print(f"  Avg duration:     No tools {avg_dur_no:.1f}s  |  With tools {avg_dur_with:.1f}s")
    if success_with > success_no:
        print("  Verdict:          With tools achieves HIGHER accuracy (more scenarios solved).")
    elif success_with == success_no and (avg_steps_with < avg_steps_no or avg_dur_with < avg_dur_no):
        print("  Verdict:          With tools same accuracy, BETTER efficiency (fewer steps or faster).")
    else:
        print("  Verdict:          With tools required for tool-only scenario; other scenarios tie or no-tools wins.")
    print("---")

    # At least one scenario should succeed with each executor
    assert any(r[1] for r in results_no), "At least one scenario should succeed without tools"
    assert any(r[1] for r in results_with), "At least one scenario should succeed with tools"

    # Tools must be strictly better on success rate (tool-required scenario only passes with tools)
    assert success_with >= success_no, "With-tools success rate should be >= no-tools"
    assert success_with > success_no, "With tools must achieve higher success (tool-required scenario needs tools)"

    # Tool-required scenario: without tools MUST fail, with tools MUST succeed (no faking)
    tool_required_name = "Web research (tools REQUIRED)"
    idx = next((i for i, sc in enumerate(scenarios_to_run) if sc.name == tool_required_name), None)
    if idx is not None:
        no_ok, with_ok = results_no[idx][1], results_with[idx][1]
        assert not no_ok, "Tool-required scenario must FAIL without tools (action has no effects)"
        assert with_ok, "Tool-required scenario must SUCCEED with tools (executor adds new_facts)"


@pytest.mark.asyncio
async def test_tool_required_fails_without_tools_passes_with_tools():
    """
    The tool-required scenario has an action with empty effects. So:
    - Without tools: SimpleLLMExecutor only applies action.apply(state) -> no new facts -> goal not satisfied -> FAIL.
    - With tools: ToolCapableExecutor runs LLM, LLM calls web_search, executor adds new_facts -> goal satisfied -> PASS.
    This test ensures we are not faking tool use; real tool invocation is required to pass.
    """
    if not _has_llm_key():
        pytest.skip("Set OPENROUTER_API_KEY or OPENAI_API_KEY to run")
    if not _has_llm_deps():
        pytest.skip("openai package required (uv sync --extra llm)")

    try:
        from web.keys_store import load_into_env
        load_into_env()
    except Exception:
        pass
    from tests.test_problems import get_scenarios_for_run
    from deliberative_agent.llm_integration import create_llm_client, LLMProvider
    from deliberative_agent.llm_executor import SimpleLLMExecutor
    from web.tools import run_tool
    from web.agent_executor import ToolCapableExecutor

    sc = next((p for p in get_scenarios_for_run() if p.name == "Web research (tools REQUIRED)"), None)
    if not sc:
        pytest.skip("Tool-required scenario not found in get_scenarios_for_run()")
    provider = LLMProvider.OPENROUTER if os.getenv("OPENROUTER_API_KEY") else LLMProvider.OPENAI
    client = create_llm_client(provider)

    exec_no = SimpleLLMExecutor(client, verbose=False)
    agent_no = DeliberativeAgent(actions=sc.available_actions, action_executor=exec_no)
    res_no = await agent_no.achieve(sc.goal, sc.initial_state.copy())

    exec_with = ToolCapableExecutor(client, tool_runner=run_tool, verbose=False)
    agent_with = DeliberativeAgent(actions=sc.available_actions, action_executor=exec_with)
    res_with = await agent_with.achieve(sc.goal, sc.initial_state.copy())

    assert not res_no.success, "Tool-required scenario must FAIL without tools"
    assert res_with.success, "Tool-required scenario must SUCCEED with tools"


@pytest.mark.asyncio
async def test_mock_executor_baseline_no_llm(scenario):
    """Run the same scenario with mock executor (no LLM). Baseline that should always succeed."""
    if scenario is None:
        pytest.skip("No test problems available")

    executor = MockExecutor()
    agent = DeliberativeAgent(
        actions=scenario.available_actions,
        action_executor=executor,
    )
    result = await agent.achieve(scenario.goal, scenario.initial_state.copy())
    steps = len(result.plan.steps) if result.plan else 0

    print(f"\nMock (no LLM): {result.status} steps={steps}")

    assert result.success, result.message or "Mock run should succeed"
    assert steps >= 1
