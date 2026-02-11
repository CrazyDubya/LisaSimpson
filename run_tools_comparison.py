#!/usr/bin/env python3
"""
Run scenarios WITH tools and WITHOUT tools; print comparison.

Usage (from repo root, with LLM key set):
  OPENROUTER_API_KEY=sk-... uv run python run_tools_comparison.py
  OPENAI_API_KEY=sk-... uv run python run_tools_comparison.py

  run_tools_comparison.py --all              # Run all scenarios (no-tools vs with-tools table)
  run_tools_comparison.py --scenario "Name"  # Single scenario (default: Sequential Project Setup)
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deliberative_agent import DeliberativeAgent
from deliberative_agent.value_benchmark import MockExecutor


async def run_single(scenario, client, run_tool, use_tools: bool):
    """Run one scenario with or without tools; return (success, steps, duration_sec)."""
    from deliberative_agent.llm_executor import SimpleLLMExecutor
    from web.agent_executor import ToolCapableExecutor

    if use_tools:
        executor = ToolCapableExecutor(client, tool_runner=run_tool, verbose=False)
    else:
        executor = SimpleLLMExecutor(client, verbose=False)
    agent = DeliberativeAgent(actions=scenario.available_actions, action_executor=executor)
    t0 = time.perf_counter()
    result = await agent.achieve(scenario.goal, scenario.initial_state.copy())
    dur = time.perf_counter() - t0
    steps = len(result.plan.steps) if result.plan else 0
    return (result.success, steps, result.status, dur)


async def run_all():
    """Run all scenarios with and without tools; print table and aggregate metrics."""
    from tests.test_problems import get_scenarios_for_run
    from deliberative_agent.llm_integration import create_llm_client, LLMProvider
    from web.tools import run_tool

    scenarios = get_scenarios_for_run()
    provider = LLMProvider.OPENROUTER if os.getenv("OPENROUTER_API_KEY") else LLMProvider.OPENAI
    client = create_llm_client(provider)

    try:
        from web.keys_store import load_into_env
        load_into_env()
    except Exception:
        pass

    results_no = []
    results_with = []
    for sc in scenarios:
        ok_no, steps_no, status_no, dur_no = await run_single(sc, client, run_tool, use_tools=False)
        results_no.append((sc.name, ok_no, steps_no, status_no, dur_no))
        ok_with, steps_with, status_with, dur_with = await run_single(sc, client, run_tool, use_tools=True)
        results_with.append((sc.name, ok_with, steps_with, status_with, dur_with))

    n = len(scenarios)
    success_no = sum(r[1] for r in results_no)
    success_with = sum(r[1] for r in results_with)
    steps_no_ok = [r[2] for r in results_no if r[1]]
    steps_with_ok = [r[2] for r in results_with if r[1]]
    avg_steps_no = sum(steps_no_ok) / len(steps_no_ok) if steps_no_ok else 0
    avg_steps_with = sum(steps_with_ok) / len(steps_with_ok) if steps_with_ok else 0
    avg_dur_no = sum(r[4] for r in results_no) / n
    avg_dur_with = sum(r[4] for r in results_with) / n

    print("\n--- Tools comparison (all scenarios) ---")
    print(f"{'Scenario':<38} {'No tools':<28} {'With tools':<28} {'Winner'}")
    print("-" * 100)
    for i, sc in enumerate(scenarios):
        name = sc.name[:36]
        no_ok, no_steps, no_status, no_dur = results_no[i][1], results_no[i][2], results_no[i][3], results_no[i][4]
        with_ok, with_steps, with_status, with_dur = results_with[i][1], results_with[i][2], results_with[i][3], results_with[i][4]
        no_str = f"ok={no_ok} steps={no_steps} {no_dur:.1f}s ({no_status})"
        with_str = f"ok={with_ok} steps={with_steps} {with_dur:.1f}s ({with_status})"
        if with_ok and not no_ok:
            winner = "WITH TOOLS"
        elif no_ok and not with_ok:
            winner = "no-tools"
        elif no_ok and with_ok:
            winner = "tie" if no_steps == with_steps and abs(no_dur - with_dur) < 0.1 else ("with-tools" if with_steps < no_steps else "no-tools")
        else:
            winner = "tie (both failed)"
        print(f"{name:<38} {no_str:<28} {with_str:<28} {winner}")
    print("-" * 100)
    print("\n--- Aggregate metrics ---")
    print(f"  Success rate:     No tools {success_no}/{n} ({100*success_no/n:.0f}%)  |  With tools {success_with}/{n} ({100*success_with/n:.0f}%)")
    print(f"  Avg steps (ok):   No tools {avg_steps_no:.1f}  |  With tools {avg_steps_with:.1f}")
    print(f"  Avg duration:     No tools {avg_dur_no:.1f}s  |  With tools {avg_dur_with:.1f}s")
    if success_with > success_no:
        print("  Verdict:          With tools achieves HIGHER accuracy (more scenarios solved).")
    else:
        print("  Verdict:          See table above.")
    print("---")
    return 0


async def main():
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Set OPENROUTER_API_KEY or OPENAI_API_KEY to run this comparison.", file=sys.stderr)
        return 1

    from tests.test_problems import get_scenarios_for_run
    from deliberative_agent.llm_integration import create_llm_client, LLMProvider
    from deliberative_agent.llm_executor import SimpleLLMExecutor
    from web.tools import run_tool
    from web.agent_executor import ToolCapableExecutor

    try:
        from web.keys_store import load_into_env
        load_into_env()
    except Exception:
        pass

    if "--all" in sys.argv:
        return await run_all()

    scenarios = get_scenarios_for_run()
    scenario_name = "Sequential Project Setup"
    if "--scenario" in sys.argv:
        i = sys.argv.index("--scenario")
        if i + 1 < len(sys.argv):
            scenario_name = sys.argv[i + 1]
    scenario = next((p for p in scenarios if p.name == scenario_name), None)
    if not scenario:
        print(f"Scenario not found: {scenario_name}", file=sys.stderr)
        return 1

    provider = LLMProvider.OPENROUTER if os.getenv("OPENROUTER_API_KEY") else LLMProvider.OPENAI
    client = create_llm_client(provider)

    print(f"Scenario: {scenario.name}")
    print("Running WITHOUT tools (SimpleLLMExecutor)...")
    executor_no_tools = SimpleLLMExecutor(client, verbose=False)
    agent_no_tools = DeliberativeAgent(
        actions=scenario.available_actions,
        action_executor=executor_no_tools,
    )
    result_no_tools = await agent_no_tools.achieve(scenario.goal, scenario.initial_state.copy())
    steps_no = len(result_no_tools.plan.steps) if result_no_tools.plan else 0
    print(f"  -> {result_no_tools.status}  steps={steps_no}  {result_no_tools.message or ''}")

    print("Running WITH tools (ToolCapableExecutor)...")
    executor_with_tools = ToolCapableExecutor(client, tool_runner=run_tool, verbose=False)
    agent_with_tools = DeliberativeAgent(
        actions=scenario.available_actions,
        action_executor=executor_with_tools,
    )
    result_with_tools = await agent_with_tools.achieve(scenario.goal, scenario.initial_state.copy())
    steps_with = len(result_with_tools.plan.steps) if result_with_tools.plan else 0
    print(f"  -> {result_with_tools.status}  steps={steps_with}  {result_with_tools.message or ''}")

    print("\n--- Summary ---")
    print(f"Without tools: success={result_no_tools.success}  steps={steps_no}")
    print(f"With tools:    success={result_with_tools.success}  steps={steps_with}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
