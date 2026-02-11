#!/usr/bin/env python3
"""
Real-world test: run the actual LLM harness with a scenario that triggers
no_plan_found, so a human sees the same feedback as in run_llm_tests.py.

Usage (as the human):
  OPENROUTER_API_KEY=sk-... uv run python scripts/run_real_world_failure_feedback.py

You should see: Testing: ..., Result: FAILED, Status: no_plan_found, then
"Why it failed" and "Suggested next steps (give further direction)".
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deliberative_agent import (
    DeliberativeAgent,
    WorldState,
    Goal,
    VerificationPlan,
)
from deliberative_agent.actions import action
from deliberative_agent.llm_integration import create_llm_client, LLMProvider
from deliberative_agent.llm_executor import SimpleLLMExecutor


def make_dependency_failure_problem():
    """A problem that will hit no_plan_found: main goal depends on prereq that we never satisfy."""
    from dataclasses import dataclass
    from tests.test_problems import Scenario, Difficulty

    prereq_goal = Goal(
        id="prereq",
        description="Prerequisite: deploy config must exist (not in initial state)",
        predicate=lambda s: s.has_fact("config_deployed") is not None,
        verification=VerificationPlan([]),
    )
    main_goal = Goal(
        id="deploy_app",
        description="Deploy application to production",
        predicate=lambda s: s.has_fact("app_deployed") is not None,
        verification=VerificationPlan([]),
        dependencies=[prereq_goal],
    )
    deploy_action = (
        action("deploy_app", "Deploy the app")
        .adds_fact("app_deployed")
        .with_cost(5.0)
        .build()
    )
    state = WorldState()
    # We do NOT add config_deployed, so prereq is never satisfied

    return Scenario(
        name="Deploy App (Missing Prereq) – real-world failure",
        description="Goal has dependency that is not satisfied; should get feedback.",
        difficulty=Difficulty.HARD,
        initial_state=state,
        goal=main_goal,
        available_actions=[deploy_action],
        success_predicate=lambda s: main_goal.is_satisfied(s),
        max_steps=5,
    )


async def main():
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY (or OPENAI_API_KEY) and run again.")
        sys.exit(1)

    provider = LLMProvider.OPENROUTER if os.getenv("OPENROUTER_API_KEY") else LLMProvider.OPENAI
    client = create_llm_client(provider)
    executor = SimpleLLMExecutor(client, verbose=True)
    problem = make_dependency_failure_problem()
    agent = DeliberativeAgent(
        actions=problem.available_actions,
        action_executor=executor,
    )

    print("\n" + "=" * 70)
    print("REAL-WORLD TEST: One problem that will fail (unsatisfied dependency)")
    print("=" * 70)
    print(f"Testing: {problem.name}")
    print(f"Provider: {provider.value} ({client.get_provider_name()})")
    print(f"Difficulty: {problem.difficulty}")
    print("=" * 70)

    result = await agent.achieve(problem.goal, problem.initial_state)

    success = result.success and problem.success_predicate(result.state or WorldState())
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
    print(f"Status: {result.status}")
    if result.lessons:
        print(f"Lessons learned: {len(result.lessons)}")

    if result.status == "no_plan_found":
        print(f"\n--- Why it failed ---")
        print(f"  {result.message}")
        if getattr(result, "concerns", None):
            for c in result.concerns:
                print(f"  Concern: {c}")
        if getattr(result, "failure_reason", None):
            print(f"  Planner: {result.failure_reason}")
        if getattr(result, "suggested_next_steps", None) and result.suggested_next_steps:
            print(f"\n--- Suggested next steps (give further direction) ---")
            for i, step in enumerate(result.suggested_next_steps, 1):
                print(f"  {i}. {step}")
        print()

    print("=" * 70)
    if success:
        print("(This run was expected to fail; if you see SUCCESS something changed.)")
    else:
        print("As the human: you got failure reason and next steps above.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
