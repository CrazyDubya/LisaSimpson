#!/usr/bin/env python3
"""
Between the agent and the human: fail first, then use the feedback to fix and succeed.

1. Run 1: Goal has unsatisfied dependency → no_plan_found + feedback.
2. Human (this script) acts on feedback: "Satisfy the listed dependencies first"
   → we add the missing prereq to state (or run a step that satisfies it).
3. Run 2: Same goal, state now has prereq satisfied → plan runs → success.

Usage:
  OPENROUTER_API_KEY=sk-... uv run python scripts/run_fail_then_succeed_with_feedback.py
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
    Confidence,
    ConfidenceSource,
    Fact,
)
from deliberative_agent.actions import action
from deliberative_agent.llm_integration import create_llm_client, LLMProvider
from deliberative_agent.llm_executor import SimpleLLMExecutor


def make_problem(state: WorldState):
    """Goal: deploy app. Prereq: config_deployed must exist in state."""
    prereq_goal = Goal(
        id="prereq",
        description="Prerequisite: deploy config must exist",
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
    return main_goal, [deploy_action]


async def main():
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY (or OPENAI_API_KEY) and run again.")
        sys.exit(1)

    provider = LLMProvider.OPENROUTER if os.getenv("OPENROUTER_API_KEY") else LLMProvider.OPENAI
    client = create_llm_client(provider)
    executor = SimpleLLMExecutor(client, verbose=True)

    # ----- Run 1: state without prereq → fail -----
    state1 = WorldState()
    goal, actions = make_problem(state1)
    agent = DeliberativeAgent(actions=actions, action_executor=executor)

    print("\n" + "=" * 70)
    print("RUN 1: Initial state has NO prereq (config_deployed)")
    print("=" * 70)

    result1 = await agent.achieve(goal, state1)

    print(f"Result: {'SUCCESS' if result1.success else 'FAILED'}")
    print(f"Status: {result1.status}")
    if not result1.success and result1.status == "no_plan_found":
        print(f"\n--- Why it failed ---")
        print(f"  {result1.message}")
        if getattr(result1, "failure_reason", None):
            print(f"  Planner: {result1.failure_reason}")
        if getattr(result1, "suggested_next_steps", None) and result1.suggested_next_steps:
            print(f"\n--- Suggested next steps ---")
            for i, step in enumerate(result1.suggested_next_steps, 1):
                print(f"  {i}. {step}")

    # ----- Human acts on feedback: satisfy the prerequisite -----
    print("\n" + "=" * 70)
    print("HUMAN (script): Acting on feedback — satisfying prerequisite.")
    print("  Adding config_deployed to state (e.g. run config deploy first).")
    print("=" * 70)

    state2 = WorldState()
    state2.add_fact(Fact("config_deployed", (), Confidence(1.0, ConfidenceSource.OBSERVATION)))

    # ----- Run 2: state with prereq → succeed -----
    goal2, actions2 = make_problem(state2)
    agent2 = DeliberativeAgent(actions=actions2, action_executor=executor)

    print("\nRUN 2: State now has prereq (config_deployed) satisfied")
    print("=" * 70)

    result2 = await agent2.achieve(goal2, state2)

    print(f"Result: {'SUCCESS' if result2.success else 'FAILED'}")
    print(f"Status: {result2.status}")
    if result2.plan:
        print(f"Steps: {[a.name for a in result2.plan.steps]}")

    print("\n" + "=" * 70)
    if result1.status == "no_plan_found" and result2.success:
        print("Between the two of us: we passed the failed state and it succeeded.")
    else:
        print(f"Run1 status={result1.status}, Run2 success={result2.success}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
