#!/usr/bin/env python3
"""
Demo: run failure scenarios and print feedback (dependencies not met, exploration limit).

Shows that the agent returns failure_reason and suggested_next_steps so the user
can get further direction. Run from repo root: uv run python scripts/run_failure_feedback_demo.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deliberative_agent import (
    DeliberativeAgent,
    WorldState,
    Goal,
    VerificationPlan,
    Planner,
)
from deliberative_agent.actions import action
from deliberative_agent.execution import ActionExecutor, ExecutionStepResult


class NoOpExecutor(ActionExecutor):
    async def execute(self, action, state):
        new_state = action.apply(state)
        return ExecutionStepResult.successful(new_state, [action.description])


async def demo_dependencies_not_met():
    """Trigger 'Dependencies not met' and print feedback."""
    dep_goal = Goal(
        id="prereq",
        description="Prerequisite (must be done first)",
        predicate=lambda s: s.has_fact("prereq_done") is not None,
        verification=VerificationPlan([]),
    )
    main_goal = Goal(
        id="main",
        description="Main goal",
        predicate=lambda s: s.has_fact("done") is not None,
        verification=VerificationPlan([]),
        dependencies=[dep_goal],
    )
    act = action("do_it", "Do the thing").adds_fact("done").with_cost(1.0).build()
    agent = DeliberativeAgent(actions=[act], action_executor=NoOpExecutor())
    state = WorldState()

    result = await agent.achieve(main_goal, state)

    print("\n" + "=" * 60)
    print("SCENARIO 1: Dependencies not met")
    print("=" * 60)
    print("Status:", result.status)
    print("Message:", result.message)
    print("Failure reason:", result.failure_reason)
    print("Suggested next steps:")
    for i, s in enumerate(result.suggested_next_steps, 1):
        print(f"  {i}. {s}")
    print("Questions (for user):", result.questions)
    print()


async def demo_exploration_limit():
    """Trigger 'Exploration limit reached' and print feedback."""
    step1 = action("step1", "First").adds_fact("a").with_cost(1.0).build()
    step2 = action("step2", "Second").requires_fact("a").adds_fact("b").with_cost(1.0).build()
    step3 = action("step3", "Third").requires_fact("b").adds_fact("goal").with_cost(1.0).build()
    goal_obj = Goal(
        id="g",
        description="Three-step goal",
        predicate=lambda s: s.has_fact("goal") is not None,
        verification=VerificationPlan([]),
    )
    planner = Planner([step1, step2, step3], max_explored=2, max_depth=10)
    agent = DeliberativeAgent(
        actions=[step1, step2, step3],
        action_executor=NoOpExecutor(),
        planner=planner,
    )
    state = WorldState()

    result = await agent.achieve(goal_obj, state)

    print("=" * 60)
    print("SCENARIO 2: Exploration limit reached")
    print("=" * 60)
    print("Status:", result.status)
    print("Message:", result.message)
    print("Failure reason:", result.failure_reason)
    print("Suggested next steps:")
    for i, s in enumerate(result.suggested_next_steps, 1):
        print(f"  {i}. {s}")
    print("Questions (for user):", result.questions)
    print()


async def main():
    await demo_dependencies_not_met()
    await demo_exploration_limit()
    print("Done. Feedback is available for user to get further direction.")


if __name__ == "__main__":
    asyncio.run(main())
