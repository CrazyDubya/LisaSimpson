"""Tests for swarm orchestration and goal DAG scheduling."""

from __future__ import annotations

import asyncio
from time import perf_counter

import pytest

from deliberative_agent.agent import DeliberativeAgent
from deliberative_agent.core import WorldState
from deliberative_agent.execution import ExecutionStepResult
from deliberative_agent.goals import Goal
from deliberative_agent.planning import HierarchicalPlanner, Planner
from deliberative_agent.swarm import SwarmManager, SwarmMember, TodoItem, build_todo_goal_graph
from deliberative_agent.verification import VerificationPlan
from deliberative_agent.actions import Action, action


class DelayExecutor:
    """Action executor that can simulate variable action runtimes."""

    def __init__(self, delays: dict[str, float]):
        self.delays = delays

    async def execute(self, action_obj: Action, state: WorldState) -> ExecutionStepResult:
        await asyncio.sleep(self.delays.get(action_obj.name, 0.0))
        new_state = action_obj.apply(state)
        return ExecutionStepResult.successful(
            new_state,
            [effect.description for effect in action_obj.effects],
        )


def _todo_goal(goal_id: str) -> Goal:
    return Goal(
        id=goal_id,
        description=f"Complete todo {goal_id}",
        predicate=lambda state, gid=goal_id: state.has_fact("todo_done", gid) is not None,
        verification=VerificationPlan([]),
    )


class TestSwarmManager:
    @pytest.mark.asyncio
    async def test_executes_parallel_branches_and_joins_dependencies(self):
        action_a = action("do_a", "Do A").adds_fact("todo_done", "a").build()
        action_b = action("do_b", "Do B").adds_fact("todo_done", "b").build()
        action_c = (
            action("do_c", "Do C")
            .requires_fact("todo_done", "a")
            .requires_fact("todo_done", "b")
            .adds_fact("todo_done", "c")
            .build()
        )

        goal_a = _todo_goal("a")
        goal_b = _todo_goal("b")
        goal_c = _todo_goal("c")
        goal_c.dependencies = [goal_a, goal_b]

        all_actions = [action_a, action_b, action_c]
        planner = HierarchicalPlanner(Planner(all_actions))

        worker_1 = SwarmMember(
            name="worker-1",
            agent=DeliberativeAgent(
                actions=all_actions,
                action_executor=DelayExecutor({"do_a": 0.30, "do_b": 0.30, "do_c": 0.05}),
            ),
        )
        worker_2 = SwarmMember(
            name="worker-2",
            agent=DeliberativeAgent(
                actions=all_actions,
                action_executor=DelayExecutor({"do_a": 0.30, "do_b": 0.30, "do_c": 0.05}),
            ),
        )

        manager = SwarmManager(
            hierarchical_planner=planner,
            members=[worker_1, worker_2],
            max_parallelism=2,
        )

        start = perf_counter()
        result = await manager.execute([goal_c], WorldState())
        elapsed = perf_counter() - start

        assert result.success
        assert result.node_results["a"].status == "success"
        assert result.node_results["b"].status == "success"
        assert result.node_results["c"].status == "success"
        assert result.final_state.has_fact("todo_done", "c") is not None
        assert elapsed < 0.55

    def test_build_goal_dag_includes_decomposed_subgoals(self):
        root = Goal(
            id="release",
            description="Release the project",
            predicate=lambda _s: False,
            verification=VerificationPlan([]),
            metadata={"type": "release"},
        )

        prep = Goal(
            id="prep",
            description="Prepare release artifacts",
            predicate=lambda _s: False,
            verification=VerificationPlan([]),
        )
        ship = Goal(
            id="ship",
            description="Ship the release",
            predicate=lambda _s: False,
            verification=VerificationPlan([]),
        )

        hierarchical = HierarchicalPlanner(Planner([]))
        hierarchical.register_decomposer("release", lambda _goal: [prep, ship])

        manager = SwarmManager(hierarchical_planner=hierarchical, members=[])
        nodes = manager.build_goal_dag([root])

        assert {"release", "prep", "ship"}.issubset(set(nodes.keys()))
        assert "prep" in nodes["release"].dependencies
        assert "ship" in nodes["release"].dependencies
        # Decomposed subgoals are sequenced like HierarchicalPlanner._plan_for_subgoals.
        assert "prep" in nodes["ship"].dependencies


class TestTodoGoalGraph:
    def test_build_todo_goal_graph_preserves_phase_and_dependencies(self):
        items = [
            TodoItem(
                id="phase1_design",
                description="Design system",
                phase="phase1",
                required_capabilities=["design"],
            ),
            TodoItem(
                id="phase2_build",
                description="Build system",
                dependencies=["phase1_design"],
                phase="phase2",
                required_capabilities=["build"],
            ),
        ]

        goals = build_todo_goal_graph(items)
        by_id = {goal.id: goal for goal in goals}

        assert by_id["phase2_build"].dependencies[0].id == "phase1_design"
        assert by_id["phase1_design"].metadata["phase"] == "phase1"
        assert by_id["phase2_build"].metadata["required_capabilities"] == ["build"]
