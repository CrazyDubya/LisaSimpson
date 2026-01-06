"""Tests for planning system."""

import pytest

from deliberative_agent.core import (
    Confidence,
    ConfidenceSource,
    Fact,
    WorldState,
)
from deliberative_agent.actions import Action, Effect, action
from deliberative_agent.goals import Goal, goal
from deliberative_agent.planning import Planner, Plan
from deliberative_agent.verification import VerificationPlan


class TestPlanner:
    """Tests for the GOAP-style planner."""

    def test_plan_already_satisfied(self):
        """Should return empty plan if goal already satisfied."""
        state = WorldState()
        state.add_fact(Fact(
            "goal_achieved",
            (),
            Confidence(1.0, ConfidenceSource.OBSERVATION)
        ))

        goal_obj = Goal(
            id="test_goal",
            description="Test goal",
            predicate=lambda s: s.has_fact("goal_achieved") is not None,
            verification=VerificationPlan([])
        )

        planner = Planner([])
        result = planner.plan(state, goal_obj)

        assert result.success
        assert result.plan.is_empty()

    def test_plan_simple_goal(self):
        """Should find a plan for a simple goal."""
        state = WorldState()

        # Action that achieves the goal
        achieve_goal = (
            action("achieve", "Achieve the goal")
            .adds_fact("goal_achieved")
            .with_cost(1.0)
            .build()
        )

        goal_obj = Goal(
            id="test_goal",
            description="Test goal",
            predicate=lambda s: s.has_fact("goal_achieved") is not None,
            verification=VerificationPlan([])
        )

        planner = Planner([achieve_goal])
        result = planner.plan(state, goal_obj)

        assert result.success
        assert len(result.plan.steps) == 1
        assert result.plan.steps[0].name == "achieve"

    def test_plan_multi_step(self):
        """Should find multi-step plans."""
        state = WorldState()

        # Two actions in sequence
        step1 = (
            action("step1", "First step")
            .adds_fact("step1_done")
            .with_cost(1.0)
            .build()
        )

        step2 = (
            action("step2", "Second step")
            .requires_fact("step1_done")
            .adds_fact("goal_achieved")
            .with_cost(1.0)
            .build()
        )

        goal_obj = Goal(
            id="test_goal",
            description="Test goal",
            predicate=lambda s: s.has_fact("goal_achieved") is not None,
            verification=VerificationPlan([])
        )

        planner = Planner([step1, step2])
        result = planner.plan(state, goal_obj)

        assert result.success
        assert len(result.plan.steps) == 2
        assert result.plan.steps[0].name == "step1"
        assert result.plan.steps[1].name == "step2"

    def test_plan_impossible(self):
        """Should fail gracefully for impossible goals."""
        state = WorldState()

        # Action that requires something that doesn't exist
        action_obj = (
            action("blocked", "Blocked action")
            .requires_fact("nonexistent")
            .adds_fact("goal_achieved")
            .build()
        )

        goal_obj = Goal(
            id="test_goal",
            description="Test goal",
            predicate=lambda s: s.has_fact("goal_achieved") is not None,
            verification=VerificationPlan([])
        )

        planner = Planner([action_obj])
        result = planner.plan(state, goal_obj)

        assert not result.success
        assert "No plan exists" in result.reason or "limit" in result.reason.lower()

    def test_plan_prefers_lower_cost(self):
        """Should prefer lower cost plans."""
        state = WorldState()

        # Two paths to goal - one cheap, one expensive
        cheap = (
            action("cheap", "Cheap option")
            .adds_fact("goal_achieved")
            .with_cost(1.0)
            .build()
        )

        expensive = (
            action("expensive", "Expensive option")
            .adds_fact("goal_achieved")
            .with_cost(100.0)
            .build()
        )

        goal_obj = Goal(
            id="test_goal",
            description="Test goal",
            predicate=lambda s: s.has_fact("goal_achieved") is not None,
            verification=VerificationPlan([])
        )

        planner = Planner([expensive, cheap])  # Order shouldn't matter
        result = planner.plan(state, goal_obj)

        assert result.success
        assert result.plan.steps[0].name == "cheap"

    def test_plan_respects_dependencies(self):
        """Should fail if goal dependencies aren't met."""
        state = WorldState()

        dep_goal = Goal(
            id="dependency",
            description="Dependency",
            predicate=lambda s: s.has_fact("dep_done") is not None,
            verification=VerificationPlan([])
        )

        main_goal = Goal(
            id="main",
            description="Main goal",
            predicate=lambda s: s.has_fact("goal_achieved") is not None,
            verification=VerificationPlan([]),
            dependencies=[dep_goal]
        )

        planner = Planner([])
        result = planner.plan(state, main_goal)

        assert not result.success
        assert "dependency" in result.reason.lower()

    def test_find_multiple_plans(self):
        """Should be able to find multiple alternative plans."""
        state = WorldState()

        option1 = (
            action("option1", "First option")
            .adds_fact("goal_achieved")
            .with_cost(1.0)
            .build()
        )

        option2 = (
            action("option2", "Second option")
            .adds_fact("goal_achieved")
            .with_cost(2.0)
            .build()
        )

        goal_obj = Goal(
            id="test_goal",
            description="Test goal",
            predicate=lambda s: s.has_fact("goal_achieved") is not None,
            verification=VerificationPlan([])
        )

        planner = Planner([option1, option2])
        plans = planner.find_all_plans(state, goal_obj, max_plans=5)

        assert len(plans) >= 2
        plan_names = [p.steps[0].name for p in plans]
        assert "option1" in plan_names
        assert "option2" in plan_names


class TestPlan:
    """Tests for Plan class."""

    def test_empty_plan(self):
        """Empty plans should be identifiable."""
        state = WorldState()
        plan = Plan.empty(state)

        assert plan.is_empty()
        assert plan.total_cost() == 0.0
        assert not plan.has_irreversible_steps()

    def test_plan_cost_calculation(self):
        """Should calculate total cost correctly."""
        actions = [
            Action(
                name="a",
                description="",
                preconditions=[],
                effects=[],
                cost=1.0,
                reversible=True
            ),
            Action(
                name="b",
                description="",
                preconditions=[],
                effects=[],
                cost=2.0,
                reversible=True
            ),
        ]

        plan = Plan(
            steps=actions,
            expected_final_state=WorldState(),
            verification_points={},
            confidence=Confidence(0.9, ConfidenceSource.INFERENCE),
            estimated_cost=3.0
        )

        assert plan.total_cost() == 3.0

    def test_plan_irreversible_detection(self):
        """Should detect irreversible steps."""
        reversible = Action(
            name="reversible",
            description="",
            preconditions=[],
            effects=[],
            cost=1.0,
            reversible=True
        )

        irreversible = Action(
            name="irreversible",
            description="",
            preconditions=[],
            effects=[],
            cost=1.0,
            reversible=False
        )

        plan1 = Plan(
            steps=[reversible],
            expected_final_state=WorldState(),
            verification_points={},
            confidence=Confidence(0.9, ConfidenceSource.INFERENCE),
            estimated_cost=1.0
        )
        assert not plan1.has_irreversible_steps()

        plan2 = Plan(
            steps=[reversible, irreversible],
            expected_final_state=WorldState(),
            verification_points={},
            confidence=Confidence(0.9, ConfidenceSource.INFERENCE),
            estimated_cost=2.0
        )
        assert plan2.has_irreversible_steps()
