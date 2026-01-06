"""Tests for the Deliberative Agent."""

import pytest
from typing import List

from deliberative_agent.core import (
    Confidence,
    ConfidenceSource,
    Fact,
    WorldState,
)
from deliberative_agent.actions import Action, action
from deliberative_agent.goals import Goal, goal
from deliberative_agent.verification import VerificationPlan
from deliberative_agent.execution import (
    ActionExecutor,
    ExecutionStepResult,
)
from deliberative_agent.memory import Memory, Lesson
from deliberative_agent.agent import DeliberativeAgent, AgentResult


class MockExecutor:
    """Mock executor for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.executed: List[str] = []

    async def execute(
        self,
        action: Action,
        state: WorldState
    ) -> ExecutionStepResult:
        self.executed.append(action.name)

        if self.should_fail:
            return ExecutionStepResult.failed(
                state,
                Exception("Mock failure")
            )

        # Apply action effects
        new_state = action.apply(state)
        return ExecutionStepResult.successful(
            new_state,
            [e.description for e in action.effects]
        )


class TestDeliberativeAgent:
    """Tests for the main agent."""

    @pytest.mark.asyncio
    async def test_agent_achieves_simple_goal(self):
        """Agent should achieve a simple goal."""
        # Setup
        achieve_action = (
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

        executor = MockExecutor()
        agent = DeliberativeAgent(
            actions=[achieve_action],
            action_executor=executor
        )

        # Execute
        state = WorldState()
        result = await agent.achieve(goal_obj, state)

        # Verify
        assert result.success
        assert "achieve" in executor.executed

    @pytest.mark.asyncio
    async def test_agent_handles_already_satisfied(self):
        """Agent should recognize already satisfied goals."""
        goal_obj = Goal(
            id="test_goal",
            description="Test goal",
            predicate=lambda s: s.has_fact("goal_achieved") is not None,
            verification=VerificationPlan([])
        )

        executor = MockExecutor()
        agent = DeliberativeAgent(
            actions=[],
            action_executor=executor
        )

        # State already has the goal achieved
        state = WorldState()
        state.add_fact(Fact(
            "goal_achieved",
            (),
            Confidence(1.0, ConfidenceSource.OBSERVATION)
        ))

        result = await agent.achieve(goal_obj, state)

        assert result.success
        assert len(executor.executed) == 0  # No actions needed

    @pytest.mark.asyncio
    async def test_agent_fails_impossible_goal(self):
        """Agent should recognize impossible goals."""
        # Action requires something that doesn't exist
        blocked_action = (
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

        executor = MockExecutor()
        agent = DeliberativeAgent(
            actions=[blocked_action],
            action_executor=executor
        )

        state = WorldState()
        result = await agent.achieve(goal_obj, state)

        assert not result.success
        assert result.status == "no_plan_found"

    @pytest.mark.asyncio
    async def test_agent_learns_from_success(self):
        """Agent should learn from successful execution."""
        achieve_action = (
            action("achieve", "Achieve the goal")
            .adds_fact("goal_achieved")
            .with_cost(1.0)
            .build()
        )

        goal_obj = Goal(
            id="test_goal",
            description="Test goal for learning",
            predicate=lambda s: s.has_fact("goal_achieved") is not None,
            verification=VerificationPlan([])
        )

        executor = MockExecutor()
        agent = DeliberativeAgent(
            actions=[achieve_action],
            action_executor=executor
        )

        state = WorldState()
        result = await agent.achieve(goal_obj, state)

        # Should have learned something
        assert len(agent.memory.lessons) > 0
        # Lesson should be about success
        success_lessons = agent.memory.retrieve_by_outcome("success")
        assert len(success_lessons) > 0

    @pytest.mark.asyncio
    async def test_agent_uses_past_experience(self):
        """Agent should use past experience to estimate confidence."""
        achieve_action = (
            action("achieve", "Achieve the goal")
            .adds_fact("goal_achieved")
            .with_cost(1.0)
            .build()
        )

        goal_obj = Goal(
            id="test_goal",
            description="Writing unit tests",
            predicate=lambda s: s.has_fact("goal_achieved") is not None,
            verification=VerificationPlan([])
        )

        # Pre-populate memory with past successes
        memory = Memory()
        memory.add_lesson(Lesson(
            situation="Writing unit tests",
            insight="Tests worked",
            outcome="success",
            confidence=Confidence(0.8, ConfidenceSource.MEMORY)
        ))

        executor = MockExecutor()
        agent = DeliberativeAgent(
            actions=[achieve_action],
            action_executor=executor,
            memory=memory
        )

        state = WorldState()
        result = await agent.achieve(goal_obj, state)

        # Should have high confidence due to past experience
        assert result.success

    @pytest.mark.asyncio
    async def test_agent_handles_execution_failure(self):
        """Agent should handle execution failures gracefully."""
        achieve_action = (
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

        # Executor that always fails
        executor = MockExecutor(should_fail=True)
        agent = DeliberativeAgent(
            actions=[achieve_action],
            action_executor=executor
        )

        state = WorldState()
        result = await agent.achieve(goal_obj, state)

        assert not result.success
        # Should have learned from failure
        failure_lessons = agent.memory.retrieve_by_outcome("failure")
        assert len(failure_lessons) > 0

    @pytest.mark.asyncio
    async def test_agent_multi_step_plan(self):
        """Agent should execute multi-step plans."""
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

        executor = MockExecutor()
        agent = DeliberativeAgent(
            actions=[step1, step2],
            action_executor=executor
        )

        state = WorldState()
        result = await agent.achieve(goal_obj, state)

        assert result.success
        assert executor.executed == ["step1", "step2"]

    def test_agent_statistics(self):
        """Agent should provide statistics."""
        actions = [
            action("a", "Action A").build(),
            action("b", "Action B").build(),
        ]

        memory = Memory()
        memory.add_lesson(Lesson(
            situation="Test",
            insight="Insight",
            outcome="success",
            confidence=Confidence(0.8, ConfidenceSource.MEMORY)
        ))

        executor = MockExecutor()
        agent = DeliberativeAgent(
            actions=actions,
            action_executor=executor,
            memory=memory
        )

        stats = agent.get_statistics()

        assert stats["actions_available"] == 2
        assert stats["lessons_learned"] == 1


class TestAgentResult:
    """Tests for AgentResult class."""

    def test_success_check(self):
        """Should correctly identify success."""
        success = AgentResult(status="success", message="Done")
        failure = AgentResult(status="failure", message="Failed")

        assert success.success
        assert not failure.success

    def test_needs_input_check(self):
        """Should identify when input is needed."""
        needs_approval = AgentResult(
            status="needs_approval",
            message="Review plan"
        )
        needs_confidence = AgentResult(
            status="insufficient_confidence",
            message="Not confident"
        )
        complete = AgentResult(status="success", message="Done")

        assert needs_approval.needs_input
        assert needs_confidence.needs_input
        assert not complete.needs_input

    def test_summary_generation(self):
        """Should generate readable summaries."""
        result = AgentResult(
            status="needs_approval",
            message="Please review",
            concerns=["Plan is long", "Has risky steps"],
            questions=["Is this the right approach?"]
        )

        summary = result.summary()

        assert "needs_approval" in summary
        assert "Please review" in summary
        assert "Plan is long" in summary
