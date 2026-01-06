"""
The main Deliberative Agent.

This is the orchestrator that brings together planning, execution,
verification, and learning into a coherent whole.

Unlike "Ralph Wiggum" style blind iteration, this agent:
- Plans before acting
- Verifies semantically
- Learns from experience
- Knows when to ask for help
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from .core import Confidence, ConfidenceSource, WorldState
from .actions import Action
from .goals import Goal
from .planning import Plan, Planner, PlanningResult
from .execution import ActionExecutor, ExecutionResult, Executor
from .memory import Memory, Lesson, Episode, LessonExtractor
from .verification import VerificationResult

if TYPE_CHECKING:
    pass


@dataclass
class AgentResult:
    """Result of the agent attempting to achieve a goal."""

    status: str
    message: str = ""
    state: Optional[WorldState] = None
    plan: Optional[Plan] = None
    verification: Optional[VerificationResult] = None
    questions: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    lessons: List[Lesson] = field(default_factory=list)
    alternatives: List[Plan] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if the goal was achieved."""
        return self.status == "success"

    @property
    def needs_input(self) -> bool:
        """Check if agent needs human input to proceed."""
        return self.status in [
            "insufficient_confidence",
            "needs_approval",
            "needs_clarification"
        ]

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [f"Status: {self.status}"]
        if self.message:
            lines.append(f"Message: {self.message}")
        if self.concerns:
            lines.append(f"Concerns: {'; '.join(self.concerns)}")
        if self.questions:
            lines.append(f"Questions: {'; '.join(self.questions)}")
        return "\n".join(lines)


class DeliberativeAgent:
    """
    The main agent that plans, executes, verifies, and learns.

    This is NOT "Ralph Wiggum". This one thinks before acting.

    Key behaviors:
    1. Assesses confidence before attempting
    2. Plans explicitly before executing
    3. Verifies semantically, not syntactically
    4. Learns from experience
    5. Knows when to ask for help
    """

    # Confidence thresholds
    MIN_CONFIDENCE_TO_ATTEMPT = 0.3
    MIN_CONFIDENCE_TO_AUTO_EXECUTE = 0.6

    def __init__(
        self,
        actions: List[Action],
        action_executor: ActionExecutor,
        memory: Optional[Memory] = None,
        planner: Optional[Planner] = None
    ):
        """
        Initialize the agent.

        Args:
            actions: Available actions
            action_executor: Implementation that executes actions
            memory: Optional memory for learning
            planner: Optional custom planner
        """
        self.actions = actions
        self.planner = planner or Planner(actions)
        self.executor = Executor(action_executor)
        self.memory = memory or Memory()
        self.lesson_extractor = LessonExtractor()

    async def achieve(self, goal: Goal, state: WorldState) -> AgentResult:
        """
        Attempt to achieve a goal.

        Unlike "Ralph Wiggum", we:
        1. Assess our confidence before acting
        2. Plan explicitly before executing
        3. Verify semantically, not syntactically
        4. Learn from the experience
        5. Know when to ask for help

        Args:
            goal: Goal to achieve
            state: Current world state

        Returns:
            AgentResult with detailed outcomes
        """
        # Step 1: Retrieve relevant experience
        relevant_lessons = self.memory.retrieve_relevant(goal)

        # Step 2: Assess our confidence
        confidence = self._estimate_confidence(goal, state, relevant_lessons)

        if float(confidence) < self.MIN_CONFIDENCE_TO_ATTEMPT:
            questions = self._generate_clarifying_questions(goal, confidence)
            return AgentResult(
                status="insufficient_confidence",
                message=f"Confidence too low ({float(confidence):.2f}) to attempt.",
                questions=questions,
                state=state
            )

        # Step 3: Plan
        planning_result = self.planner.plan(state, goal)

        if not planning_result.success:
            analysis = self._analyze_impossibility(goal, state, planning_result)
            return AgentResult(
                status="no_plan_found",
                message="Cannot find a way to achieve this goal.",
                concerns=[analysis],
                state=state
            )

        plan = planning_result.plan

        # Step 4: Evaluate plan quality
        plan_concerns = self._evaluate_plan(plan, relevant_lessons)

        if float(plan.confidence) < self.MIN_CONFIDENCE_TO_AUTO_EXECUTE:
            # Find alternatives for user to consider
            alternatives = self.planner.find_all_plans(state, goal, max_plans=3)

            return AgentResult(
                status="needs_approval",
                message="I have a plan but would like you to review it.",
                plan=plan,
                concerns=plan_concerns,
                alternatives=[p for p in alternatives if p != plan],
                state=state
            )

        # Step 5: Execute
        result = await self.executor.execute(plan, state)

        # Step 6: Learn from execution
        lessons = self.lesson_extractor.extract(goal, plan, result)
        for lesson in lessons:
            self.memory.add_lesson(lesson)

        # Record episode
        episode = Episode.from_execution(goal, plan, result, lessons)
        self.memory.add_episode(episode)

        # Step 7: Final verification against original goal
        if result.status == "success":
            final_verification = await goal.verify(result.final_state)

            if not final_verification.satisfied:
                # Execution completed but goal not satisfied
                gap_analysis = self._analyze_gap(goal, result.final_state, final_verification)
                return AgentResult(
                    status="goal_not_satisfied",
                    message="Execution completed but goal verification failed.",
                    state=result.final_state,
                    verification=final_verification,
                    concerns=[gap_analysis],
                    lessons=lessons
                )

        return AgentResult(
            status=result.status,
            message=f"Goal {'achieved' if result.status == 'success' else 'not achieved'}.",
            state=result.final_state,
            plan=plan,
            lessons=lessons
        )

    async def achieve_with_approval(
        self,
        goal: Goal,
        state: WorldState,
        approve_plan: callable
    ) -> AgentResult:
        """
        Achieve a goal, getting approval for the plan first.

        Args:
            goal: Goal to achieve
            state: Current world state
            approve_plan: Async function that approves/rejects plans

        Returns:
            AgentResult with detailed outcomes
        """
        # Get initial result (may need approval)
        result = await self.achieve(goal, state)

        if result.status == "needs_approval" and result.plan:
            # Get approval
            approved = await approve_plan(result.plan, result.alternatives)

            if approved:
                # Execute the plan
                exec_result = await self.executor.execute(result.plan, state)

                # Learn from execution
                lessons = self.lesson_extractor.extract(goal, result.plan, exec_result)
                for lesson in lessons:
                    self.memory.add_lesson(lesson)

                return AgentResult(
                    status=exec_result.status,
                    message=f"Plan executed: {exec_result.status}",
                    state=exec_result.final_state,
                    plan=result.plan,
                    lessons=lessons
                )
            else:
                return AgentResult(
                    status="rejected",
                    message="Plan was not approved.",
                    plan=result.plan,
                    state=state
                )

        return result

    def _estimate_confidence(
        self,
        goal: Goal,
        state: WorldState,
        lessons: List[Lesson]
    ) -> Confidence:
        """
        Estimate our confidence in achieving this goal.

        Takes into account:
        - Past experience with similar goals
        - Current state
        - Goal complexity
        """
        # Base confidence from goal complexity
        base = 0.5

        # Adjust based on past experience
        successes = [l for l in lessons if l.outcome == "success"]
        failures = [l for l in lessons if l.outcome == "failure"]

        if successes:
            base += 0.1 * min(len(successes), 3)
        if failures:
            base -= 0.1 * min(len(failures), 3)

        # Adjust based on dependency satisfaction
        if goal.dependencies:
            dep_satisfaction = sum(
                1 for d in goal.dependencies if d.is_satisfied(state)
            )
            dep_ratio = dep_satisfaction / len(goal.dependencies)
            base *= (0.5 + 0.5 * dep_ratio)

        # Adjust based on historical success rate
        success_rate = self.memory.get_success_rate(goal)
        base = (base + success_rate) / 2

        return Confidence(
            max(0.0, min(1.0, base)),
            ConfidenceSource.INFERENCE,
            evidence=[
                f"{len(successes)} similar successes",
                f"{len(failures)} similar failures",
                f"Historical success rate: {success_rate:.2f}"
            ]
        )

    def _generate_clarifying_questions(
        self,
        goal: Goal,
        confidence: Confidence
    ) -> List[str]:
        """Generate questions to improve understanding."""
        questions = []

        if float(confidence) < 0.3:
            questions.append(
                f"Can you provide more detail about: {goal.description}?"
            )

        if goal.dependencies:
            unsatisfied = [
                d for d in goal.dependencies
                if not d.is_satisfied(WorldState())  # Approximate check
            ]
            if unsatisfied:
                deps = [d.description for d in unsatisfied]
                questions.append(
                    f"These prerequisites may not be met: {deps}. "
                    "Should I address these first?"
                )

        questions.append("What does success look like for this goal?")

        # Add questions based on past failures
        failure_patterns = self.memory.get_common_failure_patterns()
        if failure_patterns:
            questions.append(
                f"Past attempts often failed due to: {failure_patterns[0]}. "
                "Is this still a concern?"
            )

        return questions

    def _analyze_impossibility(
        self,
        goal: Goal,
        state: WorldState,
        planning_result: PlanningResult
    ) -> str:
        """Understand why a goal can't be achieved."""
        reasons = [planning_result.reason]

        # Check dependencies
        if goal.dependencies:
            unsatisfied = goal.unsatisfied_dependencies(state)
            if unsatisfied:
                reasons.append(
                    f"Prerequisites not met: {[d.id for d in unsatisfied]}"
                )

        # Check available actions
        applicable = [a for a in self.actions if a.applicable(state)]
        if not applicable:
            reasons.append("No actions are currently applicable")
        elif len(applicable) < len(self.actions) / 2:
            reasons.append(
                f"Only {len(applicable)}/{len(self.actions)} actions available"
            )

        return "; ".join(reasons)

    def _evaluate_plan(
        self,
        plan: Plan,
        lessons: List[Lesson]
    ) -> List[str]:
        """Evaluate potential concerns with a plan."""
        concerns = []

        # Check for risky actions
        if plan.has_irreversible_steps():
            irreversible = [a for a in plan.steps if not a.reversible]
            concerns.append(
                f"Plan includes {len(irreversible)} irreversible action(s): "
                f"{[a.name for a in irreversible]}"
            )

        # Check cost
        if plan.estimated_cost > 100:
            concerns.append(f"Plan has high estimated cost: {plan.estimated_cost}")

        # Check length
        if len(plan.steps) > 20:
            concerns.append(f"Plan is long ({len(plan.steps)} steps)")

        # Check lessons for warnings
        relevant_failures = [l for l in lessons if l.outcome == "failure"]
        if relevant_failures:
            insights = [l.insight for l in relevant_failures[:3]]
            concerns.append(f"Similar goals have failed: {insights}")

        return concerns

    def _analyze_gap(
        self,
        goal: Goal,
        state: WorldState,
        verification: VerificationResult
    ) -> str:
        """Analyze why goal isn't satisfied despite execution completing."""
        failures = [f.message for f in verification.failures]
        return f"Goal verification failed: {'; '.join(failures)}"

    def get_statistics(self) -> dict:
        """Get agent statistics."""
        return {
            "actions_available": len(self.actions),
            "lessons_learned": len(self.memory.lessons),
            "episodes_recorded": len(self.memory.episodes),
            "success_rate": self.memory.get_success_rate(),
            "common_failures": self.memory.get_common_failure_patterns()
        }


class AgentBuilder:
    """Fluent builder for creating agents."""

    def __init__(self):
        self._actions: List[Action] = []
        self._executor: Optional[ActionExecutor] = None
        self._memory: Optional[Memory] = None
        self._planner: Optional[Planner] = None

    def with_actions(self, actions: List[Action]) -> AgentBuilder:
        """Add available actions."""
        self._actions.extend(actions)
        return self

    def with_action(self, action: Action) -> AgentBuilder:
        """Add a single action."""
        self._actions.append(action)
        return self

    def with_executor(self, executor: ActionExecutor) -> AgentBuilder:
        """Set the action executor."""
        self._executor = executor
        return self

    def with_memory(self, memory: Memory) -> AgentBuilder:
        """Set the memory."""
        self._memory = memory
        return self

    def with_planner(self, planner: Planner) -> AgentBuilder:
        """Set a custom planner."""
        self._planner = planner
        return self

    def build(self) -> DeliberativeAgent:
        """Build the agent."""
        if self._executor is None:
            raise ValueError("Agent requires an action executor")

        return DeliberativeAgent(
            actions=self._actions,
            action_executor=self._executor,
            memory=self._memory,
            planner=self._planner
        )


def agent() -> AgentBuilder:
    """Start building an agent."""
    return AgentBuilder()
