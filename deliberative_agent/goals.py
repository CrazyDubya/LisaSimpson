"""
Goal system for the Deliberative Agent.

Goals represent desired states with verification plans and dependencies.
Unlike simple "completion strings", goals have semantic meaning and
can be verified against specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, TYPE_CHECKING

from .core import WorldState
from .verification import VerificationPlan, VerificationResult

if TYPE_CHECKING:
    pass


@dataclass
class Goal:
    """
    A goal the agent is trying to achieve.

    Goals are more than just strings - they include:
    - A predicate for quick satisfaction checks
    - A verification plan for authoritative verification
    - Dependencies on other goals
    - Priority for scheduling

    This is fundamentally different from "Ralph Wiggum" style
    completion detection via magic strings.
    """

    id: str
    description: str
    predicate: Callable[[WorldState], bool]
    verification: VerificationPlan
    dependencies: List[Goal] = field(default_factory=list)
    priority: int = 0
    metadata: dict = field(default_factory=dict)

    def is_satisfied(self, state: WorldState) -> bool:
        """
        Quick check if goal is satisfied.

        This is a fast heuristic - use verify() for authoritative check.

        Args:
            state: Current world state

        Returns:
            True if predicate indicates goal is satisfied
        """
        try:
            return self.predicate(state)
        except Exception:
            return False

    def can_attempt(self, state: WorldState) -> bool:
        """
        Check if all dependencies are satisfied.

        Args:
            state: Current world state

        Returns:
            True if all dependencies are met
        """
        return all(dep.is_satisfied(state) for dep in self.dependencies)

    def unsatisfied_dependencies(self, state: WorldState) -> List[Goal]:
        """
        Get list of unsatisfied dependencies.

        Args:
            state: Current world state

        Returns:
            List of dependency goals that aren't satisfied
        """
        return [dep for dep in self.dependencies if not dep.is_satisfied(state)]

    async def verify(self, state: WorldState) -> VerificationResult:
        """
        Authoritatively verify if goal is satisfied.

        This runs the full verification plan, not just the predicate.

        Args:
            state: Current world state

        Returns:
            VerificationResult with detailed outcomes
        """
        return await self.verification.verify(state)

    def with_dependency(self, dependency: Goal) -> Goal:
        """
        Add a dependency to this goal.

        Args:
            dependency: Goal that must be satisfied first

        Returns:
            Self for chaining
        """
        self.dependencies.append(dependency)
        return self

    def with_priority(self, priority: int) -> Goal:
        """
        Set the priority for this goal.

        Args:
            priority: Higher = more important

        Returns:
            Self for chaining
        """
        self.priority = priority
        return self

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Goal):
            return self.id == other.id
        return False

    def __repr__(self) -> str:
        deps = f", deps={len(self.dependencies)}" if self.dependencies else ""
        return f"Goal({self.id!r}{deps})"


class GoalBuilder:
    """
    Fluent builder for creating goals.

    Makes it easier to construct complex goals with verification.
    """

    def __init__(self, id: str, description: str):
        """
        Start building a goal.

        Args:
            id: Unique identifier for the goal
            description: Human-readable description
        """
        self._id = id
        self._description = description
        self._predicate: Optional[Callable[[WorldState], bool]] = None
        self._verification: Optional[VerificationPlan] = None
        self._dependencies: List[Goal] = []
        self._priority: int = 0
        self._metadata: dict = {}

    def with_predicate(
        self,
        predicate: Callable[[WorldState], bool]
    ) -> GoalBuilder:
        """Set the satisfaction predicate."""
        self._predicate = predicate
        return self

    def with_verification(self, plan: VerificationPlan) -> GoalBuilder:
        """Set the verification plan."""
        self._verification = plan
        return self

    def with_dependency(self, goal: Goal) -> GoalBuilder:
        """Add a dependency."""
        self._dependencies.append(goal)
        return self

    def with_dependencies(self, goals: List[Goal]) -> GoalBuilder:
        """Add multiple dependencies."""
        self._dependencies.extend(goals)
        return self

    def with_priority(self, priority: int) -> GoalBuilder:
        """Set the priority."""
        self._priority = priority
        return self

    def with_metadata(self, key: str, value: object) -> GoalBuilder:
        """Add metadata."""
        self._metadata[key] = value
        return self

    def build(self) -> Goal:
        """
        Build the goal.

        Raises:
            ValueError: If required fields are missing
        """
        if self._predicate is None:
            raise ValueError("Goal requires a predicate")
        if self._verification is None:
            # Default to empty verification plan
            self._verification = VerificationPlan(checks=[])

        return Goal(
            id=self._id,
            description=self._description,
            predicate=self._predicate,
            verification=self._verification,
            dependencies=self._dependencies,
            priority=self._priority,
            metadata=self._metadata
        )


def goal(id: str, description: str) -> GoalBuilder:
    """
    Convenience function to start building a goal.

    Example:
        my_goal = (
            goal("test_pass", "All tests should pass")
            .with_predicate(lambda s: s.all_tests_pass())
            .with_verification(VerificationPlan([TestCheck(["pytest"])]))
            .build()
        )

    Args:
        id: Unique identifier
        description: Human-readable description

    Returns:
        GoalBuilder for fluent construction
    """
    return GoalBuilder(id, description)


class CompositeGoal:
    """
    A goal composed of multiple sub-goals.

    Useful for complex objectives that need to be broken down.
    """

    def __init__(
        self,
        id: str,
        description: str,
        subgoals: List[Goal],
        require_all: bool = True
    ):
        """
        Create a composite goal.

        Args:
            id: Unique identifier
            description: Human-readable description
            subgoals: List of sub-goals
            require_all: If True, all subgoals must be satisfied
        """
        self.id = id
        self.description = description
        self.subgoals = subgoals
        self.require_all = require_all

    def is_satisfied(self, state: WorldState) -> bool:
        """Check if composite goal is satisfied."""
        if self.require_all:
            return all(g.is_satisfied(state) for g in self.subgoals)
        else:
            return any(g.is_satisfied(state) for g in self.subgoals)

    def satisfied_subgoals(self, state: WorldState) -> List[Goal]:
        """Get list of satisfied subgoals."""
        return [g for g in self.subgoals if g.is_satisfied(state)]

    def pending_subgoals(self, state: WorldState) -> List[Goal]:
        """Get list of pending subgoals."""
        return [g for g in self.subgoals if not g.is_satisfied(state)]

    def progress(self, state: WorldState) -> float:
        """Get completion progress (0.0 to 1.0)."""
        if not self.subgoals:
            return 1.0
        satisfied = len(self.satisfied_subgoals(state))
        return satisfied / len(self.subgoals)

    def to_goal(self) -> Goal:
        """
        Convert to a regular Goal.

        The composite becomes a goal whose predicate checks subgoal satisfaction.
        """
        return Goal(
            id=self.id,
            description=self.description,
            predicate=self.is_satisfied,
            verification=VerificationPlan(checks=[]),
            dependencies=self.subgoals if self.require_all else []
        )
