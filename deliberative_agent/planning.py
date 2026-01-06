"""
GOAP-style planning system for the Deliberative Agent.

This is the key differentiator from "Ralph Wiggum" approaches:
we PLAN before acting, using goal-oriented action planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .core import Confidence, ConfidenceSource, WorldState
from .actions import Action
from .goals import Goal
from .verification import VerificationPlan

if TYPE_CHECKING:
    pass


@dataclass
class Plan:
    """
    A plan to achieve a goal.

    Plans are explicit sequences of actions with:
    - Verification points for safety
    - Confidence estimates
    - Cost estimates
    """

    steps: List[Action]
    expected_final_state: WorldState
    verification_points: Dict[int, VerificationPlan]  # step_index -> verification
    confidence: Confidence
    estimated_cost: float

    @classmethod
    def empty(cls, state: WorldState) -> Plan:
        """Create an empty plan (goal already satisfied)."""
        return cls(
            steps=[],
            expected_final_state=state,
            verification_points={},
            confidence=Confidence(1.0, ConfidenceSource.INFERENCE),
            estimated_cost=0.0
        )

    def is_empty(self) -> bool:
        """Check if this is an empty plan."""
        return len(self.steps) == 0

    def total_cost(self) -> float:
        """Calculate total cost of all steps."""
        return sum(step.cost for step in self.steps)

    def has_irreversible_steps(self) -> bool:
        """Check if plan contains irreversible actions."""
        return any(not step.reversible for step in self.steps)

    def reversible_prefix(self) -> List[Action]:
        """Get the reversible prefix of the plan."""
        prefix = []
        for step in self.steps:
            if not step.reversible:
                break
            prefix.append(step)
        return prefix

    def __repr__(self) -> str:
        steps_str = " -> ".join(s.name for s in self.steps) if self.steps else "(empty)"
        return f"Plan({steps_str}, cost={self.estimated_cost:.1f})"


@dataclass
class PlanningResult:
    """Result of a planning attempt."""

    success: bool
    plan: Optional[Plan]
    reason: str
    explored_states: int = 0
    planning_time_ms: float = 0.0

    @classmethod
    def success_result(cls, plan: Plan, explored: int = 0) -> PlanningResult:
        """Create a successful planning result."""
        return cls(
            success=True,
            plan=plan,
            reason="Plan found",
            explored_states=explored
        )

    @classmethod
    def failure(cls, reason: str, explored: int = 0) -> PlanningResult:
        """Create a failed planning result."""
        return cls(
            success=False,
            plan=None,
            reason=reason,
            explored_states=explored
        )


class Planner:
    """
    GOAP-style planner that finds action sequences to achieve goals.

    The key insight: we PLAN before acting, unlike "Ralph Wiggum"
    which just acts and hopes for the best.

    Uses A* search through action space to find optimal plans.
    """

    def __init__(
        self,
        actions: List[Action],
        max_depth: int = 50,
        max_explored: int = 10000,
        heuristic: Optional[Callable[[WorldState, Goal], float]] = None
    ):
        """
        Initialize the planner.

        Args:
            actions: Available actions
            max_depth: Maximum plan length
            max_explored: Maximum states to explore before giving up
            heuristic: Optional heuristic function for A* search
        """
        self.actions = actions
        self.max_depth = max_depth
        self.max_explored = max_explored
        self.heuristic = heuristic or self._default_heuristic

    def plan(self, current: WorldState, goal: Goal) -> PlanningResult:
        """
        Find a plan to achieve the goal from the current state.

        Uses A* search through action space.

        Args:
            current: Current world state
            goal: Goal to achieve

        Returns:
            PlanningResult with plan if found
        """
        # Already satisfied?
        if goal.is_satisfied(current):
            return PlanningResult.success_result(Plan.empty(current))

        # Check dependencies
        if not goal.can_attempt(current):
            unsatisfied = goal.unsatisfied_dependencies(current)
            return PlanningResult.failure(
                f"Dependencies not met: {[d.id for d in unsatisfied]}"
            )

        # A* search
        counter = 0
        explored = 0

        # Priority queue: (priority, counter, state, actions, cost)
        frontier: List[Tuple[float, int, WorldState, List[Action], float]] = []
        heappush(frontier, (0, 0, current, [], 0.0))

        visited: Set[int] = set()

        while frontier and explored < self.max_explored:
            _, _, state, actions, cost = heappop(frontier)
            explored += 1

            state_hash = hash(state)
            if state_hash in visited:
                continue
            visited.add(state_hash)

            if len(actions) >= self.max_depth:
                continue

            for action in self.actions:
                if not action.applicable(state):
                    continue

                new_state = action.apply(state)
                new_actions = actions + [action]
                new_cost = cost + action.cost

                if goal.is_satisfied(new_state):
                    plan = Plan(
                        steps=new_actions,
                        expected_final_state=new_state,
                        verification_points=self._create_verification_points(new_actions),
                        confidence=self._estimate_confidence(new_actions),
                        estimated_cost=new_cost
                    )
                    return PlanningResult.success_result(plan, explored)

                counter += 1
                priority = new_cost + self.heuristic(new_state, goal)
                heappush(frontier, (priority, counter, new_state, new_actions, new_cost))

        # No plan found
        if explored >= self.max_explored:
            return PlanningResult.failure(
                f"Exploration limit reached ({self.max_explored} states)",
                explored
            )
        else:
            return PlanningResult.failure("No plan exists", explored)

    def plan_with_dependencies(
        self,
        current: WorldState,
        goal: Goal
    ) -> PlanningResult:
        """
        Plan including satisfying dependencies first.

        Args:
            current: Current world state
            goal: Goal to achieve

        Returns:
            PlanningResult with combined plan
        """
        state = current
        all_steps: List[Action] = []

        # First, plan for dependencies
        for dep in goal.dependencies:
            if not dep.is_satisfied(state):
                dep_result = self.plan(state, dep)
                if not dep_result.success:
                    return PlanningResult.failure(
                        f"Cannot satisfy dependency {dep.id}: {dep_result.reason}"
                    )
                all_steps.extend(dep_result.plan.steps)
                state = dep_result.plan.expected_final_state

        # Then, plan for the main goal
        main_result = self.plan(state, goal)
        if not main_result.success:
            return main_result

        all_steps.extend(main_result.plan.steps)

        # Combine into single plan
        combined = Plan(
            steps=all_steps,
            expected_final_state=main_result.plan.expected_final_state,
            verification_points=self._create_verification_points(all_steps),
            confidence=self._estimate_confidence(all_steps),
            estimated_cost=sum(s.cost for s in all_steps)
        )

        return PlanningResult.success_result(combined)

    def find_all_plans(
        self,
        current: WorldState,
        goal: Goal,
        max_plans: int = 5
    ) -> List[Plan]:
        """
        Find multiple alternative plans.

        Useful for presenting options to users.

        Args:
            current: Current world state
            goal: Goal to achieve
            max_plans: Maximum number of plans to find

        Returns:
            List of alternative plans
        """
        plans: List[Plan] = []
        counter = 0
        explored = 0

        frontier: List[Tuple[float, int, WorldState, List[Action], float]] = []
        heappush(frontier, (0, 0, current, [], 0.0))

        # Don't use visited set - we want to find multiple paths
        while frontier and len(plans) < max_plans and explored < self.max_explored:
            _, _, state, actions, cost = heappop(frontier)
            explored += 1

            if len(actions) >= self.max_depth:
                continue

            for action in self.actions:
                if not action.applicable(state):
                    continue

                new_state = action.apply(state)
                new_actions = actions + [action]
                new_cost = cost + action.cost

                if goal.is_satisfied(new_state):
                    plan = Plan(
                        steps=new_actions,
                        expected_final_state=new_state,
                        verification_points=self._create_verification_points(new_actions),
                        confidence=self._estimate_confidence(new_actions),
                        estimated_cost=new_cost
                    )
                    # Only add if different from existing plans
                    if not self._is_duplicate_plan(plan, plans):
                        plans.append(plan)

                counter += 1
                priority = new_cost + self.heuristic(new_state, goal)
                heappush(frontier, (priority, counter, new_state, new_actions, new_cost))

        return plans

    def _default_heuristic(self, state: WorldState, goal: Goal) -> float:
        """
        Default heuristic for A* search.

        Currently just returns a constant - could be improved with
        learned heuristics from past experience.
        """
        return 1.0

    def _create_verification_points(
        self,
        actions: List[Action]
    ) -> Dict[int, VerificationPlan]:
        """
        Create verification points for a plan.

        Verification is inserted after irreversible actions.
        """
        points: Dict[int, VerificationPlan] = {}
        for i, action in enumerate(actions):
            if not action.reversible:
                # Verify after irreversible actions
                points[i] = VerificationPlan(checks=[])
        return points

    def _estimate_confidence(self, actions: List[Action]) -> Confidence:
        """
        Estimate confidence in a plan.

        Confidence compounds - each action can reduce it.
        """
        if not actions:
            return Confidence(1.0, ConfidenceSource.INFERENCE)

        conf = 0.9
        for action in actions:
            conf *= action.confidence_modifier

        return Confidence(
            conf,
            ConfidenceSource.INFERENCE,
            evidence=[f"Plan with {len(actions)} steps"]
        )

    def _is_duplicate_plan(self, plan: Plan, existing: List[Plan]) -> bool:
        """Check if plan is essentially the same as an existing one."""
        plan_actions = tuple(a.name for a in plan.steps)
        for existing_plan in existing:
            existing_actions = tuple(a.name for a in existing_plan.steps)
            if plan_actions == existing_actions:
                return True
        return False


class HierarchicalPlanner:
    """
    Planner that can decompose high-level goals into subgoals.

    This enables handling complex, multi-step objectives by
    breaking them down into manageable chunks.
    """

    def __init__(
        self,
        base_planner: Planner,
        decomposers: Optional[Dict[str, Callable[[Goal], List[Goal]]]] = None
    ):
        """
        Initialize the hierarchical planner.

        Args:
            base_planner: Planner for atomic goals
            decomposers: Goal type -> decomposition function
        """
        self.base_planner = base_planner
        self.decomposers = decomposers or {}

    def plan(self, current: WorldState, goal: Goal) -> PlanningResult:
        """
        Plan for a potentially complex goal.

        If the goal can be decomposed, does so recursively.
        """
        # Check if we have a decomposer for this goal type
        goal_type = goal.metadata.get("type", "")
        if goal_type in self.decomposers:
            subgoals = self.decomposers[goal_type](goal)
            return self._plan_for_subgoals(current, subgoals)

        # No decomposition - use base planner
        return self.base_planner.plan(current, goal)

    def _plan_for_subgoals(
        self,
        current: WorldState,
        subgoals: List[Goal]
    ) -> PlanningResult:
        """Plan for a sequence of subgoals."""
        state = current
        all_steps: List[Action] = []

        for subgoal in subgoals:
            result = self.plan(state, subgoal)
            if not result.success:
                return PlanningResult.failure(
                    f"Cannot achieve subgoal {subgoal.id}: {result.reason}"
                )
            all_steps.extend(result.plan.steps)
            state = result.plan.expected_final_state

        combined = Plan(
            steps=all_steps,
            expected_final_state=state,
            verification_points=self.base_planner._create_verification_points(all_steps),
            confidence=self.base_planner._estimate_confidence(all_steps),
            estimated_cost=sum(s.cost for s in all_steps)
        )

        return PlanningResult.success_result(combined)

    def register_decomposer(
        self,
        goal_type: str,
        decomposer: Callable[[Goal], List[Goal]]
    ) -> None:
        """
        Register a decomposition function for a goal type.

        Args:
            goal_type: Type of goal (from metadata)
            decomposer: Function that breaks goal into subgoals
        """
        self.decomposers[goal_type] = decomposer
