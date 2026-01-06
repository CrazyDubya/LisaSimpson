"""
Execution system for the Deliberative Agent.

Executes plans with verification at key points and rollback capability.
This is fundamentally different from blind iteration - we verify
SEMANTICALLY at each step, not just check for magic strings.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, TYPE_CHECKING

from .core import WorldState
from .actions import Action
from .planning import Plan
from .verification import VerificationResult

if TYPE_CHECKING:
    pass


@dataclass
class ExecutionStepResult:
    """Result of executing a single action."""

    success: bool
    new_state: WorldState
    actual_effects: List[str]
    unexpected_effects: List[str] = field(default_factory=list)
    error: Optional[Exception] = None
    duration_ms: float = 0.0

    @classmethod
    def successful(
        cls,
        new_state: WorldState,
        effects: List[str]
    ) -> ExecutionStepResult:
        """Create a successful result."""
        return cls(
            success=True,
            new_state=new_state,
            actual_effects=effects
        )

    @classmethod
    def failed(
        cls,
        state: WorldState,
        error: Exception
    ) -> ExecutionStepResult:
        """Create a failed result."""
        return cls(
            success=False,
            new_state=state,
            actual_effects=[],
            error=error
        )


class ActionExecutor(Protocol):
    """
    Protocol for actually executing actions.

    Implementations might:
    - Call LLMs
    - Run shell commands
    - Modify files
    - Make API calls
    """

    @abstractmethod
    async def execute(
        self,
        action: Action,
        state: WorldState
    ) -> ExecutionStepResult:
        """
        Execute an action in the given state.

        Args:
            action: Action to execute
            state: Current world state

        Returns:
            ExecutionStepResult with outcome
        """
        ...


@dataclass
class ExecutionResult:
    """Result of executing a complete plan."""

    status: str  # 'success', 'failure', 'partial', 'verification_failure'
    completed_steps: List[Action]
    final_state: WorldState
    verification: Optional[VerificationResult] = None
    error: Optional[Exception] = None
    rollback_plan: Optional[Plan] = None
    failure_diagnosis: Optional[str] = None
    step_results: List[ExecutionStepResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == "success"

    @property
    def partial_success(self) -> bool:
        """Check if at least some steps completed."""
        return len(self.completed_steps) > 0

    def summary(self) -> str:
        """Generate a human-readable summary."""
        return (
            f"Execution {self.status}: "
            f"{len(self.completed_steps)} steps completed"
            + (f", error: {self.error}" if self.error else "")
        )


class Executor:
    """
    Executes plans with verification and rollback capability.

    Key difference from "Ralph Wiggum":
    - We verify SEMANTICALLY, not by string matching
    - We understand failure causes
    - We can roll back changes
    """

    def __init__(self, action_executor: ActionExecutor):
        """
        Initialize the executor.

        Args:
            action_executor: Implementation that actually executes actions
        """
        self.action_executor = action_executor

    async def execute(
        self,
        plan: Plan,
        initial_state: WorldState
    ) -> ExecutionResult:
        """
        Execute a plan with verification at key points.

        Args:
            plan: Plan to execute
            initial_state: Starting world state

        Returns:
            ExecutionResult with detailed outcomes
        """
        current_state = initial_state
        completed_steps: List[Action] = []
        step_results: List[ExecutionStepResult] = []

        for i, action in enumerate(plan.steps):
            # Pre-check: are preconditions still met?
            if not action.applicable(current_state):
                return ExecutionResult(
                    status="precondition_failure",
                    completed_steps=completed_steps,
                    final_state=current_state,
                    failure_diagnosis=self._diagnose_precondition_failure(
                        action, current_state
                    ),
                    rollback_plan=self._create_rollback_plan(completed_steps),
                    step_results=step_results
                )

            # Execute the action
            try:
                step_result = await self.action_executor.execute(action, current_state)
                step_results.append(step_result)

                if not step_result.success:
                    return ExecutionResult(
                        status="execution_failure",
                        completed_steps=completed_steps,
                        final_state=current_state,
                        error=step_result.error,
                        failure_diagnosis=self._diagnose_execution_failure(
                            action, step_result
                        ),
                        rollback_plan=self._create_rollback_plan(completed_steps),
                        step_results=step_results
                    )

                # Check for unexpected effects
                if step_result.unexpected_effects:
                    # Log but continue - unexpected doesn't mean wrong
                    pass

                current_state = step_result.new_state
                completed_steps.append(action)

            except Exception as e:
                step_results.append(ExecutionStepResult.failed(current_state, e))
                return ExecutionResult(
                    status="execution_error",
                    completed_steps=completed_steps,
                    final_state=current_state,
                    error=e,
                    rollback_plan=self._create_rollback_plan(completed_steps),
                    step_results=step_results
                )

            # Verification point?
            if i in plan.verification_points:
                verification = await plan.verification_points[i].verify(current_state)
                if not verification.satisfied:
                    return ExecutionResult(
                        status="verification_failure",
                        completed_steps=completed_steps,
                        final_state=current_state,
                        verification=verification,
                        failure_diagnosis=self._diagnose_verification_failure(
                            verification
                        ),
                        rollback_plan=self._create_rollback_plan(completed_steps),
                        step_results=step_results
                    )

        return ExecutionResult(
            status="success",
            completed_steps=completed_steps,
            final_state=current_state,
            step_results=step_results
        )

    async def execute_with_rollback(
        self,
        plan: Plan,
        initial_state: WorldState
    ) -> ExecutionResult:
        """
        Execute a plan, rolling back on failure.

        Args:
            plan: Plan to execute
            initial_state: Starting world state

        Returns:
            ExecutionResult (state is rolled back on failure)
        """
        result = await self.execute(plan, initial_state)

        if not result.success and result.rollback_plan:
            # Attempt rollback
            rollback_result = await self.execute(
                result.rollback_plan, result.final_state
            )
            if rollback_result.success:
                result.final_state = rollback_result.final_state
                result.failure_diagnosis = (
                    (result.failure_diagnosis or "") +
                    " (rollback successful)"
                )

        return result

    def _diagnose_precondition_failure(
        self,
        action: Action,
        state: WorldState
    ) -> str:
        """
        Understand WHY preconditions aren't met.

        This is key to intelligent behavior - we don't just fail,
        we understand the failure.
        """
        failed = action.failed_preconditions(state)
        failed_count = len(failed)

        if failed_count == 0:
            return f"Action '{action.name}' preconditions mysteriously unsatisfied"

        # Try to get meaningful descriptions
        descriptions = []
        for i, p in enumerate(failed):
            # Try to extract docstring or name
            doc = getattr(p, "__doc__", None) or getattr(p, "__name__", None)
            if doc:
                descriptions.append(doc)
            else:
                descriptions.append(f"Precondition {i}")

        return (
            f"Action '{action.name}' blocked: "
            f"{', '.join(descriptions)}"
        )

    def _diagnose_execution_failure(
        self,
        action: Action,
        result: ExecutionStepResult
    ) -> str:
        """Understand WHY execution failed."""
        if result.error:
            return f"Action '{action.name}' failed: {result.error}"
        if result.unexpected_effects:
            return (
                f"Action '{action.name}' had unexpected effects: "
                f"{result.unexpected_effects}"
            )
        return f"Action '{action.name}' failed for unknown reason"

    def _diagnose_verification_failure(
        self,
        verification: VerificationResult
    ) -> str:
        """Understand WHY verification failed."""
        messages = [f.message for f in verification.failures]
        return f"Verification failed: {'; '.join(messages)}"

    def _create_rollback_plan(
        self,
        steps: List[Action]
    ) -> Optional[Plan]:
        """
        Create a plan to undo completed steps.

        Only possible if all completed steps are reversible.
        """
        reverse_actions = []
        for step in reversed(steps):
            if step.reversible and step.reverse_action:
                reverse_actions.append(step.reverse_action)
            elif not step.reversible:
                # Can't fully rollback
                return None

        if not reverse_actions:
            return None

        return Plan(
            steps=reverse_actions,
            expected_final_state=WorldState(),  # Approximate
            verification_points={},
            confidence=self._rollback_confidence(reverse_actions),
            estimated_cost=sum(a.cost for a in reverse_actions)
        )

    def _rollback_confidence(self, actions: List[Action]) -> "Confidence":
        """Estimate confidence in rollback plan."""
        from .core import Confidence, ConfidenceSource

        return Confidence(
            0.7,
            ConfidenceSource.INFERENCE,
            evidence=["Rollback plan"]
        )


class DryRunExecutor:
    """
    Executor that simulates execution without side effects.

    Useful for:
    - Testing plans
    - Estimating outcomes
    - Building confidence before real execution
    """

    async def execute(
        self,
        plan: Plan,
        initial_state: WorldState
    ) -> ExecutionResult:
        """
        Simulate plan execution.

        Just applies effects without actually executing actions.
        """
        current_state = initial_state
        completed_steps: List[Action] = []

        for action in plan.steps:
            if not action.applicable(current_state):
                return ExecutionResult(
                    status="precondition_failure",
                    completed_steps=completed_steps,
                    final_state=current_state,
                    failure_diagnosis=(
                        f"Preconditions not met for {action.name}"
                    )
                )

            # Apply effects (simulation only)
            current_state = action.apply(current_state)
            completed_steps.append(action)

        return ExecutionResult(
            status="success",
            completed_steps=completed_steps,
            final_state=current_state
        )


class TracingExecutor:
    """
    Executor that wraps another executor with tracing.

    Records all execution for debugging and learning.
    """

    def __init__(self, inner: ActionExecutor):
        self.inner = inner
        self.trace: List[dict] = []

    async def execute(
        self,
        action: Action,
        state: WorldState
    ) -> ExecutionStepResult:
        """Execute with tracing."""
        import time

        start = time.time()
        result = await self.inner.execute(action, state)
        duration = time.time() - start

        self.trace.append({
            "action": action.name,
            "success": result.success,
            "duration_ms": duration * 1000,
            "effects": result.actual_effects,
            "unexpected": result.unexpected_effects,
            "error": str(result.error) if result.error else None
        })

        return result

    def get_trace(self) -> List[dict]:
        """Get the execution trace."""
        return self.trace.copy()

    def clear_trace(self) -> None:
        """Clear the trace."""
        self.trace.clear()
