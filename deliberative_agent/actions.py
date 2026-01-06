"""
Action system for the Deliberative Agent.

Actions represent atomic operations the agent can perform, with:
- Preconditions that must hold
- Effects on the world state
- Cost estimation
- Reversibility information
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, TYPE_CHECKING

from .core import WorldState

if TYPE_CHECKING:
    pass


@dataclass
class Effect:
    """
    An effect an action has on the world state.

    Effects are the building blocks for planning - they describe
    how the world changes when an action is executed.
    """

    description: str
    apply: Callable[[WorldState], WorldState]
    undoable: bool = True

    def __repr__(self) -> str:
        return f"Effect({self.description!r})"


@dataclass
class Action:
    """
    An action the agent can take.

    Actions are the atomic units of agent behavior. Unlike blind
    LLM invocations, actions have:
    - Explicit preconditions
    - Explicit effects
    - Cost estimates
    - Reversibility information

    This enables proper planning and rollback.
    """

    name: str
    description: str
    preconditions: List[Callable[[WorldState], bool]]
    effects: List[Effect]
    cost: float  # Time/token/risk cost
    reversible: bool
    reverse_action: Optional[Action] = None
    confidence_modifier: float = 1.0  # Multiplier for confidence in this action
    metadata: dict = field(default_factory=dict)

    def applicable(self, state: WorldState) -> bool:
        """
        Check if this action can be taken in the given state.

        Args:
            state: Current world state

        Returns:
            True if all preconditions are satisfied
        """
        try:
            return all(p(state) for p in self.preconditions)
        except Exception:
            return False

    def failed_preconditions(
        self,
        state: WorldState
    ) -> List[Callable[[WorldState], bool]]:
        """
        Get list of preconditions that aren't satisfied.

        Args:
            state: Current world state

        Returns:
            List of failed precondition functions
        """
        failed = []
        for p in self.preconditions:
            try:
                if not p(state):
                    failed.append(p)
            except Exception:
                failed.append(p)
        return failed

    def apply(self, state: WorldState) -> WorldState:
        """
        Apply effects to state (for planning - doesn't actually execute).

        Args:
            state: Current world state

        Returns:
            New world state with effects applied
        """
        new_state = state.copy()
        for effect in self.effects:
            new_state = effect.apply(new_state)
        return new_state

    def can_undo(self) -> bool:
        """Check if this action can be undone."""
        return self.reversible and self.reverse_action is not None

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Action):
            return self.name == other.name
        return False

    def __repr__(self) -> str:
        rev = ", reversible" if self.reversible else ""
        return f"Action({self.name!r}, cost={self.cost}{rev})"


class ActionBuilder:
    """
    Fluent builder for creating actions.

    Makes it easier to construct actions with proper preconditions and effects.
    """

    def __init__(self, name: str, description: str):
        """
        Start building an action.

        Args:
            name: Unique identifier for the action
            description: Human-readable description
        """
        self._name = name
        self._description = description
        self._preconditions: List[Callable[[WorldState], bool]] = []
        self._effects: List[Effect] = []
        self._cost: float = 1.0
        self._reversible: bool = False
        self._reverse_action: Optional[Action] = None
        self._confidence_modifier: float = 1.0
        self._metadata: dict = {}

    def with_precondition(
        self,
        predicate: Callable[[WorldState], bool]
    ) -> ActionBuilder:
        """Add a precondition."""
        self._preconditions.append(predicate)
        return self

    def with_preconditions(
        self,
        predicates: List[Callable[[WorldState], bool]]
    ) -> ActionBuilder:
        """Add multiple preconditions."""
        self._preconditions.extend(predicates)
        return self

    def requires_fact(self, predicate: str, *args) -> ActionBuilder:
        """Add a precondition that a fact must exist."""
        self._preconditions.append(
            lambda s, p=predicate, a=args: s.has_fact(p, *a) is not None
        )
        return self

    def with_effect(
        self,
        description: str,
        apply: Callable[[WorldState], WorldState]
    ) -> ActionBuilder:
        """Add an effect."""
        self._effects.append(Effect(description, apply))
        return self

    def adds_fact(
        self,
        predicate: str,
        *args,
        confidence: float = 0.9
    ) -> ActionBuilder:
        """Add an effect that adds a fact."""
        from .core import Confidence, ConfidenceSource, Fact

        def add_fact(state: WorldState) -> WorldState:
            state.add_fact(Fact(
                predicate=predicate,
                args=args,
                confidence=Confidence(confidence, ConfidenceSource.INFERENCE)
            ))
            return state

        self._effects.append(Effect(f"adds {predicate}{args}", add_fact))
        return self

    def removes_fact(self, predicate: str, *args) -> ActionBuilder:
        """Add an effect that removes a fact."""
        def remove_fact(state: WorldState) -> WorldState:
            state.remove_fact(predicate, *args)
            return state

        self._effects.append(Effect(f"removes {predicate}{args}", remove_fact))
        return self

    def with_cost(self, cost: float) -> ActionBuilder:
        """Set the cost."""
        self._cost = cost
        return self

    def with_reversibility(
        self,
        reversible: bool,
        reverse_action: Optional[Action] = None
    ) -> ActionBuilder:
        """Set reversibility."""
        self._reversible = reversible
        self._reverse_action = reverse_action
        return self

    def with_confidence_modifier(self, modifier: float) -> ActionBuilder:
        """Set confidence modifier."""
        self._confidence_modifier = modifier
        return self

    def with_metadata(self, key: str, value: object) -> ActionBuilder:
        """Add metadata."""
        self._metadata[key] = value
        return self

    def build(self) -> Action:
        """Build the action."""
        return Action(
            name=self._name,
            description=self._description,
            preconditions=self._preconditions,
            effects=self._effects,
            cost=self._cost,
            reversible=self._reversible,
            reverse_action=self._reverse_action,
            confidence_modifier=self._confidence_modifier,
            metadata=self._metadata
        )


def action(name: str, description: str) -> ActionBuilder:
    """
    Convenience function to start building an action.

    Example:
        create_file = (
            action("create_file", "Create a new file")
            .with_precondition(lambda s: s.has_fact("directory_exists"))
            .adds_fact("file_exists", "myfile.py")
            .with_cost(1.0)
            .with_reversibility(True)
            .build()
        )

    Args:
        name: Unique identifier
        description: Human-readable description

    Returns:
        ActionBuilder for fluent construction
    """
    return ActionBuilder(name, description)


# Common action templates


def create_file_action(
    filename: str,
    directory_predicate: str = "directory_exists"
) -> Action:
    """
    Create a standard file creation action.

    Args:
        filename: Name of the file to create
        directory_predicate: Predicate indicating directory exists

    Returns:
        Action for creating the file
    """
    return (
        action(f"create_{filename}", f"Create {filename}")
        .requires_fact(directory_predicate)
        .adds_fact("file_exists", filename)
        .with_cost(1.0)
        .with_reversibility(True)
        .build()
    )


def run_command_action(
    command: str,
    requires_files: Optional[List[str]] = None,
    sets_facts: Optional[List[str]] = None
) -> Action:
    """
    Create an action for running a shell command.

    Args:
        command: Command to run
        requires_files: Files that must exist
        sets_facts: Facts that will be set after running

    Returns:
        Action for running the command
    """
    builder = action(f"run_{command}", f"Run {command}").with_cost(2.0)

    if requires_files:
        for f in requires_files:
            builder.requires_fact("file_exists", f)

    if sets_facts:
        for fact in sets_facts:
            builder.adds_fact(fact)

    return builder.with_reversibility(False).build()
