"""
Core types for the Deliberative Agent.

Provides foundational types for representing confidence, facts, and world state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ConfidenceSource(Enum):
    """Source of a confidence value."""

    INFERENCE = auto()      # Derived from reasoning
    OBSERVATION = auto()    # Direct measurement
    ASSUMPTION = auto()     # Assumed without verification
    VERIFICATION = auto()   # Result of explicit verification
    MEMORY = auto()         # From past experience


@dataclass
class Confidence:
    """
    A confidence value with provenance tracking.

    Unlike simple floats, this tracks:
    - Where the confidence came from
    - What evidence supports it
    - When it was established (for decay)
    """

    value: float  # 0.0 to 1.0
    source: ConfidenceSource
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Ensure value is clamped to [0, 1]."""
        self.value = max(0.0, min(1.0, self.value))

    def decay(self, half_life_hours: float = 24.0) -> Confidence:
        """
        Apply time-based confidence decay.

        Confidence should decrease over time as the world changes.

        Args:
            half_life_hours: Hours until confidence halves

        Returns:
            New Confidence with decayed value
        """
        age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600
        decay_factor = 0.5 ** (age_hours / half_life_hours)
        return Confidence(
            value=self.value * decay_factor,
            source=self.source,
            evidence=self.evidence,
            timestamp=self.timestamp
        )

    def combine(self, other: Confidence) -> Confidence:
        """
        Combine two confidence values using Bayesian-like update.

        Args:
            other: Another confidence value to combine with

        Returns:
            Combined confidence
        """
        # Simple approach: weighted average based on source reliability
        weight_self = self._source_weight()
        weight_other = other._source_weight()
        total_weight = weight_self + weight_other

        if total_weight == 0:
            return Confidence(0.5, ConfidenceSource.INFERENCE)

        combined_value = (
            self.value * weight_self + other.value * weight_other
        ) / total_weight

        return Confidence(
            value=combined_value,
            source=ConfidenceSource.INFERENCE,
            evidence=self.evidence + other.evidence,
            timestamp=datetime.now()
        )

    def _source_weight(self) -> float:
        """Get reliability weight for the confidence source."""
        weights = {
            ConfidenceSource.VERIFICATION: 1.0,
            ConfidenceSource.OBSERVATION: 0.9,
            ConfidenceSource.MEMORY: 0.7,
            ConfidenceSource.INFERENCE: 0.5,
            ConfidenceSource.ASSUMPTION: 0.3,
        }
        return weights.get(self.source, 0.5)

    def __float__(self) -> float:
        return self.value

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Confidence):
            return self.value < other.value
        return self.value < float(other)

    def __le__(self, other: Any) -> bool:
        if isinstance(other, Confidence):
            return self.value <= other.value
        return self.value <= float(other)

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, Confidence):
            return self.value > other.value
        return self.value > float(other)

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, Confidence):
            return self.value >= other.value
        return self.value >= float(other)

    def __repr__(self) -> str:
        return f"Confidence({self.value:.2f}, {self.source.name})"


@dataclass
class Fact:
    """
    A fact about the world with associated confidence.

    Facts are the atomic units of knowledge in the agent's world model.
    They can be queried and updated as the agent learns.
    """

    predicate: str
    args: Tuple[Any, ...]
    confidence: Confidence

    def __hash__(self) -> int:
        return hash((self.predicate, self.args))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Fact):
            return self.predicate == other.predicate and self.args == other.args
        return False

    def __repr__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"Fact({self.predicate}({args_str}), {self.confidence})"


class WorldState:
    """
    Represents the current state of the system being developed.

    This is the agent's model of the world, including:
    - Known facts with confidence
    - File states
    - Test results
    - Arbitrary metadata

    Unlike simple key-value stores, this tracks uncertainty.
    """

    def __init__(self):
        self.facts: Dict[str, Fact] = {}
        self.files: Dict[Path, str] = {}  # path -> content hash
        self.test_results: Dict[str, bool] = {}
        self.metadata: Dict[str, Any] = {}

    def add_fact(self, fact: Fact) -> None:
        """
        Add or update a fact in the world state.

        Args:
            fact: The fact to add
        """
        key = self._fact_key(fact.predicate, fact.args)
        self.facts[key] = fact

    def remove_fact(self, predicate: str, *args: Any) -> Optional[Fact]:
        """
        Remove a fact from the world state.

        Args:
            predicate: The predicate name
            *args: The predicate arguments

        Returns:
            The removed fact, or None if not found
        """
        key = self._fact_key(predicate, args)
        return self.facts.pop(key, None)

    def has_fact(self, predicate: str, *args: Any) -> Optional[Fact]:
        """
        Check if a fact exists in the world state.

        Args:
            predicate: The predicate name
            *args: The predicate arguments

        Returns:
            The fact if found, None otherwise
        """
        key = self._fact_key(predicate, args)
        return self.facts.get(key)

    def get_confidence(self, predicate: str, *args: Any) -> float:
        """
        Get the confidence value for a fact.

        Args:
            predicate: The predicate name
            *args: The predicate arguments

        Returns:
            The confidence value (0.0 if fact doesn't exist)
        """
        fact = self.has_fact(predicate, *args)
        return float(fact.confidence) if fact else 0.0

    def get_facts_by_predicate(self, predicate: str) -> List[Fact]:
        """
        Get all facts with a given predicate.

        Args:
            predicate: The predicate name

        Returns:
            List of matching facts
        """
        return [f for f in self.facts.values() if f.predicate == predicate]

    def set_file_hash(self, path: Path, content_hash: str) -> None:
        """Record a file's content hash."""
        self.files[path] = content_hash

    def get_file_hash(self, path: Path) -> Optional[str]:
        """Get a file's recorded content hash."""
        return self.files.get(path)

    def set_test_result(self, test_name: str, passed: bool) -> None:
        """Record a test result."""
        self.test_results[test_name] = passed

    def get_test_result(self, test_name: str) -> Optional[bool]:
        """Get a recorded test result."""
        return self.test_results.get(test_name)

    def all_tests_pass(self) -> bool:
        """Check if all recorded tests pass."""
        return bool(self.test_results) and all(self.test_results.values())

    def copy(self) -> WorldState:
        """Create a deep copy of the world state."""
        new_state = WorldState()
        new_state.facts = self.facts.copy()
        new_state.files = self.files.copy()
        new_state.test_results = self.test_results.copy()
        new_state.metadata = self.metadata.copy()
        return new_state

    def _fact_key(self, predicate: str, args: Tuple[Any, ...]) -> str:
        """Generate a unique key for a fact."""
        return f"{predicate}:{args}"

    def __hash__(self) -> int:
        """Hash based on fact keys for use in search algorithms."""
        return hash(frozenset(self.facts.keys()))

    def __repr__(self) -> str:
        return (
            f"WorldState(facts={len(self.facts)}, "
            f"files={len(self.files)}, "
            f"tests={len(self.test_results)})"
        )
