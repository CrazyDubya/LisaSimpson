"""Tests for core types."""

import pytest
from datetime import datetime, timedelta

from deliberative_agent.core import (
    Confidence,
    ConfidenceSource,
    Fact,
    WorldState,
)


class TestConfidence:
    """Tests for Confidence class."""

    def test_confidence_clamping(self):
        """Confidence values should be clamped to [0, 1]."""
        c1 = Confidence(1.5, ConfidenceSource.INFERENCE)
        assert c1.value == 1.0

        c2 = Confidence(-0.5, ConfidenceSource.INFERENCE)
        assert c2.value == 0.0

    def test_confidence_float_conversion(self):
        """Confidence should convert to float."""
        c = Confidence(0.7, ConfidenceSource.OBSERVATION)
        assert float(c) == 0.7

    def test_confidence_comparison(self):
        """Confidence should compare correctly."""
        c1 = Confidence(0.5, ConfidenceSource.INFERENCE)
        c2 = Confidence(0.7, ConfidenceSource.INFERENCE)

        assert c1 < c2
        assert c2 > c1
        assert c1 <= c1
        assert c1 >= c1

    def test_confidence_comparison_with_float(self):
        """Confidence should compare with floats."""
        c = Confidence(0.5, ConfidenceSource.INFERENCE)

        assert c < 0.6
        assert c > 0.4
        assert c <= 0.5
        assert c >= 0.5

    def test_confidence_decay(self):
        """Confidence should decay over time."""
        old_timestamp = datetime.now() - timedelta(hours=24)
        c = Confidence(1.0, ConfidenceSource.OBSERVATION, timestamp=old_timestamp)

        decayed = c.decay(half_life_hours=24.0)
        assert decayed.value == pytest.approx(0.5, rel=0.1)

    def test_confidence_combine(self):
        """Confidence should combine with Bayesian update."""
        c1 = Confidence(0.8, ConfidenceSource.VERIFICATION)
        c2 = Confidence(0.6, ConfidenceSource.INFERENCE)

        combined = c1.combine(c2)
        # VERIFICATION has higher weight, so result should be closer to 0.8
        assert combined.value > 0.65


class TestFact:
    """Tests for Fact class."""

    def test_fact_creation(self):
        """Facts should be created with predicate and args."""
        f = Fact(
            predicate="file_exists",
            args=("myfile.py",),
            confidence=Confidence(0.9, ConfidenceSource.OBSERVATION)
        )

        assert f.predicate == "file_exists"
        assert f.args == ("myfile.py",)

    def test_fact_equality(self):
        """Facts with same predicate and args should be equal."""
        c = Confidence(0.9, ConfidenceSource.OBSERVATION)
        f1 = Fact("file_exists", ("myfile.py",), c)
        f2 = Fact("file_exists", ("myfile.py",), c)
        f3 = Fact("file_exists", ("other.py",), c)

        assert f1 == f2
        assert f1 != f3

    def test_fact_hash(self):
        """Facts should be hashable for use in sets/dicts."""
        c = Confidence(0.9, ConfidenceSource.OBSERVATION)
        f1 = Fact("file_exists", ("myfile.py",), c)
        f2 = Fact("file_exists", ("myfile.py",), c)

        assert hash(f1) == hash(f2)

        facts = {f1}
        assert f2 in facts


class TestWorldState:
    """Tests for WorldState class."""

    def test_world_state_creation(self):
        """WorldState should initialize empty."""
        state = WorldState()

        assert len(state.facts) == 0
        assert len(state.files) == 0
        assert len(state.test_results) == 0

    def test_add_and_retrieve_fact(self):
        """Facts should be addable and retrievable."""
        state = WorldState()
        fact = Fact(
            "file_exists",
            ("myfile.py",),
            Confidence(0.9, ConfidenceSource.OBSERVATION)
        )

        state.add_fact(fact)

        retrieved = state.has_fact("file_exists", "myfile.py")
        assert retrieved is not None
        assert retrieved.predicate == "file_exists"

    def test_remove_fact(self):
        """Facts should be removable."""
        state = WorldState()
        fact = Fact(
            "file_exists",
            ("myfile.py",),
            Confidence(0.9, ConfidenceSource.OBSERVATION)
        )

        state.add_fact(fact)
        removed = state.remove_fact("file_exists", "myfile.py")

        assert removed is not None
        assert state.has_fact("file_exists", "myfile.py") is None

    def test_get_confidence(self):
        """Should get confidence for a fact."""
        state = WorldState()
        fact = Fact(
            "file_exists",
            ("myfile.py",),
            Confidence(0.9, ConfidenceSource.OBSERVATION)
        )
        state.add_fact(fact)

        assert state.get_confidence("file_exists", "myfile.py") == 0.9
        assert state.get_confidence("nonexistent", "arg") == 0.0

    def test_world_state_copy(self):
        """Copy should create independent state."""
        state = WorldState()
        fact = Fact(
            "file_exists",
            ("myfile.py",),
            Confidence(0.9, ConfidenceSource.OBSERVATION)
        )
        state.add_fact(fact)

        copy = state.copy()
        copy.remove_fact("file_exists", "myfile.py")

        # Original should be unchanged
        assert state.has_fact("file_exists", "myfile.py") is not None
        assert copy.has_fact("file_exists", "myfile.py") is None

    def test_test_results(self):
        """Test results should be trackable."""
        state = WorldState()

        state.set_test_result("test_foo", True)
        state.set_test_result("test_bar", False)

        assert state.get_test_result("test_foo") is True
        assert state.get_test_result("test_bar") is False
        assert state.all_tests_pass() is False

        state.set_test_result("test_bar", True)
        assert state.all_tests_pass() is True
