"""Tests for memory and learning system."""

import pytest
from datetime import datetime, timedelta

from deliberative_agent.core import Confidence, ConfidenceSource
from deliberative_agent.goals import Goal
from deliberative_agent.memory import Lesson, Episode, Memory
from deliberative_agent.verification import VerificationPlan


class TestLesson:
    """Tests for Lesson class."""

    def test_lesson_creation(self):
        """Lessons should be created with required fields."""
        lesson = Lesson(
            situation="Writing unit tests",
            insight="Always check edge cases",
            outcome="success",
            confidence=Confidence(0.8, ConfidenceSource.OBSERVATION)
        )

        assert lesson.situation == "Writing unit tests"
        assert lesson.outcome == "success"
        assert lesson.applications == 0

    def test_lesson_applies_to(self):
        """Should detect relevant lessons."""
        lesson = Lesson(
            situation="Writing unit tests for validation",
            insight="Check edge cases",
            outcome="success",
            confidence=Confidence(0.8, ConfidenceSource.OBSERVATION)
        )

        goal_relevant = Goal(
            id="write_tests",
            description="Write unit tests for the validation module",
            predicate=lambda s: True,
            verification=VerificationPlan([])
        )

        goal_irrelevant = Goal(
            id="deploy",
            description="Deploy to production",
            predicate=lambda s: True,
            verification=VerificationPlan([])
        )

        assert lesson.applies_to(goal_relevant)
        assert not lesson.applies_to(goal_irrelevant)

    def test_lesson_reinforce(self):
        """Reinforcement should increase confidence and applications."""
        lesson = Lesson(
            situation="Test",
            insight="Insight",
            outcome="success",
            confidence=Confidence(0.5, ConfidenceSource.MEMORY)
        )

        initial_conf = float(lesson.confidence)
        initial_apps = lesson.applications

        lesson.reinforce()

        assert float(lesson.confidence) > initial_conf
        assert lesson.applications == initial_apps + 1

    def test_lesson_weaken(self):
        """Weakening should decrease confidence."""
        lesson = Lesson(
            situation="Test",
            insight="Insight",
            outcome="success",
            confidence=Confidence(0.8, ConfidenceSource.MEMORY)
        )

        initial_conf = float(lesson.confidence)
        lesson.weaken()

        assert float(lesson.confidence) < initial_conf

    def test_lesson_staleness(self):
        """Old lessons should be marked stale."""
        old_lesson = Lesson(
            situation="Test",
            insight="Insight",
            outcome="success",
            confidence=Confidence(0.8, ConfidenceSource.MEMORY),
            timestamp=datetime.now() - timedelta(hours=200)
        )

        new_lesson = Lesson(
            situation="Test",
            insight="Insight",
            outcome="success",
            confidence=Confidence(0.8, ConfidenceSource.MEMORY)
        )

        assert old_lesson.is_stale(max_age_hours=168)
        assert not new_lesson.is_stale(max_age_hours=168)


class TestMemory:
    """Tests for Memory class."""

    def test_memory_creation(self):
        """Memory should initialize empty."""
        memory = Memory()

        assert len(memory.lessons) == 0
        assert len(memory.episodes) == 0

    def test_add_lesson(self):
        """Should add lessons."""
        memory = Memory()
        lesson = Lesson(
            situation="Test",
            insight="Insight",
            outcome="success",
            confidence=Confidence(0.8, ConfidenceSource.MEMORY)
        )

        memory.add_lesson(lesson)

        assert len(memory.lessons) == 1

    def test_lesson_reinforcement(self):
        """Adding similar lesson should reinforce existing."""
        memory = Memory()
        lesson1 = Lesson(
            situation="Test situation",
            insight="Insight",
            outcome="success",
            confidence=Confidence(0.5, ConfidenceSource.MEMORY)
        )
        lesson2 = Lesson(
            situation="Test situation",
            insight="Insight",
            outcome="success",
            confidence=Confidence(0.6, ConfidenceSource.MEMORY)
        )

        memory.add_lesson(lesson1)
        initial_conf = float(memory.lessons[0].confidence)

        memory.add_lesson(lesson2)

        # Should still have one lesson (reinforced)
        assert len(memory.lessons) == 1
        # Confidence should have increased
        assert float(memory.lessons[0].confidence) > initial_conf

    def test_lesson_contradiction(self):
        """Higher confidence lesson should win on contradiction."""
        memory = Memory()
        lesson_low = Lesson(
            situation="Test situation",
            insight="Old insight",
            outcome="failure",
            confidence=Confidence(0.4, ConfidenceSource.MEMORY)
        )
        lesson_high = Lesson(
            situation="Test situation",
            insight="New insight",
            outcome="success",
            confidence=Confidence(0.8, ConfidenceSource.MEMORY)
        )

        memory.add_lesson(lesson_low)
        memory.add_lesson(lesson_high)

        # Higher confidence lesson should win
        assert len(memory.lessons) == 1
        assert memory.lessons[0].outcome == "success"

    def test_retrieve_relevant(self):
        """Should retrieve relevant lessons."""
        memory = Memory()
        memory.add_lesson(Lesson(
            situation="Writing tests",
            insight="Check edge cases",
            outcome="success",
            confidence=Confidence(0.8, ConfidenceSource.MEMORY)
        ))
        memory.add_lesson(Lesson(
            situation="Deploying to production",
            insight="Check rollback",
            outcome="success",
            confidence=Confidence(0.7, ConfidenceSource.MEMORY)
        ))

        goal = Goal(
            id="write_tests",
            description="Writing tests for the module",
            predicate=lambda s: True,
            verification=VerificationPlan([])
        )

        relevant = memory.retrieve_relevant(goal)

        assert len(relevant) == 1
        assert "edge cases" in relevant[0].insight

    def test_retrieve_by_outcome(self):
        """Should filter by outcome."""
        memory = Memory()
        memory.add_lesson(Lesson(
            situation="Test 1",
            insight="Insight 1",
            outcome="success",
            confidence=Confidence(0.8, ConfidenceSource.MEMORY)
        ))
        memory.add_lesson(Lesson(
            situation="Test 2",
            insight="Insight 2",
            outcome="failure",
            confidence=Confidence(0.7, ConfidenceSource.MEMORY)
        ))

        successes = memory.retrieve_by_outcome("success")
        failures = memory.retrieve_by_outcome("failure")

        assert len(successes) == 1
        assert len(failures) == 1

    def test_success_rate(self):
        """Should calculate success rate."""
        memory = Memory()
        memory.add_episode(Episode(
            goal_id="g1",
            goal_description="Goal 1",
            plan_steps=["step1"],
            result_status="success",
            lessons_extracted=[]
        ))
        memory.add_episode(Episode(
            goal_id="g2",
            goal_description="Goal 2",
            plan_steps=["step1"],
            result_status="failure",
            lessons_extracted=[]
        ))

        rate = memory.get_success_rate()
        assert rate == 0.5

    def test_memory_pruning(self):
        """Should prune excess lessons."""
        memory = Memory(max_lessons=5)

        for i in range(10):
            memory.add_lesson(Lesson(
                situation=f"Situation {i}",
                insight=f"Insight {i}",
                outcome="success",
                confidence=Confidence(i / 10, ConfidenceSource.MEMORY)
            ))

        # Should have pruned to max
        assert len(memory.lessons) <= 5
        # Should keep highest confidence
        confidences = [float(l.confidence) for l in memory.lessons]
        assert min(confidences) >= 0.5  # Should have kept top 5

    def test_export_import(self):
        """Should export and import correctly."""
        memory = Memory()
        memory.add_lesson(Lesson(
            situation="Test",
            insight="Insight",
            outcome="success",
            confidence=Confidence(0.8, ConfidenceSource.MEMORY),
            tags=["tag1", "tag2"]
        ))

        exported = memory.export()
        imported = Memory.from_export(exported)

        assert len(imported.lessons) == 1
        assert imported.lessons[0].situation == "Test"
        assert imported.lessons[0].tags == ["tag1", "tag2"]
