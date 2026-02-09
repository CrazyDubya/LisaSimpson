"""
Memory and learning system for the Deliberative Agent.

Provides structured storage of experiences and lessons learned,
enabling the agent to improve over time - unlike stateless
"Ralph Wiggum" style approaches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional, TYPE_CHECKING

from .core import Confidence, ConfidenceSource

if TYPE_CHECKING:
    from .goals import Goal
    from .planning import Plan
    from .execution import ExecutionResult


@dataclass
class Lesson:
    """
    Something learned from experience.

    Lessons are the distilled wisdom from past executions.
    They help the agent make better decisions over time.
    """

    situation: str  # Description of when this applies
    insight: str    # What we learned
    outcome: str    # 'success', 'failure', 'partial'
    confidence: Confidence
    applications: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

    def applies_to(self, goal: "Goal") -> bool:
        """
        Check if this lesson applies to a goal.

        Uses simple keyword matching - could be enhanced with
        semantic similarity.

        Args:
            goal: Goal to check against

        Returns:
            True if lesson might be relevant
        """
        # Simple keyword matching
        situation_lower = self.situation.lower()
        description_lower = goal.description.lower()

        # Check for keyword overlap
        situation_words = set(situation_lower.split())
        description_words = set(description_lower.split())

        common_words = situation_words & description_words
        # Filter out common stop words
        stop_words = {"the", "a", "an", "is", "are", "to", "for", "of", "and", "or"}
        meaningful_common = common_words - stop_words

        return len(meaningful_common) >= 2 or situation_lower in description_lower

    def reinforce(self) -> None:
        """Record a successful application of this lesson."""
        self.applications += 1
        # Increase confidence slightly
        new_conf = min(1.0, float(self.confidence) + 0.05)
        self.confidence = Confidence(
            new_conf,
            ConfidenceSource.MEMORY,
            evidence=self.confidence.evidence + [
                f"Reinforced (applications: {self.applications})"
            ]
        )

    def weaken(self) -> None:
        """Record a failed application of this lesson."""
        # Decrease confidence
        new_conf = max(0.0, float(self.confidence) - 0.1)
        self.confidence = Confidence(
            new_conf,
            ConfidenceSource.MEMORY,
            evidence=self.confidence.evidence + ["Weakened after failed application"]
        )

    def is_stale(self, max_age_hours: float = 168.0) -> bool:
        """Check if lesson is too old to be reliable."""
        age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600
        return age_hours > max_age_hours

    def __repr__(self) -> str:
        return (
            f"Lesson({self.situation!r}, "
            f"outcome={self.outcome}, "
            f"confidence={float(self.confidence):.2f})"
        )


@dataclass
class Episode:
    """
    A record of a complete execution episode.

    Episodes provide detailed history of what happened,
    enabling post-hoc analysis and learning.
    """

    goal_id: str
    goal_description: str
    plan_steps: List[str]  # Action names
    result_status: str
    lessons_extracted: List[Lesson]
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_execution(
        cls,
        goal: "Goal",
        plan: "Plan",
        result: "ExecutionResult",
        lessons: List[Lesson]
    ) -> Episode:
        """Create an episode from execution artifacts."""
        return cls(
            goal_id=goal.id,
            goal_description=goal.description,
            plan_steps=[a.name for a in plan.steps],
            result_status=result.status,
            lessons_extracted=lessons,
            metadata={
                "completed_steps": len(result.completed_steps),
                "total_steps": len(plan.steps),
                "had_error": result.error is not None
            }
        )

    def __repr__(self) -> str:
        return (
            f"Episode({self.goal_id}, "
            f"status={self.result_status}, "
            f"lessons={len(self.lessons_extracted)})"
        )


@dataclass
class ChannelPolicy:
    """
    Retrieval policy for a memory channel.

    Strategies:
    - relevance: semantic matching first (task-centric)
    - recency: newest lessons first (short-term working memory)
    - persistence: stable/high-confidence lessons first (project/user memory)
    """

    strategy: Literal["relevance", "recency", "persistence"] = "relevance"
    limit: Optional[int] = None


class Memory:
    """
    Structured memory for the agent.

    Stores lessons and episodes, enabling:
    - Learning from experience
    - Pattern recognition
    - Confidence estimation
    - Avoiding past mistakes

    Unlike "Ralph Wiggum" which has no memory between runs,
    this system accumulates knowledge over time.
    """

    def __init__(
        self,
        max_lessons: int = 1000,
        max_episodes: int = 500,
        channel_policies: Optional[Dict[str, ChannelPolicy]] = None,
    ):
        """
        Initialize memory.

        Args:
            max_lessons: Maximum lessons to retain
            max_episodes: Maximum episodes to retain
        """
        self.lessons: List[Lesson] = []
        self.episodes: List[Episode] = []
        self.max_lessons = max_lessons
        self.max_episodes = max_episodes
        self.channel_policies = self._default_channel_policies()
        if channel_policies:
            self.channel_policies.update(channel_policies)

    def add_lesson(self, lesson: Lesson, channel: Optional[str] = None) -> None:
        """
        Add a lesson, handling contradictions and reinforcement.

        Args:
            lesson: Lesson to add
            channel: Optional channel name (task/recent/project/etc.)
        """
        if channel:
            tag = self._channel_tag(channel)
            if tag not in lesson.tags:
                lesson.tags.append(tag)

        # Check for contradictions
        contradicting = [
            l for l in self.lessons
            if l.situation == lesson.situation and l.outcome != lesson.outcome
        ]

        if contradicting:
            # Resolve contradiction - higher confidence wins
            for old in contradicting:
                if float(lesson.confidence) > float(old.confidence):
                    self.lessons.remove(old)
                    self.lessons.append(lesson)
                # Otherwise keep the old one
        else:
            # Check for reinforcement
            supporting = [
                l for l in self.lessons
                if l.situation == lesson.situation and l.outcome == lesson.outcome
            ]

            if supporting:
                # Reinforce existing lesson
                for old in supporting:
                    old.reinforce()
            else:
                # Novel lesson - add it
                self.lessons.append(lesson)

        # Enforce limits
        self._prune_lessons()

    def add_episode(self, episode: Episode, channel: Optional[str] = None) -> None:
        """
        Add an episode to history.

        Args:
            episode: Episode to add
            channel: Optional channel name
        """
        if channel:
            episode.metadata["channel"] = channel
        self.episodes.append(episode)
        self._prune_episodes()

    def retrieve_relevant(
        self,
        goal: "Goal",
        channel: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Lesson]:
        """
        Find lessons relevant to a goal.

        Args:
            goal: Goal to find lessons for
            channel: Optional channel name
            limit: Optional max lessons to return

        Returns:
            List of relevant lessons, sorted by confidence
        """
        if channel:
            return self.retrieve_by_channel(channel, goal=goal, limit=limit)

        relevant = [l for l in self.lessons if l.applies_to(goal)]
        # Sort by confidence, highest first
        relevant.sort(key=lambda l: float(l.confidence), reverse=True)
        if limit is not None:
            return relevant[:limit]
        return relevant

    def retrieve_by_outcome(self, outcome: str) -> List[Lesson]:
        """
        Get lessons by outcome type.

        Args:
            outcome: 'success', 'failure', or 'partial'

        Returns:
            List of lessons with that outcome
        """
        return [l for l in self.lessons if l.outcome == outcome]

    def retrieve_by_tag(self, tag: str) -> List[Lesson]:
        """
        Get lessons by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of lessons with that tag
        """
        return [l for l in self.lessons if tag in l.tags]

    def set_channel_policy(
        self,
        channel: str,
        strategy: Literal["relevance", "recency", "persistence"],
        limit: Optional[int] = None,
    ) -> None:
        """Set retrieval behavior for a memory channel."""
        if strategy not in {"relevance", "recency", "persistence"}:
            raise ValueError("strategy must be one of: relevance, recency, persistence")
        self.channel_policies[channel] = ChannelPolicy(strategy=strategy, limit=limit)

    def retrieve_by_channel(
        self,
        channel: str,
        goal: Optional["Goal"] = None,
        limit: Optional[int] = None,
    ) -> List[Lesson]:
        """
        Retrieve lessons from a specific channel using channel policy.

        Args:
            channel: Channel name
            goal: Optional goal for relevance filtering
            limit: Optional override for maximum returned lessons
        """
        lessons = self.retrieve_by_tag(self._channel_tag(channel))
        policy = self.channel_policies.get(channel, ChannelPolicy(strategy="relevance"))
        effective_limit = limit if limit is not None else policy.limit

        if policy.strategy == "relevance":
            if goal is not None:
                lessons = [lesson for lesson in lessons if lesson.applies_to(goal)]
            lessons.sort(
                key=lambda lesson: (float(lesson.confidence), lesson.timestamp),
                reverse=True,
            )
        elif policy.strategy == "recency":
            if goal is not None:
                relevant = [lesson for lesson in lessons if lesson.applies_to(goal)]
                if relevant:
                    lessons = relevant
            lessons.sort(key=lambda lesson: lesson.timestamp, reverse=True)
        else:  # persistence
            if goal is not None:
                lessons = [lesson for lesson in lessons if lesson.applies_to(goal)]
            lessons.sort(
                key=lambda lesson: (
                    lesson.applications,
                    float(lesson.confidence),
                    lesson.timestamp,
                ),
                reverse=True,
            )

        if effective_limit is not None:
            return lessons[:effective_limit]
        return lessons

    def get_success_rate(self, goal: Optional["Goal"] = None) -> float:
        """
        Calculate success rate from episodes.

        Args:
            goal: Optional goal to filter by

        Returns:
            Success rate (0.0 to 1.0)
        """
        if goal:
            relevant = [e for e in self.episodes if e.goal_id == goal.id]
        else:
            relevant = self.episodes

        if not relevant:
            return 0.5  # No data, assume average

        successes = sum(1 for e in relevant if e.result_status == "success")
        return successes / len(relevant)

    def get_common_failure_patterns(self) -> List[str]:
        """
        Identify common failure patterns.

        Returns:
            List of common failure insights
        """
        failure_lessons = self.retrieve_by_outcome("failure")

        # Group by insight
        insight_counts: dict = {}
        for lesson in failure_lessons:
            insight_counts[lesson.insight] = insight_counts.get(lesson.insight, 0) + 1

        # Return most common
        sorted_insights = sorted(
            insight_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [insight for insight, _ in sorted_insights[:5]]

    def clear_stale(self, max_age_hours: float = 168.0) -> int:
        """
        Remove stale lessons.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of lessons removed
        """
        original_count = len(self.lessons)
        self.lessons = [l for l in self.lessons if not l.is_stale(max_age_hours)]
        return original_count - len(self.lessons)

    def _prune_lessons(self) -> None:
        """Remove excess lessons, keeping highest confidence ones."""
        if len(self.lessons) > self.max_lessons:
            # Sort by confidence and keep top N
            self.lessons.sort(key=lambda l: float(l.confidence), reverse=True)
            self.lessons = self.lessons[:self.max_lessons]

    def _prune_episodes(self) -> None:
        """Remove oldest episodes."""
        if len(self.episodes) > self.max_episodes:
            # Sort by timestamp and keep most recent
            self.episodes.sort(key=lambda e: e.timestamp, reverse=True)
            self.episodes = self.episodes[:self.max_episodes]

    def export(self) -> dict:
        """Export memory to a dictionary for serialization."""
        return {
            "channel_policies": {
                channel: {
                    "strategy": policy.strategy,
                    "limit": policy.limit,
                }
                for channel, policy in self.channel_policies.items()
            },
            "lessons": [
                {
                    "situation": l.situation,
                    "insight": l.insight,
                    "outcome": l.outcome,
                    "confidence": float(l.confidence),
                    "applications": l.applications,
                    "timestamp": l.timestamp.isoformat(),
                    "tags": l.tags
                }
                for l in self.lessons
            ],
            "episodes": [
                {
                    "goal_id": e.goal_id,
                    "goal_description": e.goal_description,
                    "plan_steps": e.plan_steps,
                    "result_status": e.result_status,
                    "timestamp": e.timestamp.isoformat(),
                    "duration_ms": e.duration_ms,
                    "metadata": e.metadata,
                }
                for e in self.episodes
            ]
        }

    @classmethod
    def from_export(cls, data: dict) -> Memory:
        """Import memory from a dictionary."""
        raw_policies = data.get("channel_policies", {})
        policies = {
            channel: ChannelPolicy(
                strategy=policy_data.get("strategy", "relevance"),
                limit=policy_data.get("limit"),
            )
            for channel, policy_data in raw_policies.items()
        }
        memory = cls(channel_policies=policies)

        for l_data in data.get("lessons", []):
            lesson = Lesson(
                situation=l_data["situation"],
                insight=l_data["insight"],
                outcome=l_data["outcome"],
                confidence=Confidence(
                    l_data["confidence"],
                    ConfidenceSource.MEMORY
                ),
                applications=l_data.get("applications", 0),
                timestamp=datetime.fromisoformat(l_data["timestamp"]),
                tags=l_data.get("tags", [])
            )
            memory.lessons.append(lesson)

        for e_data in data.get("episodes", []):
            episode = Episode(
                goal_id=e_data["goal_id"],
                goal_description=e_data["goal_description"],
                plan_steps=e_data["plan_steps"],
                result_status=e_data["result_status"],
                lessons_extracted=[],  # Not preserved in export
                timestamp=datetime.fromisoformat(e_data["timestamp"]),
                duration_ms=e_data.get("duration_ms", 0.0),
                metadata=e_data.get("metadata", {}),
            )
            memory.episodes.append(episode)

        return memory

    def __repr__(self) -> str:
        return f"Memory(lessons={len(self.lessons)}, episodes={len(self.episodes)})"

    def _default_channel_policies(self) -> Dict[str, ChannelPolicy]:
        return {
            "task": ChannelPolicy(strategy="relevance", limit=25),
            "recent": ChannelPolicy(strategy="recency", limit=50),
            "conversation": ChannelPolicy(strategy="relevance", limit=40),
            "project": ChannelPolicy(strategy="persistence", limit=200),
            "temporal": ChannelPolicy(strategy="recency", limit=30),
            "user": ChannelPolicy(strategy="persistence", limit=100),
            "internal": ChannelPolicy(strategy="relevance", limit=50),
        }

    def _channel_tag(self, channel: str) -> str:
        return f"channel:{channel}"


class LessonExtractor:
    """
    Extracts lessons from execution results.

    This is where learning happens - we analyze what worked
    and what didn't to create actionable lessons.
    """

    def extract(
        self,
        goal: "Goal",
        plan: "Plan",
        result: "ExecutionResult"
    ) -> List[Lesson]:
        """
        Extract lessons from an execution.

        Args:
            goal: The goal that was attempted
            plan: The plan that was executed
            result: The execution result

        Returns:
            List of extracted lessons
        """
        lessons = []

        if result.status == "success":
            lessons.append(self._success_lesson(goal, plan))
        elif result.status in ["failure", "execution_failure"]:
            lessons.append(self._failure_lesson(goal, result))
        elif result.status == "verification_failure":
            lessons.append(self._verification_failure_lesson(goal, result))
        elif result.status == "precondition_failure":
            lessons.append(self._precondition_failure_lesson(goal, result))

        # Look for patterns in completed steps
        if result.completed_steps:
            pattern_lessons = self._extract_pattern_lessons(result)
            lessons.extend(pattern_lessons)

        return lessons

    def _success_lesson(self, goal: "Goal", plan: "Plan") -> Lesson:
        """Create a lesson from success."""
        return Lesson(
            situation=goal.description,
            insight=f"Achieved using plan with {len(plan.steps)} steps",
            outcome="success",
            confidence=Confidence(0.7, ConfidenceSource.OBSERVATION),
            tags=["success", "plan"]
        )

    def _failure_lesson(self, goal: "Goal", result: "ExecutionResult") -> Lesson:
        """Create a lesson from failure."""
        insight = result.failure_diagnosis or "Unknown failure"
        return Lesson(
            situation=goal.description,
            insight=insight,
            outcome="failure",
            confidence=Confidence(0.6, ConfidenceSource.OBSERVATION),
            tags=["failure"]
        )

    def _verification_failure_lesson(
        self,
        goal: "Goal",
        result: "ExecutionResult"
    ) -> Lesson:
        """Create a lesson from verification failure."""
        if result.verification:
            failures = [f.message for f in result.verification.failures[:3]]
            insight = f"Verification failed: {'; '.join(failures)}"
        else:
            insight = "Verification failed (no details)"

        return Lesson(
            situation=goal.description,
            insight=insight,
            outcome="failure",
            confidence=Confidence(0.7, ConfidenceSource.VERIFICATION),
            tags=["verification_failure"]
        )

    def _precondition_failure_lesson(
        self,
        goal: "Goal",
        result: "ExecutionResult"
    ) -> Lesson:
        """Create a lesson from precondition failure."""
        return Lesson(
            situation=goal.description,
            insight=result.failure_diagnosis or "Preconditions not met",
            outcome="failure",
            confidence=Confidence(0.8, ConfidenceSource.OBSERVATION),
            tags=["precondition_failure"]
        )

    def _extract_pattern_lessons(
        self,
        result: "ExecutionResult"
    ) -> List[Lesson]:
        """Extract lessons from patterns in completed steps."""
        lessons = []

        # Look for repeated actions
        action_names = [a.name for a in result.completed_steps]
        for name in set(action_names):
            count = action_names.count(name)
            if count > 2:
                lessons.append(Lesson(
                    situation=f"Plan with repeated '{name}'",
                    insight=f"Action '{name}' was repeated {count} times - consider optimization",
                    outcome="partial",
                    confidence=Confidence(0.5, ConfidenceSource.INFERENCE),
                    tags=["pattern", "optimization"]
                ))

        return lessons
