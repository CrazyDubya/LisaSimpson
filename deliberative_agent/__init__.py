"""
Deliberative Agent - A principled alternative to blind LLM iteration.

This agent plans, verifies, and learns - rather than hoping for the best.

Key features:
- GOAP-style planning before execution
- Explicit uncertainty modeling
- Semantic verification (not string matching)
- Structured memory and learning
- Knows when it doesn't know
"""

from .core import (
    Confidence,
    ConfidenceSource,
    Fact,
    WorldState,
)
from .goals import Goal
from .actions import Action, Effect
from .verification import (
    Check,
    CheckResult,
    VerificationPlan,
    VerificationResult,
    TypeCheck,
    TestCheck,
    SemanticCheck,
)
from .planning import Plan, Planner
from .execution import (
    ActionExecutor,
    ExecutionStepResult,
    ExecutionResult,
    Executor,
)
from .memory import Lesson, Episode, Memory
from .agent import DeliberativeAgent, AgentResult

__version__ = "0.1.0"
__all__ = [
    # Core types
    "Confidence",
    "ConfidenceSource",
    "Fact",
    "WorldState",
    # Goals
    "Goal",
    # Actions
    "Action",
    "Effect",
    # Verification
    "Check",
    "CheckResult",
    "VerificationPlan",
    "VerificationResult",
    "TypeCheck",
    "TestCheck",
    "SemanticCheck",
    # Planning
    "Plan",
    "Planner",
    # Execution
    "ActionExecutor",
    "ExecutionStepResult",
    "ExecutionResult",
    "Executor",
    # Memory
    "Lesson",
    "Episode",
    "Memory",
    # Agent
    "DeliberativeAgent",
    "AgentResult",
]
