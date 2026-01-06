"""
Comprehensive test problems for evaluating the Deliberative Agent
with different LLM providers.

Includes:
- Medium difficulty problems (multi-step planning, basic reasoning)
- Hard difficulty problems (complex planning, uncertainty, learning)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Literal

from deliberative_agent import (
    Action,
    Goal,
    WorldState,
    Confidence,
    ConfidenceSource,
    Fact,
    VerificationPlan,
)


class Difficulty(str, Enum):
    """Problem difficulty levels."""
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class TestProblem:
    """A test problem for evaluating the agent."""
    
    name: str
    description: str
    difficulty: Difficulty
    initial_state: WorldState
    goal: Goal
    available_actions: List[Action]
    success_predicate: Callable[[WorldState], bool]
    max_steps: int = 10


def create_medium_problems() -> List[TestProblem]:
    """Create medium difficulty test problems."""
    problems = []
    
    # Problem 1: Simple sequential task
    initial_state = WorldState()
    initial_state.add_fact(Fact(
        "project_type",
        ("web_app",),
        Confidence(1.0, ConfidenceSource.OBSERVATION)
    ))
    
    actions = [
        Action(
            name="create_directory_structure",
            description="Create the project directory structure",
            preconditions=[],
            effects=[
                Fact("directories_created", (), Confidence(0.9, ConfidenceSource.INFERENCE))
            ],
            cost=1.0,
            reversible=True
        ),
        Action(
            name="initialize_git",
            description="Initialize git repository",
            preconditions=[
                lambda s: s.has_fact("directories_created") is not None
            ],
            effects=[
                Fact("git_initialized", (), Confidence(0.9, ConfidenceSource.INFERENCE))
            ],
            cost=1.0,
            reversible=True
        ),
        Action(
            name="create_readme",
            description="Create README file",
            preconditions=[
                lambda s: s.has_fact("git_initialized") is not None
            ],
            effects=[
                Fact("readme_created", (), Confidence(0.9, ConfidenceSource.INFERENCE))
            ],
            cost=2.0,
            reversible=True
        ),
    ]
    
    goal = Goal(
        id="setup_project",
        description="Set up a new project with git and README",
        predicate=lambda s: (
            s.has_fact("directories_created") is not None and
            s.has_fact("git_initialized") is not None and
            s.has_fact("readme_created") is not None
        ),
        verification=VerificationPlan([])
    )
    
    problems.append(TestProblem(
        name="Sequential Project Setup",
        description="Set up a new project with directories, git, and README in sequence",
        difficulty=Difficulty.MEDIUM,
        initial_state=initial_state,
        goal=goal,
        available_actions=actions,
        success_predicate=lambda s: goal.is_satisfied(s),
        max_steps=5
    ))
    
    # Problem 2: Parallel tasks with dependencies
    initial_state2 = WorldState()
    initial_state2.add_fact(Fact(
        "project_initialized",
        (),
        Confidence(1.0, ConfidenceSource.OBSERVATION)
    ))
    
    actions2 = [
        Action(
            name="implement_frontend",
            description="Implement frontend components",
            preconditions=[
                lambda s: s.has_fact("project_initialized") is not None
            ],
            effects=[
                Fact("frontend_done", (), Confidence(0.8, ConfidenceSource.INFERENCE))
            ],
            cost=5.0,
            reversible=True
        ),
        Action(
            name="implement_backend",
            description="Implement backend API",
            preconditions=[
                lambda s: s.has_fact("project_initialized") is not None
            ],
            effects=[
                Fact("backend_done", (), Confidence(0.8, ConfidenceSource.INFERENCE))
            ],
            cost=5.0,
            reversible=True
        ),
        Action(
            name="integrate_components",
            description="Integrate frontend and backend",
            preconditions=[
                lambda s: (
                    s.has_fact("frontend_done") is not None and
                    s.has_fact("backend_done") is not None
                )
            ],
            effects=[
                Fact("integration_complete", (), Confidence(0.9, ConfidenceSource.INFERENCE))
            ],
            cost=3.0,
            reversible=True
        ),
    ]
    
    goal2 = Goal(
        id="complete_app",
        description="Complete application with frontend, backend, and integration",
        predicate=lambda s: s.has_fact("integration_complete") is not None,
        verification=VerificationPlan([])
    )
    
    problems.append(TestProblem(
        name="Parallel Component Development",
        description="Develop frontend and backend in parallel, then integrate",
        difficulty=Difficulty.MEDIUM,
        initial_state=initial_state2,
        goal=goal2,
        available_actions=actions2,
        success_predicate=lambda s: goal2.is_satisfied(s),
        max_steps=5
    ))
    
    # Problem 3: Conditional branching
    initial_state3 = WorldState()
    initial_state3.add_fact(Fact(
        "data_source",
        ("api",),
        Confidence(1.0, ConfidenceSource.OBSERVATION)
    ))
    
    actions3 = [
        Action(
            name="fetch_from_api",
            description="Fetch data from API",
            preconditions=[
                lambda s: s.has_fact("data_source", "api") is not None
            ],
            effects=[
                Fact("data_fetched", (), Confidence(0.9, ConfidenceSource.INFERENCE))
            ],
            cost=2.0,
            reversible=False
        ),
        Action(
            name="read_from_file",
            description="Read data from file",
            preconditions=[
                lambda s: s.has_fact("data_source", "file") is not None
            ],
            effects=[
                Fact("data_fetched", (), Confidence(0.9, ConfidenceSource.INFERENCE))
            ],
            cost=1.0,
            reversible=False
        ),
        Action(
            name="process_data",
            description="Process the fetched data",
            preconditions=[
                lambda s: s.has_fact("data_fetched") is not None
            ],
            effects=[
                Fact("data_processed", (), Confidence(0.9, ConfidenceSource.INFERENCE))
            ],
            cost=3.0,
            reversible=True
        ),
    ]
    
    goal3 = Goal(
        id="process_data",
        description="Fetch and process data",
        predicate=lambda s: s.has_fact("data_processed") is not None,
        verification=VerificationPlan([])
    )
    
    problems.append(TestProblem(
        name="Conditional Data Processing",
        description="Fetch data from appropriate source and process it",
        difficulty=Difficulty.MEDIUM,
        initial_state=initial_state3,
        goal=goal3,
        available_actions=actions3,
        success_predicate=lambda s: goal3.is_satisfied(s),
        max_steps=4
    ))
    
    return problems


def create_hard_problems() -> List[TestProblem]:
    """Create hard difficulty test problems."""
    problems = []
    
    # Problem 1: Multi-step with uncertainty and recovery
    initial_state = WorldState()
    initial_state.add_fact(Fact(
        "environment",
        ("production",),
        Confidence(1.0, ConfidenceSource.OBSERVATION)
    ))
    
    actions = [
        Action(
            name="run_tests",
            description="Run test suite",
            preconditions=[],
            effects=[
                Fact("tests_run", (), Confidence(0.95, ConfidenceSource.VERIFICATION))
            ],
            cost=2.0,
            reversible=False
        ),
        Action(
            name="create_backup",
            description="Create backup before deployment",
            preconditions=[
                lambda s: s.has_fact("tests_run") is not None
            ],
            effects=[
                Fact("backup_created", (), Confidence(0.99, ConfidenceSource.VERIFICATION))
            ],
            cost=3.0,
            reversible=False
        ),
        Action(
            name="deploy_application",
            description="Deploy application to production",
            preconditions=[
                lambda s: (
                    s.has_fact("backup_created") is not None and
                    s.has_fact("tests_run") is not None
                )
            ],
            effects=[
                Fact("deployed", (), Confidence(0.85, ConfidenceSource.INFERENCE))
            ],
            cost=5.0,
            reversible=True  # Can rollback
        ),
        Action(
            name="verify_deployment",
            description="Verify deployment is working",
            preconditions=[
                lambda s: s.has_fact("deployed") is not None
            ],
            effects=[
                Fact("deployment_verified", (), Confidence(0.95, ConfidenceSource.VERIFICATION))
            ],
            cost=2.0,
            reversible=False
        ),
        Action(
            name="rollback_deployment",
            description="Rollback deployment if verification fails",
            preconditions=[
                lambda s: (
                    s.has_fact("deployed") is not None and
                    s.has_fact("backup_created") is not None
                )
            ],
            effects=[
                Fact("rolled_back", (), Confidence(0.9, ConfidenceSource.VERIFICATION))
            ],
            cost=3.0,
            reversible=False
        ),
    ]
    
    goal = Goal(
        id="safe_deployment",
        description="Safely deploy application with backup and verification",
        predicate=lambda s: (
            s.has_fact("deployment_verified") is not None or
            s.has_fact("rolled_back") is not None
        ),
        verification=VerificationPlan([])
    )
    
    problems.append(TestProblem(
        name="Safe Production Deployment",
        description="Deploy to production with backup, testing, verification, and rollback capability",
        difficulty=Difficulty.HARD,
        initial_state=initial_state,
        goal=goal,
        available_actions=actions,
        success_predicate=lambda s: goal.is_satisfied(s),
        max_steps=8
    ))
    
    # Problem 2: Complex dependency graph
    initial_state2 = WorldState()
    initial_state2.add_fact(Fact(
        "requirements_defined",
        (),
        Confidence(1.0, ConfidenceSource.OBSERVATION)
    ))
    
    actions2 = [
        Action(
            name="design_database",
            description="Design database schema",
            preconditions=[
                lambda s: s.has_fact("requirements_defined") is not None
            ],
            effects=[
                Fact("database_designed", (), Confidence(0.85, ConfidenceSource.INFERENCE))
            ],
            cost=4.0,
            reversible=True
        ),
        Action(
            name="design_api",
            description="Design API endpoints",
            preconditions=[
                lambda s: s.has_fact("requirements_defined") is not None
            ],
            effects=[
                Fact("api_designed", (), Confidence(0.85, ConfidenceSource.INFERENCE))
            ],
            cost=4.0,
            reversible=True
        ),
        Action(
            name="implement_database",
            description="Implement database",
            preconditions=[
                lambda s: s.has_fact("database_designed") is not None
            ],
            effects=[
                Fact("database_implemented", (), Confidence(0.9, ConfidenceSource.VERIFICATION))
            ],
            cost=6.0,
            reversible=True
        ),
        Action(
            name="implement_api",
            description="Implement API",
            preconditions=[
                lambda s: (
                    s.has_fact("api_designed") is not None and
                    s.has_fact("database_implemented") is not None
                )
            ],
            effects=[
                Fact("api_implemented", (), Confidence(0.9, ConfidenceSource.VERIFICATION))
            ],
            cost=6.0,
            reversible=True
        ),
        Action(
            name="write_tests",
            description="Write integration tests",
            preconditions=[
                lambda s: (
                    s.has_fact("api_implemented") is not None
                )
            ],
            effects=[
                Fact("tests_written", (), Confidence(0.95, ConfidenceSource.VERIFICATION))
            ],
            cost=5.0,
            reversible=True
        ),
        Action(
            name="document_api",
            description="Document API",
            preconditions=[
                lambda s: s.has_fact("api_implemented") is not None
            ],
            effects=[
                Fact("api_documented", (), Confidence(0.9, ConfidenceSource.VERIFICATION))
            ],
            cost=3.0,
            reversible=True
        ),
    ]
    
    goal2 = Goal(
        id="complete_backend",
        description="Complete backend with database, API, tests, and documentation",
        predicate=lambda s: (
            s.has_fact("database_implemented") is not None and
            s.has_fact("api_implemented") is not None and
            s.has_fact("tests_written") is not None and
            s.has_fact("api_documented") is not None
        ),
        verification=VerificationPlan([])
    )
    
    problems.append(TestProblem(
        name="Complex Backend Development",
        description="Develop complete backend system with complex dependencies",
        difficulty=Difficulty.HARD,
        initial_state=initial_state2,
        goal=goal2,
        available_actions=actions2,
        success_predicate=lambda s: goal2.is_satisfied(s),
        max_steps=10
    ))
    
    # Problem 3: Optimization under constraints
    initial_state3 = WorldState()
    initial_state3.add_fact(Fact(
        "budget_available",
        (100.0,),
        Confidence(1.0, ConfidenceSource.OBSERVATION)
    ))
    initial_state3.add_fact(Fact(
        "time_limit",
        (20.0,),
        Confidence(1.0, ConfidenceSource.OBSERVATION)
    ))
    
    actions3 = [
        Action(
            name="basic_solution",
            description="Implement basic solution (cheap, fast)",
            preconditions=[],
            effects=[
                Fact("solution_implemented", ("basic",), Confidence(0.7, ConfidenceSource.INFERENCE)),
                Fact("quality_level", (60,), Confidence(0.8, ConfidenceSource.INFERENCE))
            ],
            cost=5.0,
            reversible=True
        ),
        Action(
            name="optimized_solution",
            description="Implement optimized solution (moderate cost and time)",
            preconditions=[],
            effects=[
                Fact("solution_implemented", ("optimized",), Confidence(0.85, ConfidenceSource.INFERENCE)),
                Fact("quality_level", (80,), Confidence(0.85, ConfidenceSource.INFERENCE))
            ],
            cost=15.0,
            reversible=True
        ),
        Action(
            name="premium_solution",
            description="Implement premium solution (expensive, slow, high quality)",
            preconditions=[],
            effects=[
                Fact("solution_implemented", ("premium",), Confidence(0.95, ConfidenceSource.INFERENCE)),
                Fact("quality_level", (95,), Confidence(0.9, ConfidenceSource.INFERENCE))
            ],
            cost=25.0,
            reversible=True
        ),
        Action(
            name="add_monitoring",
            description="Add monitoring and logging",
            preconditions=[
                lambda s: s.has_fact("solution_implemented") is not None
            ],
            effects=[
                Fact("monitoring_added", (), Confidence(0.9, ConfidenceSource.VERIFICATION))
            ],
            cost=5.0,
            reversible=True
        ),
        Action(
            name="add_tests",
            description="Add comprehensive test suite",
            preconditions=[
                lambda s: s.has_fact("solution_implemented") is not None
            ],
            effects=[
                Fact("tests_added", (), Confidence(0.95, ConfidenceSource.VERIFICATION))
            ],
            cost=8.0,
            reversible=True
        ),
    ]
    
    goal3 = Goal(
        id="quality_solution",
        description="Implement solution with quality >= 70, monitoring, and tests",
        predicate=lambda s: (
            s.has_fact("solution_implemented") is not None and
            s.has_fact("monitoring_added") is not None and
            s.has_fact("tests_added") is not None and
            (s.has_fact("quality_level") is not None)
        ),
        verification=VerificationPlan([])
    )
    
    problems.append(TestProblem(
        name="Quality Solution Under Constraints",
        description="Implement quality solution within budget and time constraints",
        difficulty=Difficulty.HARD,
        initial_state=initial_state3,
        goal=goal3,
        available_actions=actions3,
        success_predicate=lambda s: goal3.is_satisfied(s),
        max_steps=8
    ))
    
    return problems


def get_all_problems() -> List[TestProblem]:
    """Get all test problems (medium and hard)."""
    return create_medium_problems() + create_hard_problems()


def get_problems_by_difficulty(difficulty: str) -> List[TestProblem]:
    """
    Get problems of a specific difficulty.
    
    Args:
        difficulty: Difficulty level (Difficulty.MEDIUM or Difficulty.HARD, or string 'medium'/'hard')
        
    Returns:
        List of problems with that difficulty
    """
    # Convert string to enum if needed
    if isinstance(difficulty, str):
        difficulty = Difficulty(difficulty)
    
    all_problems = get_all_problems()
    return [p for p in all_problems if p.difficulty == difficulty]
