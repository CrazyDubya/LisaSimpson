"""
Tests for the Test Problems and Benchmark Runner modules.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

from deliberative_agent.benchmark_problems import (
    Difficulty,
    Category,
    ProblemSet,
    MEDIUM_PROBLEMS,
    HARD_PROBLEMS,
    ALL_PROBLEMS,
    get_problem_by_id,
    get_problems_summary,
    TestProblem as Problem,
    TestResult as Result,
)
from deliberative_agent.benchmark_runner import (
    ProviderResult,
    BenchmarkRun,
    BenchmarkRunner,
)
from deliberative_agent.llm_providers import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    Message,
    MessageRole,
)


class TestDifficulty:
    """Tests for Difficulty enum."""

    def test_difficulty_values(self):
        assert Difficulty.EASY.value == "easy"
        assert Difficulty.MEDIUM.value == "medium"
        assert Difficulty.HARD.value == "hard"
        assert Difficulty.EXPERT.value == "expert"


class TestCategory:
    """Tests for Category enum."""

    def test_category_values(self):
        assert Category.REASONING.value == "reasoning"
        assert Category.CODE_GENERATION.value == "code_generation"
        assert Category.MATHEMATICS.value == "mathematics"


class TestTestProblem:
    """Tests for TestProblem dataclass."""

    def test_create_problem(self):
        problem = Problem(
            id="test_01",
            name="Test Problem",
            description="A test problem",
            prompt="Solve this: 2+2",
            difficulty=Difficulty.EASY,
            category=Category.MATHEMATICS,
            expected_patterns=[r"\b4\b"]
        )
        assert problem.id == "test_01"
        assert problem.difficulty == Difficulty.EASY

    def test_validate_with_pattern(self):
        problem = Problem(
            id="test_01",
            name="Test",
            description="Test",
            prompt="What is 2+2?",
            difficulty=Difficulty.EASY,
            category=Category.MATHEMATICS,
            expected_patterns=[r"\b4\b", r"four"]
        )
        assert problem.validate_response("The answer is 4") is True
        assert problem.validate_response("The answer is four") is True
        assert problem.validate_response("The answer is 5") is False

    def test_validate_with_function(self):
        problem = Problem(
            id="test_01",
            name="Test",
            description="Test",
            prompt="Write hello world",
            difficulty=Difficulty.EASY,
            category=Category.CODE_GENERATION,
            validation_fn=lambda r: "hello" in r.lower() and "world" in r.lower()
        )
        assert problem.validate_response("Hello World!") is True
        assert problem.validate_response("Hello there") is False

    def test_validate_no_criteria_passes(self):
        problem = Problem(
            id="test_01",
            name="Test",
            description="Test",
            prompt="Do something",
            difficulty=Difficulty.EASY,
            category=Category.REASONING
        )
        assert problem.validate_response("Anything") is True


class TestProblemSet:
    """Tests for ProblemSet."""

    def test_get_by_difficulty(self):
        medium = ALL_PROBLEMS.get_by_difficulty(Difficulty.MEDIUM)
        assert len(medium) == 10
        for p in medium:
            assert p.difficulty == Difficulty.MEDIUM

    def test_get_by_category(self):
        reasoning = ALL_PROBLEMS.get_by_category(Category.REASONING)
        assert len(reasoning) >= 2
        for p in reasoning:
            assert p.category == Category.REASONING

    def test_get_by_tag(self):
        python_problems = ALL_PROBLEMS.get_by_tag("python")
        assert len(python_problems) > 0
        for p in python_problems:
            assert "python" in p.tags


class TestProblemCollection:
    """Tests for problem collections."""

    def test_medium_problems_count(self):
        assert len(MEDIUM_PROBLEMS) == 10

    def test_hard_problems_count(self):
        assert len(HARD_PROBLEMS) == 10

    def test_all_problems_count(self):
        assert len(ALL_PROBLEMS.problems) == 20

    def test_get_problem_by_id(self):
        problem = get_problem_by_id("medium_reasoning_01")
        assert problem is not None
        assert problem.name == "Syllogism Reasoning"

    def test_get_problem_by_id_not_found(self):
        problem = get_problem_by_id("nonexistent_problem")
        assert problem is None

    def test_get_problems_summary(self):
        summary = get_problems_summary()
        assert summary["total"] == 20
        assert summary["by_difficulty"]["medium"] == 10
        assert summary["by_difficulty"]["hard"] == 10


class TestProviderResult:
    """Tests for ProviderResult dataclass."""

    def test_accuracy_calculation(self):
        result = ProviderResult(
            provider_name="test",
            model="test-model",
            problems_attempted=10,
            problems_correct=7,
            total_latency_ms=1000,
            total_tokens=500
        )
        assert result.accuracy == 0.7

    def test_accuracy_zero_attempts(self):
        result = ProviderResult(
            provider_name="test",
            model="test-model",
            problems_attempted=0,
            problems_correct=0,
            total_latency_ms=0,
            total_tokens=0
        )
        assert result.accuracy == 0.0

    def test_avg_latency(self):
        result = ProviderResult(
            provider_name="test",
            model="test-model",
            problems_attempted=5,
            problems_correct=3,
            total_latency_ms=500,
            total_tokens=250
        )
        assert result.avg_latency_ms == 100.0


class TestBenchmarkRun:
    """Tests for BenchmarkRun dataclass."""

    def test_summary_generation(self):
        run = BenchmarkRun(
            run_id="test_run",
            timestamp="2024-01-01T00:00:00",
            problems_tested=["p1", "p2"],
            provider_results={
                "provider1": ProviderResult(
                    provider_name="provider1",
                    model="model1",
                    problems_attempted=2,
                    problems_correct=2,
                    total_latency_ms=200,
                    total_tokens=100
                )
            },
            duration_seconds=10.0
        )
        summary = run.summary()
        assert "test_run" in summary
        assert "provider1" in summary


class MockProvider(LLMProvider):
    """Mock provider for testing."""

    def __init__(self, responses=None):
        config = LLMConfig(api_key="test", model="mock-model")
        super().__init__(config)
        self.responses = responses or []
        self.call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    async def complete(self, messages, **kwargs):
        response_content = "Mock response"
        if self.call_count < len(self.responses):
            response_content = self.responses[self.call_count]
        self.call_count += 1

        return LLMResponse(
            content=response_content,
            model="mock-model",
            provider="mock",
            usage={"input_tokens": 10, "output_tokens": 20},
            latency_ms=50.0
        )


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_init_with_providers(self):
        providers = {"test": MockProvider()}
        runner = BenchmarkRunner(providers=providers, verbose=False)
        assert "test" in runner.providers

    def test_add_provider(self):
        runner = BenchmarkRunner(verbose=False)
        runner.add_provider("test", MockProvider())
        assert "test" in runner.providers

    @pytest.mark.asyncio
    async def test_run_single_test(self):
        provider = MockProvider(responses=["The answer is B"])
        problem = Problem(
            id="test_01",
            name="Test",
            description="Test",
            prompt="What is the answer?",
            difficulty=Difficulty.EASY,
            category=Category.REASONING,
            expected_patterns=[r"\bB\b"]
        )

        runner = BenchmarkRunner(verbose=False)
        result = await runner.run_single_test(provider, problem)

        assert result.problem_id == "test_01"
        assert result.provider == "mock"
        assert result.is_correct is True

    @pytest.mark.asyncio
    async def test_run_provider_benchmark(self):
        provider = MockProvider(responses=["B", "def func(): pass"])
        problems = [
            Problem(
                id="p1",
                name="P1",
                description="P1",
                prompt="Q1",
                difficulty=Difficulty.EASY,
                category=Category.REASONING,
                expected_patterns=[r"B"]
            ),
            Problem(
                id="p2",
                name="P2",
                description="P2",
                prompt="Q2",
                difficulty=Difficulty.EASY,
                category=Category.CODE_GENERATION,
                expected_patterns=[r"def"]
            )
        ]

        runner = BenchmarkRunner(verbose=False)
        result = await runner.run_provider_benchmark(provider, problems)

        assert result.problems_attempted == 2
        assert result.problems_correct == 2

    @pytest.mark.asyncio
    async def test_run_benchmark(self):
        providers = {
            "mock1": MockProvider(responses=["B"] * 10),
            "mock2": MockProvider(responses=["wrong"] * 10)
        }

        problems = [
            Problem(
                id=f"p{i}",
                name=f"P{i}",
                description=f"P{i}",
                prompt=f"Q{i}",
                difficulty=Difficulty.MEDIUM,
                category=Category.REASONING,
                expected_patterns=[r"B"]
            )
            for i in range(3)
        ]

        runner = BenchmarkRunner(providers=providers, verbose=False)
        run = await runner.run_benchmark(problems=problems)

        assert len(run.provider_results) == 2
        assert run.provider_results["mock1"].accuracy == 1.0
        assert run.provider_results["mock2"].accuracy == 0.0
