"""
Comprehensive test harness for evaluating the Deliberative Agent
across multiple LLM providers on various problems.

This runs extensive testing on medium and hard problems using:
- OpenAI
- Anthropic (Claude)
- XAI (Grok)
- Groq
- DeepSeek
- OpenRouter
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict

import pytest

from deliberative_agent import DeliberativeAgent, WorldState
from deliberative_agent.llm_integration import (
    LLMProvider,
    create_llm_client,
    test_llm_client,
)
from deliberative_agent.llm_executor import SimpleLLMExecutor
from test_problems import (
    TestProblem,
    get_all_problems,
    get_problems_by_difficulty,
)


@dataclass
class TestResult:
    """Result of running a test problem."""
    
    provider: str
    model: str
    problem_name: str
    problem_difficulty: str
    success: bool
    status: str
    steps_taken: int
    execution_time_seconds: float
    confidence: Optional[float] = None
    error_message: Optional[str] = None
    lessons_learned: int = 0


@dataclass
class TestSummary:
    """Summary of all test results."""
    
    total_tests: int
    successful_tests: int
    failed_tests: int
    success_rate: float
    average_execution_time: float
    results_by_provider: Dict[str, Dict]
    results_by_difficulty: Dict[str, Dict]
    all_results: List[TestResult]
    timestamp: str


class LLMTestHarness:
    """Test harness for evaluating agent with multiple LLMs."""

    def __init__(self, verbose: bool = True):
        """
        Initialize test harness.
        
        Args:
            verbose: Whether to print detailed progress
        """
        self.verbose = verbose
        self.results: List[TestResult] = []

    async def test_problem_with_provider(
        self,
        problem: TestProblem,
        provider: LLMProvider,
        model: Optional[str] = None
    ) -> TestResult:
        """
        Test a single problem with a specific LLM provider.
        
        Args:
            problem: Test problem to solve
            provider: LLM provider to use
            model: Optional specific model name
            
        Returns:
            TestResult with outcome
        """
        start_time = time.time()
        
        try:
            # Create LLM client
            client = create_llm_client(provider, model=model)
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Testing: {problem.name}")
                print(f"Provider: {provider.value} ({client.get_provider_name()})")
                print(f"Difficulty: {problem.difficulty}")
                print(f"{'='*70}")
            
            # Create executor and agent
            executor = SimpleLLMExecutor(client, verbose=self.verbose)
            agent = DeliberativeAgent(
                actions=problem.available_actions,
                action_executor=executor
            )
            
            # Run the agent
            result = await agent.achieve(problem.goal, problem.initial_state)
            
            execution_time = time.time() - start_time
            
            # Check if problem was actually solved
            success = result.success and problem.success_predicate(result.state or WorldState())
            
            if self.verbose:
                print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
                print(f"Status: {result.status}")
                print(f"Time: {execution_time:.2f}s")
                if result.lessons:
                    print(f"Lessons learned: {len(result.lessons)}")
            
            return TestResult(
                provider=provider.value,
                model=client.client._client_wrapper._client._default_model if hasattr(client.client, '_client_wrapper') else "unknown",
                problem_name=problem.name,
                problem_difficulty=problem.difficulty,
                success=success,
                status=result.status,
                steps_taken=len(result.plan.steps) if result.plan else 0,
                execution_time_seconds=execution_time,
                confidence=None,  # Could extract from result
                lessons_learned=len(result.lessons) if result.lessons else 0
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            if self.verbose:
                print(f"\nERROR: {str(e)}")
            
            return TestResult(
                provider=provider.value,
                model=model or "default",
                problem_name=problem.name,
                problem_difficulty=problem.difficulty,
                success=False,
                status="error",
                steps_taken=0,
                execution_time_seconds=execution_time,
                error_message=str(e)
            )

    async def run_comprehensive_tests(
        self,
        providers: Optional[List[LLMProvider]] = None,
        difficulty: Optional[str] = None
    ) -> TestSummary:
        """
        Run comprehensive tests across providers and problems.
        
        Args:
            providers: List of providers to test (None = all available)
            difficulty: Filter by difficulty ('medium', 'hard', or None for all)
            
        Returns:
            TestSummary with all results
        """
        if providers is None:
            # Try to detect which providers have API keys
            providers = []
            for provider in LLMProvider:
                env_var = f"{provider.value.upper()}_API_KEY"
                if os.getenv(env_var):
                    providers.append(provider)
            
            if not providers:
                raise ValueError("No LLM provider API keys found in environment")
        
        # Get test problems
        if difficulty:
            problems = get_problems_by_difficulty(difficulty)
        else:
            problems = get_all_problems()
        
        if self.verbose:
            print(f"\n{'#'*70}")
            print(f"COMPREHENSIVE LLM TESTING")
            print(f"{'#'*70}")
            print(f"Providers: {[p.value for p in providers]}")
            print(f"Problems: {len(problems)} ({difficulty or 'all difficulties'})")
            print(f"Total tests: {len(providers) * len(problems)}")
            print(f"{'#'*70}\n")
        
        # Run all tests
        self.results = []
        for provider in providers:
            for problem in problems:
                result = await self.test_problem_with_provider(problem, provider)
                self.results.append(result)
        
        # Generate summary
        return self._generate_summary()

    def _generate_summary(self) -> TestSummary:
        """Generate summary from results."""
        if not self.results:
            return TestSummary(
                total_tests=0,
                successful_tests=0,
                failed_tests=0,
                success_rate=0.0,
                average_execution_time=0.0,
                results_by_provider={},
                results_by_difficulty={},
                all_results=[],
                timestamp=datetime.now().isoformat()
            )
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful
        success_rate = successful / total if total > 0 else 0.0
        avg_time = sum(r.execution_time_seconds for r in self.results) / total
        
        # Group by provider
        by_provider = {}
        for result in self.results:
            if result.provider not in by_provider:
                by_provider[result.provider] = []
            by_provider[result.provider].append(result)
        
        provider_stats = {}
        for provider, results in by_provider.items():
            provider_success = sum(1 for r in results if r.success)
            provider_stats[provider] = {
                "total": len(results),
                "successful": provider_success,
                "failed": len(results) - provider_success,
                "success_rate": provider_success / len(results) if results else 0.0,
                "avg_time": sum(r.execution_time_seconds for r in results) / len(results)
            }
        
        # Group by difficulty
        by_difficulty = {}
        for result in self.results:
            if result.problem_difficulty not in by_difficulty:
                by_difficulty[result.problem_difficulty] = []
            by_difficulty[result.problem_difficulty].append(result)
        
        difficulty_stats = {}
        for difficulty, results in by_difficulty.items():
            difficulty_success = sum(1 for r in results if r.success)
            difficulty_stats[difficulty] = {
                "total": len(results),
                "successful": difficulty_success,
                "failed": len(results) - difficulty_success,
                "success_rate": difficulty_success / len(results) if results else 0.0,
                "avg_time": sum(r.execution_time_seconds for r in results) / len(results)
            }
        
        return TestSummary(
            total_tests=total,
            successful_tests=successful,
            failed_tests=failed,
            success_rate=success_rate,
            average_execution_time=avg_time,
            results_by_provider=provider_stats,
            results_by_difficulty=difficulty_stats,
            all_results=self.results,
            timestamp=datetime.now().isoformat()
        )

    def print_summary(self, summary: TestSummary) -> None:
        """Print a formatted summary."""
        print(f"\n{'#'*70}")
        print(f"TEST SUMMARY")
        print(f"{'#'*70}")
        print(f"Total Tests: {summary.total_tests}")
        print(f"Successful: {summary.successful_tests}")
        print(f"Failed: {summary.failed_tests}")
        print(f"Success Rate: {summary.success_rate*100:.1f}%")
        print(f"Average Time: {summary.average_execution_time:.2f}s")
        
        print(f"\n{'='*70}")
        print("RESULTS BY PROVIDER")
        print(f"{'='*70}")
        for provider, stats in summary.results_by_provider.items():
            print(f"\n{provider.upper()}:")
            print(f"  Total: {stats['total']}")
            print(f"  Success: {stats['successful']} ({stats['success_rate']*100:.1f}%)")
            print(f"  Failed: {stats['failed']}")
            print(f"  Avg Time: {stats['avg_time']:.2f}s")
        
        print(f"\n{'='*70}")
        print("RESULTS BY DIFFICULTY")
        print(f"{'='*70}")
        for difficulty, stats in summary.results_by_difficulty.items():
            print(f"\n{difficulty.upper()}:")
            print(f"  Total: {stats['total']}")
            print(f"  Success: {stats['successful']} ({stats['success_rate']*100:.1f}%)")
            print(f"  Failed: {stats['failed']}")
            print(f"  Avg Time: {stats['avg_time']:.2f}s")
        
        print(f"\n{'='*70}")
        print("DETAILED RESULTS")
        print(f"{'='*70}")
        for result in summary.all_results:
            status_icon = "✓" if result.success else "✗"
            print(f"{status_icon} [{result.provider}] {result.problem_name} - {result.status} ({result.execution_time_seconds:.2f}s)")
            if result.error_message:
                print(f"  Error: {result.error_message}")
        
        print(f"\n{'#'*70}\n")

    def save_results(self, filename: str = "test_results.json") -> None:
        """Save results to a JSON file."""
        summary = self._generate_summary()
        
        # Convert to serializable dict
        data = {
            "summary": {
                "total_tests": summary.total_tests,
                "successful_tests": summary.successful_tests,
                "failed_tests": summary.failed_tests,
                "success_rate": summary.success_rate,
                "average_execution_time": summary.average_execution_time,
                "results_by_provider": summary.results_by_provider,
                "results_by_difficulty": summary.results_by_difficulty,
                "timestamp": summary.timestamp
            },
            "detailed_results": [asdict(r) for r in summary.all_results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filename}")


# Pytest integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and
    not os.getenv("ANTHROPIC_API_KEY") and
    not os.getenv("XAI_API_KEY") and
    not os.getenv("GROQ_API_KEY") and
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="No LLM API keys configured"
)
async def test_medium_problems_all_providers():
    """Test medium difficulty problems with all available providers."""
    harness = LLMTestHarness(verbose=True)
    summary = await harness.run_comprehensive_tests(difficulty="medium")
    harness.print_summary(summary)
    harness.save_results("medium_test_results.json")
    
    # Assert reasonable success rate
    assert summary.success_rate > 0.3, f"Success rate too low: {summary.success_rate*100:.1f}%"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and
    not os.getenv("ANTHROPIC_API_KEY") and
    not os.getenv("XAI_API_KEY") and
    not os.getenv("GROQ_API_KEY") and
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="No LLM API keys configured"
)
async def test_hard_problems_all_providers():
    """Test hard difficulty problems with all available providers."""
    harness = LLMTestHarness(verbose=True)
    summary = await harness.run_comprehensive_tests(difficulty="hard")
    harness.print_summary(summary)
    harness.save_results("hard_test_results.json")
    
    # Hard problems may have lower success rate
    assert summary.success_rate > 0.2, f"Success rate too low: {summary.success_rate*100:.1f}%"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and
    not os.getenv("ANTHROPIC_API_KEY") and
    not os.getenv("XAI_API_KEY") and
    not os.getenv("GROQ_API_KEY") and
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="No LLM API keys configured"
)
async def test_all_problems_all_providers():
    """Test all problems with all available providers."""
    harness = LLMTestHarness(verbose=True)
    summary = await harness.run_comprehensive_tests()
    harness.print_summary(summary)
    harness.save_results("all_test_results.json")
    
    # Overall success rate
    assert summary.total_tests > 0
    print(f"Overall success rate: {summary.success_rate*100:.1f}%")


# Standalone execution
async def main():
    """Run comprehensive testing standalone."""
    print("Starting comprehensive LLM testing...")
    
    harness = LLMTestHarness(verbose=True)
    
    # Run all tests
    summary = await harness.run_comprehensive_tests()
    
    # Print and save results
    harness.print_summary(summary)
    harness.save_results("comprehensive_test_results.json")
    
    print("\nTesting complete!")


if __name__ == "__main__":
    asyncio.run(main())
