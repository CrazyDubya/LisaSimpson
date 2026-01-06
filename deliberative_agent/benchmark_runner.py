"""
Benchmark Runner for Multi-LLM Testing.

Runs test problems across multiple LLM providers and collects comprehensive metrics.
Provides detailed reporting and analysis capabilities.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .llm_providers import (
    LLMProvider,
    LLMResponse,
    LLMConfig,
    Message,
    MessageRole,
    ProviderFactory,
    AnthropicProvider,
    OpenAIProvider,
    XAIProvider,
    OpenRouterProvider,
    GroqProvider,
    DeepSeekProvider,
)
from .test_problems import (
    TestProblem,
    TestResult,
    Difficulty,
    Category,
    MEDIUM_PROBLEMS,
    HARD_PROBLEMS,
    ALL_PROBLEMS,
    get_problem_by_id,
)


@dataclass
class ProviderResult:
    """Results for a single provider across all problems."""
    provider_name: str
    model: str
    problems_attempted: int
    problems_correct: int
    total_latency_ms: float
    total_tokens: int
    results: List[TestResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.problems_attempted == 0:
            return 0.0
        return self.problems_correct / self.problems_attempted

    @property
    def avg_latency_ms(self) -> float:
        if self.problems_attempted == 0:
            return 0.0
        return self.total_latency_ms / self.problems_attempted


@dataclass
class BenchmarkRun:
    """Complete benchmark run results."""
    run_id: str
    timestamp: str
    problems_tested: List[str]
    provider_results: Dict[str, ProviderResult]
    duration_seconds: float
    config: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 70,
            f"BENCHMARK RUN: {self.run_id}",
            f"Timestamp: {self.timestamp}",
            f"Duration: {self.duration_seconds:.2f}s",
            f"Problems tested: {len(self.problems_tested)}",
            "=" * 70,
            "",
            "PROVIDER RESULTS:",
            "-" * 70,
        ]

        for name, result in sorted(
            self.provider_results.items(),
            key=lambda x: x[1].accuracy,
            reverse=True
        ):
            lines.append(
                f"{name:15} | Accuracy: {result.accuracy*100:5.1f}% | "
                f"Avg Latency: {result.avg_latency_ms:7.0f}ms | "
                f"Tokens: {result.total_tokens:6d}"
            )

        lines.extend(["", "-" * 70])
        return "\n".join(lines)

    def detailed_report(self) -> str:
        """Generate detailed report with per-problem results."""
        lines = [self.summary(), "", "DETAILED RESULTS BY PROBLEM:", "=" * 70]

        for problem_id in self.problems_tested:
            lines.append(f"\n{problem_id}:")
            for provider_name, result in self.provider_results.items():
                for tr in result.results:
                    if tr.problem_id == problem_id:
                        status = "PASS" if tr.is_correct else "FAIL"
                        lines.append(
                            f"  {provider_name:15}: {status} | "
                            f"{tr.latency_ms:.0f}ms | {tr.tokens_used} tokens"
                        )
                        if tr.error:
                            lines.append(f"    Error: {tr.error[:50]}...")

        return "\n".join(lines)


class BenchmarkRunner:
    """Orchestrates benchmark testing across LLM providers."""

    def __init__(
        self,
        providers: Optional[Dict[str, LLMProvider]] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the benchmark runner.

        Args:
            providers: Dict of provider_name -> LLMProvider
            system_prompt: Optional system prompt for all tests
            verbose: Print progress to stdout
        """
        self.providers = providers or {}
        self.system_prompt = system_prompt or (
            "You are an expert problem solver. Provide clear, concise, and accurate answers. "
            "Show your reasoning when appropriate. For code problems, provide working code."
        )
        self.verbose = verbose
        self._results: List[BenchmarkRun] = []

    def add_provider(self, name: str, provider: LLMProvider) -> None:
        """Add a provider to the benchmark."""
        self.providers[name] = provider

    def _log(self, message: str) -> None:
        """Log message if verbose mode is on."""
        if self.verbose:
            print(message, flush=True)

    async def run_single_test(
        self,
        provider: LLMProvider,
        problem: TestProblem
    ) -> TestResult:
        """Run a single test problem against a provider."""
        messages = [
            Message(role=MessageRole.SYSTEM, content=self.system_prompt),
            Message(role=MessageRole.USER, content=problem.prompt)
        ]

        try:
            response = await provider.complete(messages)

            is_correct = False
            if response.success:
                is_correct = problem.validate_response(response.content)

            return TestResult(
                problem_id=problem.id,
                provider=provider.name,
                model=provider.config.model,
                response=response.content[:2000] if response.content else "",
                is_correct=is_correct,
                latency_ms=response.latency_ms,
                tokens_used=response.usage.get("input_tokens", 0) + response.usage.get("output_tokens", 0),
                error=response.error
            )
        except Exception as e:
            return TestResult(
                problem_id=problem.id,
                provider=provider.name,
                model=provider.config.model,
                response="",
                is_correct=False,
                latency_ms=0,
                tokens_used=0,
                error=str(e)
            )

    async def run_provider_benchmark(
        self,
        provider: LLMProvider,
        problems: List[TestProblem],
        concurrent_limit: int = 3
    ) -> ProviderResult:
        """Run all problems against a single provider."""
        provider_result = ProviderResult(
            provider_name=provider.name,
            model=provider.config.model,
            problems_attempted=0,
            problems_correct=0,
            total_latency_ms=0,
            total_tokens=0
        )

        # Run tests with concurrency limit
        semaphore = asyncio.Semaphore(concurrent_limit)

        async def run_with_limit(problem: TestProblem) -> TestResult:
            async with semaphore:
                return await self.run_single_test(provider, problem)

        # Run all problems
        results = await asyncio.gather(
            *[run_with_limit(p) for p in problems],
            return_exceptions=True
        )

        for result in results:
            if isinstance(result, Exception):
                provider_result.errors.append(str(result))
                continue

            provider_result.results.append(result)
            provider_result.problems_attempted += 1
            provider_result.total_latency_ms += result.latency_ms
            provider_result.total_tokens += result.tokens_used

            if result.is_correct:
                provider_result.problems_correct += 1

        return provider_result

    async def run_benchmark(
        self,
        problems: Optional[List[TestProblem]] = None,
        difficulty: Optional[Difficulty] = None,
        category: Optional[Category] = None,
        provider_names: Optional[List[str]] = None,
        run_id: Optional[str] = None
    ) -> BenchmarkRun:
        """
        Run a complete benchmark.

        Args:
            problems: Specific problems to test (default: all)
            difficulty: Filter by difficulty level
            category: Filter by category
            provider_names: Specific providers to test (default: all)
            run_id: Custom run identifier

        Returns:
            BenchmarkRun with complete results
        """
        start_time = time.time()

        # Select problems
        if problems is None:
            problems = ALL_PROBLEMS.problems

        if difficulty:
            problems = [p for p in problems if p.difficulty == difficulty]

        if category:
            problems = [p for p in problems if p.category == category]

        # Select providers
        providers_to_test = self.providers
        if provider_names:
            providers_to_test = {
                k: v for k, v in self.providers.items()
                if k in provider_names
            }

        self._log(f"\n{'='*70}")
        self._log(f"Starting benchmark: {len(problems)} problems, {len(providers_to_test)} providers")
        self._log(f"{'='*70}\n")

        # Run benchmarks for each provider
        provider_results: Dict[str, ProviderResult] = {}

        for provider_name, provider in providers_to_test.items():
            self._log(f"Testing {provider_name} ({provider.config.model})...")

            result = await self.run_provider_benchmark(provider, problems)
            provider_results[provider_name] = result

            self._log(
                f"  -> {result.problems_correct}/{result.problems_attempted} correct "
                f"({result.accuracy*100:.1f}%), avg latency: {result.avg_latency_ms:.0f}ms"
            )

        duration = time.time() - start_time

        # Create run result
        run = BenchmarkRun(
            run_id=run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            problems_tested=[p.id for p in problems],
            provider_results=provider_results,
            duration_seconds=duration,
            config={
                "difficulty": difficulty.value if difficulty else None,
                "category": category.value if category else None,
                "provider_names": provider_names
            }
        )

        self._results.append(run)

        self._log("\n" + run.summary())

        return run

    def save_results(self, filepath: Path) -> None:
        """Save all results to a JSON file."""
        data = {
            "runs": [
                {
                    "run_id": run.run_id,
                    "timestamp": run.timestamp,
                    "problems_tested": run.problems_tested,
                    "duration_seconds": run.duration_seconds,
                    "config": run.config,
                    "provider_results": {
                        name: {
                            "provider_name": pr.provider_name,
                            "model": pr.model,
                            "problems_attempted": pr.problems_attempted,
                            "problems_correct": pr.problems_correct,
                            "accuracy": pr.accuracy,
                            "total_latency_ms": pr.total_latency_ms,
                            "avg_latency_ms": pr.avg_latency_ms,
                            "total_tokens": pr.total_tokens,
                            "errors": pr.errors,
                            "results": [
                                {
                                    "problem_id": tr.problem_id,
                                    "provider": tr.provider,
                                    "model": tr.model,
                                    "is_correct": tr.is_correct,
                                    "latency_ms": tr.latency_ms,
                                    "tokens_used": tr.tokens_used,
                                    "error": tr.error,
                                    "response_preview": tr.response[:500] if tr.response else ""
                                }
                                for tr in pr.results
                            ]
                        }
                        for name, pr in run.provider_results.items()
                    }
                }
                for run in self._results
            ]
        }

        filepath.write_text(json.dumps(data, indent=2))
        self._log(f"\nResults saved to: {filepath}")


async def run_full_benchmark(
    output_path: Optional[Path] = None,
    difficulty: Optional[Difficulty] = None,
    verbose: bool = True
) -> BenchmarkRun:
    """
    Convenience function to run a full benchmark with all available providers.

    Args:
        output_path: Optional path to save results
        difficulty: Optional filter by difficulty
        verbose: Print progress

    Returns:
        BenchmarkRun with results
    """
    # Create providers from environment
    providers = ProviderFactory.create_all_available()

    if not providers:
        raise RuntimeError("No LLM providers available. Set API keys in environment.")

    runner = BenchmarkRunner(providers=providers, verbose=verbose)

    result = await runner.run_benchmark(difficulty=difficulty)

    if output_path:
        runner.save_results(output_path)

    return result


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

async def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM benchmarks")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "expert"],
        help="Filter by difficulty level"
    )
    parser.add_argument(
        "--category",
        help="Filter by category"
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        help="Specific providers to test"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results JSON"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available providers and exit"
    )
    parser.add_argument(
        "--list-problems",
        action="store_true",
        help="List available problems and exit"
    )

    args = parser.parse_args()

    if args.list_providers:
        providers = ProviderFactory.create_all_available()
        print("\nAvailable providers:")
        for name, provider in providers.items():
            print(f"  - {name}: {provider.config.model}")
        if not providers:
            print("  (none - set API keys in environment)")
        return

    if args.list_problems:
        print("\nAvailable problems:")
        for problem in ALL_PROBLEMS.problems:
            print(f"  - [{problem.difficulty.value:6}] {problem.id}: {problem.name}")
        return

    # Run benchmark
    providers = ProviderFactory.create_all_available()

    if not providers:
        print("Error: No LLM providers available.")
        print("Set API keys in environment (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)")
        sys.exit(1)

    if args.providers:
        providers = {k: v for k, v in providers.items() if k in args.providers}

    difficulty = Difficulty(args.difficulty) if args.difficulty else None

    runner = BenchmarkRunner(providers=providers, verbose=not args.quiet)
    result = await runner.run_benchmark(difficulty=difficulty)

    if args.output:
        runner.save_results(args.output)

    print("\n" + result.detailed_report())


if __name__ == "__main__":
    asyncio.run(main())
