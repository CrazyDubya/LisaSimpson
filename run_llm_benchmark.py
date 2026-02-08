#!/usr/bin/env python3
"""
Comprehensive LLM Benchmark Runner.

Tests the Deliberative Agent system across multiple LLM providers
with medium and hard problems.

Usage:
    python run_llm_benchmark.py [--difficulty medium|hard] [--output results.json]
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from deliberative_agent.llm_providers import (
    ProviderFactory,
    LLMConfig,
    AnthropicProvider,
    OpenAIProvider,
    XAIProvider,
    OpenRouterProvider,
    GroqProvider,
    DeepSeekProvider,
)
from deliberative_agent.benchmark_problems import (
    MEDIUM_PROBLEMS,
    HARD_PROBLEMS,
    ALL_PROBLEMS,
    Difficulty,
    Category,
    get_problems_summary,
)
from deliberative_agent.benchmark_runner import BenchmarkRunner, BenchmarkRun


# Provider configurations with specific models to test
PROVIDER_CONFIGS = {
    "anthropic_claude": {
        "class": AnthropicProvider,
        "env_key": "ANTHROPIC_API_KEY",
        "model": "claude-3-5-sonnet-20241022",
        "display_name": "Claude 3.5 Sonnet"
    },
    "openai_gpt4o": {
        "class": OpenAIProvider,
        "env_key": "OPENAI_API_KEY",
        "model": "gpt-4o",
        "display_name": "GPT-4o"
    },
    "xai_grok": {
        "class": XAIProvider,
        "env_key": "XAI_API_KEY",
        "model": "grok-2-latest",
        "display_name": "Grok 2"
    },
    "openrouter_claude": {
        "class": OpenRouterProvider,
        "env_key": "OPENROUTER_API_KEY",
        "model": "anthropic/claude-3.5-sonnet",
        "display_name": "OR: Claude 3.5"
    },
    "openrouter_gpt4": {
        "class": OpenRouterProvider,
        "env_key": "OPENROUTER_API_KEY",
        "model": "openai/gpt-4-turbo",
        "display_name": "OR: GPT-4 Turbo"
    },
    "openrouter_llama": {
        "class": OpenRouterProvider,
        "env_key": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.1-405b-instruct",
        "display_name": "OR: Llama 3.1 405B"
    },
    "groq_llama": {
        "class": GroqProvider,
        "env_key": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
        "display_name": "Groq: Llama 3.3 70B"
    },
    "groq_mixtral": {
        "class": GroqProvider,
        "env_key": "GROQ_API_KEY",
        "model": "mixtral-8x7b-32768",
        "display_name": "Groq: Mixtral 8x7B"
    },
    "deepseek_chat": {
        "class": DeepSeekProvider,
        "env_key": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
        "display_name": "DeepSeek Chat"
    },
}


def create_providers() -> dict:
    """Create all available providers based on environment variables."""
    providers = {}

    for name, config in PROVIDER_CONFIGS.items():
        api_key = os.environ.get(config["env_key"])
        if api_key:
            try:
                provider_class = config["class"]
                llm_config = LLMConfig(
                    api_key=api_key,
                    model=config["model"],
                    temperature=0.3,  # Lower temperature for more consistent results
                    max_tokens=4096,
                    timeout=120.0
                )
                providers[name] = provider_class(llm_config)
                print(f"  + {config['display_name']}: ready")
            except Exception as e:
                print(f"  ! {config['display_name']}: failed to initialize ({e})")

    return providers


def print_banner():
    """Print benchmark banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║          LisaSimpson Deliberative Agent - LLM Benchmark              ║
║                                                                      ║
║  Testing Medium and Hard Problems Across Multiple LLM Providers      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def print_problem_summary():
    """Print summary of available problems."""
    summary = get_problems_summary()
    print("\n📋 Test Problems Summary:")
    print(f"   Total problems: {summary['total']}")
    print(f"   By difficulty:")
    for diff, count in summary['by_difficulty'].items():
        if count > 0:
            print(f"      - {diff}: {count}")
    print(f"   By category:")
    for cat, count in summary['by_category'].items():
        if count > 0:
            print(f"      - {cat}: {count}")


async def run_benchmark_suite(
    providers: dict,
    difficulty: Difficulty = None,
    output_path: Path = None
) -> BenchmarkRun:
    """Run the full benchmark suite."""

    # Select problems based on difficulty
    if difficulty == Difficulty.MEDIUM:
        problems = MEDIUM_PROBLEMS
        print(f"\n🎯 Running MEDIUM difficulty problems ({len(problems)} total)")
    elif difficulty == Difficulty.HARD:
        problems = HARD_PROBLEMS
        print(f"\n🎯 Running HARD difficulty problems ({len(problems)} total)")
    else:
        problems = ALL_PROBLEMS.problems
        print(f"\n🎯 Running ALL problems ({len(problems)} total)")

    # Create benchmark runner
    runner = BenchmarkRunner(
        providers=providers,
        system_prompt=(
            "You are an expert problem solver with deep knowledge in programming, "
            "mathematics, logic, and reasoning. Provide accurate, concise answers. "
            "For code problems, provide working, correct code. "
            "Show your reasoning when it helps explain your answer. "
            "Be precise and avoid unnecessary verbosity."
        ),
        verbose=True
    )

    # Run the benchmark
    print("\n" + "=" * 70)
    print("Starting benchmark execution...")
    print("=" * 70)

    result = await runner.run_benchmark(
        problems=problems,
        run_id=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Save results if output path specified
    if output_path:
        runner.save_results(output_path)

    return result


def print_detailed_results(result: BenchmarkRun):
    """Print detailed results analysis."""
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS")
    print("=" * 70)

    # Sort providers by accuracy
    sorted_results = sorted(
        result.provider_results.items(),
        key=lambda x: x[1].accuracy,
        reverse=True
    )

    print("\n🏆 Provider Ranking (by accuracy):\n")
    for rank, (name, pr) in enumerate(sorted_results, 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{emoji} {rank}. {name}")
        print(f"      Accuracy: {pr.accuracy*100:.1f}% ({pr.problems_correct}/{pr.problems_attempted})")
        print(f"      Avg Latency: {pr.avg_latency_ms:.0f}ms")
        print(f"      Total Tokens: {pr.total_tokens:,}")
        if pr.errors:
            print(f"      Errors: {len(pr.errors)}")
        print()

    # Problem difficulty analysis
    print("\n📊 Performance by Problem:\n")

    for problem_id in result.problems_tested:
        results_for_problem = []
        for pname, pr in result.provider_results.items():
            for tr in pr.results:
                if tr.problem_id == problem_id:
                    results_for_problem.append((pname, tr))

        correct_count = sum(1 for _, tr in results_for_problem if tr.is_correct)
        total_count = len(results_for_problem)

        difficulty_indicator = "🟢" if correct_count == total_count else "🟡" if correct_count > 0 else "🔴"

        print(f"{difficulty_indicator} {problem_id}: {correct_count}/{total_count} correct")

        # Show which providers got it right/wrong
        correct_providers = [n for n, tr in results_for_problem if tr.is_correct]
        wrong_providers = [n for n, tr in results_for_problem if not tr.is_correct]

        if correct_providers:
            print(f"      ✓ {', '.join(correct_providers)}")
        if wrong_providers:
            print(f"      ✗ {', '.join(wrong_providers)}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM Benchmark Suite")
    parser.add_argument(
        "--difficulty",
        choices=["medium", "hard", "all"],
        default="all",
        help="Difficulty level to test (default: all)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.json"),
        help="Output file for results (default: benchmark_results.json)"
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        help="Specific providers to test (default: all available)"
    )

    args = parser.parse_args()

    print_banner()

    # Show available problems
    print_problem_summary()

    # Initialize providers
    print("\n🔌 Initializing LLM Providers:")
    providers = create_providers()

    if not providers:
        print("\n❌ No providers available!")
        print("Please set API keys in environment variables:")
        print("  - ANTHROPIC_API_KEY")
        print("  - OPENAI_API_KEY")
        print("  - XAI_API_KEY")
        print("  - OPENROUTER_API_KEY")
        print("  - GROQ_API_KEY")
        print("  - DEEPSEEK_API_KEY")
        sys.exit(1)

    # Filter providers if specified
    if args.providers:
        providers = {k: v for k, v in providers.items() if k in args.providers}
        print(f"\n📌 Testing specific providers: {list(providers.keys())}")

    print(f"\n✅ {len(providers)} provider(s) ready for testing")

    # Determine difficulty
    difficulty = None
    if args.difficulty == "medium":
        difficulty = Difficulty.MEDIUM
    elif args.difficulty == "hard":
        difficulty = Difficulty.HARD

    # Run benchmark
    try:
        result = await run_benchmark_suite(
            providers=providers,
            difficulty=difficulty,
            output_path=args.output
        )

        # Print detailed analysis
        print_detailed_results(result)

        print(f"\n✅ Benchmark complete!")
        print(f"📄 Results saved to: {args.output}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
