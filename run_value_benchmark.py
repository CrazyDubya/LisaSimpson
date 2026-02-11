#!/usr/bin/env python3
"""
Run the value benchmark: Deliberative Agent vs "Ralph Wiggum" baseline.

Shows whether planning + verification add real value (success rate, steps, cost).

Usage:
  # Fast run with mock executor (no LLM) - default
  uv run python run_value_benchmark.py

  # With real LLM (OpenRouter) - slower, uses API
  OPENROUTER_API_KEY=sk-... uv run python run_value_benchmark.py --llm

  # More baseline runs for smoother stats
  uv run python run_value_benchmark.py --baseline-runs 20

  # Save JSON report
  uv run python run_value_benchmark.py --output value_report.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deliberative_agent.value_benchmark import (
    run_value_benchmark,
    run_memory_value_demo,
    get_real_world_scenarios,
    ValueReport,
    ValueRunResult,
    MemoryRunPair,
)
from deliberative_agent.llm_integration import create_llm_client, LLMProvider
from deliberative_agent.llm_executor import SimpleLLMExecutor


def _serialize_result(r: ValueRunResult) -> dict:
    return {
        "scenario_id": r.scenario_id,
        "run_type": r.run_type,
        "success": r.success,
        "steps": r.steps,
        "total_cost": r.total_cost,
        "duration_seconds": r.duration_seconds,
        "failure_reason": r.failure_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Value benchmark: Deliberative vs Ralph Wiggum baseline"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use real LLM executor (OpenRouter). Default: mock executor (no API).",
    )
    parser.add_argument(
        "--baseline-runs",
        type=int,
        default=10,
        metavar="N",
        help="Number of baseline (random-order) runs per scenario (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for baseline (default: 42)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="FILE",
        help="Write JSON report to FILE",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary table, no extra messages",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Also run memory-value demo: same scenario twice with shared agent memory",
    )
    args = parser.parse_args()

    executor = None
    if args.llm:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: --llm requires OPENROUTER_API_KEY or OPENAI_API_KEY", file=sys.stderr)
            return 1
        provider = LLMProvider.OPENROUTER if os.getenv("OPENROUTER_API_KEY") else LLMProvider.OPENAI
        client = create_llm_client(provider)
        executor = SimpleLLMExecutor(client, verbose=not args.quiet)
        if not args.quiet:
            print("Using LLM executor (this will be slower and use API credits).\n")
    else:
        if not args.quiet:
            print("Using mock executor (no LLM). Use --llm for real API runs.\n")

    async def run() -> ValueReport:
        return await run_value_benchmark(
            executor=executor,
            baseline_runs_per_scenario=args.baseline_runs,
            baseline_seed=args.seed,
        )

    async def run_all():
        rep = await run()
        if args.memory:
            memory_pairs = await run_memory_value_demo(executor=executor)
            return rep, memory_pairs
        return rep, None

    report, memory_pairs = asyncio.run(run_all())

    print(report.summary_table())

    if report.summary:
        print("\nAggregate:")
        print(f"  Deliberative success rate: {report.summary.get('deliberative_success_rate', 0):.0%}")
        print(f"  Deliberative avg steps (when success): {report.summary.get('deliberative_avg_steps_when_success', 0):.1f}")
        print(f"  Baseline success rate: {report.summary.get('baseline_success_rate', 0):.0%}")
        print(f"  Baseline avg steps (when success): {report.summary.get('baseline_avg_steps_when_success', 0):.1f}")
        print(f"  Baseline runs per scenario: {report.summary.get('baseline_runs_per_scenario', 0)}")

    if memory_pairs and not args.quiet:
        print("\nMemory value (run same scenario twice with shared memory):")
        for p in memory_pairs:
            print(f"  {p.scenario_id}: run1 steps={p.run1_steps} success={p.run1_success}  run2 steps={p.run2_steps} success={p.run2_success}")

    if args.output:
        out_path = Path(args.output)
        data = {
            "summary": report.summary,
            "scenarios": report.scenarios,
            "deliberative_results": [_serialize_result(r) for r in report.deliberative_results],
            "baseline_results": [_serialize_result(r) for r in report.baseline_results],
        }
        out_path.write_text(json.dumps(data, indent=2))
        if not args.quiet:
            print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
