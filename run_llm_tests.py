#!/usr/bin/env python3
"""
Standalone script to run comprehensive LLM testing.

Usage:
    python run_llm_tests.py [--provider PROVIDER] [--difficulty DIFFICULTY]

Examples:
    # Test all providers on all problems
    python run_llm_tests.py

    # Test only OpenAI on medium problems
    python run_llm_tests.py --provider openai --difficulty medium

    # Test multiple providers on hard problems
    python run_llm_tests.py --provider openai --provider anthropic --difficulty hard
"""

import argparse
import asyncio
import sys
import os

# Add project root to path so "tests" and "deliberative_agent" resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_llm_comprehensive import LLMTestHarness
from deliberative_agent.llm_integration import LLMProvider


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive LLM testing for Deliberative Agent"
    )
    
    parser.add_argument(
        "--provider",
        action="append",
        choices=[p.value for p in LLMProvider],
        help="LLM provider to test (can specify multiple times). If not specified, tests all available."
    )
    
    parser.add_argument(
        "--difficulty",
        choices=["medium", "hard"],
        help="Problem difficulty to test. If not specified, tests both."
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    parser.add_argument(
        "--output",
        default="comprehensive_test_results.json",
        help="Output file for results (default: comprehensive_test_results.json)"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Convert provider strings to enum values
    providers = None
    if args.provider:
        providers = [LLMProvider(p) for p in args.provider]
    
    # Create test harness
    harness = LLMTestHarness(verbose=not args.quiet)
    
    print("="*70)
    print("DELIBERATIVE AGENT - COMPREHENSIVE LLM TESTING")
    print("="*70)
    
    # Check for API keys
    available_keys = []
    for provider in LLMProvider:
        env_var = f"{provider.value.upper()}_API_KEY"
        if os.getenv(env_var):
            available_keys.append(provider.value)
    
    if not available_keys:
        print("\nERROR: No LLM API keys found in environment!")
        print("\nPlease set at least one of:")
        for provider in LLMProvider:
            env_var = f"{provider.value.upper()}_API_KEY"
            print(f"  export {env_var}=your_key_here")
        print("\nSee tests/README_TESTING.md for more details.")
        return 1
    
    print(f"\nAvailable providers: {', '.join(available_keys)}")
    
    if providers:
        print(f"Testing providers: {', '.join(p.value for p in providers)}")
    else:
        print("Testing ALL available providers")
    
    if args.difficulty:
        print(f"Testing difficulty: {args.difficulty}")
    else:
        print("Testing ALL difficulty levels")
    
    print()
    
    # Run tests
    try:
        summary = await harness.run_comprehensive_tests(
            providers=providers,
            difficulty=args.difficulty
        )
        
        # Print summary
        harness.print_summary(summary)
        
        # Save results
        harness.save_results(args.output)
        
        print(f"\n✓ Testing complete! Results saved to {args.output}")
        
        # Return exit code based on success rate
        if summary.success_rate >= 0.5:
            return 0
        elif summary.success_rate >= 0.3:
            print("\nWARNING: Success rate below 50%")
            return 0
        else:
            print("\nERROR: Success rate below 30%")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
