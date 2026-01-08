#!/usr/bin/env python3
"""
Example of using the LLM testing infrastructure.

This demonstrates how to:
1. Create test problems
2. Run them with different LLM providers
3. Analyze results

Note: Requires API keys to be set in environment variables.
"""

import asyncio
import os
from deliberative_agent import WorldState, Goal, Action, Fact, Confidence, ConfidenceSource, VerificationPlan
from deliberative_agent.llm_integration import LLMProvider, create_llm_client
from deliberative_agent.llm_executor import SimpleLLMExecutor
from deliberative_agent import DeliberativeAgent


async def simple_example():
    """Simple example: solve a basic problem with an LLM."""
    
    print("="*70)
    print("SIMPLE EXAMPLE: Solve a basic problem with LLM")
    print("="*70)
    
    # Check if we have any API key
    if not any(os.getenv(f"{p.value.upper()}_API_KEY") for p in LLMProvider):
        print("\nERROR: No API keys found!")
        print("Please set at least one API key:")
        print("  export OPENAI_API_KEY=your_key")
        print("  export ANTHROPIC_API_KEY=your_key")
        print("  # etc.")
        return
    
    # Find first available provider
    provider = None
    for p in LLMProvider:
        if os.getenv(f"{p.value.upper()}_API_KEY"):
            provider = p
            break
    
    print(f"\nUsing provider: {provider.value}")
    
    # Create a simple problem
    initial_state = WorldState()
    initial_state.add_fact(Fact(
        "task_type",
        ("greeting",),
        Confidence(1.0, ConfidenceSource.OBSERVATION)
    ))
    
    # Define actions
    actions = [
        Action(
            name="create_greeting",
            description="Create a friendly greeting message",
            preconditions=[],
            effects=[
                Fact("greeting_created", (), Confidence(0.9, ConfidenceSource.INFERENCE))
            ],
            cost=1.0,
            reversible=True
        ),
        Action(
            name="format_greeting",
            description="Format the greeting nicely",
            preconditions=[
                lambda s: s.has_fact("greeting_created") is not None
            ],
            effects=[
                Fact("greeting_formatted", (), Confidence(0.9, ConfidenceSource.INFERENCE))
            ],
            cost=1.0,
            reversible=True
        ),
    ]
    
    # Define goal
    goal = Goal(
        id="greeting_task",
        description="Create and format a greeting",
        predicate=lambda s: s.has_fact("greeting_formatted") is not None,
        verification=VerificationPlan([])
    )
    
    # Create LLM client and executor
    client = create_llm_client(provider)
    executor = SimpleLLMExecutor(client, verbose=True)
    
    # Create and run agent
    agent = DeliberativeAgent(
        actions=actions,
        action_executor=executor
    )
    
    print("\nRunning agent...")
    result = await agent.achieve(goal, initial_state)
    
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(f"Success: {result.success}")
    print(f"Status: {result.status}")
    if result.plan:
        print(f"Steps taken: {len(result.plan.steps)}")
        print(f"Actions: {[a.name for a in result.plan.steps]}")
    if result.lessons:
        print(f"Lessons learned: {len(result.lessons)}")
    
    return result


async def compare_providers():
    """Compare multiple providers on the same problem."""
    
    print("\n" + "="*70)
    print("COMPARISON: Same problem with different providers")
    print("="*70)
    
    # Find available providers
    available = []
    for p in LLMProvider:
        if os.getenv(f"{p.value.upper()}_API_KEY"):
            available.append(p)
    
    if len(available) < 2:
        print("\nNeed at least 2 API keys to compare providers")
        print(f"Found: {[p.value for p in available]}")
        return
    
    print(f"\nComparing providers: {[p.value for p in available]}")
    
    # Simple problem
    initial_state = WorldState()
    initial_state.add_fact(Fact(
        "task",
        ("summary",),
        Confidence(1.0, ConfidenceSource.OBSERVATION)
    ))
    
    actions = [
        Action(
            name="analyze_text",
            description="Analyze the input text",
            preconditions=[],
            effects=[
                Fact("text_analyzed", (), Confidence(0.9, ConfidenceSource.INFERENCE))
            ],
            cost=2.0,
            reversible=True
        ),
        Action(
            name="write_summary",
            description="Write a concise summary",
            preconditions=[
                lambda s: s.has_fact("text_analyzed") is not None
            ],
            effects=[
                Fact("summary_written", (), Confidence(0.9, ConfidenceSource.INFERENCE))
            ],
            cost=3.0,
            reversible=True
        ),
    ]
    
    goal = Goal(
        id="summary_task",
        description="Analyze text and write summary",
        predicate=lambda s: s.has_fact("summary_written") is not None,
        verification=VerificationPlan([])
    )
    
    results = {}
    
    for provider in available[:3]:  # Limit to 3 to avoid too many API calls
        print(f"\n--- Testing with {provider.value} ---")
        
        try:
            client = create_llm_client(provider)
            executor = SimpleLLMExecutor(client, verbose=False)
            agent = DeliberativeAgent(
                actions=actions,
                action_executor=executor
            )
            
            import time
            start = time.time()
            result = await agent.achieve(goal, initial_state)
            duration = time.time() - start
            
            results[provider.value] = {
                "success": result.success,
                "status": result.status,
                "duration": duration,
                "steps": len(result.plan.steps) if result.plan else 0
            }
            
            print(f"  Success: {result.success}")
            print(f"  Duration: {duration:.2f}s")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[provider.value] = {"error": str(e)}
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    for provider, data in results.items():
        print(f"\n{provider}:")
        for key, value in data.items():
            print(f"  {key}: {value}")


async def main():
    """Run examples."""
    print("\n" + "#"*70)
    print("LLM TESTING EXAMPLES")
    print("#"*70)
    
    # Run simple example
    await simple_example()
    
    # Compare providers if multiple available
    await compare_providers()
    
    print("\n" + "#"*70)
    print("For comprehensive testing, run:")
    print("  python run_llm_tests.py")
    print("or")
    print("  pytest tests/test_llm_comprehensive.py -v -s")
    print("#"*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
