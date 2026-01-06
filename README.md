# LisaSimpson: Deliberative Agent

> *"Me fail English? That's unpossible!"* - Ralph Wiggum
>
> *"I know that I know nothing."* - Socrates
>
> The difference is the second one can improve.

A principled alternative to "Ralph Wiggum" style blind LLM iteration. This agent **plans**, **verifies**, and **learns** - rather than hoping for the best.

## The Problem with "Ralph Wiggum"

The "Ralph Wiggum" approach to LLM-based development is essentially:

```bash
while :; do cat PROMPT.md | llm ; done
```

This is problematic because:

1. **No Planning** - Actions are taken without understanding consequences
2. **No Verification** - "Completion" is detected by magic strings, not semantic correctness
3. **No Learning** - Each run is stateless with no accumulated knowledge
4. **No Understanding** - Failures are retried blindly without diagnosis

## The Deliberative Agent Approach

This library implements a fundamentally different architecture:

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  GOAL SYSTEM     | --> |  PLANNING        | --> |  EXECUTION       |
|  - Verification  |     |  - GOAP-style    |     |  - Monitored     |
|  - Dependencies  |     |  - Cost-aware    |     |  - Rollback      |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
         ^                                                 |
         |                                                 v
+------------------+                         +------------------+
|                  |                         |                  |
|  MEMORY          | <---------------------- |  LEARNING        |
|  - Lessons       |                         |  - From success  |
|  - Episodes      |                         |  - From failure  |
|                  |                         |                  |
+------------------+                         +------------------+
```

### Key Differences

| Aspect | Ralph Wiggum | Deliberative Agent |
|--------|--------------|-------------------|
| **Planning** | None - just executes | GOAP-style planning |
| **Verification** | Magic string match | Semantic verification |
| **Failure** | Retry blindly | Diagnose, understand, adapt |
| **Learning** | Context injection | Structured memory |
| **Uncertainty** | Ignored | Explicit modeling |
| **Termination** | Claimed completion | Verified goal satisfaction |
| **Recovery** | Git reset | Planned rollback |

## Installation

```bash
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import asyncio
from deliberative_agent import (
    DeliberativeAgent,
    WorldState,
    Goal,
    Action,
    action,
    goal,
    VerificationPlan,
    TestCheck,
)

# Define actions
create_module = (
    action("create_module", "Create the Python module")
    .adds_fact("module_exists", "calculator.py")
    .with_cost(2.0)
    .with_reversibility(True)
    .build()
)

write_tests = (
    action("write_tests", "Write unit tests")
    .requires_fact("module_exists", "calculator.py")
    .adds_fact("tests_exist", "test_calculator.py")
    .with_cost(3.0)
    .with_reversibility(True)
    .build()
)

run_tests = (
    action("run_tests", "Run the test suite")
    .requires_fact("tests_exist", "test_calculator.py")
    .adds_fact("tests_pass")
    .with_cost(1.0)
    .with_reversibility(False)  # Can't un-run tests
    .build()
)

# Define the goal with verification
calculator_goal = (
    goal("calculator", "Create a working calculator with tests")
    .with_predicate(lambda s: s.has_fact("tests_pass") is not None)
    .with_verification(VerificationPlan(
        checks=[TestCheck(["pytest", "test_calculator.py", "-v"])],
        required_confidence=0.8
    ))
    .build()
)

# Create your action executor (implements the actual work)
class MyExecutor:
    async def execute(self, action, state):
        # Your implementation here
        ...

# Create and run the agent
async def main():
    agent = DeliberativeAgent(
        actions=[create_module, write_tests, run_tests],
        action_executor=MyExecutor()
    )

    state = WorldState()
    result = await agent.achieve(calculator_goal, state)

    if result.success:
        print("Goal achieved!")
    elif result.needs_input:
        print(f"Need input: {result.questions}")
    else:
        print(f"Failed: {result.message}")
        print(f"Learned: {result.lessons}")

asyncio.run(main())
```

## Core Concepts

### Goals

Goals are not just strings - they include:

- **Predicate**: Quick check for satisfaction
- **Verification Plan**: Authoritative verification with multiple checks
- **Dependencies**: Other goals that must be satisfied first
- **Priority**: For scheduling multiple goals

```python
from deliberative_agent import goal, VerificationPlan, TestCheck, TypeCheck

my_goal = (
    goal("feature_complete", "Implement the user authentication feature")
    .with_predicate(lambda s: s.has_fact("auth_implemented"))
    .with_verification(VerificationPlan(
        checks=[
            TypeCheck(["src/auth.py"]),
            TestCheck(["pytest", "tests/test_auth.py"]),
        ],
        required_confidence=0.9
    ))
    .with_dependency(database_ready_goal)
    .with_priority(10)
    .build()
)
```

### Actions

Actions have explicit preconditions, effects, and costs:

```python
from deliberative_agent import action

deploy_action = (
    action("deploy", "Deploy to production")
    .requires_fact("tests_pass")
    .requires_fact("approved")
    .adds_fact("deployed", "production")
    .with_cost(10.0)
    .with_reversibility(True, rollback_action)
    .with_confidence_modifier(0.8)  # Slightly risky
    .build()
)
```

### Verification

Unlike magic string matching, verification is semantic:

```python
from deliberative_agent import VerificationPlan, TypeCheck, TestCheck, SemanticCheck

verification = VerificationPlan(
    checks=[
        TypeCheck(["src/"], tool="pyright"),
        TestCheck(["pytest", "-v", "--cov"]),
        SemanticCheck(
            "All API endpoints return proper error responses",
            context_files=["src/api.py", "src/errors.py"]
        ),
    ],
    required_confidence=0.85
)
```

### Memory & Learning

The agent learns from experience:

```python
# Agent automatically learns from execution
result = await agent.achieve(goal, state)

# Check what was learned
for lesson in result.lessons:
    print(f"Learned: {lesson.insight} (confidence: {lesson.confidence})")

# Memory persists across sessions
memory_data = agent.memory.export()
# ... save to disk ...

# Later, restore memory
from deliberative_agent import Memory
restored_memory = Memory.from_export(memory_data)
agent = DeliberativeAgent(..., memory=restored_memory)
```

## Architecture

### Confidence System

Every belief has explicit confidence with provenance:

```python
from deliberative_agent import Confidence, ConfidenceSource

# High confidence from verification
conf = Confidence(0.95, ConfidenceSource.VERIFICATION)

# Lower confidence from inference
conf = Confidence(0.6, ConfidenceSource.INFERENCE)

# Confidence decays over time
decayed = conf.decay(half_life_hours=24.0)
```

### Planning (GOAP-style)

The planner uses A* search through action space:

```python
from deliberative_agent import Planner

planner = Planner(
    actions=my_actions,
    max_depth=50,
    max_explored=10000
)

result = planner.plan(current_state, goal)
if result.success:
    print(f"Found plan with {len(result.plan.steps)} steps")
    print(f"Estimated cost: {result.plan.estimated_cost}")
```

### Execution with Rollback

Execution includes verification points and rollback capability:

```python
from deliberative_agent import Executor

executor = Executor(action_executor)
result = await executor.execute_with_rollback(plan, initial_state)

if not result.success:
    print(f"Failed at step: {result.failure_diagnosis}")
    print(f"Rolled back to: {result.final_state}")
```

## Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=deliberative_agent

# Just unit tests
pytest tests/test_core.py tests/test_planning.py
```

## Philosophy

This library embodies several key principles:

1. **Think Before Acting**: Plan first, execute second
2. **Verify Semantically**: Don't check for magic strings; verify actual properties
3. **Learn from Experience**: Build knowledge over time
4. **Know Your Limits**: Explicit uncertainty modeling
5. **Fail Gracefully**: Understand failures, roll back when possible

## License

MIT
