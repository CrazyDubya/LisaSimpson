# LLM Testing for Deliberative Agent

This directory contains comprehensive testing infrastructure for evaluating the Deliberative Agent across multiple LLM providers on various problem difficulties.

## Setup

### 1. Install Dependencies

```bash
pip install -e ".[dev,llm]"
```

This installs:
- Core dependencies
- Development tools (pytest, ruff, mypy)
- LLM provider SDKs (openai, anthropic, groq)

### 2. Configure API Keys

Export your API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY=your_openai_api_key_here

# Anthropic (Claude)
export ANTHROPIC_API_KEY=your_anthropic_api_key_here

# XAI (Grok)
export XAI_API_KEY=your_xai_api_key_here

# Groq
export GROQ_API_KEY=your_groq_api_key_here

# DeepSeek
export DEEPSEEK_API_KEY=your_deepseek_api_key_here

# OpenRouter (optional)
export OPENROUTER_API_KEY=your_openrouter_api_key_here
```

You can also create a `.env` file:

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
XAI_API_KEY=your_xai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
EOF

# Load environment variables
source .env
```

## Running Tests

### Quick Test - Single Provider

Test a specific provider:

```bash
# Test with OpenAI
python -c "
import asyncio
from tests.test_llm_comprehensive import LLMTestHarness
from deliberative_agent.llm_integration import LLMProvider

async def test():
    harness = LLMTestHarness(verbose=True)
    await harness.run_comprehensive_tests(
        providers=[LLMProvider.OPENAI],
        difficulty='medium'
    )

asyncio.run(test())
"
```

### Comprehensive Testing

Run all tests across all providers:

```bash
# Using pytest
pytest tests/test_llm_comprehensive.py -v -s

# Or run standalone
python tests/test_llm_comprehensive.py
```

### Test Specific Difficulties

```bash
# Medium problems only
pytest tests/test_llm_comprehensive.py::test_medium_problems_all_providers -v -s

# Hard problems only
pytest tests/test_llm_comprehensive.py::test_hard_problems_all_providers -v -s

# All problems
pytest tests/test_llm_comprehensive.py::test_all_problems_all_providers -v -s
```

## Test Problems

The test suite includes:

### Medium Difficulty Problems
1. **Sequential Project Setup** - Multi-step sequential task with dependencies
2. **Parallel Component Development** - Parallel tasks that converge
3. **Conditional Data Processing** - Branching logic based on state

### Hard Difficulty Problems
1. **Safe Production Deployment** - Complex workflow with backup, testing, verification, and rollback
2. **Complex Backend Development** - Multi-component system with complex dependency graph
3. **Quality Solution Under Constraints** - Optimization problem with cost/time constraints

## Results

Test results are saved to JSON files:
- `medium_test_results.json` - Results for medium problems
- `hard_test_results.json` - Results for hard problems
- `all_test_results.json` - Combined results
- `comprehensive_test_results.json` - Results from standalone execution

Each result file contains:
- Summary statistics (success rate, timing, etc.)
- Results grouped by provider
- Results grouped by difficulty
- Detailed results for each test

### Example Output

```
##########################################################################
TEST SUMMARY
##########################################################################
Total Tests: 18
Successful: 14
Failed: 4
Success Rate: 77.8%
Average Time: 3.45s

======================================================================
RESULTS BY PROVIDER
======================================================================

OPENAI:
  Total: 6
  Success: 5 (83.3%)
  Failed: 1
  Avg Time: 3.12s

ANTHROPIC:
  Total: 6
  Success: 5 (83.3%)
  Failed: 1
  Avg Time: 3.89s

GROQ:
  Total: 6
  Success: 4 (66.7%)
  Failed: 2
  Avg Time: 3.34s
```

## Architecture

### LLM Integration (`deliberative_agent/llm_integration.py`)
- Abstract `LLMClient` interface
- Implementations for OpenAI, Anthropic, XAI, Groq, DeepSeek, OpenRouter
- Unified API across all providers
- Factory function for easy client creation

### LLM Executor (`deliberative_agent/llm_executor.py`)
- `LLMActionExecutor` - Uses LLM to implement action execution
- `SimpleLLMExecutor` - Simplified executor for testing
- Integrates with the agent's execution system

### Test Problems (`tests/test_problems.py`)
- Structured test problem definitions
- Medium and hard difficulty levels
- Success predicates for verification
- Action definitions with preconditions and effects

### Test Harness (`tests/test_llm_comprehensive.py`)
- `LLMTestHarness` - Main test orchestrator
- Runs problems across multiple providers
- Collects and analyzes results
- Generates comprehensive reports
- Pytest integration

## Interpreting Results

### Success Rates
- **80%+** - Excellent, agent handles problems well
- **60-80%** - Good, some challenges but generally capable
- **40-60%** - Moderate, struggles with harder aspects
- **<40%** - Poor, significant issues to address

### By Provider
Compare different LLMs to understand:
- Which models are most reliable
- Which handle complexity better
- Performance vs. cost tradeoffs

### By Difficulty
- **Medium problems** should have higher success rates (70%+)
- **Hard problems** are expected to be more challenging (50%+)
- Large gaps may indicate specific weaknesses

## Troubleshooting

### Missing API Keys
```
ValueError: OPENAI_API_KEY environment variable not set
```
**Solution:** Export the required API key as shown in setup section

### Import Errors
```
ImportError: openai package not installed
```
**Solution:** Install LLM dependencies: `pip install -e ".[llm]"`

### Rate Limiting
Some providers have rate limits. If you hit them:
- Add delays between tests
- Test fewer problems at once
- Use different API keys/accounts

### Timeout Errors
Some models are slower. Increase timeouts in the test harness if needed.

## Customization

### Adding New Problems

Edit `tests/test_problems.py`:

```python
def create_custom_problem():
    initial_state = WorldState()
    # ... set up state ...
    
    actions = [
        # ... define actions ...
    ]
    
    goal = Goal(
        id="my_goal",
        description="My custom goal",
        predicate=lambda s: # ... check satisfaction ...,
        verification=VerificationPlan([])
    )
    
    return TestProblem(
        name="My Custom Problem",
        description="...",
        difficulty="medium",
        initial_state=initial_state,
        goal=goal,
        available_actions=actions,
        success_predicate=lambda s: goal.is_satisfied(s),
        max_steps=10
    )
```

### Adding New Providers

Edit `deliberative_agent/llm_integration.py`:

```python
class MyProviderClient(LLMClient):
    def __init__(self, api_key=None, model="default"):
        # Initialize your client
        ...
    
    async def complete(self, messages, temperature, max_tokens):
        # Implement completion
        ...
    
    def get_provider_name(self):
        return "myprovider"
```

Then add to the factory:

```python
def create_llm_client(provider, api_key=None, model=None):
    # ...
    elif provider == LLMProvider.MYPROVIDER:
        return MyProviderClient(api_key, model)
```

## Contributing

When adding tests:
1. Ensure problems are well-defined with clear success criteria
2. Test with at least 2 providers before submitting
3. Document expected behavior and any known issues
4. Update this README with new features

## License

MIT
