"""
Pytest fixtures for LisaSimpson tests.
"""

import os

import pytest

# Only provide LLM client fixture when we have the integration and optional deps
try:
    from deliberative_agent.llm_integration import LLMProvider, create_llm_client
except ImportError:
    LLMProvider = None
    create_llm_client = None


def _env_key(provider: LLMProvider) -> str:
    return f"{provider.value.upper()}_API_KEY"


@pytest.fixture
def client():
    """
    Provide an LLM client for the first provider that has an API key set.

    Skips the test if no provider has a key (so test_llm_client can run when keys
    are present and skip when they are not).
    """
    if LLMProvider is None or create_llm_client is None:
        pytest.skip("LLM integration not installed (install with [llm] extra)")
    for provider in LLMProvider:
        if os.getenv(_env_key(provider)):
            return create_llm_client(provider)
    pytest.skip("No LLM API key set (set OPENAI_API_KEY, OPENROUTER_API_KEY, etc.)")
