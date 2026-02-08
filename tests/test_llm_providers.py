"""
Tests for the LLM Providers module.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json

from deliberative_agent.llm_providers import (
    Message,
    MessageRole,
    LLMResponse,
    LLMConfig,
    LLMProvider,
    AnthropicProvider,
    OpenAIProvider,
    XAIProvider,
    OpenRouterProvider,
    GroqProvider,
    DeepSeekProvider,
    ProviderFactory,
)


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_user_message(self):
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_create_system_message(self):
        msg = Message(role=MessageRole.SYSTEM, content="You are helpful")
        assert msg.role == MessageRole.SYSTEM

    def test_create_assistant_message(self):
        msg = Message(role=MessageRole.ASSISTANT, content="Hi there!")
        assert msg.role == MessageRole.ASSISTANT


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_successful_response(self):
        response = LLMResponse(
            content="Hello!",
            model="gpt-4",
            provider="openai",
            usage={"input_tokens": 10, "output_tokens": 5},
            latency_ms=100.0
        )
        assert response.success is True
        assert response.content == "Hello!"
        assert response.error is None

    def test_failed_response(self):
        response = LLMResponse(
            content="",
            model="gpt-4",
            provider="openai",
            error="API Error"
        )
        assert response.success is False
        assert response.error == "API Error"

    def test_empty_content_is_failure(self):
        response = LLMResponse(
            content="",
            model="gpt-4",
            provider="openai"
        )
        assert response.success is False


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_values(self):
        config = LLMConfig(api_key="test-key", model="test-model")
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout == 60.0
        assert config.base_url is None

    def test_custom_values(self):
        config = LLMConfig(
            api_key="test-key",
            model="test-model",
            temperature=0.3,
            max_tokens=2048,
            timeout=30.0,
            base_url="https://custom.api.com"
        )
        assert config.temperature == 0.3
        assert config.max_tokens == 2048
        assert config.base_url == "https://custom.api.com"


class TestProviderFactory:
    """Tests for ProviderFactory."""

    def test_provider_map_has_all_providers(self):
        expected_providers = ["anthropic", "openai", "xai", "openrouter", "groq", "deepseek"]
        for provider in expected_providers:
            assert provider in ProviderFactory.PROVIDER_MAP

    def test_create_returns_none_without_key(self):
        with patch.dict('os.environ', {}, clear=True):
            provider = ProviderFactory.create("anthropic")
            assert provider is None

    def test_create_with_explicit_key(self):
        provider = ProviderFactory.create("anthropic", api_key="test-key")
        assert provider is not None
        assert isinstance(provider, AnthropicProvider)
        assert provider.config.api_key == "test-key"

    def test_create_with_custom_model(self):
        provider = ProviderFactory.create(
            "anthropic",
            api_key="test-key",
            model="claude-3-opus-20240229"
        )
        assert provider.config.model == "claude-3-opus-20240229"

    def test_create_unknown_provider(self):
        provider = ProviderFactory.create("unknown-provider", api_key="test")
        assert provider is None

    def test_create_all_available_with_env(self):
        env = {
            "ANTHROPIC_API_KEY": "test1",
            "OPENAI_API_KEY": "test2",
            "DEEPSEEK_API_KEY": "test3"
        }
        with patch.dict('os.environ', env, clear=True):
            providers = ProviderFactory.create_all_available()
            assert len(providers) == 3
            assert "anthropic" in providers
            assert "openai" in providers
            assert "deepseek" in providers


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_provider_name(self):
        config = LLMConfig(api_key="test", model="claude-3-5-sonnet-20241022")
        provider = AnthropicProvider(config)
        assert provider.name == "anthropic"

    @pytest.mark.asyncio
    async def test_complete_formats_messages(self):
        config = LLMConfig(api_key="test", model="claude-3-5-sonnet-20241022")
        provider = AnthropicProvider(config)

        mock_response = {
            "content": [{"text": "Hello!"}],
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }

        with patch.object(provider, '_make_request', return_value=mock_response):
            messages = [
                Message(role=MessageRole.SYSTEM, content="Be helpful"),
                Message(role=MessageRole.USER, content="Hi")
            ]
            response = await provider.complete(messages)

            assert response.success
            assert response.content == "Hello!"
            assert response.provider == "anthropic"


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_provider_name(self):
        config = LLMConfig(api_key="test", model="gpt-4o")
        provider = OpenAIProvider(config)
        assert provider.name == "openai"

    @pytest.mark.asyncio
    async def test_complete_formats_messages(self):
        config = LLMConfig(api_key="test", model="gpt-4o")
        provider = OpenAIProvider(config)

        mock_response = {
            "choices": [{"message": {"content": "Hello!"}}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }

        with patch.object(provider, '_make_request', return_value=mock_response):
            messages = [Message(role=MessageRole.USER, content="Hi")]
            response = await provider.complete(messages)

            assert response.success
            assert response.content == "Hello!"
            assert response.provider == "openai"


class TestXAIProvider:
    """Tests for XAIProvider."""

    def test_provider_name(self):
        config = LLMConfig(api_key="test", model="grok-2-latest")
        provider = XAIProvider(config)
        assert provider.name == "xai"


class TestOpenRouterProvider:
    """Tests for OpenRouterProvider."""

    def test_provider_name(self):
        config = LLMConfig(api_key="test", model="anthropic/claude-3.5-sonnet")
        provider = OpenRouterProvider(config)
        assert provider.name == "openrouter"


class TestGroqProvider:
    """Tests for GroqProvider."""

    def test_provider_name(self):
        config = LLMConfig(api_key="test", model="llama-3.3-70b-versatile")
        provider = GroqProvider(config)
        assert provider.name == "groq"


class TestDeepSeekProvider:
    """Tests for DeepSeekProvider."""

    def test_provider_name(self):
        config = LLMConfig(api_key="test", model="deepseek-chat")
        provider = DeepSeekProvider(config)
        assert provider.name == "deepseek"

    @pytest.mark.asyncio
    async def test_complete_handles_error(self):
        config = LLMConfig(api_key="test", model="deepseek-chat")
        provider = DeepSeekProvider(config)

        with patch.object(provider, '_make_request', side_effect=RuntimeError("API Error")):
            messages = [Message(role=MessageRole.USER, content="Hi")]
            response = await provider.complete(messages)

            assert not response.success
            assert "API Error" in response.error
