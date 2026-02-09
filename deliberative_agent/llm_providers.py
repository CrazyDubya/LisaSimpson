"""
Multi-LLM Provider Abstraction Layer.

Provides a unified interface for interacting with various LLM providers:
- xAI (Grok)
- Anthropic (Claude)
- OpenAI (GPT-4, GPT-3.5)
- OpenRouter (multiple models)
- Groq (fast inference)
- DeepSeek (DeepSeek-V3)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict
import urllib.request
import urllib.error
import ssl


class MessageRole(str, Enum):
    """Role of message in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a conversation."""
    role: MessageRole
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.content != ""


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 60.0
    base_url: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._ssl_context = ssl.create_default_context()

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        **kwargs: Any
    ) -> LLMResponse:
        """
        Generate a completion from the model.

        Args:
            messages: Conversation history
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with generated content
        """
        pass

    def _make_request(
        self,
        url: str,
        data: Dict[str, Any],
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Make a synchronous HTTP request."""
        json_data = json.dumps(data).encode('utf-8')
        request = urllib.request.Request(
            url,
            data=json_data,
            headers=headers,
            method='POST'
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout,
                context=self._ssl_context
            ) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else str(e)
            raise RuntimeError(f"HTTP {e.code}: {error_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"URL Error: {e.reason}")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    DEFAULT_BASE_URL = "https://api.anthropic.com/v1/messages"

    @property
    def name(self) -> str:
        return "anthropic"

    async def complete(
        self,
        messages: List[Message],
        **kwargs: Any
    ) -> LLMResponse:
        start_time = time.time()

        # Separate system message
        system_content = ""
        conversation = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_content = msg.content
            else:
                conversation.append({
                    "role": msg.role.value,
                    "content": msg.content
                })

        data = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": conversation
        }

        if system_content:
            data["system"] = system_content

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01"
        }

        url = self.config.base_url or self.DEFAULT_BASE_URL

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._make_request(url, data, headers)
            )

            latency = (time.time() - start_time) * 1000

            content = ""
            if response.get("content"):
                content = response["content"][0].get("text", "")

            usage = response.get("usage", {})

            return LLMResponse(
                content=content,
                model=response.get("model", self.config.model),
                provider=self.name,
                usage={
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0)
                },
                latency_ms=latency,
                raw_response=response
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.config.model,
                provider=self.name,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"

    @property
    def name(self) -> str:
        return "openai"

    async def complete(
        self,
        messages: List[Message],
        **kwargs: Any
    ) -> LLMResponse:
        start_time = time.time()

        formatted_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]

        data = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": formatted_messages
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        url = self.config.base_url or self.DEFAULT_BASE_URL

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._make_request(url, data, headers)
            )

            latency = (time.time() - start_time) * 1000

            content = ""
            if response.get("choices"):
                content = response["choices"][0].get("message", {}).get("content", "")

            usage = response.get("usage", {})

            return LLMResponse(
                content=content,
                model=response.get("model", self.config.model),
                provider=self.name,
                usage={
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0)
                },
                latency_ms=latency,
                raw_response=response
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.config.model,
                provider=self.name,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )


class XAIProvider(LLMProvider):
    """xAI Grok provider."""

    DEFAULT_BASE_URL = "https://api.x.ai/v1/chat/completions"

    @property
    def name(self) -> str:
        return "xai"

    async def complete(
        self,
        messages: List[Message],
        **kwargs: Any
    ) -> LLMResponse:
        start_time = time.time()

        formatted_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]

        data = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": formatted_messages
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        url = self.config.base_url or self.DEFAULT_BASE_URL

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._make_request(url, data, headers)
            )

            latency = (time.time() - start_time) * 1000

            content = ""
            if response.get("choices"):
                content = response["choices"][0].get("message", {}).get("content", "")

            usage = response.get("usage", {})

            return LLMResponse(
                content=content,
                model=response.get("model", self.config.model),
                provider=self.name,
                usage={
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0)
                },
                latency_ms=latency,
                raw_response=response
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.config.model,
                provider=self.name,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )


class OpenRouterProvider(LLMProvider):
    """OpenRouter multi-model provider."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    @property
    def name(self) -> str:
        return "openrouter"

    async def complete(
        self,
        messages: List[Message],
        **kwargs: Any
    ) -> LLMResponse:
        start_time = time.time()

        formatted_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]

        data = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": formatted_messages
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": "https://github.com/CrazyDubya/LisaSimpson",
            "X-Title": "LisaSimpson Deliberative Agent"
        }

        url = self.config.base_url or self.DEFAULT_BASE_URL

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._make_request(url, data, headers)
            )

            latency = (time.time() - start_time) * 1000

            content = ""
            if response.get("choices"):
                content = response["choices"][0].get("message", {}).get("content", "")

            usage = response.get("usage", {})

            return LLMResponse(
                content=content,
                model=response.get("model", self.config.model),
                provider=self.name,
                usage={
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0)
                },
                latency_ms=latency,
                raw_response=response
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.config.model,
                provider=self.name,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )


class GroqProvider(LLMProvider):
    """Groq fast inference provider."""

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    @property
    def name(self) -> str:
        return "groq"

    async def complete(
        self,
        messages: List[Message],
        **kwargs: Any
    ) -> LLMResponse:
        start_time = time.time()

        formatted_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]

        data = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": formatted_messages
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        url = self.config.base_url or self.DEFAULT_BASE_URL

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._make_request(url, data, headers)
            )

            latency = (time.time() - start_time) * 1000

            content = ""
            if response.get("choices"):
                content = response["choices"][0].get("message", {}).get("content", "")

            usage = response.get("usage", {})

            return LLMResponse(
                content=content,
                model=response.get("model", self.config.model),
                provider=self.name,
                usage={
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0)
                },
                latency_ms=latency,
                raw_response=response
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.config.model,
                provider=self.name,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )


class DeepSeekProvider(LLMProvider):
    """DeepSeek provider."""

    DEFAULT_BASE_URL = "https://api.deepseek.com/chat/completions"

    @property
    def name(self) -> str:
        return "deepseek"

    async def complete(
        self,
        messages: List[Message],
        **kwargs: Any
    ) -> LLMResponse:
        start_time = time.time()

        formatted_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]

        data = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": formatted_messages
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        url = self.config.base_url or self.DEFAULT_BASE_URL

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._make_request(url, data, headers)
            )

            latency = (time.time() - start_time) * 1000

            content = ""
            if response.get("choices"):
                content = response["choices"][0].get("message", {}).get("content", "")

            usage = response.get("usage", {})

            return LLMResponse(
                content=content,
                model=response.get("model", self.config.model),
                provider=self.name,
                usage={
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0)
                },
                latency_ms=latency,
                raw_response=response
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.config.model,
                provider=self.name,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )


class ProviderFactory:
    """Factory for creating LLM providers from environment variables."""

    PROVIDER_MAP = {
        "anthropic": (AnthropicProvider, "ANTHROPIC_API_KEY", "claude-3-5-sonnet-20241022"),
        "openai": (OpenAIProvider, "OPENAI_API_KEY", "gpt-4o"),
        "xai": (XAIProvider, "XAI_API_KEY", "grok-2-latest"),
        "openrouter": (OpenRouterProvider, "OPENROUTER_API_KEY", "anthropic/claude-3.5-sonnet"),
        "groq": (GroqProvider, "GROQ_API_KEY", "llama-3.3-70b-versatile"),
        "deepseek": (DeepSeekProvider, "DEEPSEEK_API_KEY", "deepseek-chat"),
    }

    @classmethod
    def create(
        cls,
        provider_name: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any
    ) -> Optional[LLMProvider]:
        """
        Create an LLM provider instance.

        Args:
            provider_name: Name of the provider (anthropic, openai, etc.)
            api_key: API key (uses env var if not provided)
            model: Model to use (uses default if not provided)
            **kwargs: Additional config parameters

        Returns:
            LLMProvider instance or None if provider not available
        """
        if provider_name not in cls.PROVIDER_MAP:
            return None

        provider_class, env_var, default_model = cls.PROVIDER_MAP[provider_name]

        key = api_key or os.environ.get(env_var)
        if not key:
            return None

        config = LLMConfig(
            api_key=key,
            model=model or default_model,
            **kwargs
        )

        return provider_class(config)

    @classmethod
    def create_all_available(
        cls,
        models: Optional[Dict[str, str]] = None
    ) -> Dict[str, LLMProvider]:
        """
        Create all providers that have API keys available.

        Args:
            models: Optional dict mapping provider names to specific models

        Returns:
            Dict of provider_name -> provider instance
        """
        models = models or {}
        providers = {}

        for name in cls.PROVIDER_MAP:
            model = models.get(name)
            provider = cls.create(name, model=model)
            if provider:
                providers[name] = provider

        return providers


# Convenience function for quick testing
async def test_provider(provider: LLMProvider, prompt: str = "Say 'Hello, World!'") -> LLMResponse:
    """Quick test of a provider."""
    messages = [Message(role=MessageRole.USER, content=prompt)]
    return await provider.complete(messages)
