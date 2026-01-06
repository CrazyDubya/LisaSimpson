"""
LLM integration for the Deliberative Agent.

Supports multiple LLM providers:
- OpenAI
- Anthropic (Claude)
- XAI (Grok)
- Groq
- DeepSeek

This allows testing the agent across different LLM backends.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    XAI = "xai"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"


@dataclass
class LLMMessage:
    """A message in an LLM conversation."""
    role: str  # 'user', 'assistant', 'system'
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """
        Get a completion from the LLM.
        
        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse with the completion
        """
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        ...


class OpenAIClient(LLMClient):
    """OpenAI client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            model: Model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """Get completion from OpenAI."""
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider="openai",
            tokens_used=response.usage.total_tokens if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "openai"


class AnthropicClient(LLMClient):
    """Anthropic (Claude) client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022"
    ):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """Get completion from Anthropic."""
        # Separate system messages from conversation
        system_msgs = [m.content for m in messages if m.role == "system"]
        system = "\n".join(system_msgs) if system_msgs else None
        
        conv_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role != "system"
        ]
        
        response = await self.client.messages.create(
            model=self.model,
            messages=conv_messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            provider="anthropic",
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason
        )

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "anthropic"


class XAIClient(LLMClient):
    """XAI (Grok) client - uses OpenAI-compatible API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "grok-beta"):
        """
        Initialize XAI client.
        
        Args:
            api_key: API key (defaults to XAI_API_KEY env var)
            model: Model to use
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("XAI_API_KEY environment variable not set")
        
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1"
            )
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """Get completion from XAI."""
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider="xai",
            tokens_used=response.usage.total_tokens if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "xai"


class GroqClient(LLMClient):
    """Groq client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-70b-versatile"
    ):
        """
        Initialize Groq client.
        
        Args:
            api_key: API key (defaults to GROQ_API_KEY env var)
            model: Model to use
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        try:
            from groq import AsyncGroq
            self.client = AsyncGroq(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "groq package not installed. Install with: pip install groq"
            )

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """Get completion from Groq."""
        groq_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=groq_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider="groq",
            tokens_used=response.usage.total_tokens if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "groq"


class DeepSeekClient(LLMClient):
    """DeepSeek client - uses OpenAI-compatible API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat"
    ):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key: API key (defaults to DEEPSEEK_API_KEY env var)
            model: Model to use
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """Get completion from DeepSeek."""
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider="deepseek",
            tokens_used=response.usage.total_tokens if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "deepseek"


class OpenRouterClient(LLMClient):
    """OpenRouter client - aggregator supporting multiple models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-3.5-sonnet"
    ):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: API key (defaults to OPENROUTER_API_KEY env var)
            model: Model to use (e.g., "anthropic/claude-3.5-sonnet")
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """Get completion from OpenRouter."""
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider="openrouter",
            tokens_used=response.usage.total_tokens if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "openrouter"


def create_llm_client(
    provider: LLMProvider,
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> LLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        provider: Which provider to use
        api_key: Optional API key (uses env var if not provided)
        model: Optional model name (uses default if not provided)
        
    Returns:
        LLMClient instance
        
    Raises:
        ValueError: If provider is not supported
    """
    if provider == LLMProvider.OPENAI:
        return OpenAIClient(api_key=api_key, model=model or "gpt-4")
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicClient(api_key=api_key, model=model or "claude-3-5-sonnet-20241022")
    elif provider == LLMProvider.XAI:
        return XAIClient(api_key=api_key, model=model or "grok-beta")
    elif provider == LLMProvider.GROQ:
        return GroqClient(api_key=api_key, model=model or "llama-3.1-70b-versatile")
    elif provider == LLMProvider.DEEPSEEK:
        return DeepSeekClient(api_key=api_key, model=model or "deepseek-chat")
    elif provider == LLMProvider.OPENROUTER:
        return OpenRouterClient(api_key=api_key, model=model or "anthropic/claude-3.5-sonnet")
    else:
        raise ValueError(f"Unsupported provider: {provider}")


async def test_llm_client(client: LLMClient) -> bool:
    """
    Test if an LLM client is working correctly.
    
    Args:
        client: LLM client to test
        
    Returns:
        True if the client works, False otherwise
    """
    try:
        messages = [
            LLMMessage(role="user", content="Say 'Hello, world!' and nothing else.")
        ]
        response = await client.complete(messages, temperature=0.0, max_tokens=20)
        return "hello" in response.content.lower()
    except Exception as e:
        print(f"Error testing {client.get_provider_name()}: {e}")
        return False
