# LLM Layers

How the codebase wires LLM clients and executors.

## Public API: `llm_integration`

**File:** `deliberative_agent/llm_integration.py`

- **Purpose:** Simple public API for creating and using LLM clients.
- **Use when:** Building an app or script that runs the agent with a real LLM (e.g. OpenRouter, OpenAI).
- **Key symbols:**
  - `create_llm_client(provider, ...)` – factory for an LLM client.
  - `LLMProvider` – enum of supported providers (OpenAI, Anthropic, OpenRouter, etc.).
  - `LLMClient` – abstract interface; concrete clients (`OpenAIClient`, `OpenRouterClient`, …) implement it.
  - `test_llm_client(client)` – optional connectivity check.

Typical usage: create a client with `create_llm_client(LLMProvider.OPENROUTER)` (or another provider), pass it to `SimpleLLMExecutor` from `llm_executor`, then pass that executor into `DeliberativeAgent`.

## Benchmark / config: `llm_providers`

**File:** `deliberative_agent/llm_providers.py`

- **Purpose:** Benchmark runner and configuration (model names, API base URLs, env var names).
- **Use when:** Running the LLM benchmark or need to resolve provider config (e.g. `ProviderFactory`, `LLMConfig`).
- **Not required** for normal “create client → executor → agent” usage; that path uses `llm_integration` only.

## Executor: `llm_executor`

**File:** `deliberative_agent/llm_executor.py`

- **Purpose:** Turns agent actions into LLM calls and execution results.
- **Key symbols:** `SimpleLLMExecutor(client, ...)`, `LLMActionExecutor`.
- **Uses:** A client from `llm_integration` (or any `LLMClient` implementation).

Summary: use **`llm_integration`** for the simple public API; use **`llm_providers`** when working with the benchmark runner or provider config.
