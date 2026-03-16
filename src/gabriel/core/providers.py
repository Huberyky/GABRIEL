"""Multi-provider routing registry for GABRIEL.

Providers are identified by a slash-prefixed model name convention::

    "gpt-5-mini"                  → OpenAI (default, no prefix needed)
    "deepseek/deepseek-chat"      → DeepSeek
    "qwen/qwen-plus"              → Alibaba Qwen (DashScope)
    "ollama/llama3.2"             → Local Ollama instance
    "openrouter/meta-llama/..."   → OpenRouter unified gateway
    "lmstudio/my-local-model"     → LM Studio local server

Any provider not listed in :data:`PROVIDER_CONFIGS` is treated as a
custom OpenAI-compatible endpoint.  In that case the API key is read from
an environment variable named ``<PROVIDER_UPPER>_API_KEY`` and you must
supply a ``base_url`` manually (or set ``<PROVIDER_UPPER>_BASE_URL``).

Environment variables
---------------------
Each built-in provider reads its API key from a dedicated variable:

* OpenAI      → ``OPENAI_API_KEY``
* DeepSeek    → ``DEEPSEEK_API_KEY``
* Qwen        → ``DASHSCOPE_API_KEY``
* Ollama      → ``OLLAMA_API_KEY`` (optional; falls back to ``"ollama"``)
* OpenRouter  → ``OPENROUTER_API_KEY``
* LM Studio   → ``LMSTUDIO_API_KEY`` (optional; falls back to ``"lmstudio"``)

Base URLs can also be overridden per provider via
``<PROVIDER_UPPER>_BASE_URL`` (e.g. ``OLLAMA_BASE_URL`` to point at a
remote Ollama instance).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

#: Configuration for each known provider.
#:
#: Keys per entry:
#:   base_url            Default API base URL (``None`` → use SDK default)
#:   api_key_env         Environment variable that holds the API key
#:   api_key_default     Fallback key when the env-var is absent (e.g. local servers)
#:   use_chat_completions  ``True`` → use Chat Completions endpoint;
#:                          ``False`` → use OpenAI Responses API (OpenAI only)
#:   supports_batch      Whether the provider supports OpenAI Batch API mode
#:   description         Human-readable description shown in log output
PROVIDER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "api_key_default": None,
        "use_chat_completions": False,  # Uses newer Responses API
        "supports_batch": True,
        "description": "OpenAI (GPT-5, GPT-4o, o3, o4, etc.)",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "api_key_default": None,
        "use_chat_completions": True,
        "supports_batch": False,
        "description": "DeepSeek (deepseek-chat, deepseek-reasoner, etc.)",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "api_key_default": None,
        "use_chat_completions": True,
        "supports_batch": False,
        "description": "Alibaba Qwen via DashScope (qwen-plus, qwen-turbo, qwen-max, etc.)",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "OLLAMA_API_KEY",
        "api_key_default": "ollama",  # Placeholder required by the OpenAI SDK
        "use_chat_completions": True,
        "supports_batch": False,
        "description": "Ollama local server (llama3.2, mistral, phi4, gemma3, etc.)",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_key_default": None,
        "use_chat_completions": True,
        "supports_batch": False,
        "description": "OpenRouter unified gateway (Claude, Gemini, Llama, Mistral, etc.)",
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key_env": "LMSTUDIO_API_KEY",
        "api_key_default": "lmstudio",  # Placeholder required by the OpenAI SDK
        "use_chat_completions": True,
        "supports_batch": False,
        "description": "LM Studio local server",
    },
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def parse_model_provider(model: str) -> Tuple[str, str]:
    """Split a ``provider/model`` string into ``(provider, bare_model)``.

    When *model* contains no ``/`` separator it is assumed to belong to the
    ``"openai"`` provider and is returned as-is.

    Examples::

        parse_model_provider("gpt-5-mini")
        # → ("openai", "gpt-5-mini")

        parse_model_provider("deepseek/deepseek-chat")
        # → ("deepseek", "deepseek-chat")

        parse_model_provider("qwen/qwen-plus")
        # → ("qwen", "qwen-plus")

        parse_model_provider("ollama/llama3.2")
        # → ("ollama", "llama3.2")

        parse_model_provider("openrouter/meta-llama/llama-3-8b-instruct")
        # → ("openrouter", "meta-llama/llama-3-8b-instruct")
    """
    if "/" not in model:
        return "openai", model
    provider, rest = model.split("/", 1)
    return provider.lower(), rest


def get_provider_config(provider: str) -> Dict[str, Any]:
    """Return the config dict for *provider*, falling back to a generic entry."""
    return PROVIDER_CONFIGS.get(
        provider,
        {
            "base_url": os.getenv(f"{provider.upper()}_BASE_URL"),
            "api_key_env": f"{provider.upper()}_API_KEY",
            "api_key_default": None,
            "use_chat_completions": True,
            "supports_batch": False,
            "description": f"Custom provider: {provider}",
        },
    )


def get_provider_api_key(provider: str) -> Optional[str]:
    """Return the API key for *provider* from environment variables.

    Resolution order:

    1. Environment variable named by ``api_key_env`` in :data:`PROVIDER_CONFIGS`.
    2. ``api_key_default`` from :data:`PROVIDER_CONFIGS` (used for local servers
       that require a non-empty but otherwise arbitrary key).
    3. ``None`` (caller should raise an error if the key is required).
    """
    config = get_provider_config(provider)
    env_var = config.get("api_key_env")
    if env_var:
        key = os.getenv(env_var)
        if key:
            return key
    default = config.get("api_key_default")
    if default:
        return default
    return None


def get_provider_base_url(provider: str, override: Optional[str] = None) -> Optional[str]:
    """Return the effective base URL for *provider*.

    Priority:

    1. Explicit *override* argument.
    2. ``<PROVIDER_UPPER>_BASE_URL`` environment variable.
    3. ``base_url`` from :data:`PROVIDER_CONFIGS`.
    4. ``OPENAI_BASE_URL`` env-var (only for the ``"openai"`` provider).
    """
    if override:
        return override
    env_override = os.getenv(f"{provider.upper()}_BASE_URL")
    if env_override:
        return env_override
    config = get_provider_config(provider)
    url = config.get("base_url")
    if url:
        return url
    if provider == "openai":
        return os.getenv("OPENAI_BASE_URL")
    return None


def uses_chat_completions(provider: str) -> bool:
    """Return ``True`` when *provider* uses the Chat Completions API.

    OpenAI uses the newer Responses API by default; all other providers use
    the Chat Completions endpoint (which is widely supported by OpenAI-
    compatible servers).
    """
    return bool(get_provider_config(provider).get("use_chat_completions", True))


def supports_batch(provider: str) -> bool:
    """Return ``True`` when the provider supports OpenAI Batch API mode."""
    return bool(get_provider_config(provider).get("supports_batch", False))


def list_providers() -> Dict[str, str]:
    """Return a ``{provider_name: description}`` mapping for all built-in providers."""
    return {name: cfg["description"] for name, cfg in PROVIDER_CONFIGS.items()}
