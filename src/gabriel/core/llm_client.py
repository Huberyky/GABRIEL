"""Abstract LLM client interface and provider-agnostic implementation."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os

from ..utils.openai_utils import get_all_responses, get_response
from .providers import (
    get_provider_api_key,
    get_provider_base_url,
    get_provider_config,
    list_providers,
    parse_model_provider,
)


class LLMClient(ABC):
    """Minimal interface for language model providers."""

    @abstractmethod
    async def acall(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Asynchronously call the LLM with provided messages."""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """Concrete LLM client leveraging :func:`get_response`.

    .. deprecated::
        Prefer :class:`ModelClient` which supports any provider via model name
        prefixes (e.g. ``"deepseek/deepseek-chat"``).  ``OpenAIClient`` is kept
        for backwards compatibility.
    """

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None
    ) -> None:
        # ``get_response`` manages a shared client; API key via env or arg
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        if base_url:
            os.environ.setdefault("OPENAI_BASE_URL", base_url)

    async def acall(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        prompt = "\n".join(m.get("content", "") for m in messages)
        responses, _ = await get_response(prompt, **kwargs)
        return responses[0] if responses else ""

    async def get_all_responses(self, **kwargs: Any):
        """Convenience wrapper around :func:`get_all_responses`."""
        return await get_all_responses(**kwargs)


class ModelClient(LLMClient):
    """Provider-agnostic LLM client supporting any OpenAI-compatible backend.

    Providers are identified by a slash-prefixed model name convention::

        ModelClient()                            # uses default model
        ModelClient(model="gpt-5-mini")          # OpenAI (default)
        ModelClient(model="deepseek/deepseek-chat")   # DeepSeek
        ModelClient(model="qwen/qwen-plus")      # Alibaba Qwen
        ModelClient(model="ollama/llama3.2")     # Local Ollama
        ModelClient(model="openrouter/meta-llama/llama-3-8b-instruct")

    When no *api_key* is supplied the client reads from the matching
    ``<PROVIDER_UPPER>_API_KEY`` environment variable (e.g.
    ``DEEPSEEK_API_KEY`` for DeepSeek models).

    Parameters
    ----------
    model:
        Default model to use for :meth:`acall` and :meth:`get_all_responses`
        when no ``model`` kwarg is supplied at call time.
    api_key:
        Optional explicit API key.  When provided it is stored in the
        appropriate environment variable for the detected provider.
    base_url:
        Optional API base URL override.  When provided it takes precedence
        over the built-in default for the detected provider.

    Examples
    --------
    >>> import asyncio, gabriel
    >>> client = gabriel.ModelClient(model="deepseek/deepseek-chat")
    >>> text = asyncio.run(client.acall([{"role": "user", "content": "Hello!"}]))

    >>> import asyncio, gabriel
    >>> client = gabriel.ModelClient(model="ollama/llama3.2")
    >>> text = asyncio.run(client.acall([{"role": "user", "content": "Hello!"}]))
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        provider, _ = parse_model_provider(model)
        cfg = get_provider_config(provider)
        key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
        if api_key:
            os.environ[key_env] = api_key
        if base_url:
            os.environ[f"{provider.upper()}_BASE_URL"] = base_url

    async def acall(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Call the configured model with the given message list.

        The ``messages`` list follows the standard OpenAI chat format::

            [{"role": "user", "content": "Hello!"}, ...]

        Only the textual content of each message is concatenated and sent as a
        single prompt.  For structured multi-turn conversations prefer using
        :func:`~gabriel.utils.openai_utils.get_response` directly.
        """
        prompt = "\n".join(m.get("content", "") for m in messages)
        kwargs.setdefault("model", self.model)
        responses, _ = await get_response(prompt, **kwargs)
        return responses[0] if responses else ""

    async def get_all_responses(self, **kwargs: Any):
        """Convenience wrapper around :func:`~gabriel.utils.openai_utils.get_all_responses`.

        Injects the client's default model when ``model`` is not supplied via
        ``kwargs``.
        """
        kwargs.setdefault("model", self.model)
        return await get_all_responses(**kwargs)

    @staticmethod
    def list_providers() -> Dict[str, str]:
        """Return a ``{provider_name: description}`` mapping of all built-in providers.

        Example::

            >>> gabriel.ModelClient.list_providers()
            {
                'openai':     'OpenAI (GPT-5, GPT-4o, o3, o4, etc.)',
                'deepseek':   'DeepSeek (deepseek-chat, deepseek-reasoner, etc.)',
                'qwen':       'Alibaba Qwen via DashScope ...',
                'ollama':     'Ollama local server ...',
                'openrouter': 'OpenRouter unified gateway ...',
                'lmstudio':   'LM Studio local server',
            }
        """
        return list_providers()
