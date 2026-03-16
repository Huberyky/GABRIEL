"""GABRIEL: LLM-based social science analysis toolkit.

Multi-model support
-------------------
GABRIEL supports any OpenAI-compatible backend through a provider prefix
convention on the model name::

    # OpenAI (default – no prefix needed)
    await gabriel.rate(df, "text", attributes={...}, model="gpt-5-mini", ...)

    # DeepSeek  (set DEEPSEEK_API_KEY)
    await gabriel.rate(df, "text", attributes={...}, model="deepseek/deepseek-chat", ...)

    # Alibaba Qwen  (set DASHSCOPE_API_KEY)
    await gabriel.rate(df, "text", attributes={...}, model="qwen/qwen-plus", ...)

    # Local Ollama  (no API key required)
    await gabriel.rate(df, "text", attributes={...}, model="ollama/llama3.2", ...)

    # OpenRouter  (set OPENROUTER_API_KEY)
    await gabriel.rate(df, "text", attributes={...}, model="openrouter/meta-llama/llama-3-8b-instruct", ...)

Use :class:`ModelClient` for lower-level access or to list all supported
providers::

    gabriel.ModelClient.list_providers()
"""

from importlib.metadata import PackageNotFoundError, version as _v

from . import tasks as _tasks
from .api import (
    rate,
    classify,
    extract,
    deidentify,
    rank,
    codify,
    paraphrase,
    compare,
    discover,
    deduplicate,
    merge,
    filter,
    debias,
    ideate,
    id8,
    whatever,
    view,
    bucket,
    seed,
    poll,
)
from .utils import load
from .core.llm_client import ModelClient, OpenAIClient
from .core.providers import (
    PROVIDER_CONFIGS,
    get_provider_api_key,
    get_provider_base_url,
    get_provider_config,
    list_providers,
    parse_model_provider,
    supports_batch,
    uses_chat_completions,
)

try:
    __version__ = _v("gabriel")
except PackageNotFoundError:  # pragma: no cover - package not installed
    from ._version import __version__

__all__ = list(_tasks.__all__) + [
    "rate",
    "classify",
    "extract",
    "deidentify",
    "rank",
    "codify",
    "paraphrase",
    "compare",
    "discover",
    "seed",
    "poll",
    "deduplicate",
    "merge",
    "filter",
    "debias",
    "ideate",
    "id8",
    "whatever",
    "view",
    "bucket",
    "load",
    # Multi-model support
    "ModelClient",
    "OpenAIClient",
    "PROVIDER_CONFIGS",
    "list_providers",
    "parse_model_provider",
    "get_provider_config",
    "get_provider_api_key",
    "get_provider_base_url",
    "supports_batch",
    "uses_chat_completions",
]


def __getattr__(name: str):
    if name in _tasks.__all__:
        return getattr(_tasks, name)
    raise AttributeError(name)
