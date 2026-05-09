import re
from typing import Callable

from langchain_core.language_models import BaseChatModel

from .anthropic import AnthropicThinkingExtractor
from .base import ThinkingExtractor
from .deepseek import DeepSeekThinkingExtractor
from .noop import NoOpExtractor
from .openai import OpenAIReasoningExtractor

_OPENAI_REASONING_PATTERN = re.compile(
    r"^(o[1-9](?:-|$)|gpt-5|gpt-o[1-9])", re.IGNORECASE
)
_OVERRIDES: list[tuple[Callable[[BaseChatModel], bool], ThinkingExtractor]] = []


def register_thinking_extractor(
    predicate: Callable[[BaseChatModel], bool],
    extractor: ThinkingExtractor,
) -> None:
    """Resgisters a custom extractor for CoT/reasoning events. Latest take priority"""

    _OVERRIDES.insert(0, (predicate, extractor))


def _model_name(model: BaseChatModel) -> str:
    name = getattr(model, "model_name", None) or getattr(model, "model", None) or ""
    return str(name)


def _is_openai_reasoning_model(model: BaseChatModel) -> bool:
    """Necessary for filtering out non-OpenAI reasoning models -> send them to no-op"""

    if getattr(model, "reasoning_effort", None) is not None:
        return True
    if getattr(model, "reasoning", None) is not None:
        return True
    return bool(_OPENAI_REASONING_PATTERN.match(_model_name(model)))


def resolve_extractor(model: BaseChatModel) -> ThinkingExtractor:
    for predicate, extractor in _OVERRIDES:
        try:
            if predicate(model):
                return extractor
        except Exception:
            continue

    module = type(model).__module__ or ""

    if module.startswith("langchain_anthropic"):
        return AnthropicThinkingExtractor()

    if module.startswith("langchain_openai"):
        if _is_openai_reasoning_model(model):
            return OpenAIReasoningExtractor()
        return NoOpExtractor()

    model_name = _model_name(model)
    if "deepseek" in module.lower() or (
        isinstance(model_name, str) and "deepseek" in model_name.lower()
    ):
        return DeepSeekThinkingExtractor()

    return NoOpExtractor()
