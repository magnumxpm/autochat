import warnings
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from langchain.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from autochat.config import ChatConfig
from autochat.runtime import ChatRuntime

from .strategies import SummarizeAll
from .types import CompressionStrategy

TContext = TypeVar("TContext")


def _message_content_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


# TODO: Check if this approach is good
def _approximate_message_tokens(messages: list[BaseMessage]) -> int:
    # Rough fallback used when the model cannot count tokens without extra deps.
    return max(1, sum(len(_message_content_text(message)) for message in messages) // 4)


def _count_message_tokens(
    model: BaseChatModel,
    messages: list[BaseMessage],
) -> int:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Using fallback GPT-2 tokenizer.*",
                category=UserWarning,
            )
            return model.get_num_tokens_from_messages(messages)
    except (ImportError, ValueError, NotImplementedError):
        return _approximate_message_tokens(messages)


@dataclass(frozen=True, slots=True)
class AutoCompress(Generic[TContext]):
    """Automatic chat-history compression policy."""

    at: float = 0.6
    strategy: CompressionStrategy[TContext] = field(
        default_factory=lambda: cast(CompressionStrategy[TContext], SummarizeAll())
    )
    context_window: int | None = None
    model: BaseChatModel | None = None

    def __post_init__(self) -> None:
        if not 0 < self.at <= 1:
            raise ValueError(
                "AutoCompress.at must be greater than 0 and less than or equal to 1."
            )

    async def acompress_if_needed(
        self,
        messages: list[BaseMessage],
        *,
        runtime: ChatRuntime[TContext],
        config: ChatConfig,
    ) -> list[BaseMessage] | None:
        context_window = self.context_window or config.context_window
        if context_window is None:
            raise ValueError(
                "AutoCompress requires ChatConfig.context_window or "
                "AutoCompress.context_window."
            )

        model = self.model or config.model
        used_tokens = _count_message_tokens(model, messages)
        if used_tokens < context_window * self.at:
            return None

        return await self.strategy.acompress(
            messages,
            runtime=runtime,
            model=model,
        )
