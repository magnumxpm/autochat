from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar

from langchain.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from autochat.runtime import ChatRuntime

TContext = TypeVar("TContext")


class CompressionStrategy(Protocol[TContext]):
    """Strategy used by AutoCompress to compact chat history."""

    async def acompress(
        self,
        messages: Sequence[BaseMessage],
        *,
        runtime: ChatRuntime[TContext],
        model: BaseChatModel,
    ) -> list[BaseMessage]: ...
