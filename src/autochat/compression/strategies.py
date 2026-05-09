from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from langchain.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage

from autochat.runtime import ChatRuntime

TContext = TypeVar("TContext")

_SUMMARY_METADATA = {
    "autochat": {
        "kind": "compression_summary",
    }
}


def _summary_message(summary: str) -> SystemMessage:
    return SystemMessage(
        content=f"Previous conversation summary:\n{summary}",
        additional_kwargs=_SUMMARY_METADATA,
    )


def _replace_messages(
    messages: Sequence[BaseMessage],
) -> list[BaseMessage]:
    return [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        *messages,
    ]


def _replace_with_summary_between(
    messages_before: Sequence[BaseMessage],
    summary: str,
    messages_after: Sequence[BaseMessage],
) -> list[BaseMessage]:
    return _replace_messages(
        [
            *messages_before,
            _summary_message(summary),
            *messages_after,
        ]
    )


def _replace_with_latest(
    messages_to_keep: Sequence[BaseMessage],
) -> list[BaseMessage]:
    return _replace_messages(messages_to_keep)


def _split_preserved_tail(
    messages: Sequence[BaseMessage],
    preserve_latest: int,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    message_list = list(messages)
    if preserve_latest <= 0:
        return message_list, []
    return message_list[:-preserve_latest], message_list[-preserve_latest:]


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def _format_messages_for_summary(messages: Sequence[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.type
        name = getattr(message, "name", None)
        label = f"{role}:{name}" if name else role
        lines.append(f"{label}: {_message_text(message)}")
    return "\n".join(lines)


async def _summarize_messages(
    messages: Sequence[BaseMessage],
    *,
    model: BaseChatModel,
) -> str:
    prompt = (
        "Summarize this conversation for future continuation. "
        "Preserve user goals, decisions, constraints, tool results, and unresolved tasks. "
        "Do not add facts that are not present.\n\n"
        f"Conversation:\n{_format_messages_for_summary(messages)}"
    )
    response = await model.ainvoke([HumanMessage(content=prompt)])
    content = response.content
    if isinstance(content, str):
        return content
    return str(content)


@dataclass(frozen=True, slots=True)
class KeepLatestN(Generic[TContext]):
    """Drop older history and keep only the latest N messages."""

    n: int

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("KeepLatestN.n must be at least 1.")

    async def acompress(
        self,
        messages: Sequence[BaseMessage],
        *,
        runtime: ChatRuntime[TContext],
        model: BaseChatModel,
    ) -> list[BaseMessage]:
        del runtime, model
        return _replace_with_latest(list(messages)[-self.n :])


@dataclass(frozen=True, slots=True)
class SummarizeAll(Generic[TContext]):
    """Summarize history into one summary and preserve the active latest messages."""

    preserve_latest: int = 1

    def __post_init__(self) -> None:
        if self.preserve_latest < 0:
            raise ValueError("SummarizeAll.preserve_latest cannot be negative.")

    async def acompress(
        self,
        messages: Sequence[BaseMessage],
        *,
        runtime: ChatRuntime[TContext],
        model: BaseChatModel,
    ) -> list[BaseMessage]:
        del runtime
        messages_to_summarize, messages_to_keep = _split_preserved_tail(
            messages,
            self.preserve_latest,
        )
        if not messages_to_summarize:
            return _replace_with_latest(messages_to_keep)

        summary = await _summarize_messages(messages_to_summarize, model=model)
        return _replace_with_summary_between([], summary, messages_to_keep)


@dataclass(frozen=True, slots=True)
class SummarizeLatestN(Generic[TContext]):
    """Summarize the latest N history messages and preserve the active latest messages."""

    n: int
    preserve_latest: int = 1

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("SummarizeLatestN.n must be at least 1.")
        if self.preserve_latest < 0:
            raise ValueError("SummarizeLatestN.preserve_latest cannot be negative.")

    async def acompress(
        self,
        messages: Sequence[BaseMessage],
        *,
        runtime: ChatRuntime[TContext],
        model: BaseChatModel,
    ) -> list[BaseMessage]:
        del runtime
        history_messages, preserved_messages = _split_preserved_tail(
            messages,
            self.preserve_latest,
        )
        messages_to_summarize = history_messages[-self.n :]
        messages_to_keep = history_messages[: -self.n]
        if not messages_to_summarize:
            return _replace_with_latest(preserved_messages)

        summary = await _summarize_messages(messages_to_summarize, model=model)
        return _replace_with_summary_between(
            messages_to_keep,
            summary,
            preserved_messages,
        )
