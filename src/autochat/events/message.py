from dataclasses import dataclass, field
from typing import Any, Mapping

from langchain_core.messages import AIMessage


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    """A single chain-of-thought / reasoning block extracted from model response"""

    id: str
    text: str
    signature: str | None = None
    redacted: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolCallSpec:
    """Tool call requested by the model"""

    id: str
    name: str
    args: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class UsageInfo:
    """Usage information of the model run"""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """AutoChat assistant message"""

    id: str
    content: str
    thinking: tuple[ThinkingBlock, ...] = ()
    tool_calls: tuple[ToolCallSpec, ...] = ()
    usage: UsageInfo | None = None

    @classmethod
    def from_ai_message(
        cls,
        message: AIMessage,
        thinking: tuple[ThinkingBlock, ...] = (),
    ) -> "AssistantMessage":
        text = _extract_text(message)

        tool_calls = tuple(
            ToolCallSpec(
                id=call.get("id") or "",
                name=call.get("name") or "",
                args=dict(call.get("args") or {}),
            )
            for call in (message.tool_calls or [])
        )

        usage = _extract_usage(message)
        return cls(
            id=message.id or "",
            content=text,
            thinking=tuple(thinking or ()),
            tool_calls=tool_calls,
            usage=usage,
        )


def _extract_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)

        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")

    return "".join(parts)


def _extract_usage(message: AIMessage) -> UsageInfo | None:
    metadata = getattr(message, "usage_metadata", None) or {}
    if not metadata:
        return None

    return UsageInfo(
        input_tokens=metadata.get("input_tokens"),
        output_tokens=metadata.get("output_tokens"),
        total_tokens=metadata.get("total_tokens"),
    )
