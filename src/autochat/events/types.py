from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Mapping, Union

from pydantic import BaseModel, Field

from autochat.hitl import HITLRequest
from autochat.retrieval import RetrievedDocument

from .message import AssistantMessage, ThinkingBlock


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _EventBase(BaseModel):
    """Common fields for all AutoChat events"""

    run_id: str
    thread_id: str
    timestamp: datetime = Field(default_factory=_utcnow)

    model_config = {"arbitrary_types_allowed": True}


class RunStartEvent(_EventBase):
    type: Literal["run.start"] = "run.start"
    input: str


class RunEndEvent(_EventBase):
    type: Literal["run.end"] = "run.end"


class MessageStartEvent(_EventBase):
    type: Literal["message.start"] = "message.start"
    message_id: str


class MessageDeltaEvent(_EventBase):
    type: Literal["message.delta"] = "message.delta"
    message_id: str
    delta: str


class MessageEndEvent(_EventBase):
    type: Literal["message.end"] = "message.end"
    message: AssistantMessage


class ThinkingStartEvent(_EventBase):
    type: Literal["thinking.start"] = "thinking.start"
    block_id: str


class ThinkingDeltaEvent(_EventBase):
    type: Literal["thinking.delta"] = "thinking.delta"
    block_id: str
    delta: str


class ThinkingEndEvent(_EventBase):
    type: Literal["thinking.end"] = "thinking.end"
    block: ThinkingBlock


class ToolCallRequestEvent(_EventBase):
    type: Literal["tool.request"] = "tool.request"
    tool_call_id: str
    name: str
    args: Mapping[str, Any]


class ToolCallResponseEvent(_EventBase):
    type: Literal["tool.response"] = "tool.response"
    tool_call_id: str
    name: str
    result: Any | None = None
    error: str | None = None


class RetrieverRequestEvent(_EventBase):
    type: Literal["retriever.request"] = "retriever.request"
    tool_call_id: str
    name: str
    query: str


class RetrieverResponseEvent(_EventBase):
    type: Literal["retriever.response"] = "retriever.response"
    tool_call_id: str
    name: str
    documents: list[RetrievedDocument]
    error: str | None = None


class CompressionStartEvent(_EventBase):
    type: Literal["compression.start"] = "compression.start"
    strategy: str
    message_count_before: int


class CompressionEndEvent(_EventBase):
    type: Literal["compression.end"] = "compression.end"
    strategy: str
    compressed: bool
    message_count_after: int


class HITLRequestedEvent(_EventBase):
    type: Literal["hitl.requested"] = "hitl.requested"
    request: HITLRequest


class HITLResolvedEvent(_EventBase):
    type: Literal["hitl.resolved"] = "hitl.resolved"
    request_id: str
    response: Mapping[str, Any]


class ErrorEvent(_EventBase):
    type: Literal["error"] = "error"
    node: str
    error: str


# each event chunk will be an AutoChatEvent type
AutoChatEvent = Annotated[
    Union[
        RunStartEvent,
        RunEndEvent,
        MessageStartEvent,
        MessageDeltaEvent,
        MessageEndEvent,
        ThinkingStartEvent,
        ThinkingDeltaEvent,
        ThinkingEndEvent,
        ToolCallRequestEvent,
        ToolCallResponseEvent,
        RetrieverRequestEvent,
        RetrieverResponseEvent,
        CompressionStartEvent,
        CompressionEndEvent,
        HITLRequestedEvent,
        HITLResolvedEvent,
        ErrorEvent,
    ],
    Field(discriminator="type"),
]
