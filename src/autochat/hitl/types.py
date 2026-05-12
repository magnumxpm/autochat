from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

HITLKind = Literal[
    "tool_approval",
    "retriever_approval",
    "user_question",
    "custom",
]


class ApprovalResponse(BaseModel):
    """Default response shape for approval HITL gates."""

    approved: bool


class UserQuestionResponse(BaseModel):
    """Response shape for model-asked user questions."""

    answer: str


class HITLRequest(BaseModel):
    """A pending HITL interaction surfaced to the host application.

    Carried on `HITLRequestedEvent` and produced by `ApprovalSpec` gates or
    by `runtime.hitl.request_approval` / `runtime.hitl.ask_user`.
    """

    request_id: str
    kind: HITLKind
    summary: str | None = None
    payload: Mapping[str, Any] = Field(default_factory=dict)
    response_schema: type[BaseModel] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
