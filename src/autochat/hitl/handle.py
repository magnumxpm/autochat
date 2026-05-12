from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar
from uuid import uuid4

from langgraph.types import interrupt
from pydantic import BaseModel

from .spec import (
    ApprovalSpec,
    resolve_payload,
    resolve_summary,
)
from .types import ApprovalResponse, HITLRequest, UserQuestionResponse

TResponse = TypeVar("TResponse", bound=BaseModel)


def new_request_id() -> str:
    return f"hitl_{uuid4().hex[:12]}"


@dataclass(slots=True)
class HITLHandle:
    """Per-invocation handle exposed on `ChatRuntime.hitl`.

    Tools and retrievers use this to:
      - read `approval` (the validated response from a declarative gate, if any)
      - call `request_approval(...)` to issue a custom approval interrupt
      - call `ask_user(...)` to ask the end-user a free-form question

    `runtime.hitl` is `None` when AutoChat was not configured with `hitl=...`,
    so tool code should branch on `if runtime.hitl is not None`.
    """

    allow_questions: bool = False
    approval: BaseModel | None = field(default=None)

    async def request_approval(
        self,
        *,
        summary: str | None = None,
        response_schema: type[TResponse] = ApprovalResponse,  # type: ignore[assignment]
        payload: Mapping[str, Any] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> TResponse:
        """Issue a custom approval interrupt from inside a tool or retriever.

        Returns a validated instance of `response_schema`. Raises `GraphInterrupt`
        on the first call (LangGraph halts the graph); on resume returns the
        validated response.
        """
        from autochat.events.dispatch import (
            emit_hitl_requested,
            emit_hitl_resolved,
        )

        request = HITLRequest(
            request_id=new_request_id(),
            kind="custom",
            summary=summary,
            payload=dict(payload or {}),
            response_schema=response_schema,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )
        await emit_hitl_requested(request)
        raw = interrupt(serialize_request(request))
        response = response_schema.model_validate(raw)
        await emit_hitl_resolved(request.request_id, response.model_dump())
        return response

    async def ask_user(
        self,
        question: str,
        *,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> str:
        """Ask the end-user a free-form question and return their reply."""
        from autochat.events.dispatch import (
            emit_hitl_requested,
            emit_hitl_resolved,
        )

        request = HITLRequest(
            request_id=new_request_id(),
            kind="user_question",
            summary=question,
            payload={"question": question},
            response_schema=UserQuestionResponse,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )
        await emit_hitl_requested(request)
        raw = interrupt(serialize_request(request))
        response = UserQuestionResponse.model_validate(raw)
        await emit_hitl_resolved(request.request_id, response.model_dump())
        return response.answer


async def request_declarative_approval(
    spec: ApprovalSpec,
    invocation: Any,
    *,
    kind: str,
    tool_call_id: str | None,
    tool_name: str | None,
) -> BaseModel:
    """Helper used by graph nodes to gate a tool/retriever with an `ApprovalSpec`.

    Issues the interrupt, validates the response against the spec's schema, and
    returns the validated model. The caller decides whether to invoke the
    underlying tool based on `spec.decide(response)`.
    """
    from autochat.events.dispatch import emit_hitl_requested, emit_hitl_resolved

    request = HITLRequest(
        request_id=new_request_id(),
        kind=kind,  # type: ignore[arg-type]
        summary=resolve_summary(spec, invocation),
        payload=dict(resolve_payload(spec, invocation)),
        response_schema=spec.response_schema,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
    )
    await emit_hitl_requested(request)
    raw = interrupt(serialize_request(request))
    response = spec.response_schema.model_validate(raw)
    await emit_hitl_resolved(request.request_id, response.model_dump())
    return response


def serialize_request(request: HITLRequest) -> dict[str, Any]:
    """Render an HITLRequest into a JSON-friendly dict for the interrupt payload.

    The response schema class itself can't be checkpointed; we send its name and
    JSON Schema so hosts that need to reconstruct field definitions can.
    """
    schema_info: dict[str, Any] | None = None
    if request.response_schema is not None:
        schema_info = {
            "name": request.response_schema.__name__,
            "schema": request.response_schema.model_json_schema(),
        }
    return {
        "request_id": request.request_id,
        "kind": request.kind,
        "summary": request.summary,
        "payload": dict(request.payload),
        "response_schema": schema_info,
        "tool_call_id": request.tool_call_id,
        "tool_name": request.tool_name,
    }
