from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from autochat.exceptions import ApprovalDecisionUndefinedError

from .types import ApprovalResponse

TResponse = TypeVar("TResponse", bound=BaseModel)

SummaryFn = Callable[[Any], str]
PayloadFn = Callable[[Any], Mapping[str, Any]]
DecideFn = Callable[[Any], bool]
DenialMessageFn = Callable[[Any], str]


@dataclass(frozen=True, slots=True)
class ApprovalSpec(Generic[TResponse]):
    """Declarative HITL approval gate for a tool or retriever.

    `summary` and `payload` describe the pending action to the user, either as
    static values or as callables that receive the in-flight invocation.

    `response_schema` is the pydantic model the host returns; it defaults to
    `ApprovalResponse` (a single `approved: bool` field).

    `decide` interprets the response as approve/deny. When omitted, it is
    inferred from the schema:
      1. If the schema has an `approved: bool` field, use it.
      2. Else if the schema has exactly one bool field, use that field.
      3. Else raise `ApprovalDecisionUndefinedError` at construction.

    `denial_message` overrides the `ToolMessage` content sent back to the
    model when the user denies the action. May be a string or a callable
    receiving the validated response.
    """

    summary: str | SummaryFn | None = None
    payload: Mapping[str, Any] | PayloadFn | None = None
    response_schema: type[BaseModel] = field(default=ApprovalResponse)
    decide: DecideFn | None = None
    denial_message: str | DenialMessageFn | None = None

    def __post_init__(self) -> None:
        if self.decide is None:
            object.__setattr__(self, "decide", _derive_decide(self.response_schema))


def _derive_decide(schema: type[BaseModel]) -> DecideFn:
    fields = schema.model_fields
    approved = fields.get("approved")
    if approved is not None and approved.annotation is bool:
        return lambda r: bool(getattr(r, "approved"))

    bool_fields = [name for name, info in fields.items() if info.annotation is bool]
    if len(bool_fields) == 1:
        name = bool_fields[0]
        return lambda r: bool(getattr(r, name))

    raise ApprovalDecisionUndefinedError(
        f"Cannot infer `decide` for ApprovalSpec(response_schema={schema.__name__}). "
        f"Provide an explicit `decide=` callable, add an `approved: bool` field, "
        f"or reduce the schema to a single bool field."
    )


def resolve_summary(spec: ApprovalSpec, invocation: Any) -> str | None:
    if callable(spec.summary):
        return spec.summary(invocation)
    return spec.summary


def resolve_payload(spec: ApprovalSpec, invocation: Any) -> Mapping[str, Any]:
    if spec.payload is None:
        return {}
    if callable(spec.payload):
        return spec.payload(invocation)
    return spec.payload


def resolve_denial_message(spec: ApprovalSpec, response: BaseModel) -> str:
    if spec.denial_message is None:
        return f"User declined: {response.model_dump()}"
    if callable(spec.denial_message):
        return spec.denial_message(response)
    return spec.denial_message


def normalize_approval(
    approval: ApprovalSpec | bool | None,
) -> ApprovalSpec | None:
    """Coerce the `approval=` parameter on tools/retrievers into an `ApprovalSpec`.

    `True` → default `ApprovalSpec()` (Approve/Deny with no summary).
    `False` / `None` → no gate.
    `ApprovalSpec` → passed through.
    """
    if approval is None or approval is False:
        return None
    if approval is True:
        return ApprovalSpec()
    return approval
