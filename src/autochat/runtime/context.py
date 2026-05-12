from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from autochat.hitl import HITLHandle

TContext = TypeVar("TContext")


@dataclass(frozen=True, slots=True)
class ChatRuntime(Generic[TContext]):
    """
    Invocation-scoped runtime context.
    Runtime can hold request-local objects agnostic to graph state.

    `hitl` is the per-run HITL handle. It is `None` when AutoChat was not
    configured with `hitl=...`; tools and retrievers should branch on
    `if runtime.hitl is not None` before issuing HITL requests.
    """

    thread_id: str
    context: TContext
    run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    hitl: "HITLHandle | None" = None
