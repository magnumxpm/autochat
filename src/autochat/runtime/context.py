from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

TContext = TypeVar("TContext")


@dataclass(frozen=True, slots=True)
class ChatRuntime(Generic[TContext]):
    """
    Invocation-scoped runtime context.
    Runtime can hold request-local objects agnostic to graph state
    """

    thread_id: str
    context: TContext
    run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
