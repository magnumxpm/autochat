from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    """Normalized document returned by an AutoChat retriever."""

    content: str
    retriever_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    score: float | None = None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    documents: list[RetrievedDocument]
