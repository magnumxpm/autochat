from dataclasses import dataclass
from typing import Generic, TypeVar

from .strategies import RAGStrategy

TContext = TypeVar("TContext")


@dataclass(frozen=True, slots=True)
class RetrievalConfig(Generic[TContext]):
    strategy: RAGStrategy[TContext] | None = None
    max_context_chars: int = 12_000
    include_sources: bool = True
    context_header: str = "Relevant retrieved context:"
