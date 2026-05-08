from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeAlias, TypeVar

from langchain_core.documents import Document

from autochat.runtime import ChatRuntime

from .documents import RetrievedDocument

T = TypeVar("T")
TContext = TypeVar("TContext")

MaybeAwaitable: TypeAlias = T | Awaitable[T]
RetrieverDocument: TypeAlias = RetrievedDocument | Document | str


@dataclass(frozen=True, slots=True)
class RetrievalInvocation(Generic[TContext]):
    """Data passed to retriever processors."""

    name: str
    query: str
    runtime: ChatRuntime[TContext]
    raw_retriever: Any


class RetrieverFn(Protocol[TContext]):
    def __call__(
        self,
        query: str,
        runtime: ChatRuntime[TContext],
    ) -> MaybeAwaitable[Sequence[RetrieverDocument]]: ...


class RetrieverPreprocessor(Protocol[TContext]):
    def __call__(
        self,
        invocation: RetrievalInvocation[TContext],
    ) -> MaybeAwaitable[str]: ...


class RetrieverPostprocessor(Protocol[TContext]):
    def __call__(
        self,
        invocation: RetrievalInvocation[TContext],
        documents: list[RetrievedDocument],
    ) -> MaybeAwaitable[list[RetrievedDocument]]: ...
