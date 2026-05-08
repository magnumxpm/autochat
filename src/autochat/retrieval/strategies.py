import asyncio
from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar

from autochat.runtime import ChatRuntime

from .base import ChatRetriever
from .documents import RetrievalResult

TContext = TypeVar("TContext")


class RAGStrategy(Protocol[TContext]):
    async def aretrieve(
        self,
        query: str,
        *,
        runtime: ChatRuntime[TContext],
        retrievers: Sequence[ChatRetriever[TContext]],
    ) -> RetrievalResult: ...


class SimpleRAGStrategy(Generic[TContext]):
    """Run all retrievers in parallel and concatenate their results."""

    async def aretrieve(
        self,
        query: str,
        *,
        runtime: ChatRuntime[TContext],
        retrievers: Sequence[ChatRetriever[TContext]],
    ) -> RetrievalResult:
        batches = await asyncio.gather(
            *(retriever.aretrieve(query, runtime) for retriever in retrievers)
        )
        documents = [document for batch in batches for document in batch]
        return RetrievalResult(documents=documents)
