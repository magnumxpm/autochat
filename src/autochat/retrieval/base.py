import inspect
from collections.abc import Sequence
from typing import Any, Generic, TypeVar, overload

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from autochat.runtime import ChatRuntime
from autochat.tools.base import maybe_await

from .documents import RetrievedDocument
from .types import (
    RetrievalInvocation,
    RetrieverDocument,
    RetrieverFn,
    RetrieverPostprocessor,
    RetrieverPreprocessor,
)

TContext = TypeVar("TContext")


class ChatRetriever(Generic[TContext]):
    """Retriever wrapper that gives retrieval access to ChatRuntime."""

    @overload
    def __init__(
        self,
        retriever: BaseRetriever,
        *,
        name: str,
        description: str | None = None,
        top_k: int = 5,
        tags: Sequence[str] = (),
        metadata: dict[str, Any] | None = None,
        preprocessors: Sequence[RetrieverPreprocessor[TContext]] = (),
        postprocessors: Sequence[RetrieverPostprocessor[TContext]] = (),
    ) -> None: ...

    @overload
    def __init__(
        self,
        retriever: RetrieverFn[TContext],
        *,
        name: str,
        description: str | None = None,
        top_k: int = 5,
        tags: Sequence[str] = (),
        metadata: dict[str, Any] | None = None,
        preprocessors: Sequence[RetrieverPreprocessor[TContext]] = (),
        postprocessors: Sequence[RetrieverPostprocessor[TContext]] = (),
    ) -> None: ...

    def __init__(
        self,
        retriever: BaseRetriever | RetrieverFn[TContext],
        *,
        name: str,
        description: str | None = None,
        top_k: int = 5,
        tags: Sequence[str] = (),
        metadata: dict[str, Any] | None = None,
        preprocessors: Sequence[RetrieverPreprocessor[TContext]] = (),
        postprocessors: Sequence[RetrieverPostprocessor[TContext]] = (),
    ) -> None:
        if top_k < 1:
            raise ValueError("ChatRetriever top_k must be greater than zero.")

        self._retriever = retriever
        self._name = name
        self._description = description or self._infer_description(retriever)
        self._top_k = top_k
        self._tags = tuple(tags)
        self._metadata = metadata or {}
        self._preprocessors = tuple(preprocessors)
        self._postprocessors = tuple(postprocessors)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def raw_retriever(self) -> BaseRetriever | RetrieverFn[TContext]:
        return self._retriever

    async def aretrieve(
        self,
        query: str,
        runtime: ChatRuntime[TContext],
    ) -> list[RetrievedDocument]:
        invocation = self._make_invocation(query, runtime)

        for preprocessor in self._preprocessors:
            query = await maybe_await(preprocessor(invocation))
            invocation = self._make_invocation(query, runtime)

        raw_documents = await self._arun_underlying_retriever(query, runtime)
        documents = self._normalize_documents(raw_documents)[: self._top_k]

        for postprocessor in self._postprocessors:
            documents = await maybe_await(postprocessor(invocation, documents))

        return documents

    @staticmethod
    def _infer_description(
        retriever: BaseRetriever | RetrieverFn[TContext],
    ) -> str | None:
        if isinstance(retriever, BaseRetriever):
            return getattr(retriever, "description", None)

        return inspect.getdoc(retriever)

    def model_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description
                or f"Retrieve relevant context from {self.name}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to retrieve relevant context for.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    def _make_invocation(
        self,
        query: str,
        runtime: ChatRuntime[TContext],
    ) -> RetrievalInvocation[TContext]:
        return RetrievalInvocation(
            name=self._name,
            query=query,
            runtime=runtime,
            raw_retriever=self._retriever,
        )

    async def _arun_underlying_retriever(
        self,
        query: str,
        runtime: ChatRuntime[TContext],
    ) -> Sequence[RetrieverDocument]:
        if isinstance(self._retriever, BaseRetriever):
            return await self._retriever.ainvoke(query)

        return await maybe_await(self._retriever(query, runtime))

    def _normalize_documents(
        self,
        documents: Sequence[RetrieverDocument],
    ) -> list[RetrievedDocument]:
        return [self._normalize_document(document) for document in documents]

    def _normalize_document(
        self,
        document: RetrieverDocument,
    ) -> RetrievedDocument:
        metadata: dict[str, Any] = {
            **self._metadata,
            "retriever": self._name,
        }

        if self._tags:
            metadata["tags"] = list(self._tags)

        if isinstance(document, RetrievedDocument):
            return RetrievedDocument(
                content=document.content,
                retriever_name=document.retriever_name or self._name,
                metadata={**metadata, **dict(document.metadata)},
                score=document.score,
            )

        if isinstance(document, Document):
            return RetrievedDocument(
                content=document.page_content,
                retriever_name=self._name,
                metadata={**metadata, **dict(document.metadata)},
            )

        return RetrievedDocument(
            content=str(document),
            retriever_name=self._name,
            metadata=metadata,
        )
