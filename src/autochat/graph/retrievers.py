import json
from typing import Any, Sequence

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from autochat.events.dispatch import (
    emit_error,
    emit_retriever_request,
    emit_retriever_response,
)
from autochat.graph.runtime import get_runtime
from autochat.retrieval import ChatRetriever, RetrievedDocument


def serialize_retrieved_documents(documents: Sequence[RetrievedDocument]) -> str:
    payload = [
        {
            "content": document.content,
            "retriever": document.retriever_name,
            "metadata": dict(document.metadata),
            "score": document.score,
        }
        for document in documents
    ]

    return json.dumps(payload, default=str)


async def run_retriever_call(
    *,
    name: str,
    args: dict[str, Any],
    tool_call_id: str,
    retrievers: Sequence[ChatRetriever[Any]],
    config: RunnableConfig,
) -> ToolMessage:
    runtime = get_runtime(config)
    retrievers_by_name = {retriever.name: retriever for retriever in retrievers}
    retriever = retrievers_by_name.get(name)

    if retriever is None:
        content = f"Unknown retriever: {name}"
        await emit_retriever_response(
            name=name,
            tool_call_id=tool_call_id,
            error=content,
        )
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )

    query = args.get("query")
    if not isinstance(query, str):
        content = "Retriever tool call requires a string 'query' argument"

        await emit_retriever_response(
            name=name,
            tool_call_id=tool_call_id,
            error=content,
        )
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )

    await emit_retriever_request(
        name=name,
        tool_call_id=tool_call_id,
        query=query,
    )
    try:
        documents = await retriever.aretrieve(query, runtime)
    except Exception as e:
        await emit_retriever_response(
            name=name,
            tool_call_id=tool_call_id,
            error=str(e),
        )
        await emit_error(node="retrievers", error=str(e))
        raise

    await emit_retriever_response(
        name=name,
        tool_call_id=tool_call_id,
        documents=documents,
    )

    content = serialize_retrieved_documents(documents)
    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        name=name,
    )
