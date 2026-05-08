import json
from typing import Any, Sequence

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

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
    else:
        query = args.get("query")
        if not isinstance(query, str):
            content = "Retriever tool call requires a string 'query' argument"
        else:
            documents = await retriever.aretrieve(query, runtime)
            content = serialize_retrieved_documents(documents)

    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        name=name,
    )
