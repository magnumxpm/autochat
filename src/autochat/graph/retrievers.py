import json
from typing import Any, Sequence

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt

from autochat.events.dispatch import (
    emit_error,
    emit_retriever_request,
    emit_retriever_response,
)
from autochat.graph.runtime import get_runtime
from autochat.hitl import (
    request_declarative_approval,
    resolve_denial_message,
)
from autochat.retrieval import ChatRetriever, RetrievedDocument
from autochat.runtime import ChatRuntime


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


async def _gate_retriever_approval(
    retriever: ChatRetriever[Any],
    query: str,
    runtime: ChatRuntime[Any],
    tool_call_id: str,
) -> tuple[bool, str | None]:
    spec = retriever.approval
    if spec is None or runtime.hitl is None:
        return True, None

    invocation = retriever.make_invocation(query, runtime)
    response = await request_declarative_approval(
        spec,
        invocation,
        kind="retriever_approval",
        tool_call_id=tool_call_id,
        tool_name=retriever.name,
    )

    decide = spec.decide
    if decide is None:
        return True, None

    if not decide(response):
        return False, resolve_denial_message(spec, response)

    runtime.hitl.approval = response
    return True, None


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
        allowed, denial_message = await _gate_retriever_approval(
            retriever, query, runtime, tool_call_id
        )
    except GraphInterrupt:
        # Approval is pending; LangGraph will checkpoint and pause.
        raise
    except Exception as e:
        await emit_retriever_response(
            name=name,
            tool_call_id=tool_call_id,
            error=str(e),
        )
        raise

    if not allowed:
        content = denial_message or "User declined the retrieval."
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
    finally:
        if runtime.hitl is not None:
            runtime.hitl.approval = None

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
