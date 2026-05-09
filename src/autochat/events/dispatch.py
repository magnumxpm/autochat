from typing import Any, Mapping, Sequence

from langchain_core.callbacks.manager import adispatch_custom_event

from autochat.retrieval import RetrievedDocument

TOOL_REQUEST = "autochat.tool.request"
TOOL_RESPONSE = "autochat.tool.response"
RETRIEVER_REQUEST = "autochat.retriever.request"
RETRIEVER_RESPONSE = "autochat.retriever.response"
COMPRESSION_START = "autochat.compression.start"
COMPRESSION_END = "autochat.compression.end"
ERROR = "autochat.error"


async def emit_tool_request(
    *, name: str, tool_call_id: str, args: Mapping[str, Any]
) -> None:
    await adispatch_custom_event(
        TOOL_REQUEST,
        {"name": name, "tool_call_id": tool_call_id, "args": dict(args)},
    )


async def emit_tool_response(
    *,
    name: str,
    tool_call_id: str,
    result: Any | None = None,
    error: str | None = None,
) -> None:
    await adispatch_custom_event(
        TOOL_RESPONSE,
        {
            "name": name,
            "tool_call_id": tool_call_id,
            "result": result,
            "error": error,
        },
    )


async def emit_retriever_request(
    *,
    name: str,
    tool_call_id: str,
    query: str,
) -> None:
    await adispatch_custom_event(
        RETRIEVER_REQUEST,
        {
            "name": name,
            "tool_call_id": tool_call_id,
            "query": query,
        },
    )


async def emit_retriever_response(
    *,
    name: str,
    tool_call_id: str,
    documents: Sequence[RetrievedDocument] = (),
    error: str | None = None,
) -> None:
    await adispatch_custom_event(
        RETRIEVER_RESPONSE,
        {
            "name": name,
            "tool_call_id": tool_call_id,
            "documents": list(documents),
            "error": error,
        },
    )


async def emit_compression_start(
    *,
    strategy: str,
    message_count_before: int,
) -> None:
    await adispatch_custom_event(
        COMPRESSION_START,
        {
            "strategy": strategy,
            "message_count_before": message_count_before,
        },
    )


async def emit_compression_end(
    *,
    strategy: str,
    compressed: bool,
    message_count_after: int,
) -> None:
    await adispatch_custom_event(
        COMPRESSION_END,
        {
            "strategy": strategy,
            "compressed": compressed,
            "message_count_after": message_count_after,
        },
    )


async def emit_error(*, node: str, error: str) -> None:
    await adispatch_custom_event(
        ERROR,
        {"node": node, "error": error},
    )
