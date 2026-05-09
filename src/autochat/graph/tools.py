import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from autochat.events.dispatch import (
    emit_error,
    emit_tool_request,
    emit_tool_response,
)
from autochat.graph.retrievers import run_retriever_call
from autochat.graph.runtime import get_runtime
from autochat.retrieval import ChatRetriever
from autochat.tools import ChatTool


def serialize_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result

    try:
        return json.dumps(result, default=str)
    except TypeError:
        return str(result)


async def run_tool_calls(
    last_message: AIMessage,
    tools: Sequence[ChatTool[Any, Any, Any]],
    retrievers: Sequence[ChatRetriever[Any]],
    config: RunnableConfig,
) -> list[ToolMessage]:
    runtime = get_runtime(config)
    tools_by_name = {tool.name: tool for tool in tools}
    retrievers_by_name = {retriever.name: retriever for retriever in retrievers}

    messages: list[ToolMessage] = []

    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        tool_call_id = tool_call.get("id") or ""
        tool = tools_by_name.get(name)

        if tool is not None:
            args = tool_call.get("args", {})

            await emit_tool_request(
                name=name,
                tool_call_id=tool_call_id,
                args=args,
            )
            try:
                result = await tool.ainvoke(args, runtime)
            except Exception as e:
                await emit_tool_response(
                    name=name,
                    tool_call_id=tool_call_id,
                    error=str(e),
                )
                await emit_error(node="tools", error=str(e))
                raise

            content = serialize_tool_result(result)
            await emit_tool_response(
                name=name,
                tool_call_id=tool_call_id,
                result=result,
            )

        elif name in retrievers_by_name:
            args = tool_call.get("args", {})
            if not isinstance(args, dict):
                args = {}

            message = await run_retriever_call(
                name=name,
                args=args,
                tool_call_id=tool_call_id,
                retrievers=retrievers,
                config=config,
            )

            messages.append(message)
            continue

        else:
            content = f"Unknown tool or retriever: {name}"

        messages.append(
            ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                name=name,
            )
        )

    return messages
