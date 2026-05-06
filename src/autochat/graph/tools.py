import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from autochat.graph.runtime import get_runtime
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
    config: RunnableConfig,
) -> list[ToolMessage]:
    runtime = get_runtime(config)
    tools_by_name = {tool.name: tool for tool in tools}

    messages: list[ToolMessage] = []

    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        tool = tools_by_name.get(name)

        if tool is None:
            content = f"Unknown tool: {name}"
        else:
            # TODO: Run pre/post processors

            args = tool_call.get("args", {})
            result = await tool.ainvoke(args, runtime)
            content = serialize_tool_result(result)

        messages.append(
            ToolMessage(
                content=content,
                tool_call_id=tool_call["id"],
                name=name,
            )
        )

    return messages
