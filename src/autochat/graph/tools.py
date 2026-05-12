import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt

from autochat.events.dispatch import (
    emit_error,
    emit_tool_request,
    emit_tool_response,
)
from autochat.graph.retrievers import run_retriever_call
from autochat.graph.runtime import get_runtime
from autochat.hitl import (
    request_declarative_approval,
    resolve_denial_message,
)
from autochat.retrieval import ChatRetriever
from autochat.runtime import ChatRuntime
from autochat.tools import ChatTool


def serialize_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result

    try:
        return json.dumps(result, default=str)
    except TypeError:
        return str(result)


async def _gate_tool_approval(
    tool: ChatTool[Any, Any, Any],
    args: Any,
    runtime: ChatRuntime[Any],
    tool_call_id: str,
) -> tuple[bool, str | None]:
    """Run the declarative approval gate for a tool, if any.

    Returns `(allowed, denial_message)`. `allowed=True` means the tool should run;
    `denial_message` is set when the user denied the request.

    On the first call this raises a `GraphInterrupt` (LangGraph halts the graph);
    on resume `interrupt()` returns the host-provided response and execution
    proceeds.
    """
    spec = tool.approval
    if spec is None or runtime.hitl is None:
        return True, None

    invocation = tool.make_invocation(args, runtime)
    response = await request_declarative_approval(
        spec,
        invocation,
        kind="tool_approval",
        tool_call_id=tool_call_id,
        tool_name=tool.name,
    )

    decide = spec.decide
    if decide is None:
        return True, None

    if not decide(response):
        return False, resolve_denial_message(spec, response)

    runtime.hitl.approval = response
    return True, None


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
                allowed, denial_message = await _gate_tool_approval(
                    tool, args, runtime, tool_call_id
                )
            except GraphInterrupt:
                # Approval is pending; LangGraph will checkpoint and pause.
                # Do not emit a tool response — the tool hasn't run yet.
                raise
            except Exception as e:
                await emit_tool_response(
                    name=name,
                    tool_call_id=tool_call_id,
                    error=str(e),
                )
                raise

            if not allowed:
                content = denial_message or "User declined the tool invocation."
                await emit_tool_response(
                    name=name,
                    tool_call_id=tool_call_id,
                    result=content,
                )
                messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id,
                        name=name,
                    )
                )
                continue

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
            finally:
                if runtime.hitl is not None:
                    runtime.hitl.approval = None

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
