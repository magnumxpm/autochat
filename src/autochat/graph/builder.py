from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from autochat.compression import AutoCompress
from autochat.config import ChatConfig
from autochat.events.dispatch import (
    emit_compression_end,
    emit_compression_start,
    emit_error,
)
from autochat.graph.runtime import get_runtime
from autochat.graph.state import ChatGraphState, ChatGraphUpdate
from autochat.graph.tools import run_tool_calls
from autochat.guidelines import ChatGuideline
from autochat.retrieval import (
    ChatRetriever,
    RetrievalConfig,
)
from autochat.tools import ChatTool


def validate_callable_names(
    tools: Sequence[ChatTool[Any, Any, Any]],
    retrievers: Sequence[ChatRetriever[Any]],
) -> None:
    """Validate that tool and retriever names are unique."""
    names = [tool.name for tool in tools] + [retriever.name for retriever in retrievers]
    duplicates = {name for name in names if names.count(name) > 1}

    if duplicates:
        duplicate_names = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate tool/retriever names: {duplicate_names}")


def build_system_messages(
    system_message: str | None,
    guidelines: Sequence[ChatGuideline],
) -> list[SystemMessage]:
    messages: list[SystemMessage] = []

    if system_message:
        messages.append(SystemMessage(content=system_message))

    if guidelines:
        guideline_text = "\n".join(f"- {guideline.content}" for guideline in guidelines)
        messages.append(SystemMessage(content=f"Guidelines:\n{guideline_text}"))

    return messages


def should_continue(state: ChatGraphState) -> str:
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return END


def build_chat_graph(
    *,
    config: ChatConfig,
    tools: Sequence[ChatTool[Any, Any, Any]],
    retrievers: Sequence[ChatRetriever[Any]],
    retrieval: RetrievalConfig[Any],
    compression: AutoCompress[Any] | None,
    system_message: str | None,
    guidelines: Sequence[ChatGuideline],
    persistence: BaseCheckpointSaver | None,
) -> CompiledStateGraph:
    graph = StateGraph(ChatGraphState)
    system_messages = build_system_messages(system_message, guidelines)

    chat_config = config
    model = chat_config.model
    if tools or retrievers:
        validate_callable_names(tools, retrievers)

        model = model.bind_tools(
            [
                *(tool.model_tool() for tool in tools),
                *(retriever.model_tool() for retriever in retrievers),
            ]
        )

    async def call_model(
        state: ChatGraphState,
        config: RunnableConfig,
    ) -> ChatGraphUpdate:
        try:
            messages: list[BaseMessage] = [
                *system_messages,
                *state["messages"],
            ]

            response = await model.ainvoke(
                messages, config=config, **(chat_config.model_kwargs or {})
            )

            response_messages: list[BaseMessage] = [response]
            return {"messages": response_messages}
        except Exception as e:
            await emit_error(node="agent", error=str(e))
            raise

    async def call_tools(
        state: ChatGraphState,
        config: RunnableConfig,
    ) -> ChatGraphUpdate:
        last_message = state["messages"][-1]

        if not isinstance(last_message, AIMessage):
            return {"messages": []}

        tool_messages = await run_tool_calls(
            last_message,
            tools,
            retrievers,
            config,
        )
        messages: list[BaseMessage] = list(tool_messages)
        return {"messages": messages}

    async def compress_messages(
        state: ChatGraphState,
        config: RunnableConfig,
    ) -> ChatGraphUpdate:
        if compression is None:
            return {"messages": []}

        strategy_name = type(compression.strategy).__name__
        before_count = len(state["messages"])
        await emit_compression_start(
            strategy=strategy_name,
            message_count_before=before_count,
        )

        try:
            messages = await compression.acompress_if_needed(
                list(state["messages"]),
                runtime=get_runtime(config),
                config=chat_config,
            )
        except Exception as e:
            await emit_compression_end(
                strategy=strategy_name,
                compressed=False,
                message_count_after=before_count,
            )
            await emit_error(node="compression", error=str(e))
            raise

        if messages is None:
            await emit_compression_end(
                strategy=strategy_name,
                compressed=False,
                message_count_after=before_count,
            )

            return {"messages": []}

        await emit_compression_end(
            strategy=strategy_name,
            compressed=True,
            message_count_after=len(messages),
        )

        return {"messages": messages}

    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)

    if compression:
        graph.add_node("compression", compress_messages)
        graph.add_edge(START, "compression")
        graph.add_edge("compression", "agent")
    else:
        graph.add_edge(START, "agent")

    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=persistence)
