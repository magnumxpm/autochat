from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from autochat.config import ChatConfig
from autochat.graph.state import ChatGraphState, ChatGraphUpdate
from autochat.graph.tools import run_tool_calls
from autochat.guidelines import ChatGuideline
from autochat.tools import ChatTool


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
    system_message: str | None,
    guidelines: Sequence[ChatGuideline],
) -> CompiledStateGraph:
    graph = StateGraph(ChatGraphState)
    system_messages = build_system_messages(system_message, guidelines)

    chat_config = config
    model = chat_config.model
    if tools:
        model = model.bind_tools([tool.model_tool() for tool in tools])

    async def call_model(
        state: ChatGraphState,
        config: RunnableConfig,
    ) -> ChatGraphUpdate:
        messages: list[BaseMessage] = [
            *system_messages,
            *state["messages"],
        ]

        response = await model.ainvoke(
            messages, config=config, **(chat_config.model_kwargs or {})
        )

        response_messages: list[BaseMessage] = [response]
        return {"messages": response_messages}

    async def call_tools(
        state: ChatGraphState,
        config: RunnableConfig,
    ) -> ChatGraphUpdate:
        last_message = state["messages"][-1]

        if not isinstance(last_message, AIMessage):
            return {"messages": []}

        tool_messages = await run_tool_calls(last_message, tools, config)
        messages: list[BaseMessage] = list(tool_messages)
        return {"messages": messages}

    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()
