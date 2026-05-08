from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from autochat.config import ChatConfig
from autochat.graph.runtime import get_runtime
from autochat.graph.state import ChatGraphState, ChatGraphUpdate
from autochat.graph.tools import run_tool_calls
from autochat.guidelines import ChatGuideline
from autochat.retrieval import (
    ChatRetriever,
    RetrievalConfig,
    RetrievedDocument,
    SimpleRAGStrategy,
)
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


def latest_user_query(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content

            return str(content)

    return ""


def format_retrieved_context(
    documents: Sequence[RetrievedDocument],
    retrieval: RetrievalConfig[Any],
) -> SystemMessage | None:
    if not documents:
        return None

    sections: list[str] = [retrieval.context_header]
    used_chars = len(retrieval.context_header)

    for index, document in enumerate(documents, start=1):
        source = document.retriever_name or document.metadata.get("retriever", "retriever")
        title = f"[{index}] {source}"

        if retrieval.include_sources and document.metadata.get("source"):
            title = f"{title} source={document.metadata['source']}"

        body = document.content.strip()
        if not body:
            continue

        section = f"{title}\n{body}"
        next_used_chars = used_chars + len(section) + 2
        if next_used_chars > retrieval.max_context_chars:
            remaining = retrieval.max_context_chars - used_chars - len(title) - 3
            if remaining > 0:
                sections.append(f"{title}\n{body[:remaining]}")
            break

        sections.append(section)
        used_chars = next_used_chars

    if len(sections) == 1:
        return None

    return SystemMessage(content="\n\n".join(sections))


def build_chat_graph(
    *,
    config: ChatConfig,
    tools: Sequence[ChatTool[Any, Any, Any]],
    retrievers: Sequence[ChatRetriever[Any]],
    retrieval: RetrievalConfig[Any],
    system_message: str | None,
    guidelines: Sequence[ChatGuideline],
) -> CompiledStateGraph:
    graph = StateGraph(ChatGraphState)
    system_messages = build_system_messages(system_message, guidelines)
    rag_strategy = retrieval.strategy or SimpleRAGStrategy()

    chat_config = config
    model = chat_config.model
    if tools:
        model = model.bind_tools([tool.model_tool() for tool in tools])

    async def call_model(
        state: ChatGraphState,
        config: RunnableConfig,
    ) -> ChatGraphUpdate:
        retrieved_context = format_retrieved_context(
            state.get("retrieved_documents", []),
            retrieval,
        )
        messages: list[BaseMessage] = [
            *system_messages,
            *([retrieved_context] if retrieved_context else []),
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

    async def retrieve_context(
        state: ChatGraphState,
        config: RunnableConfig,
    ) -> ChatGraphUpdate:
        query = latest_user_query(state["messages"])
        if not query:
            return {"retrieved_documents": []}

        result = await rag_strategy.aretrieve(
            query,
            runtime=get_runtime(config),
            retrievers=retrievers,
        )
        return {"retrieved_documents": result.documents}

    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)

    if retrievers:
        graph.add_node("retrieve", retrieve_context)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "agent")
    else:
        graph.add_edge(START, "agent")

    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()
