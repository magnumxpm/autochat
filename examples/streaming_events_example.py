"""Demonstrates AutoChat's typed astream_events with a tool, retriever,
and AutoCompress firing at 10% of the tiny context window.

Run:
    OPENAI_API_KEY=... uv run python examples/streaming_events_example.py
"""

import asyncio
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from autochat import (
    AssistantMessage,
    AutoChat,
    AutoChatEvent,
    AutoCompress,
    ChatConfig,
    ChatRetriever,
    ChatRuntime,
    CompressionEndEvent,
    CompressionStartEvent,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    RetrievedDocument,
    RetrieverRequestEvent,
    RetrieverResponseEvent,
    RunEndEvent,
    RunStartEvent,
    SummarizeAll,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallRequestEvent,
    ToolCallResponseEvent,
    chat_tool,
)


@dataclass(frozen=True, slots=True)
class AppContext:
    user_id: str
    org_id: str


@chat_tool(
    name="get_weather",
    description="Get the current weather for a city.",
)
async def get_weather(
    city: str,
    runtime: ChatRuntime[AppContext],
) -> dict:
    del runtime
    return {
        "city": city,
        "temperature_c": 21,
        "conditions": "partly cloudy",
    }


KNOWLEDGE_BASE = {
    "org_42": [
        RetrievedDocument(
            content=(
                "AutoChat is an opinionated chat harness primitive built on "
                "LangGraph. It provides typed streaming events for tools, "
                "retrievers, reasoning, and compression."
            ),
            metadata={"source": "internal_overview.md"},
        ),
        RetrievedDocument(
            content=(
                "Compression policies are configured via AutoCompress. The "
                "'at' parameter controls when compression fires as a fraction "
                "of the configured context window."
            ),
            metadata={"source": "compression_guide.md"},
        ),
    ]
}


async def lookup_internal_docs(
    query: str,
    runtime: ChatRuntime[AppContext],
) -> list[RetrievedDocument]:
    del query
    return KNOWLEDGE_BASE.get(runtime.context.org_id, [])


# Console rendering ------------------------------------------------------------


CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def _print_status(color: str, label: str, message: str = "") -> None:
    suffix = f" {message}" if message else ""
    print(f"\n{color}[{label}]{RESET}{suffix}", flush=True)


def _print_assistant_summary(message: AssistantMessage) -> None:
    print(f"\n{DIM}--- assistant message ---{RESET}", flush=True)
    if message.thinking:
        print(f"{DIM}thinking blocks: {len(message.thinking)}{RESET}", flush=True)
    if message.tool_calls:
        for call in message.tool_calls:
            print(
                f"{DIM}requested tool: {call.name}({dict(call.args)}){RESET}",
                flush=True,
            )
    if message.usage:
        print(
            f"{DIM}usage: in={message.usage.input_tokens} "
            f"out={message.usage.output_tokens}{RESET}",
            flush=True,
        )


def render_event(event: AutoChatEvent) -> None:
    if isinstance(event, RunStartEvent):
        _print_status(BLUE, "run.start", f"input={event.input!r}")

    elif isinstance(event, MessageStartEvent):
        _print_status(GREEN, "message.start", event.message_id)

    elif isinstance(event, MessageDeltaEvent):
        # Stream text deltas inline without newlines.
        print(event.delta, end="", flush=True)

    elif isinstance(event, MessageEndEvent):
        _print_assistant_summary(event.message)

    elif isinstance(event, ThinkingStartEvent):
        _print_status(MAGENTA, "thinking.start", event.block_id)

    elif isinstance(event, ThinkingDeltaEvent):
        print(f"{MAGENTA}{event.delta}{RESET}", end="", flush=True)

    elif isinstance(event, ThinkingEndEvent):
        _print_status(MAGENTA, "thinking.end", f"({len(event.block.text)} chars)")

    elif isinstance(event, ToolCallRequestEvent):
        _print_status(
            YELLOW,
            "tool.request",
            f"{event.name}({dict(event.args)})",
        )

    elif isinstance(event, ToolCallResponseEvent):
        if event.error:
            _print_status(RED, "tool.response", f"{event.name} error: {event.error}")
        else:
            _print_status(YELLOW, "tool.response", f"{event.name} -> {event.result!r}")

    elif isinstance(event, RetrieverRequestEvent):
        _print_status(CYAN, "retriever.request", f"{event.name}({event.query!r})")

    elif isinstance(event, RetrieverResponseEvent):
        if event.error:
            _print_status(RED, "retriever.response", event.error)
        else:
            _print_status(
                CYAN,
                "retriever.response",
                f"{event.name} -> {len(event.documents)} docs",
            )
            for doc in event.documents:
                source = doc.metadata.get("source", "?")
                preview = doc.content[:60].replace("\n", " ")
                print(f"  {DIM}- [{source}] {preview}...{RESET}", flush=True)

    elif isinstance(event, CompressionStartEvent):
        _print_status(
            BLUE,
            "compression.start",
            f"{event.strategy} (msgs={event.message_count_before})",
        )

    elif isinstance(event, CompressionEndEvent):
        verb = "ran" if event.compressed else "skipped"
        _print_status(
            BLUE,
            "compression.end",
            f"{event.strategy} {verb} (msgs={event.message_count_after})",
        )

    elif isinstance(event, ErrorEvent):
        _print_status(RED, "error", f"{event.node}: {event.error}")

    elif isinstance(event, RunEndEvent):
        _print_status(BLUE, "run.end")


# Main -------------------------------------------------------------------------


async def main() -> None:
    # GPT-5 reasoning model. AutoChat will pick OpenAIReasoningExtractor
    # automatically based on the model name pattern.
    model = ChatOpenAI(
        model="gpt-5-mini",
        reasoning_effort="low",
        use_responses_api=True,
    )

    chat = AutoChat[AppContext](
        config=ChatConfig(
            model=model,
            # Small context window so AutoCompress is easy to trigger in a demo.
            context_window=300,
        ),
        tools=[get_weather],
        retrievers=[
            ChatRetriever(
                lookup_internal_docs,
                name="internal_docs",
                description="Internal AutoChat documentation lookup.",
                top_k=3,
            )
        ],
        compression=AutoCompress(
            at=0.1,  # compress at 10% of context window
            strategy=SummarizeAll(),
        ),
        persistence=InMemorySaver(),
        system_message=(
            "You are a helpful assistant. Use the get_weather tool when asked "
            "about weather, and the internal_docs retriever when asked about "
            "AutoChat itself."
        ),
    )

    context = AppContext(user_id="user_1", org_id="org_42")
    thread_id = "thread_streaming_events_example"

    prompts = [
        "What is AutoChat? Look it up in our internal docs.",
        "Now get the weather in Tokyo and summarize what you found in the docs earlier.",
    ]

    for prompt in prompts:
        print(f"\n{DIM}>>> user: {prompt}{RESET}", flush=True)
        async for event in chat.astream_events(
            prompt,
            thread_id=thread_id,
            context=context,
        ):
            render_event(event)
        print()


if __name__ == "__main__":
    asyncio.run(main())
