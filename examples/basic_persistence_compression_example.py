import asyncio
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from autochat import (
    AutoChat,
    AutoCompress,
    ChatConfig,
    # KeepLatestN,
    SummarizeAll,
    # SummarizeLatestN,
)


@dataclass(frozen=True, slots=True)
class AppContext:
    user_id: str


def chunk_text(event: dict) -> str:
    if event.get("event") != "on_chat_model_stream":
        return ""

    chunk = event.get("data", {}).get("chunk")
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content

    return str(content)


async def main() -> None:
    model = ChatOpenAI(model="gpt-5-nano")

    chat = AutoChat[AppContext](
        config=ChatConfig(
            model=model,
            # Used by AutoCompress to decide when the persisted thread is too large.
            context_window=128_000,
        ),
        # LangGraph persists graph state by thread_id. Swap this with a durable
        # saver, such as SQLite/Postgres, when running outside a demo.
        persistence=InMemorySaver(),
        compression=AutoCompress(
            at=0.6,
            strategy=SummarizeAll(),
            # Other options:
            # strategy=SummarizeLatestN(n=20),
            # strategy=KeepLatestN(n=20),
            # context_window=64_000,  # override ChatConfig.context_window
            # model=summary_model,    # use a cheaper/different summarizer
        ),
        system_message="You are concise and remember the conversation.",
    )

    context = AppContext(user_id="user_1")
    thread_id = "thread_persistence_compression_example"

    await chat.ainvoke(
        "My name is Pritam. I am building a chat harness called autochat.",
        thread_id=thread_id,
        context=context,
    )

    async for event in chat.astream_events(
        "What am I building, and what is my name?",
        thread_id=thread_id,
        context=context,
    ):
        print(chunk_text(event), end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())
