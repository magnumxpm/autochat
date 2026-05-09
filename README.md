# autochat

`autochat` is a small Python library for building context-aware chat applications on top of LangGraph and LangChain.

It gives your app a clean chat harness primitive:

```python
from autochat import AutoChat

chat = AutoChat(...)
```

Then you can invoke or stream the graph while passing your own runtime context into tools, retrievers, processors, and graph execution.

`autochat` is still under construction. It is not published to PyPI yet, but it will soon be installable as `autochat` with `pip`, `uv`, and other standard Python package managers.

## Why

Most production chat apps need the same foundation:

- model and tool orchestration
- runtime context for auth, tenancy, request metadata, and app services
- context-aware tools and retrievers
- thread persistence
- optional history compression
- a simple async invoke/stream API

`autochat` packages those pieces into a small, typed, async-first interface.

## Installation

For local development:

```bash
uv sync
```

For examples that use OpenAI models:

```bash
uv sync --dev
```

Future install flow:

```bash
pip install autochat
```

or:

```bash
uv add autochat
```

## Quick Start

```python
import asyncio
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from autochat import AutoChat, ChatConfig, ChatRuntime, chat_tool


@dataclass(frozen=True)
class AppContext:
    user_id: str
    plan: str


@chat_tool(name="current_plan")
async def current_plan(runtime: ChatRuntime[AppContext]) -> str:
    return f"The user is on the {runtime.context.plan} plan."


async def main() -> None:
    chat = AutoChat[AppContext](
        config=ChatConfig(model=ChatOpenAI(model="gpt-5-nano")),
        tools=[current_plan],
        system_message="You are concise and practical.",
    )

    result = await chat.ainvoke(
        "What plan am I on?",
        thread_id="thread_123",
        context=AppContext(user_id="user_1", plan="pro"),
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
```

Run the included examples:

```bash
uv run python examples/basic_tool_example.py
OPENAI_API_KEY=... uv run --dev python examples/basic_chat_example.py
OPENAI_API_KEY=... uv run --dev python examples/basic_retriever_example.py
OPENAI_API_KEY=... uv run --dev python examples/basic_persistence_compression_example.py
```

## Runtime Context

`ChatRuntime[TContext]` is created for each chat run and passed through the graph layer.

Use it to carry app-specific data like user IDs, org IDs, permissions, request metadata, database handles, or tenant config.

```python
@dataclass(frozen=True)
class AppContext:
    org_id: str
    permissions: set[str]


@chat_tool(name="billing_status")
async def billing_status(runtime: ChatRuntime[AppContext]) -> str:
    return f"Billing is active for {runtime.context.org_id}."
```

## Tools

Use `@chat_tool` for native AutoChat tools. Function schemas are inferred from normal function parameters, and `runtime` is injected automatically.

```python
@chat_tool(name="calculator")
async def calculator(
    a: float,
    b: float,
    runtime: ChatRuntime[AppContext],
) -> float:
    return a + b
```

You can also wrap LangChain tools:

```python
from autochat import ChatTool

chat = AutoChat(
    config=ChatConfig(model=model),
    tools=[ChatTool(langchain_tool)],
)
```

## Tool Processors

Preprocessors and postprocessors wrap tool execution with app logic such as auth checks, input normalization, logging, or cleanup.

```python
from autochat import ToolInvocation


def require(permission: str):
    def processor(invocation: ToolInvocation[AppContext, object]) -> object:
        if permission not in invocation.runtime.context.permissions:
            raise PermissionError(f"Missing permission: {permission}")
        return invocation.input

    return processor


@chat_tool(name="billing_status", preprocessors=[require("billing.read")])
async def billing_status(runtime: ChatRuntime[AppContext]) -> str:
    return "Billing is active."
```

## Retrieval

Retrievers are exposed to the model as callable retrieval tools. The model decides when to call them, and AutoChat executes the retriever with the current `ChatRuntime`.

```python
from autochat import ChatRetriever, ChatRuntime


async def search_docs(query: str, runtime: ChatRuntime[AppContext]) -> list[str]:
    return [f"Docs for {runtime.context.org_id}: {query}"]


chat = AutoChat[AppContext](
    config=ChatConfig(model=model),
    retrievers=[
        ChatRetriever(
            search_docs,
            name="docs",
            description="Search organization documentation.",
        )
    ],
    system_message="Use the docs retriever for policy or product questions.",
)
```

## Persistence

AutoChat uses LangGraph checkpointers for thread persistence. Pass a checkpointer with `persistence=...`, and LangGraph stores graph state by `thread_id`.

```python
from langgraph.checkpoint.memory import InMemorySaver

chat = AutoChat(
    config=ChatConfig(model=model),
    persistence=InMemorySaver(),
)
```

For production, swap `InMemorySaver` for a durable LangGraph saver.

## Compression

Compression is optional. It runs before the model call, after persisted thread state has been loaded.

```python
from autochat import AutoCompress, SummarizeAll
from langgraph.checkpoint.memory import InMemorySaver

chat = AutoChat(
    config=ChatConfig(
        model=model,
        context_window=128_000,
    ),
    persistence=InMemorySaver(),
    compression=AutoCompress(
        at=0.6,
        strategy=SummarizeAll(),
    ),
)
```

Available strategies:

- `SummarizeAll()`: summarize older history into one summary message
- `SummarizeLatestN(n=20)`: summarize only the latest `n` historical messages
- `KeepLatestN(n=20)`: keep only the latest `n` messages without summarizing

Summaries replace graph history using LangGraph message removal, so future turns see a compacted thread state.

## Core Pieces

- `AutoChat`: public chat harness for invoke and stream workflows
- `ChatConfig`: model configuration and context-window metadata
- `ChatRuntime[TContext]`: per-run context passed through graph execution
- `ChatTool` / `@chat_tool`: LangChain-compatible and native context-aware tools
- `ChatRetriever`: LangChain-compatible and native context-aware retrievers
- `AutoCompress`: optional automatic thread compression
- `ChatGuideline`: lightweight reusable instruction primitive

## Project Structure

A high-level map for contributors:

```text
src/autochat/
  chat.py              AutoChat public harness API
  config.py            ChatConfig and model configuration
  guidelines.py        Lightweight guideline primitives
  runtime/             Invocation-scoped runtime context
  tools/               ChatTool, @chat_tool, processor types
  retrieval/           ChatRetriever, retrieval config, RAG strategies
  compression/         AutoCompress and compression strategies
  graph/               LangGraph state, builder, runtime wiring, execution
  exceptions/          Library exception types

examples/
  basic_tool_example.py                    Context-aware tool + processor
  basic_chat_example.py                    AutoChat + model + tools
  basic_retriever_example.py               AutoChat + retriever
  basic_persistence_compression_example.py Persistence + compression
```

The intended dependency direction is:

```text
AutoChat
  -> graph
      -> tools / retrieval / compression
          -> runtime
```

## Contributing

The library is early and the API is still being shaped. Contributions should keep the surface area small, typed, and pleasant for application developers.

Before changing internals, run the examples when relevant:

```bash
uv run python examples/basic_tool_example.py
uv run --dev python examples/basic_chat_example.py
uv run --dev python examples/basic_retriever_example.py
uv run --dev python examples/basic_persistence_compression_example.py
```

Design preferences:

- async-first internally
- explicit runtime context
- native primitives with LangChain compatibility
- LangGraph persistence instead of custom thread storage
- minimal graph details in user-facing APIs

## License

`autochat` is released under the MIT License. See [LICENSE](LICENSE).
