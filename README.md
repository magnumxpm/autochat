# autochat

`autochat` is a small Python library for building context-aware chat applications with LangGraph, LangChain models, and typed tools.

It is meant to give you a clean application primitive:

```python
from autochat import AutoChat

chat = AutoChat(...)
```

and then let you invoke or stream a graph while passing your own runtime context into tools, processors, and future subgraphs.

`autochat` is still under construction. It is not published to PyPI yet, but it will soon be available as `autochat` and installable with `pip`, `uv`, and other standard Python package managers.

## Why

Most chat apps need the same harness around the model:

- a graph to manage model and tool turns
- tools that can see request/runtime context
- authorization and cleanup around tool calls
- a simple invoke/stream API
- enough structure to stay maintainable as the app grows

`autochat` packages those pieces into a small async-first interface.

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

A minimal chat app with one context-aware tool:

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
    return f"The current user is on the {runtime.context.plan} plan."


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
```

## Tool Processors

Preprocessors and postprocessors let you wrap tool execution with app logic such as auth checks, input normalization, logging, or cleanup.

```python
from dataclasses import dataclass

from autochat import ChatRuntime, ToolInvocation, chat_tool


@dataclass(frozen=True)
class AppContext:
    permissions: set[str]


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

Retrievers can be added beside tools. AutoChat runs them before the model and injects the retrieved context into the graph.

```python
from autochat import ChatRetriever, ChatRuntime


async def search_docs(query: str, runtime: ChatRuntime[AppContext]) -> list[str]:
    return [f"Docs for {runtime.context.user_id}: {query}"]


chat = AutoChat[AppContext](
    config=ChatConfig(model=model),
    retrievers=[ChatRetriever(search_docs, name="docs")],
)
```

## Core Pieces

- `AutoChat`: public chat harness for invoke and stream workflows
- `ChatConfig`: model and chat configuration
- `ChatRuntime[TContext]`: per-run context passed through graph/tool execution
- `ChatTool`: wrapper for LangChain tools and AutoChat-native tools
- `@chat_tool`: decorator for native context-aware tools
- `ChatRetriever`: wrapper for LangChain retrievers and AutoChat-native retrievers
- `RetrievalConfig`: graph-level retrieval strategy configuration
- `ChatGuideline`: lightweight instruction primitive for reusable behavior rules

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
  graph/               LangGraph state, builder, runtime wiring, tool execution
  exceptions/          Library exception types

examples/
  basic_tool_example.py       Context-aware tool and processor example
  basic_chat_example.py       AutoChat + model + tools example
  basic_retriever_example.py  AutoChat + retriever example
```

The intended dependency direction is:

```text
AutoChat
  -> graph
      -> tools / retrieval
          -> runtime
```

## Contributing

The library is early and the API is still being shaped. Contributions should keep the surface area small, typed, and pleasant for application developers.

Before changing internals, run the examples when relevant:

```bash
uv run python examples/basic_tool_example.py
uv run --dev python examples/basic_chat_example.py
uv run --dev python examples/basic_retriever_example.py
```

Design preferences:

- async-first internally
- explicit runtime context
- clean native tools
- LangChain compatibility where useful
- minimal graph details in user-facing APIs

## License

`autochat` is released under the MIT License. See [LICENSE](LICENSE).
