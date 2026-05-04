import asyncio
import inspect
from collections.abc import Sequence
from typing import Generic, TypeVar, cast, overload

from langchain_core.tools import BaseTool

from autochat.runtime import ChatRuntime

from .types import (
    ContextToolFn,
    LangChainToolInput,
    MaybeAwaitable,
    ToolInvocation,
    ToolPostprocessor,
    ToolPreprocessor,
)

T = TypeVar("T")
TContext = TypeVar("TContext")
TInput = TypeVar("TInput")
TResult = TypeVar("TResult")


async def maybe_await(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


class ChatTool(Generic[TContext, TInput, TResult]):
    """AutoChat tool wrapper primitive.

    Either use:\n

    1. LangChain `BaseTool`:
    ```python
    tool = ChatTool(langchain_tool)
    ```

    2. AutoChat-native context-aware callable:
    ```python
    tool = ChatTool(my_tool_func)
    ```
    """

    @overload
    def __init__(
        self,
        tool: BaseTool,
        *,
        name: str | None = None,
        description: str | None = None,
        preprocessors: Sequence[ToolPreprocessor[TContext, TInput]] = (),
        postprocessors: Sequence[ToolPostprocessor[TContext, TInput, TResult]] = (),
    ) -> None: ...

    @overload
    def __init__(
        self,
        tool: ContextToolFn[TContext, TInput, TResult],
        *,
        name: str | None = None,
        description: str | None = None,
        preprocessors: Sequence[ToolPreprocessor[TContext, TInput]] = (),
        postprocessors: Sequence[ToolPostprocessor[TContext, TInput, TResult]] = (),
    ) -> None: ...

    def __init__(
        self,
        tool: BaseTool | ContextToolFn[TContext, TInput, TResult],
        *,
        name: str | None = None,
        description: str | None = None,
        preprocessors: Sequence[ToolPreprocessor[TContext, TInput]] = (),
        postprocessors: Sequence[ToolPostprocessor[TContext, TInput, TResult]] = (),
    ) -> None:
        self._tool = tool
        self._name = name or self._infer_name(tool)
        self._description = description or self._infer_description(tool)
        self._preprocessors = preprocessors
        self._postprocessors = postprocessors

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def raw_tool(self) -> BaseTool | ContextToolFn[TContext, TInput, TResult]:
        return self._tool

    async def ainvoke(
        self,
        input: TInput,
        runtime: ChatRuntime[TContext],
    ) -> TResult:
        invocation = self._make_invocation(input, runtime)

        for preprocessor in self._preprocessors:
            input = await maybe_await(preprocessor(invocation))
            invocation = self._make_invocation(input, runtime)

        result = await self._arun_underlying_tool(input, runtime)

        for postprocessor in self._postprocessors:
            result = await maybe_await(postprocessor(invocation, result))

        return result

    def invoke(
        self,
        input: TInput,
        runtime: ChatRuntime[TContext],
    ) -> TResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(input, runtime))

        raise RuntimeError(
            "ChatTool.invoke() cannot be called from a running event loop. "
            "Use await ChatTool.ainvoke(...) instead."
        )

    @staticmethod
    def _infer_name(
        tool: BaseTool | ContextToolFn[TContext, TInput, TResult],
    ) -> str:
        if isinstance(tool, BaseTool):
            return tool.name

        return getattr(tool, "__name__", tool.__class__.__name__)

    @staticmethod
    def _infer_description(
        tool: BaseTool | ContextToolFn[TContext, TInput, TResult],
    ) -> str | None:
        if isinstance(tool, BaseTool):
            return tool.description

        return inspect.getdoc(tool)

    def _make_invocation(
        self,
        input: TInput,
        runtime: ChatRuntime[TContext],
    ) -> ToolInvocation[TContext, TInput]:
        return ToolInvocation(
            name=self._name,
            input=input,
            runtime=runtime,
            raw_tool=self._tool,
        )

    async def _arun_underlying_tool(
        self,
        input: TInput,
        runtime: ChatRuntime[TContext],
    ) -> TResult:
        if isinstance(self._tool, BaseTool):
            result = await self._tool.ainvoke(cast(LangChainToolInput, input))
            return cast(TResult, result)

        result = self._tool(input, runtime)
        return await maybe_await(result)
