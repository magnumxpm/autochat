from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeAlias, TypeVar

from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool

from autochat.runtime import ChatRuntime

T = TypeVar("T")
TContext = TypeVar("TContext")
TInput = TypeVar("TInput")
TResult = TypeVar("TResult")

TInput_contra = TypeVar("TInput_contra", contravariant=True)
TResult_co = TypeVar("TResult_co", covariant=True)


# TInput is too generic for LangChain Tool's expected input type.
# Used in _tool.ainvoke() in base.
LangChainToolInput: TypeAlias = str | dict[str, Any] | ToolCall

MaybeAwaitable: TypeAlias = T | Awaitable[T]


@dataclass(frozen=True, slots=True)
class ToolInvocation(Generic[TContext, TInput]):
    """Data passed to `processors` when invoking a tool."""

    name: str
    input: TInput
    runtime: ChatRuntime[TContext]
    raw_tool: BaseTool | Any


class ToolPreprocessor(Protocol[TContext, TInput]):
    def __call__(
        self, invocation: ToolInvocation[TContext, TInput]
    ) -> MaybeAwaitable[TInput]: ...


class ToolPostprocessor(Protocol[TContext, TInput, TResult]):
    def __call__(
        self, invocation: ToolInvocation[TContext, TInput], result: TResult
    ) -> MaybeAwaitable[TResult]: ...


class ContextToolFn(Protocol[TContext, TInput_contra, TResult_co]):
    def __call__(
        self,
        input: TInput_contra,
        runtime: ChatRuntime[TContext],
    ) -> MaybeAwaitable[TResult_co]: ...
