from collections.abc import Sequence
from typing import Callable, TypeVar

from .base import ChatTool
from .types import ContextToolFn, ToolPostprocessor, ToolPreprocessor

TContext = TypeVar("TContext")
TInput = TypeVar("TInput")
TResult = TypeVar("TResult")


def chat_tool(
    *,
    name: str | None = None,
    description: str | None = None,
    preprocessors: Sequence[ToolPreprocessor[TContext, TInput]] = (),
    postprocessors: Sequence[ToolPostprocessor[TContext, TInput, TResult]] = (),
) -> Callable[
    [ContextToolFn[TContext, TInput, TResult]],
    ChatTool[TContext, TInput, TResult],
]:
    def decorator(
        fn: ContextToolFn[TContext, TInput, TResult],
    ) -> ChatTool[TContext, TInput, TResult]:
        return ChatTool(
            fn,
            name=name,
            description=description,
            preprocessors=preprocessors,
            postprocessors=postprocessors,
        )

    return decorator
