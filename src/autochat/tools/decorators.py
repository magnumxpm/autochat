from collections.abc import Sequence
from typing import Callable, TypeVar

from pydantic import BaseModel

from .base import ChatTool
from .types import ContextToolFn, ToolPostprocessor, ToolPreprocessor

TContext = TypeVar("TContext")
TInput = TypeVar("TInput")
TResult = TypeVar("TResult")


def chat_tool(
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
    preprocessors: Sequence[ToolPreprocessor[TContext, TInput]] = (),
    postprocessors: Sequence[ToolPostprocessor[TContext, TInput, TResult]] = (),
) -> Callable[
    [ContextToolFn],
    ChatTool[TContext, TInput, TResult],
]:
    def decorator(
        fn: ContextToolFn,
    ) -> ChatTool[TContext, TInput, TResult]:
        return ChatTool(
            fn,
            name=name,
            description=description,
            args_schema=args_schema,
            preprocessors=preprocessors,
            postprocessors=postprocessors,
        )

    return decorator
