from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, TypeVar

from pydantic import BaseModel

from .base import ChatTool
from .types import ContextToolFn, ToolPostprocessor, ToolPreprocessor

if TYPE_CHECKING:
    from autochat.hitl import ApprovalSpec

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
    approval: "ApprovalSpec | bool | None" = None,
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
            approval=approval,
        )

    return decorator
