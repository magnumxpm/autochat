import asyncio
import inspect
from collections.abc import Sequence
from typing import Any, Callable, Generic, TypeVar, cast, get_type_hints, overload

from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

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

    3. @chat_tool decorator:
    ```python
    from autochat.tools import chat_tool

    @chat_tool(...)
    def my_tool_func():
        ...
    ```
    """

    @overload
    def __init__(
        self,
        tool: BaseTool,
        *,
        name: str | None = None,
        description: str | None = None,
        args_schema: type[BaseModel] | None = None,
        preprocessors: Sequence[ToolPreprocessor[TContext, TInput]] = (),
        postprocessors: Sequence[ToolPostprocessor[TContext, TInput, TResult]] = (),
    ) -> None: ...

    @overload
    def __init__(
        self,
        tool: ContextToolFn,
        *,
        name: str | None = None,
        description: str | None = None,
        args_schema: type[BaseModel] | None = None,
        preprocessors: Sequence[ToolPreprocessor[TContext, TInput]] = (),
        postprocessors: Sequence[ToolPostprocessor[TContext, TInput, TResult]] = (),
    ) -> None: ...

    def __init__(
        self,
        tool: BaseTool | ContextToolFn,
        *,
        name: str | None = None,
        description: str | None = None,
        args_schema: type[BaseModel] | None = None,
        preprocessors: Sequence[ToolPreprocessor[TContext, TInput]] = (),
        postprocessors: Sequence[ToolPostprocessor[TContext, TInput, TResult]] = (),
    ) -> None:
        self._tool = tool
        self._name = name or self._infer_name(tool)
        self._description = description or self._infer_description(tool)
        self._preprocessors = preprocessors
        self._postprocessors = postprocessors
        self._args_schema = args_schema or self._infer_args_schema(tool)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def raw_tool(self) -> BaseTool | ContextToolFn:
        return self._tool

    def model_tool(self) -> BaseTool | dict[str, Any]:
        if isinstance(self._tool, BaseTool):
            return self._tool

        if self._args_schema is None:
            raise ValueError(
                f"Native ChatTool '{self.name}' requires args_schema to be bound to a model."
            )

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
                "parameters": self._args_schema.model_json_schema(),
            },
        }

    def _coerce_input(self, input: Any) -> TInput:
        if isinstance(self._tool, BaseTool):
            return cast(TInput, input)

        if self._args_schema is not None and isinstance(input, dict):
            return cast(TInput, self._args_schema.model_validate(input))

        return cast(TInput, input)

    async def ainvoke(
        self,
        input: TInput,
        runtime: ChatRuntime[TContext],
    ) -> TResult:
        input = self._coerce_input(input)
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
    def _validate_native_signature(tool: Callable[..., Any]) -> None:
        signature = inspect.signature(tool)

        for parameter in signature.parameters.values():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise ValueError("@chat_tool does not support *args or **kwargs.")

    @staticmethod
    def _is_runtime_parameter(name: str) -> bool:
        return name == "runtime"

    @staticmethod
    def _is_base_model_type(annotation: Any) -> bool:
        return isinstance(annotation, type) and issubclass(annotation, BaseModel)

    @classmethod
    def _infer_native_args_schema(
        cls,
        tool: Callable[..., Any],
    ) -> type[BaseModel]:
        cls._validate_native_signature(tool)
        signature = inspect.signature(tool)

        try:
            hints = get_type_hints(tool)
        except Exception:
            hints = {}

        fields = {}
        for name, parameter in signature.parameters.items():
            if cls._is_runtime_parameter(name):
                continue

            annotation = hints.get(name, parameter.annotation)
            if annotation is inspect.Parameter.empty:
                annotation = Any

            default: Any
            if parameter.default is inspect.Parameter.empty:
                default = ...
            else:
                default = parameter.default

            fields[name] = (annotation, default)

        if not fields:
            return create_model(f"{getattr(tool, '__name__', ChatTool)}Args")

        if len(fields) == 1:
            only_annotation, _ = next(iter(fields.values()))
            if cls._is_base_model_type(only_annotation):
                return only_annotation

        return create_model(f"{getattr(tool, '__name__', ChatTool)}Args", **fields)

    @classmethod
    def _infer_args_schema(
        cls,
        tool: BaseTool | ContextToolFn,
    ) -> type[BaseModel] | None:
        if isinstance(tool, BaseTool):
            args_schema = getattr(tool, "args_schema", None)
            if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
                return args_schema
            return None

        return cls._infer_native_args_schema(tool)

    @staticmethod
    def _infer_name(
        tool: BaseTool | ContextToolFn,
    ) -> str:
        if isinstance(tool, BaseTool):
            return tool.name

        return getattr(tool, "__name__", tool.__class__.__name__)

    @staticmethod
    def _infer_description(
        tool: BaseTool | ContextToolFn,
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

    def _native_call_kwargs(
        self,
        input: Any,
        runtime: ChatRuntime[TContext],
    ) -> dict[str, Any]:
        if isinstance(self._tool, BaseTool):
            raise TypeError("LangChain BaseTool does not use native call kwargs.")

        signature = inspect.signature(self._tool)

        if isinstance(input, BaseModel):
            input_data = input.model_dump()
        elif isinstance(input, dict):
            input_data = input
        else:
            input_data = {"input": input}

        kwargs: dict[str, Any] = {}

        for name, parameter in signature.parameters.items():
            if self._is_runtime_parameter(name):
                kwargs[name] = runtime
                continue

            if name in input_data:
                kwargs[name] = input_data[name]
                continue

            if parameter.default is inspect.Parameter.empty:
                raise TypeError(f"Missing required tool argument: {name}")

        return kwargs

    async def _arun_underlying_tool(
        self,
        input: TInput,
        runtime: ChatRuntime[TContext],
    ) -> TResult:
        if isinstance(self._tool, BaseTool):
            result = await self._tool.ainvoke(cast(LangChainToolInput, input))
            return cast(TResult, result)

        kwargs = self._native_call_kwargs(input, runtime)
        result = self._tool(**kwargs)
        return await maybe_await(cast(MaybeAwaitable[TResult], result))
