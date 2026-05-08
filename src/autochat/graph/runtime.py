from typing import Any, Mapping, TypeVar, cast

from langchain_core.runnables import RunnableConfig

from autochat.runtime import ChatRuntime

TContext = TypeVar("TContext")
_RUNTIME_CONFIG_KEY = "autochat_runtime"


def with_runtime_config(
    config: RunnableConfig | None,
    runtime: ChatRuntime[TContext],
) -> RunnableConfig:
    merged: dict[str, Any] = dict(config or {})
    configurable = dict(merged.get("configurable", {}))

    configurable["thread_id"] = runtime.thread_id
    configurable[_RUNTIME_CONFIG_KEY] = runtime

    merged["configurable"] = configurable
    return cast(RunnableConfig, merged)


def get_runtime(
    config: RunnableConfig,
) -> ChatRuntime[Any]:
    configurable = cast(Mapping[str, Any], config.get("configurable") or {})
    runtime = configurable.get(_RUNTIME_CONFIG_KEY)

    if runtime is None:
        raise RuntimeError("AutoChat runtime missing from LangGraph config.")

    return cast(ChatRuntime[Any], runtime)
