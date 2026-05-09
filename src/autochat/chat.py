from typing import Any, AsyncIterator, Generic, Literal, Mapping, Sequence, TypeVar
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

from autochat.compression import AutoCompress
from autochat.config import ChatConfig
from autochat.events import AutoChatEvent, ErrorEvent, EventTranslator
from autochat.events.thinking import ThinkingExtractor, resolve_extractor
from autochat.graph.builder import build_chat_graph
from autochat.graph.runtime import with_runtime_config
from autochat.guidelines import ChatGuideline
from autochat.retrieval import ChatRetriever, RetrievalConfig
from autochat.runtime import ChatRuntime
from autochat.tools import ChatTool

TContext = TypeVar("TContext")


class AutoChat(Generic[TContext]):
    def __init__(
        self,
        *,
        config: ChatConfig,
        tools: Sequence[ChatTool[TContext, Any, Any]] = (),
        retrievers: Sequence[ChatRetriever[TContext]] = (),
        retrieval: RetrievalConfig[TContext] | None = None,
        persistence: BaseCheckpointSaver | None = None,
        compression: AutoCompress[TContext] | None = None,
        system_message: str | None = None,
        guidelines: Sequence[ChatGuideline] = (),
        thinking_extractor: ThinkingExtractor | None = None,
    ) -> None:
        self.config = config
        self.tools = tuple(tools)
        self.retrievers = tuple(retrievers)
        self.retrieval = retrieval or RetrievalConfig()
        self.system_message = system_message
        self.guidelines = tuple(guidelines)
        self.persistence = persistence
        self.compression = compression
        self._thinking_extractor = thinking_extractor or resolve_extractor(config.model)

        self._graph = build_chat_graph(
            config=config,
            tools=self.tools,
            retrievers=self.retrievers,
            retrieval=self.retrieval,
            compression=compression,
            system_message=system_message,
            guidelines=self.guidelines,
            persistence=persistence,
        )

    def _runtime(
        self,
        *,
        thread_id: str,
        context: TContext,
        run_id: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> ChatRuntime[TContext]:
        return ChatRuntime(
            thread_id=thread_id,
            context=context,
            run_id=run_id or str(uuid4()),
            metadata=metadata or {},
        )

    async def ainvoke(
        self,
        input: str,
        *,
        thread_id: str,
        context: TContext,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        config: RunnableConfig | None = None,
    ) -> Any:
        runtime = self._runtime(
            thread_id=thread_id,
            context=context,
            run_id=run_id,
            metadata=metadata,
        )

        graph_config = with_runtime_config(config, runtime)

        return await self._graph.ainvoke(
            {"messages": [HumanMessage(content=input)]}, config=graph_config
        )

    async def astream_events(
        self,
        input: str,
        *,
        thread_id: str,
        context: TContext,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        config: RunnableConfig | None = None,
        version: Literal["v1", "v2"] = "v2",
    ) -> AsyncIterator[AutoChatEvent]:
        runtime = self._runtime(
            thread_id=thread_id,
            context=context,
            run_id=run_id,
            metadata=metadata,
        )

        graph_config = with_runtime_config(config, runtime)
        translator = EventTranslator(
            runtime=runtime,
            extractor=self._thinking_extractor,
            input_text=input,
        )

        try:
            async for raw in self._graph.astream_events(
                {"messages": [HumanMessage(content=input)]},
                config=graph_config,
                version=version,
            ):
                for typed in translator.translate(raw):
                    yield typed
        except Exception as e:
            yield ErrorEvent(
                run_id=runtime.run_id or "",
                thread_id=runtime.thread_id,
                node="autochat",
                error=str(e),
            )
            raise

    async def astream_raw_events(
        self,
        input: str,
        *,
        thread_id: str,
        context: TContext,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        config: RunnableConfig | None = None,
        version: Literal["v1", "v2"] = "v2",
    ) -> AsyncIterator[Any]:
        """Escape hatch: yields raw LangChain stream events (v2)."""
        runtime = self._runtime(
            thread_id=thread_id,
            context=context,
            run_id=run_id,
            metadata=metadata,
        )
        graph_config = with_runtime_config(config, runtime)
        async for event in self._graph.astream_events(
            {"messages": [HumanMessage(content=input)]},
            config=graph_config,
            version=version,
        ):
            yield event
