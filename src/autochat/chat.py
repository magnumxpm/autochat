from typing import Any, AsyncIterator, Generic, Literal, Mapping, Sequence, TypeVar
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from autochat.compression import AutoCompress
from autochat.config import ChatConfig
from autochat.events import AutoChatEvent, ErrorEvent, EventTranslator
from autochat.events.thinking import ThinkingExtractor, resolve_extractor
from autochat.exceptions import HITLPersistenceRequiredError
from autochat.graph.builder import build_chat_graph
from autochat.graph.runtime import with_runtime_config
from autochat.guidelines import ChatGuideline
from autochat.hitl import HITL, HITLHandle
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
        hitl: HITL | None = None,
    ) -> None:
        if hitl is not None and persistence is None:
            raise HITLPersistenceRequiredError(
                "AutoChat was configured with `hitl=...` but no `persistence` "
                "checkpointer was provided. HITL requires a checkpointer to pause "
                "and resume graph execution across requests."
            )

        self.config = config
        self.tools = tuple(tools)
        self.retrievers = tuple(retrievers)
        self.retrieval = retrieval or RetrievalConfig()
        self.system_message = system_message
        self.guidelines = tuple(guidelines)
        self.persistence = persistence
        self.compression = compression
        self.hitl = hitl
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
            hitl=hitl,
        )

    def _runtime(
        self,
        *,
        thread_id: str,
        context: TContext,
        run_id: str | None,
        metadata: Mapping[str, Any] | None,
        hitl_enabled: bool,
    ) -> ChatRuntime[TContext]:
        handle: HITLHandle | None = None
        if hitl_enabled and self.hitl is not None:
            handle = HITLHandle(allow_questions=self.hitl.allow_questions)

        return ChatRuntime(
            thread_id=thread_id,
            context=context,
            run_id=run_id or str(uuid4()),
            metadata=metadata or {},
            hitl=handle,
        )

    def _graph_input(self, input: str | None, resume: Any) -> Any:
        """Build the LangGraph input for ainvoke/astream_events.

        Passes `Command(resume=...)` when resuming a paused HITL interrupt, or a
        normal `{"messages": [HumanMessage(...)]}` payload for a fresh user turn.
        """
        if resume is not None:
            return Command(resume=resume)
        if input is None:
            raise ValueError(
                "Either `input` (a new user message) or `resume` (a HITL response) "
                "must be provided."
            )
        return {"messages": [HumanMessage(content=input)]}

    async def ainvoke(
        self,
        input: str | None = None,
        *,
        thread_id: str,
        context: TContext,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        config: RunnableConfig | None = None,
        resume: Any = None,
        hitl: bool | None = None,
    ) -> Any:
        runtime = self._runtime(
            thread_id=thread_id,
            context=context,
            run_id=run_id,
            metadata=metadata,
            hitl_enabled=self._hitl_enabled(hitl),
        )

        graph_config = with_runtime_config(config, runtime)
        graph_input = self._graph_input(input, resume)

        return await self._graph.ainvoke(graph_input, config=graph_config)

    async def astream_events(
        self,
        input: str | None = None,
        *,
        thread_id: str,
        context: TContext,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        config: RunnableConfig | None = None,
        version: Literal["v1", "v2"] = "v2",
        resume: Any = None,
        hitl: bool | None = None,
    ) -> AsyncIterator[AutoChatEvent]:
        runtime = self._runtime(
            thread_id=thread_id,
            context=context,
            run_id=run_id,
            metadata=metadata,
            hitl_enabled=self._hitl_enabled(hitl),
        )

        graph_config = with_runtime_config(config, runtime)
        graph_input = self._graph_input(input, resume)
        translator = EventTranslator(
            runtime=runtime,
            extractor=self._thinking_extractor,
            input_text=input or "",
        )

        try:
            async for raw in self._graph.astream_events(
                graph_input,
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
        input: str | None = None,
        *,
        thread_id: str,
        context: TContext,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        config: RunnableConfig | None = None,
        version: Literal["v1", "v2"] = "v2",
        resume: Any = None,
        hitl: bool | None = None,
    ) -> AsyncIterator[Any]:
        """Escape hatch: yields raw LangChain stream events (v2)."""
        runtime = self._runtime(
            thread_id=thread_id,
            context=context,
            run_id=run_id,
            metadata=metadata,
            hitl_enabled=self._hitl_enabled(hitl),
        )
        graph_config = with_runtime_config(config, runtime)
        graph_input = self._graph_input(input, resume)
        async for event in self._graph.astream_events(
            graph_input,
            config=graph_config,
            version=version,
        ):
            yield event

    def _hitl_enabled(self, override: bool | None) -> bool:
        """Resolve the effective HITL setting for a single call.

        Defaults to whether AutoChat was constructed with `hitl=...`. A per-call
        `hitl=False` disables HITL for that run (useful for evals / cron). A
        per-call `hitl=True` is rejected when the chat instance has no HITL config.
        """
        if override is None:
            return self.hitl is not None
        if override and self.hitl is None:
            raise ValueError(
                "Per-call `hitl=True` requires AutoChat to be constructed with "
                "`hitl=HITL(...)`."
            )
        return bool(override)
