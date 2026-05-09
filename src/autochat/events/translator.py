from typing import Any, Iterable, Mapping
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk

from autochat.retrieval import RetrievedDocument
from autochat.runtime import ChatRuntime

from .dispatch import (
    COMPRESSION_END,
    COMPRESSION_START,
    ERROR,
    RETRIEVER_REQUEST,
    RETRIEVER_RESPONSE,
    TOOL_REQUEST,
    TOOL_RESPONSE,
)
from .message import AssistantMessage
from .thinking.base import ExtractorState, ThinkingExtractor
from .types import (
    AutoChatEvent,
    CompressionEndEvent,
    CompressionStartEvent,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    RetrieverRequestEvent,
    RetrieverResponseEvent,
    RunEndEvent,
    RunStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallRequestEvent,
    ToolCallResponseEvent,
)


class EventTranslator:
    def __init__(
        self,
        *,
        runtime: ChatRuntime[Any],
        extractor: ThinkingExtractor,
        input_text: str,
    ) -> None:
        self._runtime = runtime
        self._extractor = extractor
        self._input_text = input_text
        self._thinking_state = ExtractorState()
        self._current_message_id: str | None = None
        self._root_chain_id: str | None = None
        self._run_started = False
        self._run_ended = False

    def translate(self, lc_event: Mapping[str, Any]) -> Iterable[AutoChatEvent]:
        event = lc_event.get("event")
        if event == "on_chain_start" and self._is_root_chain(lc_event):
            yield from self._on_run_start(lc_event)
        elif event == "on_chain_end" and self._is_root_chain(lc_event):
            yield from self._on_run_end(lc_event)
        elif event == "on_chat_model_start":
            yield from self._on_model_start(lc_event)
        elif event == "on_chat_model_stream":
            yield from self._on_model_stream(lc_event)
        elif event == "on_chat_model_end":
            yield from self._on_model_end(lc_event)
        elif event == "on_custom_event":
            yield from self._on_custom_event(lc_event)

    def _meta(self) -> dict[str, Any]:
        return {
            "run_id": self._runtime.run_id or "",
            "thread_id": self._runtime.thread_id,
        }

    def _is_root_chain(self, lc_event: Mapping[str, Any]) -> bool:
        # The first on_chain_start we observe is the graph root.
        run_id = lc_event.get("run_id")
        if self._root_chain_id is None and lc_event.get("event") == "on_chain_start":
            self._root_chain_id = run_id
            return True
        return run_id == self._root_chain_id

    def _on_run_start(self, lc_event: Mapping[str, Any]) -> Iterable[AutoChatEvent]:
        if self._run_started:
            return
        self._run_started = True
        yield RunStartEvent(input=self._input_text, **self._meta())

    def _on_run_end(self, lc_event: Mapping[str, Any]) -> Iterable[AutoChatEvent]:
        if self._run_ended:
            return
        self._run_ended = True
        yield RunEndEvent(**self._meta())

    def _on_model_start(self, lc_event: Mapping[str, Any]) -> Iterable[AutoChatEvent]:
        message_id = str(lc_event.get("run_id") or uuid4())
        self._current_message_id = message_id
        self._thinking_state = ExtractorState()
        yield MessageStartEvent(message_id=message_id, **self._meta())

    def _on_model_stream(self, lc_event: Mapping[str, Any]) -> Iterable[AutoChatEvent]:
        if self._current_message_id is None:
            return
        chunk = (lc_event.get("data") or {}).get("chunk")
        if not isinstance(chunk, AIMessageChunk):
            return

        for delta in self._extractor.extract_from_chunk(
            chunk, lc_event, self._thinking_state
        ):
            if delta.kind == "start":
                yield ThinkingStartEvent(block_id=delta.block_id, **self._meta())
            elif delta.kind == "delta":
                yield ThinkingDeltaEvent(
                    block_id=delta.block_id, delta=delta.text, **self._meta()
                )
            elif delta.kind == "end" and delta.block is not None:
                yield ThinkingEndEvent(block=delta.block, **self._meta())

        text = self._extract_text_delta(chunk)
        if text:
            yield MessageDeltaEvent(
                message_id=self._current_message_id, delta=text, **self._meta()
            )

    def _on_model_end(self, lc_event: Mapping[str, Any]) -> Iterable[AutoChatEvent]:
        if self._current_message_id is None:
            return
        output = (lc_event.get("data") or {}).get("output")
        if not isinstance(output, AIMessage):
            self._current_message_id = None
            return

        thinking_blocks = self._extractor.extract_from_final(
            output, self._thinking_state
        )
        message = AssistantMessage.from_ai_message(
            output, thinking=tuple(thinking_blocks)
        )
        self._current_message_id = None
        yield MessageEndEvent(message=message, **self._meta())

    def _on_custom_event(self, lc_event: Mapping[str, Any]) -> Iterable[AutoChatEvent]:
        name = lc_event.get("name")
        data = lc_event.get("data") or {}

        if name == TOOL_REQUEST:
            yield ToolCallRequestEvent(
                tool_call_id=data.get("tool_call_id") or "",
                name=data.get("name") or "",
                args=data.get("args") or {},
                **self._meta(),
            )
        elif name == TOOL_RESPONSE:
            yield ToolCallResponseEvent(
                tool_call_id=data.get("tool_call_id") or "",
                name=data.get("name") or "",
                result=data.get("result"),
                error=data.get("error"),
                **self._meta(),
            )
        elif name == RETRIEVER_REQUEST:
            yield RetrieverRequestEvent(
                tool_call_id=data.get("tool_call_id") or "",
                name=data.get("name") or "",
                query=data.get("query") or "",
                **self._meta(),
            )
        elif name == RETRIEVER_RESPONSE:
            documents = data.get("documents") or []
            typed_docs = [d for d in documents if isinstance(d, RetrievedDocument)]
            yield RetrieverResponseEvent(
                tool_call_id=data.get("tool_call_id") or "",
                name=data.get("name") or "",
                documents=typed_docs,
                error=data.get("error"),
                **self._meta(),
            )
        elif name == COMPRESSION_START:
            yield CompressionStartEvent(
                strategy=data.get("strategy") or "",
                message_count_before=int(data.get("message_count_before") or 0),
                **self._meta(),
            )
        elif name == COMPRESSION_END:
            yield CompressionEndEvent(
                strategy=data.get("strategy") or "",
                compressed=bool(data.get("compressed")),
                message_count_after=int(data.get("message_count_after") or 0),
                **self._meta(),
            )
        elif name == ERROR:
            yield ErrorEvent(
                node=data.get("node") or "",
                error=data.get("error") or "",
                **self._meta(),
            )

    @staticmethod
    def _extract_text_delta(chunk: AIMessageChunk) -> str:
        content = chunk.content
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for block in content or []:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "".join(parts)
