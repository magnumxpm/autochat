from typing import Any, Iterable, Mapping

from langchain_core.messages import AIMessage, BaseMessageChunk

from autochat.events.message import ThinkingBlock

from .base import ExtractorState, ThinkingDelta


class NoOpExtractor:
    name = "noop"

    def extract_from_chunk(
        self,
        chunk: BaseMessageChunk,
        raw_event: Mapping[str, Any],
        state: ExtractorState,
    ) -> Iterable[ThinkingDelta]:
        return ()

    def extract_from_final(
        self,
        message: AIMessage,
        state: ExtractorState,
    ) -> list[ThinkingBlock]:
        return list(state.finalized)
