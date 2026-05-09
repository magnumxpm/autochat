from typing import Any, Iterable, Mapping

from langchain_core.messages import AIMessage, BaseMessageChunk

from autochat.events.message import ThinkingBlock

from .base import ExtractorState, ThinkingDelta, _OpenBlock

_BLOCK_ID = "reasoning"


class DeepSeekThinkingExtractor:
    """Extracts reasoning_content from DeepSeek-style chunks."""

    name = "deepseek"

    def extract_from_chunk(
        self,
        chunk: BaseMessageChunk,
        raw_event: Mapping[str, Any],
        state: ExtractorState,
    ) -> Iterable[ThinkingDelta]:
        text = self._read_reasoning(chunk)
        if not text:
            return

        existing = state.open_blocks.get(_BLOCK_ID)
        if existing is None:
            existing = _OpenBlock(id=_BLOCK_ID)
            state.open_blocks[_BLOCK_ID] = existing
            yield ThinkingDelta(block_id=_BLOCK_ID, kind="start")

        existing.append(text)
        yield ThinkingDelta(block_id=_BLOCK_ID, kind="delta", text=text)

    def extract_from_final(
        self,
        message: AIMessage,
        state: ExtractorState,
    ) -> list[ThinkingBlock]:
        finalized = list(state.finalized)
        for open_block in state.open_blocks.values():
            finalized.append(open_block.finalize())
        state.open_blocks.clear()
        return finalized

    @staticmethod
    def _read_reasoning(chunk: BaseMessageChunk) -> str:
        kwargs = getattr(chunk, "additional_kwargs", None) or {}
        text = kwargs.get("reasoning_content")
        if isinstance(text, str):
            return text
        return ""
