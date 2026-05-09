from typing import Any, Iterable, Mapping

from langchain_core.messages import AIMessage, BaseMessageChunk

from autochat.events.message import ThinkingBlock

from .base import ExtractorState, ThinkingDelta, _OpenBlock


class AnthropicThinkingExtractor:
    """Extracts thinking/redacted_thinking blocks from Claude responses.

    Anthropic streams content blocks of the form
    `{"type": "thinking", "thinking": "...", "signature": "..."}` or
    `{"type": "redacted_thinking", "data": "..."}` alongside text blocks.
    """

    name = "anthropic"

    def extract_from_chunk(
        self,
        chunk: BaseMessageChunk,
        raw_event: Mapping[str, Any],
        state: ExtractorState,
    ) -> Iterable[ThinkingDelta]:
        content = chunk.content
        if isinstance(content, str) or content is None:
            return
        for index, block in enumerate(content):
            if not isinstance(block, dict):
                continue

            btype = block.get("type")
            if btype == "thinking":
                block_id = str(block.get("index", index))
                text = block.get("thinking") or ""
                signature = block.get("signature")
                yield from self._handle_open_block(
                    state,
                    block_id=block_id,
                    text=text,
                    signature=signature,
                    redacted=False,
                )
            elif btype == "redacted_thinking":
                block_id = str(block.get("index", index))
                yield from self._handle_open_block(
                    state,
                    block_id=block_id,
                    text=block.get("data") or "",
                    signature=None,
                    redacted=True,
                )
            elif btype == "text":
                # Text block — implicitly closes any open thinking blocks.
                yield from self._close_all(state)

    def extract_from_final(
        self,
        message: AIMessage,
        state: ExtractorState,
    ) -> list[ThinkingBlock]:
        # Close any blocks that never saw an explicit terminator.
        finalized = list(state.finalized)
        for open_block in state.open_blocks.values():
            finalized.append(open_block.finalize())
        state.open_blocks.clear()
        return finalized

    def _handle_open_block(
        self,
        state: ExtractorState,
        *,
        block_id: str,
        text: str,
        signature: str | None,
        redacted: bool,
    ) -> Iterable[ThinkingDelta]:
        existing = state.open_blocks.get(block_id)
        if existing is None:
            existing = _OpenBlock(id=block_id, redacted=redacted)
            state.open_blocks[block_id] = existing
            yield ThinkingDelta(block_id=block_id, kind="start")

        if signature:
            existing.signature = signature
        if text:
            existing.append(text)
            yield ThinkingDelta(block_id=block_id, kind="delta", text=text)

    def _close_all(self, state: ExtractorState) -> Iterable[ThinkingDelta]:
        for block_id in list(state.open_blocks):
            open_block = state.open_blocks.pop(block_id)
            block = open_block.finalize()
            state.finalized.append(block)
            yield ThinkingDelta(block_id=block_id, kind="end", block=block)
