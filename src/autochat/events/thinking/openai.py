from typing import Any, Iterable, Mapping
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessageChunk

from autochat.events.message import ThinkingBlock

from .base import ExtractorState, ThinkingDelta


class OpenAIReasoningExtractor:
    """Extracts reasoning summaries from OpenAI o-series / GPT-5 models.

    OpenAI does not stream reasoning tokens. A summary appears post-hoc on
    the final message under additional_kwargs.reasoning or
    response_metadata.reasoning. This extractor emits a single
    start/delta/end triplet at message-end via extract_from_final.
    """

    name = "openai-reasoning"

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
        # Responses API: reasoning shows up as content blocks.
        state.finalized.extend(self._read_content_blocks(message))

        # Chat Completions / older shapes: reasoning summary in metadata.
        summary = self._read_summary(message)
        if summary:
            state.finalized.append(ThinkingBlock(id=str(uuid4()), text=summary))

        return list(state.finalized)

    @staticmethod
    def _read_content_blocks(message: AIMessage) -> list[ThinkingBlock]:
        content = message.content
        if not isinstance(content, list):
            return []
        blocks: list[ThinkingBlock] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "reasoning":
                continue
            block_id = str(block.get("id") or uuid4())
            summary_items = block.get("summary") or []
            text_parts: list[str] = []
            for item in summary_items:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("summary_text") or ""
                    if text:
                        text_parts.append(text)
            text = "\n".join(text_parts)
            if not text:
                # Skip empty-summary reasoning blocks to avoid noisy empty events.
                continue
            blocks.append(ThinkingBlock(id=block_id, text=text))
        return blocks

    @staticmethod
    def _read_summary(message: AIMessage) -> str:
        kwargs = getattr(message, "additional_kwargs", None) or {}
        reasoning = kwargs.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
        if isinstance(reasoning, dict):
            text = reasoning.get("summary") or reasoning.get("text")
            if isinstance(text, str) and text.strip():
                return text
        metadata = getattr(message, "response_metadata", None) or {}
        text = metadata.get("reasoning_summary") or metadata.get("reasoning")
        if isinstance(text, str) and text.strip():
            return text
        return ""
