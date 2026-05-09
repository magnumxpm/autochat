from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from langchain_core.messages import AIMessage, BaseMessageChunk

from autochat.events.message import ThinkingBlock


@dataclass
class _OpenBlock:
    id: str
    text_parts: list[str] = field(default_factory=list)
    signature: str | None = None
    redacted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, text: str) -> None:
        self.text_parts.append(text)

    def finalize(self) -> ThinkingBlock:
        return ThinkingBlock(
            id=self.id,
            text="".join(self.text_parts),
            signature=self.signature,
            redacted=self.redacted,
            metadata=dict(self.metadata),
        )


@dataclass
class ExtractorState:
    """Per-stream extractor state, mutated by the extractor across chunks"""

    open_blocks: dict[str, _OpenBlock] = field(default_factory=dict)
    finalized: list[ThinkingBlock] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    """Outcome from a chunk: zero or more transitions"""

    block_id: str
    kind: str  # "start", "end", "delta"
    text: str = ""
    block: ThinkingBlock | None = None  # populated for "end" blocks


@runtime_checkable
class ThinkingExtractor(Protocol):
    """Strategy for extracting reasoning/CoT blocks from model response"""

    name: str

    def extract_from_chunk(
        self,
        chunk: BaseMessageChunk,
        raw_event: Mapping[str, Any],
        state: ExtractorState,
    ) -> Iterable[ThinkingDelta]: ...

    def extract_from_final(
        self,
        message: AIMessage,
        state: ExtractorState,
    ) -> list[ThinkingBlock]: ...
