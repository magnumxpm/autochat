from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChatGuideline:
    content: str
