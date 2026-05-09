from dataclasses import dataclass
from typing import Any

from langchain.chat_models import BaseChatModel


@dataclass(frozen=True, slots=True)
class ChatConfig:
    model: BaseChatModel
    model_kwargs: dict[str, Any] | None = None
    context_window: int | None = None
