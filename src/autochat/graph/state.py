from typing import Annotated, TypedDict

from langgraph.graph import add_messages
from langgraph.graph.message import BaseMessage


class ChatGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class ChatGraphUpdate(TypedDict, total=False):
    messages: list[BaseMessage]
