from typing import Annotated, NotRequired, TypedDict

from langgraph.graph import add_messages
from langgraph.graph.message import BaseMessage

from autochat.retrieval import RetrievedDocument


class ChatGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_documents: NotRequired[list[RetrievedDocument]]


class ChatGraphUpdate(TypedDict, total=False):
    messages: list[BaseMessage]
    retrieved_documents: list[RetrievedDocument]
