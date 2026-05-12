from typing import Any

from autochat.exceptions import HITLError
from autochat.runtime import ChatRuntime


async def _ask_user_impl(question: str, runtime: ChatRuntime[Any]) -> str:
    if runtime.hitl is None:
        raise HITLError(
            "The built-in 'ask_user' tool was invoked but HITL is not configured "
            "for this AutoChat run."
        )
    return await runtime.hitl.ask_user(question, tool_name="ask_user")


def build_ask_user_tool() -> Any:
    """Construct the built-in `ask_user` ChatTool.

    Imported lazily here (not at module load) to avoid a circular import between
    `autochat.tools` and `autochat.hitl`.
    """
    from autochat.tools import ChatTool

    _ask_user_impl.__name__ = "ask_user"
    _ask_user_impl.__doc__ = (
        "Ask the end-user a clarifying or follow-up question and return their reply. "
        "Use this when you need information from the user to proceed."
    )

    return ChatTool(
        _ask_user_impl,
        name="ask_user",
        description=(
            "Ask the end-user a clarifying or follow-up question and return their "
            "reply. Use when you need information from the user to proceed."
        ),
    )
