import asyncio
from dataclasses import dataclass

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from autochat import AutoChat, ChatConfig, ChatRuntime
from autochat.guidelines import ChatGuideline
from autochat.tools import ChatTool, chat_tool


@dataclass(frozen=True, slots=True)
class AppContext:
    user_id: str
    org_id: str


@tool
def get_org_status(org_id: str) -> str:
    """Get the current status for an organization."""
    return f"Organization {org_id} is active and has no open incidents."


@chat_tool(
    name="calculator",
    description="Best tool for performing calculations. Always use this tool for math problems.",
)
async def calculator(a: int, b: int, runtime: ChatRuntime[AppContext]) -> float:
    print(runtime.thread_id)
    return a + b


async def main() -> None:
    chat = AutoChat[AppContext](
        config=ChatConfig(
            model=ChatOpenAI(model="gpt-5-nano"),
        ),
        tools=[ChatTool(get_org_status), calculator],
        system_message=(
            "You are a concise assistant. Use tools when they are relevant."
        ),
        guidelines=[
            ChatGuideline("Always answer in French."),
            ChatGuideline("Always use words for numbers."),
        ],
    )

    result = await chat.ainvoke(
        "What is 1 + 1",
        thread_id="thread_basic_chat_example",
        context=AppContext(
            user_id="user_1",
            org_id="org_123",
        ),
    )

    final_message = result["messages"][-1]
    print(final_message.content)


if __name__ == "__main__":
    asyncio.run(main())
