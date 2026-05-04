from dataclasses import dataclass

from autochat import (
    ChatRuntime,
    ToolInvocation,
    chat_tool,
)


class UserNotAuthorizedForTool(Exception):
    pass


@dataclass
class AppContext:
    user_id: str
    org_id: str
    permissions: set[str]


def require_permission(permission: str):
    def processor(invocation: ToolInvocation[AppContext, str]) -> str:
        if permission not in invocation.runtime.context.permissions:
            raise UserNotAuthorizedForTool(f"Missing permission: {permission}")

        return invocation.input

    return processor


@chat_tool(
    name="get_billing_info",
    description="Fetch billing information for the current organization.",
    preprocessors=[require_permission("billing.read")],
)
async def get_billing_info(
    input: str,
    runtime: ChatRuntime[AppContext],
) -> str:
    org_id = runtime.context.org_id
    return f"Billing info for org {org_id}: {input}"


runtime = ChatRuntime(
    thread_id="thread_123",
    context=AppContext(
        user_id="user_1",
        org_id="org_1",
        permissions={"billing.read"},
    ),
)


async def main():
    result = await get_billing_info.ainvoke("latest invoice", runtime)
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
