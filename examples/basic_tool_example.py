from dataclasses import dataclass

from pydantic import BaseModel

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


class BillingInput(BaseModel):
    invoice_id: str


def require_permission(permission: str):
    def processor(invocation: ToolInvocation[AppContext, BillingInput]) -> BillingInput:
        if permission not in invocation.runtime.context.permissions:
            raise UserNotAuthorizedForTool(f"Missing permission: {permission}")

        return invocation.input

    return processor


def hello_world():
    def processor(invocation):
        print("Hello World!")
        invocation.input.invoice_id = "Hello World!"
        return invocation.input

    return processor


@chat_tool(
    name="get_billing_info",
    description="Fetch billing information for the current organization.",
    preprocessors=[require_permission("billing.read"), hello_world()],
    args_schema=BillingInput,
)
async def get_billing_info(
    input: BillingInput,
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
    result = await get_billing_info.ainvoke(BillingInput(invoice_id="123"), runtime)
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
