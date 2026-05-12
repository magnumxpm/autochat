"""Demonstrates AutoChat's HITL (human-in-the-loop) support.

Three HITL flavors are shown in a single example:

1. Declarative tool approval (before invocation) — `refund_order` is gated
   with `ApprovalSpec(...)`. The model proposes a refund; AutoChat halts the
   graph; the host approves (and overrides the amount via a custom schema);
   the tool then runs with `runtime.hitl.approval` populated.

2. In-tool custom HITL (mid-execution) — `cancel_order` checks admin auth
   and order existence first, and only then asks the user to confirm via
   `runtime.hitl.request_approval` with its own response schema that
   includes a `reason` field.

3. Model-asked questions — `HITL(allow_questions=True)` injects a built-in
   `ask_user` tool. When the user prompt is missing required info, the
   model calls `ask_user("Which order?")` and the host replies.

The driver simulates a host application: it streams events until the run
either ends or pauses on a HITL request, prints what's pending, returns a
canned response, and resumes by calling `astream_events` again with
`resume=<response>`.

Run:
    OPENAI_API_KEY=... uv run --dev python examples/hitl_example.py
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from autochat import (
    HITL,
    ApprovalSpec,
    AutoChat,
    AutoChatEvent,
    ChatConfig,
    ChatRuntime,
    HITLRequest,
    HITLRequestedEvent,
    HITLResolvedEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    RunEndEvent,
    RunStartEvent,
    ToolCallRequestEvent,
    ToolCallResponseEvent,
    chat_tool,
)


# Domain ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AppContext:
    user_id: str
    is_admin: bool


ORDERS: dict[str, dict[str, Any]] = {
    "ORD-42": {"customer": "Alice", "total": 79.00, "status": "shipped"},
    "ORD-77": {"customer": "Bob", "total": 250.00, "status": "delivered"},
}


# Tools -----------------------------------------------------------------------


@chat_tool(
    name="lookup_order",
    description="Fetch order details by order id. Call before refund or cancel.",
)
async def lookup_order(
    order_id: str,
    runtime: ChatRuntime[AppContext],
) -> dict[str, Any]:
    del runtime
    return ORDERS.get(order_id, {"error": f"order {order_id} not found"})


# (1) Declarative approval — host can override the refund amount through a
# custom response schema. The tool reads the validated decision from
# `runtime.hitl.approval`.
class RefundApproval(BaseModel):
    approved: bool
    final_amount: float | None = None
    note: str | None = None


@chat_tool(
    name="refund_order",
    description="Refund an order. Always look up the order first.",
    approval=ApprovalSpec(
        summary=lambda inv: (
            f"Refund {inv.input['order_id']} for ${inv.input['amount']:.2f}?"
        ),
        payload=lambda inv: dict(inv.input),
        response_schema=RefundApproval,
    ),
)
async def refund_order(
    order_id: str,
    amount: float,
    runtime: ChatRuntime[AppContext],
) -> str:
    approval: RefundApproval = runtime.hitl.approval  # type: ignore[assignment]
    final = approval.final_amount if approval.final_amount is not None else amount
    note = f" (note: {approval.note})" if approval.note else ""
    return f"Refunded ${final:.2f} to {order_id}{note}."


# (2) In-tool custom HITL — auth and existence checks run first; only then
# do we pause to ask the user, so denied callers get a fast error and never
# trigger an approval prompt.
class CancelApproval(BaseModel):
    approved: bool
    reason: str = "no reason given"


@chat_tool(
    name="cancel_order",
    description="Cancel an order. Requires admin and explicit user confirmation.",
)
async def cancel_order(
    order_id: str,
    runtime: ChatRuntime[AppContext],
) -> str:
    if not runtime.context.is_admin:
        return "Cancel denied: admin only."
    if order_id not in ORDERS:
        return f"Cancel denied: {order_id} not found."

    decision = await runtime.hitl.request_approval(
        summary=f"Cancel {order_id}?",
        response_schema=CancelApproval,
        payload={
            "order_id": order_id,
            "current_status": ORDERS[order_id]["status"],
        },
    )
    if not decision.approved:
        return f"Cancel skipped: {decision.reason}"

    ORDERS[order_id]["status"] = "cancelled"
    return f"Cancelled {order_id} (reason: {decision.reason})."


# Simulated host --------------------------------------------------------------


def simulate_user(request: HITLRequest) -> Mapping[str, Any]:
    """Return a host response for a pending HITL request.

    In a real application this is where you would surface a UI to the end-user
    and collect their input. Here we just return canned responses keyed off
    the request kind and tool name.
    """
    if request.kind == "user_question":
        # The model is asking a free-form clarifying question.
        return {"answer": "ORD-42"}

    if request.kind == "tool_approval" and request.tool_name == "refund_order":
        # Approve, but cap the refund and attach a note. The tool will read
        # these values from `runtime.hitl.approval` and adjust the refund.
        return {"approved": True, "final_amount": 50.00, "note": "partial per policy"}

    if request.kind == "custom" and request.tool_name == "cancel_order":
        return {"approved": True, "reason": "customer requested"}

    return {"approved": True}


# Console rendering -----------------------------------------------------------


DIM = "\033[2m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
GREEN = "\033[32m"
RESET = "\033[0m"


def render(event: AutoChatEvent) -> None:
    if isinstance(event, RunStartEvent):
        print(f"\n{BLUE}[run.start]{RESET} input={event.input!r}", flush=True)
    elif isinstance(event, MessageDeltaEvent):
        print(event.delta, end="", flush=True)
    elif isinstance(event, MessageEndEvent):
        print()
    elif isinstance(event, ToolCallRequestEvent):
        print(
            f"\n{YELLOW}[tool.request]{RESET} {event.name}({dict(event.args)})",
            flush=True,
        )
    elif isinstance(event, ToolCallResponseEvent):
        print(
            f"{YELLOW}[tool.response]{RESET} {event.name} -> {event.result!r}",
            flush=True,
        )
    elif isinstance(event, HITLRequestedEvent):
        req = event.request
        print(
            f"\n{MAGENTA}[hitl.requested:{req.kind}]{RESET} "
            f"{req.summary or ''} (tool={req.tool_name})",
            flush=True,
        )
        if req.payload:
            print(f"  {DIM}payload: {dict(req.payload)}{RESET}", flush=True)
        if req.response_schema is not None:
            fields = ", ".join(req.response_schema.model_fields)
            print(
                f"  {DIM}expected response fields: {fields}{RESET}", flush=True
            )
    elif isinstance(event, HITLResolvedEvent):
        print(
            f"{MAGENTA}[hitl.resolved]{RESET} {dict(event.response)}", flush=True
        )
    elif isinstance(event, RunEndEvent):
        print(f"{BLUE}[run.end]{RESET}", flush=True)


# Driver: stream → detect HITL pause → resume until done ----------------------


async def stream_one_segment(
    chat: AutoChat[AppContext],
    *,
    prompt: str | None,
    resume: Any,
    thread_id: str,
    context: AppContext,
) -> HITLRequest | None:
    """Stream one execution segment (a fresh turn OR a resume).

    Returns the unresolved `HITLRequest` if the graph paused on a HITL
    interrupt, or `None` if the run completed. On a resumed stream the
    graph re-executes the paused node from the top, so `HITLRequestedEvent`
    can re-fire for already-resolved requests — we treat a request as
    "pending" only when no matching `HITLResolvedEvent` followed it.
    """
    requested: dict[str, HITLRequest] = {}
    resolved_ids: set[str] = set()

    async for event in chat.astream_events(
        prompt,
        thread_id=thread_id,
        context=context,
        resume=resume,
    ):
        render(event)
        if isinstance(event, HITLRequestedEvent):
            requested[event.request.request_id] = event.request
        elif isinstance(event, HITLResolvedEvent):
            resolved_ids.add(event.request_id)

    for request_id, request in requested.items():
        if request_id not in resolved_ids:
            return request
    return None


async def run_turn(
    chat: AutoChat[AppContext],
    *,
    prompt: str,
    thread_id: str,
    context: AppContext,
) -> None:
    print(f"\n{GREEN}>>> user: {prompt}{RESET}", flush=True)

    pending = await stream_one_segment(
        chat,
        prompt=prompt,
        resume=None,
        thread_id=thread_id,
        context=context,
    )

    while pending is not None:
        response = simulate_user(pending)
        print(f"\n{DIM}--- host responds: {response} ---{RESET}", flush=True)
        pending = await stream_one_segment(
            chat,
            prompt=None,
            resume=response,
            thread_id=thread_id,
            context=context,
        )


# Main ------------------------------------------------------------------------


async def main() -> None:
    chat = AutoChat[AppContext](
        config=ChatConfig(model=ChatOpenAI(model="gpt-5-nano")),
        tools=[lookup_order, refund_order, cancel_order],
        persistence=InMemorySaver(),
        hitl=HITL(allow_questions=True),
        system_message=(
            "You are an operations assistant for an order management system. "
            "If the user doesn't give you the order id you need, use the "
            "ask_user tool to ask them for it. Always look up an order with "
            "lookup_order before issuing a refund or cancellation."
        ),
    )

    admin = AppContext(user_id="alice", is_admin=True)
    thread_id = "thread_hitl_example"

    # Turn 1: user omits the order id → model calls ask_user. After the host
    # supplies it, the model looks up the order and calls refund_order, which
    # is gated by an ApprovalSpec — two HITL pauses in a single turn.
    await run_turn(
        chat,
        prompt="I want to refund my last order. Charge back $80 please.",
        thread_id=thread_id,
        context=admin,
    )

    # Turn 2: a known order id is given. cancel_order runs auth + existence
    # checks first, and only then issues an in-tool HITL request with its
    # own custom schema (CancelApproval).
    await run_turn(
        chat,
        prompt="Please cancel order ORD-77 — the customer asked us to.",
        thread_id=thread_id,
        context=admin,
    )


if __name__ == "__main__":
    asyncio.run(main())
