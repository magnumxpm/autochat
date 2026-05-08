import asyncio
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from autochat import (
    AutoChat,
    ChatConfig,
    ChatRetriever,
    ChatRuntime,
    RetrievalConfig,
    RetrievedDocument,
)


@dataclass(frozen=True, slots=True)
class AppContext:
    user_id: str
    org_id: str


DOCUMENTS = {
    "org_123": [
        RetrievedDocument(
            content="Refunds are available within fourteen days of purchase.",
            metadata={"source": "billing_policy.md"},
        ),
        RetrievedDocument(
            content="Enterprise plans include priority support and audit logs.",
            metadata={"source": "enterprise_plan.md"},
        ),
    ]
}


async def search_org_docs(
    query: str,
    runtime: ChatRuntime[AppContext],
) -> list[RetrievedDocument]:
    print(f"Searching docs for query: {query}")

    # TODO: implement search logic
    return DOCUMENTS.get(runtime.context.org_id, [])


async def main() -> None:
    chat = AutoChat[AppContext](
        config=ChatConfig(model=ChatOpenAI(model="gpt-5-nano")),
        retrievers=[
            ChatRetriever(
                search_org_docs,
                name="org_docs",
                description="Organization-specific policy and plan documents.",
                top_k=3,
                preprocessors=[],
                postprocessors=[],
            )
        ],
        retrieval=RetrievalConfig(max_context_chars=2_000),
        system_message="Answer using retrieved context when it is relevant.",
    )

    result = await chat.ainvoke(
        "What is our refund policy?",
        thread_id="thread_basic_retriever_example",
        context=AppContext(user_id="user_1", org_id="org_123"),
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
