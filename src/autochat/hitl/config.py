from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HITL:
    """Top-level HITL configuration for `AutoChat(hitl=...)`.

    Opting in unlocks:
      - declarative `approval=...` on tools and retrievers
      - `runtime.hitl` for custom in-tool approval / question requests
      - the built-in `ask_user` tool (only when `allow_questions=True`)

    Requires a `persistence` checkpointer on `AutoChat`; without one,
    `AutoChat` will raise `HITLPersistenceRequiredError`.
    """

    allow_questions: bool = False
