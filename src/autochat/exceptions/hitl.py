from .errors import AutoChatError


class HITLError(AutoChatError):
    """Base exception for HITL-related errors."""


class HITLPersistenceRequiredError(HITLError):
    """Raised when AutoChat is constructed with `hitl=...` but no `persistence`
    checkpointer was provided.

    HITL relies on LangGraph's interrupt/resume mechanism, which requires a
    durable checkpointer to pause and resume graph execution.
    """


class ApprovalDecisionUndefinedError(HITLError):
    """Raised at `ApprovalSpec` construction when `decide` is not provided and
    cannot be inferred from the response schema (no `approved: bool` field and
    not exactly one bool field on the model)."""
