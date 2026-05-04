class AutoChatError(Exception):
    """Base exception for AutoChat."""


class AutoChatToolError(AutoChatError):
    """Base exception for AutoChat tool execution errors."""
