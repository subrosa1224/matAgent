"""Stage 3.5 agent errors (S3.5-M1)."""


class AgentError(Exception):
    """Base exception for the agent layer."""


class AgentInvariantError(AgentError):
    """Raised when agent state or routing invariants are violated."""


class AgentConversationError(AgentError):
    """Raised when conversation operations fail (missing/busy/limits)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class AgentToolError(AgentError):
    """Raised when tool registry, argument or execution fails."""


class AgentModelError(AgentError):
    """Raised when the agent model or API call fails."""


class WorkflowResultReadError(AgentToolError):
    """Raised when a validated screening result cannot be read."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
