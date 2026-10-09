"""LLM provider error hierarchy (D2-M1)."""

from typing import Literal


class LLMError(Exception):
    """Base exception for all LLM provider errors."""


class LLMConfigurationError(LLMError):
    """Raised when LLM configuration is invalid."""


class LLMAuthenticationError(LLMError):
    """Raised when the LLM API rejects credentials."""


class LLMPermissionError(LLMError):
    """Raised when the LLM API denies permission."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM API rate limit is exceeded."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""


class LLMConnectionError(LLMError):
    """Raised when an LLM connection cannot be established."""


class LLMServiceUnavailableError(LLMError):
    """Raised when the LLM service is unavailable."""


class LLMModelNotSupportedError(LLMError):
    """Raised when the requested model is not supported."""


class LLMRefusalError(LLMError):
    """Raised when the LLM refuses or content filtering blocks output."""


class LLMTruncatedOutputError(LLMError):
    """Raised when the LLM output is truncated."""


class LLMStructuredOutputError(LLMError):
    """Raised when the LLM output cannot be parsed into the target model."""

    def __init__(
        self, message: str, *, failure_kind: Literal["truncated_output"] | None = None
    ):
        super().__init__(message)
        self.failure_kind = failure_kind
