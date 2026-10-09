"""Structured LLM response container and protocol (D2-M1/D2-M3)."""

from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound=BaseModel)


class StructuredProviderResponse(BaseModel, Generic[T]):
    """Parsed structured output plus non-sensitive provider metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parsed: T
    provider: str
    model: str
    request_id: str | None
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    raw_output_sha256: str | None


class StructuredLLM(Protocol):
    """Protocol for providers that return structured model output."""

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        output_model: type[T],
        schema_name: str,
        max_output_tokens: int,
    ) -> StructuredProviderResponse[T]: ...
