"""Provider metadata for audit purposes (D2-M1/D2-M3)."""

from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel, ConfigDict

from materials_screening.llm.base import StructuredProviderResponse

T = TypeVar("T", bound=BaseModel)


class ProviderMetadata(BaseModel):
    """Non-sensitive audit metadata for one LLM call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    request_id: str | None
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    reasoning_effort: str
    prompt_version: str
    prompt_hash: str | None
    schema_version: str
    raw_output_sha256: str | None
    retrieved_at: datetime | None = None


def build_provider_metadata(
    *,
    response: StructuredProviderResponse[T],
    prompt_version: str,
    prompt_hash: str,
    schema_version: str,
    reasoning_effort: str,
    retrieved_at: datetime,
) -> ProviderMetadata:
    """Build non-sensitive audit metadata from a provider response."""
    return ProviderMetadata(
        provider=response.provider,
        model=response.model,
        request_id=response.request_id,
        latency_ms=response.latency_ms,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        reasoning_tokens=response.reasoning_tokens,
        reasoning_effort=reasoning_effort,
        prompt_version=prompt_version,
        prompt_hash=prompt_hash,
        schema_version=schema_version,
        raw_output_sha256=response.raw_output_sha256,
        retrieved_at=retrieved_at,
    )
