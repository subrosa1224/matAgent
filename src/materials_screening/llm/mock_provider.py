"""Offline mock provider implementing StructuredLLM (D2-M3)."""

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import NoReturn, TypeVar

from pydantic import BaseModel

from materials_screening.llm.base import StructuredProviderResponse
from materials_screening.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMServiceUnavailableError,
    LLMStructuredOutputError,
    LLMTimeoutError,
    LLMTruncatedOutputError,
)
from materials_screening.planner.models import PlannerDraft
from materials_screening.planner.prompt_builder import build_user_message

T = TypeVar("T", bound=BaseModel)
_X = TypeVar("_X")


class MockErrorKind(StrEnum):
    """Simulated failure modes for the mock provider."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    SERVICE_ERROR = "service_error"
    INCOMPLETE_MAX_TOKENS = "incomplete_max_tokens"
    CONTENT_FILTER = "content_filter"
    EMPTY_OUTPUT = "empty_output"
    INVALID_JSON = "invalid_json"
    SCHEMA_MISMATCH = "schema_mismatch"


def _raise_mock_error(kind: MockErrorKind) -> NoReturn:
    if kind is MockErrorKind.TIMEOUT:
        raise LLMTimeoutError("mock timeout")
    if kind is MockErrorKind.RATE_LIMIT:
        raise LLMRateLimitError("mock rate limit")
    if kind is MockErrorKind.AUTHENTICATION:
        raise LLMAuthenticationError("mock authentication failure")
    if kind is MockErrorKind.SERVICE_ERROR:
        raise LLMServiceUnavailableError("mock service error")
    if kind is MockErrorKind.INCOMPLETE_MAX_TOKENS:
        raise LLMTruncatedOutputError("mock max_output_tokens reached")
    if kind is MockErrorKind.CONTENT_FILTER:
        raise LLMRefusalError("mock content filter blocked output")
    if kind is MockErrorKind.EMPTY_OUTPUT:
        raise LLMStructuredOutputError("mock returned empty output")
    if kind is MockErrorKind.INVALID_JSON:
        raise LLMStructuredOutputError("mock returned invalid JSON")
    if kind is MockErrorKind.SCHEMA_MISMATCH:
        raise LLMStructuredOutputError("mock output does not match schema")
    raise LLMConfigurationError(f"unknown mock error kind: {kind!r}")


class MockStructuredProvider:
    """Deterministic offline mock of the structured planner provider."""

    def __init__(
        self,
        fixtures: Mapping[str, PlannerDraft] | None = None,
        error_fixtures: Mapping[str, MockErrorKind] | None = None,
    ) -> None:
        self._fixtures = {
            build_user_message(query): draft
            for query, draft in (fixtures or {}).items()
        }
        self._error_fixtures = {
            build_user_message(query): kind
            for query, kind in (error_fixtures or {}).items()
        }

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        output_model: type[T],
        schema_name: str,
        max_output_tokens: int,
    ) -> StructuredProviderResponse[T]:
        error_kind = self._lookup(user_text, self._error_fixtures)
        if error_kind is not None:
            _raise_mock_error(error_kind)

        draft = self._lookup(user_text, self._fixtures)
        if draft is None:
            raise LLMStructuredOutputError(
                f"no mock fixture registered for query: {user_text!r}"
            )

        payload = draft.model_dump(mode="json")
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        parsed = output_model.model_validate(payload)
        return StructuredProviderResponse[T](
            parsed=parsed,
            provider="mock",
            model="mock",
            request_id="mock-"
            + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8],
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            raw_output_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def _lookup(
        self,
        user_text: str,
        mapping: Mapping[str, _X],
    ) -> _X | None:
        if user_text in mapping:
            return mapping[user_text]
        return mapping.get(build_user_message(user_text))
