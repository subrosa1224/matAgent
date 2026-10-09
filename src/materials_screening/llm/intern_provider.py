"""Structured planner provider for Intern Chat Completions."""

from __future__ import annotations

import hashlib
import json
import time
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from materials_screening.llm.base import StructuredProviderResponse
from materials_screening.llm.errors import (
    LLMConfigurationError,
    LLMStructuredOutputError,
)
from materials_screening.llm.intern_transport import (
    create_completion,
    create_intern_client,
)
from materials_screening.planner.settings import Settings

T = TypeVar("T", bound=BaseModel)


class InternProvider:
    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        if settings.intern_api_key is None:
            raise LLMConfigurationError("INTERN_API_KEY is not set")
        self._settings = settings
        self._client = client or create_intern_client(
            api_key=settings.intern_api_key.get_secret_value(),
            base_url=settings.intern_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            use_system_proxy=settings.intern_use_system_proxy,
        )

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        output_model: type[T],
        schema_name: str,
        max_output_tokens: int,
    ) -> StructuredProviderResponse[T]:
        schema = json.dumps(output_model.model_json_schema(), ensure_ascii=False)
        started = time.perf_counter()
        response = create_completion(
            self._client,
            {
                "model": self._settings.intern_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"{system_prompt}\n\n"
                            f"Return only JSON matching this schema: {schema}"
                        ),
                    },
                    {"role": "user", "content": user_text},
                ],
                "temperature": 0.0,
                "max_tokens": max_output_tokens,
                "stream": False,
                "extra_body": {"thinking_mode": self._settings.intern_thinking_mode},
            },
            max_attempts=self._settings.llm_max_attempts,
            timeout_seconds=self._settings.llm_timeout_seconds,
        )
        choice = response.choices[0] if response.choices else None
        if choice is not None and choice.finish_reason == "length":
            raise LLMStructuredOutputError(
                "Intern response exceeded max_tokens", failure_kind="truncated_output"
            )
        if choice is None or not choice.message.content:
            raise LLMStructuredOutputError("Intern returned empty structured output")
        raw = choice.message.content.strip()
        try:
            payload = json.loads(_json_object_text(raw))
        except json.JSONDecodeError as exc:
            detail = (
                " (provider reported prompt processing error)"
                if raw.strip() == "in prompt processing error"
                else ""
            )
            raise LLMStructuredOutputError(
                "Intern output is not valid JSON" + detail
            ) from exc
        try:
            parsed = output_model.model_validate(payload)
        except ValidationError as exc:
            first = exc.errors(include_url=False)[0]
            location = ".".join(str(part) for part in first["loc"])
            raise LLMStructuredOutputError(
                f"Intern output does not match schema at {location}: {first['msg']}"
            ) from exc
        usage = getattr(response, "usage", None)
        return StructuredProviderResponse(
            parsed=parsed,
            provider="intern",
            model=str(response.model),
            request_id=str(response.id),
            latency_ms=round((time.perf_counter() - started) * 1000),
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            reasoning_tokens=None,
            raw_output_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )


def _json_object_text(raw: str) -> str:
    """Accept a JSON object wrapped by common model formatting noise."""
    candidate = raw.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        return candidate
    return candidate[start : end + 1]
