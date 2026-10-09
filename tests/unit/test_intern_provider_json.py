from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from materials_screening.llm.errors import LLMStructuredOutputError
from materials_screening.llm.intern_provider import InternProvider, _json_object_text
from materials_screening.planner.settings import Settings


def test_json_object_text_accepts_fence_and_thinking_noise() -> None:
    raw = '<think>reasoning</think>\n```json\n{"items": []}\n```'

    assert _json_object_text(raw) == '{"items": []}'


class _Probe(BaseModel):
    answer: str


class _Completions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(
            id="response-1",
            model="intern-test",
            choices=(
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='reasoning\n```json\n{"answer":"ok"}\n```'
                    ),
                ),
            ),
            usage=None,
        )


def test_intern_provider_merges_system_messages() -> None:
    completions = _Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = InternProvider(Settings(intern_api_key="test-key"), client=client)

    response = provider.generate_structured(
        system_prompt="System task.",
        user_text="User evidence.",
        output_model=_Probe,
        schema_name="probe",
        max_output_tokens=512,
    )

    assert response.parsed.answer == "ok"
    assert [message["role"] for message in completions.kwargs["messages"]] == [
        "system",
        "user",
    ]
    assert completions.kwargs["extra_body"] == {"thinking_mode": True}


def test_intern_provider_can_disable_thinking_mode() -> None:
    completions = _Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = InternProvider(
        Settings(intern_api_key="test-key", intern_thinking_mode=False),
        client=client,
    )

    provider.generate_structured(
        system_prompt="System task.",
        user_text="User evidence.",
        output_model=_Probe,
        schema_name="probe",
        max_output_tokens=512,
    )

    assert completions.kwargs["extra_body"] == {"thinking_mode": False}


@pytest.mark.parametrize("content", ['{"answer":"ok"}', ""])
def test_truncated_intern_output_has_safe_category_without_parsing_message(content):
    class Truncated(_Completions):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            response.choices[0].finish_reason = "length"
            response.choices[0].message.content = content
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=Truncated()))
    provider = InternProvider(Settings(intern_api_key="test-key"), client=client)
    with pytest.raises(LLMStructuredOutputError) as caught:
        provider.generate_structured(
            system_prompt="task",
            user_text="evidence",
            output_model=_Probe,
            schema_name="probe",
            max_output_tokens=512,
        )
    assert caught.value.failure_kind == "truncated_output"
