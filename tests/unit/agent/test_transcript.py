"""Unit tests for the agent transcript (S3.5)."""

import json

import pytest
from pydantic import ValidationError

from materials_screening.agent.errors import AgentInvariantError
from materials_screening.agent.models import AgentMessageItem
from materials_screening.agent.transcript import AgentTranscript


def _multi_turn_transcript() -> AgentTranscript:
    transcript = AgentTranscript()
    transcript.append(AgentTranscript.user("寻找不含 Pb 的半导体"))
    transcript.append(
        AgentTranscript.function_call(
            call_id="call_1",
            name="run_screening_workflow",
            arguments='{"query":"不含 Pb 的半导体"}',
        )
    )
    transcript.append(
        AgentTranscript.function_output(
            call_id="call_1",
            output='{"status":"completed","thread_id":"t1"}',
        )
    )
    transcript.append(AgentTranscript.assistant("筛选任务已完成。"))
    return transcript


class TestAgentTranscriptBuilders:
    def test_builders_produce_typed_items(self) -> None:
        assert AgentTranscript.user("hi").role == "user"
        assert AgentTranscript.assistant("hi").role == "assistant"
        call = AgentTranscript.function_call(
            call_id="c1", name="get_workflow_status", arguments="{}"
        )
        assert call.type == "function_call"
        output = AgentTranscript.function_output(call_id="c1", output="{}")
        assert output.type == "function_call_output"


class TestAgentTranscriptValidation:
    def test_normal_multi_turn_preserves_order(self) -> None:
        transcript = _multi_turn_transcript()
        transcript.validate()
        items = transcript.items()
        assert [item.type for item in items] == [
            "message",
            "function_call",
            "function_call_output",
            "message",
        ]
        assert items[0].role == "user"
        assert items[1].call_id == "call_1"
        assert items[3].role == "assistant"

    def test_extend_appends_all_items(self) -> None:
        transcript = AgentTranscript()
        transcript.extend(
            [
                AgentTranscript.user("hi"),
                AgentTranscript.function_call(
                    call_id="c1", name="get_workflow_status", arguments="{}"
                ),
                AgentTranscript.function_output(call_id="c1", output="{}"),
            ]
        )
        transcript.validate()
        assert len(transcript.items()) == 3

    def test_missing_output_rejected(self) -> None:
        transcript = AgentTranscript()
        transcript.append(
            AgentTranscript.function_call(
                call_id="call_1",
                name="get_workflow_status",
                arguments="{}",
            )
        )
        with pytest.raises(AgentInvariantError, match="missing output"):
            transcript.validate()

    def test_duplicate_call_id_rejected(self) -> None:
        transcript = AgentTranscript()
        transcript.append(
            AgentTranscript.function_call(call_id="call_1", name="a", arguments="{}")
        )
        transcript.append(
            AgentTranscript.function_call(call_id="call_1", name="b", arguments="{}")
        )
        transcript.append(
            AgentTranscript.function_output(call_id="call_1", output="{}")
        )
        with pytest.raises(AgentInvariantError, match="duplicate function call"):
            transcript.validate()

    def test_orphan_output_rejected(self) -> None:
        transcript = AgentTranscript()
        transcript.append(
            AgentTranscript.function_output(call_id="call_1", output="{}")
        )
        with pytest.raises(AgentInvariantError, match="orphan"):
            transcript.validate()

    def test_output_before_call_rejected(self) -> None:
        transcript = AgentTranscript()
        transcript.append(
            AgentTranscript.function_output(call_id="call_1", output="{}")
        )
        transcript.append(
            AgentTranscript.function_call(call_id="call_1", name="a", arguments="{}")
        )
        with pytest.raises(AgentInvariantError, match="orphan"):
            transcript.validate()

    def test_duplicate_output_rejected(self) -> None:
        transcript = AgentTranscript()
        transcript.append(
            AgentTranscript.function_call(call_id="call_1", name="a", arguments="{}")
        )
        transcript.append(
            AgentTranscript.function_output(call_id="call_1", output="{}")
        )
        transcript.append(
            AgentTranscript.function_output(call_id="call_1", output="{}")
        )
        with pytest.raises(AgentInvariantError, match="duplicate function output"):
            transcript.validate()

    def test_max_chars_exceeded(self) -> None:
        transcript = AgentTranscript(max_chars=10)
        transcript.append(AgentTranscript.user("x" * 11))
        with pytest.raises(AgentInvariantError, match="characters"):
            transcript.validate()

    def test_max_bytes_exceeded(self) -> None:
        transcript = AgentTranscript(max_bytes=20)
        transcript.append(AgentTranscript.user("中文内容长度超出字节限制"))
        with pytest.raises(AgentInvariantError, match="bytes"):
            transcript.validate()

    def test_unknown_item_rejected(self) -> None:
        transcript = AgentTranscript()
        with pytest.raises(AgentInvariantError, match="unknown transcript item"):
            transcript.append({"type": "reasoning", "content": "..."})
        with pytest.raises(AgentInvariantError, match="unknown transcript item"):
            transcript.append("not-an-item")

    def test_reasoning_item_cannot_be_constructed(self) -> None:
        with pytest.raises(ValidationError):
            AgentMessageItem.model_validate({"type": "reasoning", "content": "..."})


class TestAgentTranscriptSerialization:
    def test_to_json_round_trips(self) -> None:
        transcript = _multi_turn_transcript()
        text = transcript.to_json()
        parsed = json.loads(text)
        assert parsed == transcript.as_dicts()
        assert "\n" not in text

    def test_unicode_preserved(self) -> None:
        transcript = AgentTranscript()
        transcript.append(AgentTranscript.user("寻找不含铅（Pb）的材料"))
        transcript.append(AgentTranscript.assistant("完成。"))
        text = transcript.to_json()
        assert "铅（Pb）" in text
        assert json.loads(text)[0]["content"] == "寻找不含铅（Pb）的材料"

    def test_no_secret_fields_in_serialization(self) -> None:
        transcript = _multi_turn_transcript()
        text = transcript.to_json()
        parsed = json.loads(text)
        allowed_keys = {
            "type",
            "role",
            "content",
            "call_id",
            "name",
            "arguments",
            "output",
        }
        for item in parsed:
            assert set(item) <= allowed_keys
        lowered = text.lower()
        assert "api_key" not in lowered
        assert "authorization" not in lowered
        assert "reasoning" not in lowered
        assert "web_search_call" not in lowered

    def test_as_dicts_json_safe(self) -> None:
        transcript = _multi_turn_transcript()
        dumped = json.dumps(transcript.as_dicts(), ensure_ascii=False)
        assert json.loads(dumped) == transcript.as_dicts()
