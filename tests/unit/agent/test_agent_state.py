"""Unit tests for stage 3.5 agent state (S3.5-M1)."""

import json
from datetime import datetime
from operator import add
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from materials_screening.agent.state import MaterialAgentState


def _full_state() -> dict[str, Any]:
    event = {
        "event_id": "evt_1",
        "conversation_id": "c1",
        "node": "prepare_turn",
        "event_type": "started",
        "created_at": "2026-08-06T00:00:00Z",
        "status": "running",
        "message": "turn started",
        "metrics": {},
    }
    return {
        "conversation_id": "c1",
        "user_turn_id": "turn_1",
        "status": "running",
        "current_node": "call_agent_model",
        "input_items": [
            {"type": "message", "role": "user", "content": "find materials"}
        ],
        "active_workflow_thread_id": "thread-1",
        "model_call_count": 1,
        "tool_call_count": 0,
        "workflow_run_count": 0,
        "pending_tool_calls": [
            {
                "call_id": "call_1",
                "name": "run_screening_workflow",
                "arguments_json": "{}",
            }
        ],
        "executed_call_ids": [],
        "evidence_ids": [],
        "final_draft": None,
        "final_response": None,
        "error": None,
        "events": [event],
    }


class TestMaterialAgentState:
    def test_full_state_is_json_serializable(self) -> None:
        state = _full_state()
        dumped = json.dumps(state)
        assert json.loads(dumped) == state

    def test_state_contains_no_forbidden_values(self) -> None:
        state = _full_state()
        forbidden = (BaseModel, Path, datetime, Exception)
        assert not any(isinstance(value, forbidden) for value in state.values())

    def test_events_reducer_appends(self) -> None:
        first = [{"event_id": "evt_1"}]
        second = [{"event_id": "evt_2"}]
        assert add(first, second) == [
            {"event_id": "evt_1"},
            {"event_id": "evt_2"},
        ]

    def test_events_field_uses_append_reducer(self) -> None:
        annotation = MaterialAgentState.__annotations__["events"]
        assert annotation.__metadata__ == (add,)

    def test_no_reasoning_or_secret_fields(self) -> None:
        annotations = MaterialAgentState.__annotations__
        assert "reasoning" not in annotations
        assert "chain_of_thought" not in annotations
        assert "api_key" not in annotations
        assert "authorization" not in annotations
        assert "workflow_state" not in annotations
        assert "connection" not in annotations

    def test_all_state_fields_optional(self) -> None:
        assert MaterialAgentState.__required_keys__ == frozenset()

    def test_field_set(self) -> None:
        expected = {
            "conversation_id",
            "user_turn_id",
            "user_message",
            "status",
            "current_node",
            "input_items",
            "active_workflow_thread_id",
            "model_call_count",
            "tool_call_count",
            "workflow_run_count",
            "pending_tool_calls",
            "executed_call_ids",
            "evidence_ids",
            "cancelled",
            "final_draft",
            "final_response",
            "error",
            "events",
        }
        assert set(MaterialAgentState.__annotations__) == expected
