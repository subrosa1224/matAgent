"""Unit tests for stage 3.5 tool abstraction (S3.5-M2)."""

import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from materials_screening.agent.tool_base import (
    AgentToolDefinition,
    ToolSideEffect,
    to_function_definition,
)


class _StatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str | None = None


def _parameters() -> dict[str, Any]:
    return _StatusInput.model_json_schema()


def _definition(**overrides: Any) -> AgentToolDefinition:
    values: dict[str, Any] = {
        "name": "get_workflow_status",
        "description": "Read workflow status",
        "parameters": _parameters(),
        "side_effect": ToolSideEffect.READ_ONLY,
        "version": "1",
    }
    values.update(overrides)
    return AgentToolDefinition.model_validate(values)


class TestToolSideEffect:
    def test_values(self) -> None:
        assert ToolSideEffect.READ_ONLY.value == "read_only"
        assert ToolSideEffect.CREATE_WORKFLOW_RUN.value == "create_workflow_run"


class TestAgentToolDefinition:
    def test_valid(self) -> None:
        definition = _definition()
        assert definition.name == "get_workflow_status"
        assert definition.parameters["type"] == "object"
        assert definition.parameters["additionalProperties"] is False

    def test_frozen_and_extra_rejected(self) -> None:
        assert AgentToolDefinition.model_config.get("frozen") is True
        with pytest.raises(ValidationError):
            _definition(junk=1)

    def test_non_object_schema_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _definition(
                parameters={
                    "type": "array",
                    "items": {"type": "string"},
                }
            )

    def test_additional_properties_not_false_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _definition(
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                }
            )

    def test_missing_additional_properties_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _definition(parameters={"type": "object", "properties": {}})


class TestFunctionConversion:
    def test_converts_definition(self) -> None:
        definition = _definition()
        converted = to_function_definition(definition)
        assert converted == {
            "type": "function",
            "name": "get_workflow_status",
            "description": "Read workflow status",
            "parameters": _parameters(),
        }

    def test_conversion_is_json_serializable(self) -> None:
        converted = to_function_definition(_definition())
        dumped = json.dumps(converted, ensure_ascii=False)
        assert json.loads(dumped)["name"] == "get_workflow_status"
