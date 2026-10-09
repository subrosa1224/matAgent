"""Unit tests for the stage 3.5 tool registry (S3.5-M2)."""

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from materials_screening.agent.errors import AgentToolError
from materials_screening.agent.tool_base import ToolSideEffect
from materials_screening.agent.tool_registry import AgentToolRegistry


class _StatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str | None = None


class _StatusOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str


class _LooseInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    thread_id: str | None = None


def _make_tool(
    name: str,
    *,
    side_effect: object = ToolSideEffect.READ_ONLY,
    input_model: type[BaseModel] = _StatusInput,
) -> Any:
    input_cls = input_model
    effect = side_effect

    class _Tool:
        description = f"{name} tool"
        input_model = input_cls
        output_model = _StatusOutput
        side_effect = effect

        def __init__(self) -> None:
            self.name = name

        def execute(
            self,
            arguments: BaseModel,
            context: object,
        ) -> BaseModel:
            raise AssertionError("tools must not execute in S3.5-M2")

    return _Tool()


_ALL_WHITELISTED = (
    "run_screening_workflow",
    "get_workflow_status",
    "get_workflow_history",
    "get_screening_result",
    "compare_ranked_materials",
)


class TestAgentToolRegistry:
    def test_registers_whitelisted_tools(self) -> None:
        registry = AgentToolRegistry([_make_tool(name) for name in _ALL_WHITELISTED])
        assert registry.names() == tuple(sorted(_ALL_WHITELISTED))
        assert registry.get("get_workflow_status").name == "get_workflow_status"

    def test_get_unknown_tool_rejected(self) -> None:
        registry = AgentToolRegistry([_make_tool("get_workflow_status")])
        with pytest.raises(AgentToolError, match="unknown tool"):
            registry.get("not_a_tool")

    def test_duplicate_name_rejected(self) -> None:
        with pytest.raises(AgentToolError, match="duplicate"):
            AgentToolRegistry(
                [
                    _make_tool("get_workflow_status"),
                    _make_tool("get_workflow_status"),
                ]
            )

    @pytest.mark.parametrize(
        "name",
        ["", "bad name", "run/tool", "tool\\x", "x" * 129],
    )
    def test_invalid_name_rejected(self, name: str) -> None:
        with pytest.raises(AgentToolError, match="invalid tool name"):
            AgentToolRegistry([_make_tool(name)])

    def test_non_whitelisted_tool_rejected(self) -> None:
        with pytest.raises(AgentToolError, match="not in whitelist"):
            AgentToolRegistry([_make_tool("delete_run")])

    def test_web_search_rejected(self) -> None:
        with pytest.raises(AgentToolError, match="web search"):
            AgentToolRegistry([_make_tool("web_search")])

    def test_non_conforming_tool_rejected(self) -> None:
        class _BrokenTool:
            name = "get_workflow_status"

        with pytest.raises(AgentToolError, match="conform"):
            AgentToolRegistry([_BrokenTool()])

    def test_invalid_side_effect_rejected(self) -> None:
        with pytest.raises(AgentToolError, match="side effect"):
            AgentToolRegistry(
                [_make_tool("get_workflow_status", side_effect="read_only")]
            )

    def test_loose_input_schema_rejected(self) -> None:
        with pytest.raises(AgentToolError, match="invalid input schema"):
            AgentToolRegistry(
                [_make_tool("get_workflow_status", input_model=_LooseInput)]
            )

    def test_definitions_for_model(self) -> None:
        registry = AgentToolRegistry([_make_tool(name) for name in _ALL_WHITELISTED])
        definitions = registry.definitions_for_model()
        assert [item["name"] for item in definitions] == sorted(_ALL_WHITELISTED)
        for definition in definitions:
            assert definition["type"] == "function"
            assert definition["parameters"]["type"] == "object"
            assert definition["parameters"]["additionalProperties"] is False

    def test_registry_never_executes_tools(self) -> None:
        tools = [_make_tool(name) for name in _ALL_WHITELISTED]
        registry = AgentToolRegistry(tools)
        registry.names()
        registry.get("get_workflow_status")
        registry.definitions_for_model()

    def test_empty_registry(self) -> None:
        registry = AgentToolRegistry()
        assert registry.names() == ()
        assert registry.definitions_for_model() == []
        with pytest.raises(AgentToolError, match="unknown tool"):
            registry.get("anything")
