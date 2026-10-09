"""Tests for the sub-agent registry."""

import pytest

from materials_screening.agent.errors import AgentToolError
from materials_screening.master import (
    SubAgentRegistry,
    SubAgentSpec,
)


def _dummy_spec(name: str = "test_agent") -> SubAgentSpec:
    return SubAgentSpec(
        name=name,
        description="A test sub-agent.",
        system_prompt="You are a test agent.",
        tool_definitions=(),
        runner_factory=lambda: None,
    )


class TestSubAgentRegistry:
    def test_register_single(self) -> None:
        registry = SubAgentRegistry()
        spec = _dummy_spec("materials_screening")
        registry.register(spec)
        assert "materials_screening" in registry.names()

    def test_register_multiple(self) -> None:
        registry = SubAgentRegistry()
        registry.register(_dummy_spec("agent_a"))
        registry.register(_dummy_spec("agent_b"))
        assert registry.names() == ("agent_a", "agent_b")

    def test_constructor_registers(self) -> None:
        spec = _dummy_spec("agent_a")
        registry = SubAgentRegistry([spec])
        assert registry.names() == ("agent_a",)

    def test_duplicate_name_rejected(self) -> None:
        registry = SubAgentRegistry([_dummy_spec("agent_a")])
        with pytest.raises(AgentToolError, match="duplicate"):
            registry.register(_dummy_spec("agent_a"))

    def test_invalid_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid sub-agent name"):
            _dummy_spec("bad name!")

    def test_get_returns_spec(self) -> None:
        spec = _dummy_spec("agent_a")
        registry = SubAgentRegistry([spec])
        assert registry.get("agent_a") is spec

    def test_get_unknown_raises(self) -> None:
        registry = SubAgentRegistry()
        with pytest.raises(AgentToolError, match="unknown sub-agent"):
            registry.get("nonexistent")

    def test_resolve_by_delegate_function(self) -> None:
        spec = _dummy_spec("agent_a")
        registry = SubAgentRegistry([spec])
        resolved = registry.resolve_by_delegate_function("delegate_to_agent_a")
        assert resolved is spec

    def test_resolve_by_delegate_function_not_found(self) -> None:
        registry = SubAgentRegistry([_dummy_spec("agent_a")])
        assert registry.resolve_by_delegate_function("unknown_func") is None

    def test_definitions_for_model(self) -> None:
        registry = SubAgentRegistry([_dummy_spec("agent_a")])
        defs = registry.definitions_for_model()
        assert len(defs) == 1
        assert defs[0]["type"] == "function"
        assert defs[0]["name"] == "delegate_to_agent_a"
        params = defs[0]["parameters"]
        assert params["type"] == "object"
        assert "task" in params["properties"]
        assert "task" in params["required"]

    def test_definitions(self) -> None:
        registry = SubAgentRegistry([_dummy_spec("agent_a")])
        defs = registry.definitions()
        assert len(defs) == 1
        assert defs[0].name == "delegate_to_agent_a"

    def test_custom_delegate_function_name(self) -> None:
        spec = SubAgentSpec(
            name="agent_b",
            description="desc",
            system_prompt="prompt",
            tool_definitions=(),
            runner_factory=lambda: None,
            delegate_function_name="custom_delegate",
        )
        registry = SubAgentRegistry([spec])
        resolved = registry.resolve_by_delegate_function("custom_delegate")
        assert resolved is spec
        assert resolved.delegate_function_name == "custom_delegate"
