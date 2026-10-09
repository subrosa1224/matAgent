"""MA-5 plugin registry tests."""

from __future__ import annotations

from typing import Any

import pytest

from materials_screening.master import (
    SubAgentUiPlugin,
    SubAgentUiPluginRegistry,
    SubAgentUiSpec,
)


def _plugin(
    name: str = "test_agent", display_name: str = "测试 Agent"
) -> SubAgentUiPlugin:
    def handler(*args: Any, **kwargs: Any) -> Any:
        return iter(())

    return SubAgentUiPlugin(
        spec=SubAgentUiSpec(
            name=name,
            display_name=display_name,
            description="Minimal test agent.",
            renderer_name="system_message",
        ),
        handler=handler,
    )


def test_plugin_registry_exposes_modes_and_rejects_duplicates() -> None:
    registry = SubAgentUiPluginRegistry((_plugin(),))
    assert registry.modes() == ("测试 Agent",)
    assert registry.for_mode("测试 Agent").spec.name == "test_agent"
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(_plugin())
    with pytest.raises(KeyError, match="unknown"):
        registry.for_mode("missing")
