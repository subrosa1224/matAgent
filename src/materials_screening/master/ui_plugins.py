"""Plugin registry for explicit-mode sub-agent UI adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from materials_screening.master.application_contracts import SubAgentUiSpec

UiDispatchHandler = Callable[..., Any]


class SubAgentUiPlugin(BaseModel):
    """One declarative UI spec paired with its application adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    spec: SubAgentUiSpec
    handler: UiDispatchHandler


class SubAgentUiPluginRegistry:
    """Duplicate-safe registry used by the unified explicit-mode dispatcher."""

    def __init__(self, plugins: tuple[SubAgentUiPlugin, ...] = ()) -> None:
        self._by_name: dict[str, SubAgentUiPlugin] = {}
        self._by_display_name: dict[str, SubAgentUiPlugin] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: SubAgentUiPlugin) -> None:
        name = plugin.spec.name
        display_name = plugin.spec.display_name
        if name in self._by_name:
            raise ValueError(f"duplicate sub-agent UI plugin: {name!r}")
        if display_name in self._by_display_name:
            raise ValueError(f"duplicate sub-agent display name: {display_name!r}")
        self._by_name[name] = plugin
        self._by_display_name[display_name] = plugin

    def get(self, name: str) -> SubAgentUiPlugin:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown sub-agent UI plugin: {name!r}") from exc

    def for_mode(self, display_name: str) -> SubAgentUiPlugin:
        try:
            return self._by_display_name[display_name]
        except KeyError as exc:
            raise KeyError(f"unknown sub-agent UI mode: {display_name!r}") from exc

    def modes(self) -> tuple[str, ...]:
        return tuple(plugin.spec.display_name for plugin in self.plugins())

    def plugins(self) -> tuple[SubAgentUiPlugin, ...]:
        return tuple(self._by_name[name] for name in sorted(self._by_name))
