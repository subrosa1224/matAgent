"""Whitelist-based sub-agent registry (parallel to AgentToolRegistry)."""

import re
from collections.abc import Sequence
from typing import Any

from materials_screening.agent.errors import AgentToolError

from .sub_agent_spec import SubAgentDefinition, SubAgentSpec

_SUB_AGENT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


class SubAgentRegistry:
    """Statically registered, whitelisted sub-agent collection.

    Same pattern as ``AgentToolRegistry`` (tool_registry.py:30-96). Sub-agents
    are instantiated by code at startup, never imported dynamically.
    """

    def __init__(self, sub_agents: Sequence[SubAgentSpec] = ()) -> None:
        self._specs: dict[str, SubAgentSpec] = {}
        self._definitions: dict[str, SubAgentDefinition] = {}
        for spec in sub_agents:
            self.register(spec)

    def register(self, spec: SubAgentSpec) -> None:
        if not _SUB_AGENT_NAME_PATTERN.fullmatch(spec.name):
            raise AgentToolError(f"invalid sub-agent name: {spec.name!r}")
        if spec.name in self._specs:
            raise AgentToolError(f"duplicate sub-agent: {spec.name!r}")
        self._specs[spec.name] = spec
        self._definitions[spec.name] = SubAgentDefinition(
            name=spec.delegate_function_name,
            description=spec.description,
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            f"Clear, specific task description for the "
                            f"{spec.name} sub-agent. Include all relevant "
                            f"details from the user request."
                        ),
                    }
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        )

    def get(self, name: str) -> SubAgentSpec:
        if name not in self._specs:
            raise AgentToolError(f"unknown sub-agent: {name!r}")
        return self._specs[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def definitions_for_model(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": d.name,
                "description": d.description,
                "parameters": d.parameters,
            }
            for d in self.definitions()
        ]

    def definitions(self) -> tuple[SubAgentDefinition, ...]:
        return tuple(self._definitions[name] for name in self.names())

    def resolve_by_delegate_function(self, function_name: str) -> SubAgentSpec | None:
        for spec in self._specs.values():
            if spec.delegate_function_name == function_name:
                return spec
        return None
