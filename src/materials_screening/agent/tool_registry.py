"""Stage 3.5 agent tool registry (S3.5-M2)."""

import re
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from materials_screening.agent.errors import AgentToolError
from materials_screening.agent.tool_base import (
    AgentTool,
    AgentToolDefinition,
    ToolSideEffect,
    to_function_definition,
)

_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
ALLOWED_TOOL_NAMES = frozenset(
    {
        "run_screening_workflow",
        "get_workflow_status",
        "get_workflow_history",
        "get_screening_result",
        "compare_ranked_materials",
        "run_outlier_detection",
        "detect_property_outliers",
        "detect_multivariate_outliers",
        "search_materials",
        "get_material_details",
        "get_query_result",
        "compare_materials",
        "describe_materials",
        "detect_material_outliers",
        "export_materials",
        "openalex_search",
        "s2_search",
        "literature_search",
        "screen_candidate_literature",
        "ingest_literature_documents",
        "rag_retrieve",
        "extract_experimental_data",
        "assemble_literature_result",
        "inspect_dataset",
        "assess_data_quality",
        "describe_dataset",
        "analyze_correlations",
        "run_statistical_test",
        "transform_dataset",
        "create_analysis_plot",
        "create_analysis_report",
    }
)
_TOOL_VERSION = "1"


class AgentToolRegistry:
    """Statically registered, whitelisted tool collection.

    Tools are instantiated by code, never imported dynamically or added at
    runtime. Registration validates structural conformance, name format and
    uniqueness, whitelist membership, and the input JSON schema.
    """

    def __init__(self, tools: Sequence[AgentTool] = ()) -> None:
        self._tools: dict[str, AgentTool] = {}
        self._definitions: dict[str, AgentToolDefinition] = {}
        for tool in tools:
            self._register(tool)

    def _register(self, tool: AgentTool) -> None:
        required = (
            "name",
            "description",
            "input_model",
            "output_model",
            "side_effect",
            "execute",
        )
        if not all(hasattr(tool, attribute) for attribute in required):
            raise AgentToolError(f"tool does not conform to AgentTool: {tool!r}")
        name = tool.name
        if not _TOOL_NAME_PATTERN.fullmatch(name):
            raise AgentToolError(f"invalid tool name: {name!r}")
        if name in self._tools:
            raise AgentToolError(f"duplicate tool name: {name!r}")
        if name == "web_search":
            raise AgentToolError("web search tool is not allowed")
        if name not in ALLOWED_TOOL_NAMES:
            raise AgentToolError(f"tool not in whitelist: {name!r}")
        if not (isinstance(tool.side_effect, ToolSideEffect)):
            raise AgentToolError(f"tool {name!r} has an invalid side effect")
        try:
            definition = AgentToolDefinition(
                name=name,
                description=tool.description,
                parameters=tool.input_model.model_json_schema(),
                side_effect=tool.side_effect,
                version=_TOOL_VERSION,
            )
        except ValidationError as exc:
            raise AgentToolError(f"tool {name!r} has an invalid input schema") from exc
        self._tools[name] = tool
        self._definitions[name] = definition

    def get(self, name: str) -> AgentTool:
        if name not in self._tools:
            raise AgentToolError(f"unknown tool: {name!r}")
        return self._tools[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def definitions_for_model(self) -> list[dict[str, Any]]:
        """OpenAI-compatible function definitions, sorted by name."""
        return [
            to_function_definition(self._definitions[name]) for name in self.names()
        ]

    def definitions(self) -> tuple[AgentToolDefinition, ...]:
        """Registered definitions, sorted by name."""
        return tuple(self._definitions[name] for name in self.names())
