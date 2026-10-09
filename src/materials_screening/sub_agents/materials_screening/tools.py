"""Materials Screening 子Agent — 包装现有 5 个筛选工具."""

from materials_screening.agent.tool_registry import AgentToolRegistry

from materials_screening.agent_tools import (
    CompareRankedMaterialsTool,
    GetScreeningResultTool,
    GetWorkflowHistoryTool,
    GetWorkflowStatusTool,
    RunScreeningWorkflowTool,
)


def build_tool_registry() -> AgentToolRegistry:
    """Build the tool registry with all 5 whitelisted screening tools."""
    return AgentToolRegistry((
        RunScreeningWorkflowTool(),
        GetWorkflowStatusTool(),
        GetWorkflowHistoryTool(),
        GetScreeningResultTool(),
        CompareRankedMaterialsTool(),
    ))
