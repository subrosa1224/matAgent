"""Factory for the DataAnalysisAgent SubAgentSpec."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import SecretStr

from materials_screening.agent.conversation_store import SqliteConversationStore
from materials_screening.agent.intern_model import InternAgentModel
from materials_screening.agent.model_base import MaterialAgentModel
from materials_screening.agent.runner import MaterialAgentRunner
from materials_screening.agent.settings import AgentSettings, shared_intern_settings
from materials_screening.agent_tools.result_reader import FileWorkflowResultReader
from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.reporting import DataAnalysisReportingService
from materials_screening.data_analysis.service import DataAnalysisService
from materials_screening.data_analysis.statistics import DataStatisticsService
from materials_screening.data_analysis.transform import DatasetTransformService
from materials_screening.master import SubAgentSpec

from .mock_model import DataAnalysisMockModel
from .prompt import SYSTEM_PROMPT
from .routing_model import DataAnalysisRoutingModel
from .tools import build_tool_registry


def create_spec(
    *,
    workflow_runner: object,
    result_reader: FileWorkflowResultReader,
    data_root: Path = Path("data/data_analysis"),
    intern_api_key: str | None = None,
) -> SubAgentSpec:
    settings = AgentSettings(
        agent_system_prompt=SYSTEM_PROMPT,
        agent_max_model_calls_per_turn=6,
        agent_max_tool_calls_per_turn=5,
        agent_allow_multi_step_tools=True,
        agent_max_tool_output_bytes=262_144,
        agent_max_input_bytes=524_288,
    )
    store = DatasetStore(data_root)
    registry = build_tool_registry(
        DataAnalysisService(store),
        DataStatisticsService(store),
        DatasetTransformService(store),
        DataAnalysisReportingService(store),
    )

    def runner_factory() -> MaterialAgentRunner:
        conversation_store = SqliteConversationStore(
            data_root / "conversations.sqlite"
        )
        checkpoint_path = data_root / "checkpoints.sqlite"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
        saver = SqliteSaver(connection)
        model: MaterialAgentModel
        if intern_api_key:
            delegate = InternAgentModel(
                shared_intern_settings(settings),
                api_key=SecretStr(intern_api_key),
            )
            model = DataAnalysisRoutingModel(delegate)
        else:
            model = DataAnalysisMockModel()
        return MaterialAgentRunner(
            settings=settings,
            store=conversation_store,
            workflow_runner=workflow_runner,  # type: ignore[arg-type]
            workflow_result_reader=result_reader,
            tool_registry=registry,
            agent_model=model,
            checkpointer=saver,
        )

    return SubAgentSpec(
        name="data_analysis",
        description=(
            "Analyzes registered CSV/JSON datasets: schema and data quality, "
            "descriptive statistics, correlations, explicit statistical tests, "
            "immutable cleaning, fixed plots, reports and safe exports."
        ),
        system_prompt=SYSTEM_PROMPT,
        tool_definitions=registry.definitions(),
        runner_factory=runner_factory,
    )
