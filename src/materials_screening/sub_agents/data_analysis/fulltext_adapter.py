"""Real bounded DataAnalysisAgent graph for explicit source-isolated descriptions."""

import sqlite3
from contextlib import contextmanager

from langgraph.checkpoint.sqlite import SqliteSaver

from materials_screening.agent.conversation_store import SqliteConversationStore
from materials_screening.agent.runner import MaterialAgentRunner
from materials_screening.agent.settings import AgentSettings
from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.reporting import DataAnalysisReportingService
from materials_screening.data_analysis.service import DataAnalysisService
from materials_screening.data_analysis.statistics import DataStatisticsService
from materials_screening.data_analysis.transform import DatasetTransformService

from .prompt import SYSTEM_PROMPT
from .routing_model import DataAnalysisRoutingModel
from .tools import build_tool_registry


class _NoFallback:
    def generate(self, request):
        raise ValueError("Only the explicit fulltext descriptive handoff is allowed")


@contextmanager
def bounded_description_agent(*, data_root, workflow_runner, result_reader):
    datasets = DatasetStore(data_root)
    registry = build_tool_registry(
        DataAnalysisService(datasets),
        DataStatisticsService(datasets),
        DatasetTransformService(datasets),
        DataAnalysisReportingService(datasets),
    )
    # Keep this adapter's mutable graph state out of existing agent conversations.
    state_root = data_root / "fulltext_agent"
    state_root.mkdir(parents=True, exist_ok=True)
    store = SqliteConversationStore(state_root / "conversations.sqlite")
    try:
        with sqlite3.connect(
            state_root / "checkpoints.sqlite", check_same_thread=False
        ) as connection:
            yield MaterialAgentRunner(
                settings=AgentSettings(
                    agent_system_prompt=SYSTEM_PROMPT,
                    agent_max_model_calls_per_turn=3,
                    agent_max_tool_calls_per_turn=2,
                    agent_allow_multi_step_tools=True,
                    agent_max_tool_output_bytes=262144,
                ),
                store=store,
                workflow_runner=workflow_runner,
                workflow_result_reader=result_reader,
                tool_registry=registry,
                agent_model=DataAnalysisRoutingModel(_NoFallback()),
                checkpointer=SqliteSaver(connection),
            )
    finally:
        store.close()
