"""Factory for the unified Materials Database SubAgentSpec."""

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
from materials_screening.master import SubAgentSpec
from materials_screening.repositories.base import MaterialsRepository
from materials_screening.services.material_database_service import (
    MaterialDatabaseService,
)
from materials_screening.services.query_result_store import QueryResultStore

from .deterministic_model import DeterministicDatabaseModel
from .mock_model import MaterialsDatabaseMockModel
from .prompt import SYSTEM_PROMPT
from .tools import build_tool_registry


def create_spec(
    *,
    repository: MaterialsRepository,
    workflow_runner: object,
    result_reader: FileWorkflowResultReader,
    query_root: Path = Path("data/material_queries"),
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
    service = MaterialDatabaseService(repository, QueryResultStore(query_root))
    registry = build_tool_registry(service)

    def runner_factory() -> MaterialAgentRunner:
        store = SqliteConversationStore(
            Path("data/material_database_conversations.sqlite")
        )
        checkpoint = Path("data/material_database_checkpoints.sqlite")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(checkpoint), check_same_thread=False)
        saver = SqliteSaver(connection)
        model: MaterialAgentModel
        if intern_api_key:
            model = DeterministicDatabaseModel(
                InternAgentModel(
                    shared_intern_settings(settings),
                    api_key=SecretStr(intern_api_key),
                )
            )
        else:
            model = DeterministicDatabaseModel(MaterialsDatabaseMockModel())
        return MaterialAgentRunner(
            settings=settings,
            store=store,
            workflow_runner=workflow_runner,  # type: ignore[arg-type]
            workflow_result_reader=result_reader,
            tool_registry=registry,
            agent_model=model,
            checkpointer=saver,
        )

    return SubAgentSpec(
        name="materials_database",
        description=(
            "Unified Materials Project database querying and analysis: "
            "structured filtering, sorting and Top-K, material details, "
            "comparison, descriptive statistics, explicit outlier detection, "
            "and explicit CSV/JSON/Markdown export."
        ),
        system_prompt=SYSTEM_PROMPT,
        tool_definitions=registry.definitions(),
        runner_factory=runner_factory,
    )
