"""Factory for the Outlier Detection SubAgentSpec."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import SecretStr

from materials_screening.agent.conversation_store import SqliteConversationStore
from materials_screening.agent.runner import MaterialAgentRunner
from materials_screening.agent.settings import AgentSettings, shared_intern_settings
from materials_screening.agent.tool_registry import AgentToolRegistry
from materials_screening.agent_tools.result_reader import FileWorkflowResultReader
from materials_screening.master import SubAgentSpec
from materials_screening.parsers.data_file_parser import DataFileParser
from materials_screening.repositories.base import MaterialsRepository
from materials_screening.services.material_set_resolver import MaterialSetResolver

from .prompt import OUTLIER_DETECTION_SYSTEM_PROMPT
from .routing_model import OutlierDetectionRoutingModel
from .tools import RunOutlierDetectionTool


def create_spec(
    *,
    result_reader: FileWorkflowResultReader,
    repository: MaterialsRepository,
    workflow_runner: object,
    intern_api_key: str | None = None,
) -> SubAgentSpec:
    """Create the outlier detection sub-agent spec.

    Args:
        result_reader: Reader for prior screening workflow results.
        repository: Material repository for formula-based lookups.
        workflow_runner: WorkflowRunner instance (required by runner,
            not used by outlier tools).
        intern_api_key: Intern API token (real model mode).
    """
    settings = AgentSettings(
        agent_system_prompt=OUTLIER_DETECTION_SYSTEM_PROMPT,
        # Formula queries intentionally expand to all MP polymorphs.  Their
        # evidence can exceed the generic agent's 32 KiB tool envelope while
        # still being a bounded, read-only result for a handful of formulas.
        agent_max_tool_output_bytes=262_144,
        agent_max_input_bytes=524_288,
    )

    parser = DataFileParser()
    resolver = MaterialSetResolver(
        workflow_reader=result_reader,
        repository=repository,
        parser=parser,
    )

    tool_registry = AgentToolRegistry((RunOutlierDetectionTool(resolver=resolver),))

    def runner_factory() -> MaterialAgentRunner:
        store = SqliteConversationStore(
            Path("data/outlier_detection_conversations.sqlite")
        )
        checkpoint_path = (
            settings.agent_checkpoint_db.parent
            / "outlier_detection_checkpoint.sqlite"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            str(checkpoint_path), check_same_thread=False
        )
        saver = SqliteSaver(connection)

        if intern_api_key:
            from materials_screening.agent.intern_model import InternAgentModel
            model = OutlierDetectionRoutingModel(
                InternAgentModel(
                    shared_intern_settings(settings),
                    api_key=SecretStr(intern_api_key),
                )
            )
        else:
            from .mock_model import OutlierDetectionMockAgentModel

            model = OutlierDetectionMockAgentModel()

        return MaterialAgentRunner(
            settings=settings,
            store=store,
            workflow_runner=workflow_runner,  # type: ignore[arg-type]
            workflow_result_reader=result_reader,
            tool_registry=tool_registry,
            agent_model=model,
            checkpointer=saver,
        )

    return SubAgentSpec(
        name="outlier_detection",
        description=(
            "Detects outlier materials in a material set: single-property "
            "outliers (Z-score / IQR) and multivariate outliers (Mahalanobis "
            "distance, Isolation Forest). Materials can be specified by "
            "workflow_thread_ids, material_formulas, or a CSV/JSON data file. "
            "Provides anomaly scores and reasons for flagged materials."
        ),
        system_prompt=OUTLIER_DETECTION_SYSTEM_PROMPT,
        tool_definitions=tool_registry.definitions(),
        runner_factory=runner_factory,
    )
