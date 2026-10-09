"""产出 Materials Screening SubAgentSpec 的工厂函数."""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from materials_screening.agent.conversation_store import SqliteConversationStore
from materials_screening.agent.runner import MaterialAgentRunner
from materials_screening.agent.settings import AgentSettings, shared_intern_settings
from materials_screening.agent_tools.result_reader import FileWorkflowResultReader
from materials_screening.master import SubAgentSpec

from .prompt import SYSTEM_PROMPT
from .tools import build_tool_registry


def create_spec(
    *,
    workflow_runner: object,
    run_root: Path = Path("data/workflow_runs"),
    intern_api_key: str | None = None,
) -> SubAgentSpec:
    """Create the materials screening sub-agent spec.

    Args:
        workflow_runner: An open WorkflowRunner instance.
        run_root: Workflow run root directory.
        intern_api_key: Intern API token (for real model mode).
    """
    settings = AgentSettings(agent_system_prompt=SYSTEM_PROMPT)
    tool_registry = build_tool_registry()

    def _runner_factory():
        """Create a fresh MaterialAgentRunner per delegation call."""
        store = SqliteConversationStore(Path("data/agent_conversations.sqlite"))
        checkpoint_path = settings.agent_checkpoint_db
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            str(checkpoint_path), check_same_thread=False
        )
        saver = SqliteSaver(connection)

        if intern_api_key:
            from pydantic import SecretStr

            from materials_screening.agent.intern_model import InternAgentModel
            model = InternAgentModel(
                shared_intern_settings(settings),
                api_key=SecretStr(intern_api_key)
            )
        else:
            from materials_screening.agent.mock_model import (
                WorkflowDrivenMockAgentModel,
            )
            model = WorkflowDrivenMockAgentModel()

        return MaterialAgentRunner(
            settings=settings,
            store=store,
            workflow_runner=workflow_runner,
            workflow_result_reader=FileWorkflowResultReader(run_root),
            tool_registry=tool_registry,
            agent_model=model,
            checkpointer=saver,
        )

    return SubAgentSpec(
        name="materials_screening",
        description=(
            "Performs inorganic materials screening: accepts natural-language "
            "descriptions of material requirements (e.g., 'find stable oxides "
            "with band gap > 2 eV'), runs the screening workflow, and returns "
            "ranked candidate materials with properties and scores."
        ),
        system_prompt=SYSTEM_PROMPT,
        tool_definitions=tool_registry.definitions(),
        runner_factory=_runner_factory,
    )
