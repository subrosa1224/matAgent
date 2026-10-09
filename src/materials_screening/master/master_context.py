"""Immutable runtime dependencies for the master agent graph."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from materials_screening.agent.conversation_store import ConversationStore
from materials_screening.agent.model_base import MaterialAgentModel
from materials_screening.workflow.context import Clock, IdGenerator

from .sub_agent_executor import SubAgentExecutor
from .sub_agent_registry import SubAgentRegistry

if TYPE_CHECKING:
    from materials_screening.agent.settings import AgentSettings

    from .data_analysis_handoffs import DataAnalysisCrossAgentCoordinator


@dataclass(frozen=True)
class MasterAgentContext:
    """Immutable runtime dependencies for the master agent graph.

    Same pattern as ``MaterialAgentContext`` (context.py:43-63). Injected via
    LangGraph ``Runtime.context``; never serialized into state/checkpoints.
    """

    sub_agent_registry: SubAgentRegistry
    sub_agent_executor: SubAgentExecutor
    master_model: MaterialAgentModel
    settings: AgentSettings
    conversation_store: ConversationStore
    clock: Clock
    id_generator: IdGenerator
    data_analysis_coordinator: DataAnalysisCrossAgentCoordinator | None = None
    cancel_event: threading.Event | None = None
