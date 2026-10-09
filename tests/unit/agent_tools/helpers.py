"""Shared test helpers for agent tool tests (S3.5-M3)."""

from datetime import UTC, datetime
from typing import Any

from materials_screening.agent.context import AgentToolContext
from materials_screening.agent.ledger import ToolExecutionLedger
from materials_screening.agent.policy import ConversationWorkflowLink
from materials_screening.workflow.state import (
    WorkflowCheckpointView,
    WorkflowStateView,
)


class SeqIdGenerator:
    def __init__(self) -> None:
        self.count = 0

    def new_id(self) -> str:
        self.count += 1
        return f"evt_{self.count}"


def fixed_clock() -> datetime:
    return datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


def state_view(
    *,
    status: str = "completed",
    thread_id: str = "thread-1",
    retrieved: int = 0,
    filtered: int = 0,
    returned: int = 0,
    validation: bool | None = None,
    current_node: str | None = None,
    exports: tuple[str, ...] = (),
) -> WorkflowStateView:
    return WorkflowStateView(
        run_id="run-1",
        thread_id=thread_id,
        workflow_version="workflow-v1",
        status=status,
        current_node=current_node,
        started_at=None,
        finished_at=None,
        planner_status=None,
        retrieved_count=retrieved,
        filtered_count=filtered,
        returned_count=returned,
        validation_passed=validation,
        exports=exports,
        warnings=(),
        error=None,
    )


def checkpoint_view(
    *,
    step: int = 1,
    checkpoint_id: str = "cp-1",
    node: str = "filter_materials",
    status: str = "filtering",
    created_at: str = "2026-08-06T00:00:00Z",
) -> WorkflowCheckpointView:
    return WorkflowCheckpointView(
        step=step,
        checkpoint_id=checkpoint_id,
        source="loop",
        current_node=node,
        status=status,
        next_nodes=(),
        created_at=created_at,
    )


class FakeRunner:
    def __init__(
        self,
        view: WorkflowStateView | None = None,
        history: tuple[WorkflowCheckpointView, ...] = (),
    ) -> None:
        self._view = view
        self._history = history
        self.get_state_calls: list[str] = []
        self.get_history_calls: list[tuple[str, int | None]] = []

    def get_state(self, thread_id: str) -> WorkflowStateView:
        self.get_state_calls.append(thread_id)
        if self._view is None:
            raise AssertionError("no view configured")
        return self._view

    def get_history(
        self,
        thread_id: str,
        limit: int | None = None,
    ) -> tuple[WorkflowCheckpointView, ...]:
        self.get_history_calls.append((thread_id, limit))
        return self._history


class FakeReader:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[str] = []

    def read(self, thread_id: str) -> dict[str, Any]:
        self.calls.append(thread_id)
        return self._payload


def context(
    *,
    runner: Any,
    reader: Any = None,
    ledger: ToolExecutionLedger | None = None,
    call_id: str = "call_1",
    user_turn_id: str = "turn_1",
    conversation_id: str = "c1",
    active_thread: str | None = "thread-1",
    links: tuple[ConversationWorkflowLink, ...] | None = None,
) -> AgentToolContext:
    if links is None:
        links = (
            ConversationWorkflowLink(
                conversation_id="c1",
                thread_id="thread-linked",
                created_at="now",
            ),
        )
    return AgentToolContext(
        workflow_runner=runner,
        workflow_result_reader=reader if reader is not None else FakeReader({}),
        clock=fixed_clock,
        id_generator=SeqIdGenerator(),
        ledger=ledger if ledger is not None else ToolExecutionLedger(),
        call_id=call_id,
        user_turn_id=user_turn_id,
        conversation_id=conversation_id,
        active_workflow_thread_id=active_thread,
        conversation_links=links,
    )


def result_payload(*, n: int = 2) -> dict[str, Any]:
    return {
        "request": {
            "limit": 10,
            "band_gap_ev": {"min": 1.0, "max": 2.0},
        },
        "validation": {
            "passed": True,
            "errors": [],
            "warnings": [],
            "checked_material_ids": [f"mp-{i}" for i in range(1, n + 1)],
        },
        "ranked_materials": [
            {
                "rank": index,
                "record": {
                    "material_id": f"mp-{index}",
                    "formula_pretty": f"F{index}",
                    "band_gap_ev": 1.5,
                    "energy_above_hull_ev_atom": 0.0,
                    "formation_energy_ev_atom": -2.0,
                    "is_gap_direct": True,
                    "symmetry": {"crystal_system": "Cubic"},
                    "source": "mock",
                    "provenance": [{"value_type": "dft_calculated"}],
                },
                "total_score": 0.8,
                "score_breakdown": {
                    "stability": 0.45,
                    "band_gap_match": 0.3,
                    "completeness": 0.1,
                    "direct_gap": 0.05,
                },
            }
            for index in range(1, n + 1)
        ],
    }
