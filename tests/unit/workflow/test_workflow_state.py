"""Unit tests for Stage 3 workflow state models (S3-M2)."""

import json
from datetime import datetime
from operator import add
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from materials_screening.workflow.state import (
    TERMINAL_WORKFLOW_STATUSES,
    ArtifactRef,
    WorkflowCheckpointView,
    WorkflowState,
    WorkflowStateView,
    WorkflowStatus,
)

_VALID_REF: dict[str, Any] = {
    "name": "retrieval",
    "relative_path": "artifacts/retrieval.json",
    "sha256": "a" * 64,
    "media_type": "application/json",
    "size_bytes": 12,
}


class TestWorkflowStatus:
    def test_values(self) -> None:
        assert WorkflowStatus.INITIALIZING.value == "initializing"
        assert WorkflowStatus.RESOLVING_REQUEST.value == "resolving_request"
        assert WorkflowStatus.READY_FOR_RETRIEVAL.value == "ready_for_retrieval"
        assert WorkflowStatus.RETRIEVING.value == "retrieving"
        assert WorkflowStatus.FILTERING.value == "filtering"
        assert WorkflowStatus.RANKING.value == "ranking"
        assert WorkflowStatus.VALIDATING.value == "validating"
        assert WorkflowStatus.EXPORTING.value == "exporting"
        assert WorkflowStatus.NEEDS_CLARIFICATION.value == "needs_clarification"
        assert WorkflowStatus.INVALID_REQUEST.value == "invalid_request"
        assert WorkflowStatus.UNSUPPORTED_REQUEST.value == "unsupported_request"
        assert WorkflowStatus.NO_RESULTS.value == "no_results"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"

    def test_terminal_statuses(self) -> None:
        assert (
            frozenset(
                {
                    WorkflowStatus.NEEDS_CLARIFICATION,
                    WorkflowStatus.INVALID_REQUEST,
                    WorkflowStatus.UNSUPPORTED_REQUEST,
                    WorkflowStatus.NO_RESULTS,
                    WorkflowStatus.COMPLETED,
                    WorkflowStatus.FAILED,
                }
            )
            == TERMINAL_WORKFLOW_STATUSES
        )

    def test_intermediate_statuses_are_not_terminal(self) -> None:
        intermediate = {
            WorkflowStatus.INITIALIZING,
            WorkflowStatus.RESOLVING_REQUEST,
            WorkflowStatus.READY_FOR_RETRIEVAL,
            WorkflowStatus.RETRIEVING,
            WorkflowStatus.FILTERING,
            WorkflowStatus.RANKING,
            WorkflowStatus.VALIDATING,
            WorkflowStatus.EXPORTING,
        }
        assert intermediate.isdisjoint(TERMINAL_WORKFLOW_STATUSES)


class TestArtifactRef:
    def test_valid(self) -> None:
        ref = ArtifactRef.model_validate(_VALID_REF)
        assert ref.name == "retrieval"
        assert ref.relative_path == "artifacts/retrieval.json"
        assert ref.sha256 == "a" * 64
        assert ref.item_count is None
        assert ref.schema_name is None

    def test_optional_fields(self) -> None:
        ref = ArtifactRef.model_validate(
            {
                **_VALID_REF,
                "item_count": 7,
                "schema_name": "retrieval-v1",
                "schema_version": "1",
            }
        )
        assert ref.item_count == 7
        assert ref.schema_name == "retrieval-v1"

    def test_json_serializable(self) -> None:
        ref = ArtifactRef.model_validate(_VALID_REF)
        dumped = ref.model_dump(mode="json")
        assert json.loads(json.dumps(dumped)) == dumped

    def test_frozen(self) -> None:
        assert ArtifactRef.model_config.get("frozen") is True

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRef.model_validate({**_VALID_REF, "junk": 1})

    @pytest.mark.parametrize(
        "relative_path",
        ["/artifacts/retrieval.json", "artifacts/../retrieval.json", "a\\b.json"],
    )
    def test_unsafe_relative_path_rejected(self, relative_path: str) -> None:
        with pytest.raises(ValidationError):
            ArtifactRef.model_validate({**_VALID_REF, "relative_path": relative_path})

    @pytest.mark.parametrize("sha256", ["abc", "g" + "0" * 63, "A" + "0" * 63])
    def test_invalid_sha256_rejected(self, sha256: str) -> None:
        with pytest.raises(ValidationError):
            ArtifactRef.model_validate({**_VALID_REF, "sha256": sha256})

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRef.model_validate({**_VALID_REF, "size_bytes": -1})

    def test_zero_item_count_accepted(self) -> None:
        ref = ArtifactRef.model_validate({**_VALID_REF, "item_count": 0})
        assert ref.item_count == 0

    def test_negative_item_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRef.model_validate({**_VALID_REF, "item_count": -1})


def _full_state() -> dict[str, Any]:
    event = {
        "event_id": "event-1",
        "run_id": "run-1",
        "node": "initialize_run",
        "event_type": "started",
        "created_at": "2026-08-06T00:00:00Z",
        "status": "initializing",
        "message": "run started",
        "metrics": {"retrieved": 0},
    }
    return {
        "run_id": "run-1",
        "workflow_version": "workflow-v1",
        "input_mode": "query",
        "user_query": "find materials",
        "raw_request": None,
        "output_root": "data/workflow_runs",
        "export_cif": True,
        "status": "completed",
        "current_node": "finalize_success",
        "started_at": "2026-08-06T00:00:00Z",
        "finished_at": "2026-08-06T00:00:01Z",
        "planner_result": {"status": "ready"},
        "planner_status": "ready",
        "clarification_question": None,
        "screening_request": {"limit": 10},
        "retrieval_ref": dict(_VALID_REF),
        "filtered_ref": dict(_VALID_REF),
        "filter_trace_ref": dict(_VALID_REF),
        "ranked_ref": dict(_VALID_REF),
        "validation_ref": dict(_VALID_REF),
        "screening_result_ref": dict(_VALID_REF),
        "export_manifest_ref": dict(_VALID_REF),
        "retrieved_count": 5,
        "filtered_count": 3,
        "returned_count": 2,
        "validation_passed": True,
        "exports": ["exports/request.json"],
        "warnings": [],
        "error": None,
        "events": [event],
    }


class TestWorkflowState:
    def test_full_state_is_json_serializable(self) -> None:
        state = _full_state()
        dumped = json.dumps(state)
        assert json.loads(dumped) == state

    def test_state_contains_no_pydantic_path_datetime_or_exception(self) -> None:
        state = _full_state()
        forbidden = (BaseModel, Path, datetime, Exception)
        assert not any(isinstance(value, forbidden) for value in state.values())
        assert not any(
            isinstance(value, forbidden)
            for event in state["events"]
            for value in event.values()
        )

    def test_events_reducer_appends(self) -> None:
        first = [{"event_id": "event-1"}]
        second = [{"event_id": "event-2"}]
        assert add(first, second) == [
            {"event_id": "event-1"},
            {"event_id": "event-2"},
        ]

    def test_events_field_uses_append_reducer(self) -> None:
        annotation = WorkflowState.__annotations__["events"]
        assert annotation.__metadata__ == (add,)

    def test_all_state_fields_optional(self) -> None:
        assert WorkflowState.__required_keys__ == frozenset()
        expected = {
            "run_id",
            "workflow_version",
            "input_mode",
            "user_query",
            "raw_request",
            "output_root",
            "export_cif",
            "status",
            "current_node",
            "started_at",
            "finished_at",
            "planner_result",
            "planner_status",
            "clarification_question",
            "screening_request",
            "retrieval_ref",
            "filtered_ref",
            "filter_trace_ref",
            "ranked_ref",
            "validation_ref",
            "screening_result_ref",
            "export_manifest_ref",
            "retrieved_count",
            "filtered_count",
            "returned_count",
            "validation_passed",
            "exports",
            "warnings",
            "error",
            "events",
        }
        assert set(WorkflowState.__annotations__) == expected


class TestWorkflowStateView:
    def test_valid(self) -> None:
        view = WorkflowStateView(
            run_id="run-1",
            thread_id="thread-1",
            workflow_version="workflow-v1",
            status="completed",
            current_node="finalize_success",
            started_at="2026-08-06T00:00:00Z",
            finished_at="2026-08-06T00:00:01Z",
            planner_status="ready",
            retrieved_count=5,
            filtered_count=3,
            returned_count=2,
            validation_passed=True,
            exports=("exports/request.json",),
            warnings=(),
            error=None,
        )
        dumped = view.model_dump(mode="json")
        assert json.loads(json.dumps(dumped)) == dumped

    def test_frozen_and_extra_rejected(self) -> None:
        assert WorkflowStateView.model_config.get("frozen") is True
        with pytest.raises(ValidationError):
            WorkflowStateView(
                run_id="run-1",
                thread_id="thread-1",
                workflow_version="workflow-v1",
                status="completed",
                junk=1,
            )

    def test_view_exposes_no_artifact_refs(self) -> None:
        assert not any(name.endswith("_ref") for name in WorkflowStateView.model_fields)


class TestWorkflowCheckpointView:
    def test_valid(self) -> None:
        view = WorkflowCheckpointView(
            step=3,
            checkpoint_id="checkpoint-1",
            source="super-step",
            current_node="filter_materials",
            status="filtering",
            next_nodes=("rank_materials",),
            created_at="2026-08-06T00:00:00Z",
        )
        dumped = view.model_dump(mode="json")
        assert json.loads(json.dumps(dumped)) == dumped

    def test_frozen_and_extra_rejected(self) -> None:
        assert WorkflowCheckpointView.model_config.get("frozen") is True
        with pytest.raises(ValidationError):
            WorkflowCheckpointView(
                step=0,
                checkpoint_id="checkpoint-1",
                source="super-step",
                junk=1,
            )

    def test_negative_step_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowCheckpointView(
                step=-1,
                checkpoint_id="checkpoint-1",
                source="super-step",
                status="",
                next_nodes=(),
                created_at=None,
            )
