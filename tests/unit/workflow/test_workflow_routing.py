"""Unit tests for S3-M4 route_after_request (pure routing)."""

from typing import Any

import pytest

from materials_screening.workflow.errors import WorkflowInvariantError
from materials_screening.workflow.routing import (
    route_after_export,
    route_after_filter,
    route_after_ranking,
    route_after_request,
    route_after_retrieval,
    route_after_validation,
)
from materials_screening.workflow.state import WorkflowStatus


class TestRouteAfterRequest:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (WorkflowStatus.READY_FOR_RETRIEVAL, "retrieve_materials"),
            (WorkflowStatus.NEEDS_CLARIFICATION, "finalize_planner_stop"),
            (WorkflowStatus.INVALID_REQUEST, "finalize_planner_stop"),
            (WorkflowStatus.UNSUPPORTED_REQUEST, "finalize_planner_stop"),
            (WorkflowStatus.FAILED, "finalize_failure"),
        ],
    )
    def test_maps_each_status_to_single_edge(
        self, status: WorkflowStatus, expected: str
    ) -> None:
        assert route_after_request({"status": status.value}) == expected

    def test_missing_status_raises(self) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_request({})

    @pytest.mark.parametrize("raw_status", ["bogus", "completed"])
    def test_unknown_or_unroutable_status_raises(self, raw_status: str) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_request({"status": raw_status})

    @pytest.mark.parametrize(
        "status",
        [
            WorkflowStatus.INITIALIZING,
            WorkflowStatus.RESOLVING_REQUEST,
            WorkflowStatus.RETRIEVING,
            WorkflowStatus.FILTERING,
            WorkflowStatus.RANKING,
            WorkflowStatus.VALIDATING,
            WorkflowStatus.EXPORTING,
        ],
    )
    def test_intermediate_statuses_raise(self, status: WorkflowStatus) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_request({"status": status.value})

    def test_is_pure_and_does_not_mutate_state(self) -> None:
        state: dict[str, Any] = {
            "status": WorkflowStatus.READY_FOR_RETRIEVAL.value,
            "retrieved_count": 0,
        }
        before = dict(state)
        assert route_after_request(state) == "retrieve_materials"
        assert state == before


class TestRouteAfterRetrieval:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (WorkflowStatus.RETRIEVING, "filter_materials"),
            (WorkflowStatus.FAILED, "finalize_failure"),
        ],
    )
    def test_maps_each_status_to_single_edge(
        self, status: WorkflowStatus, expected: str
    ) -> None:
        assert route_after_retrieval({"status": status.value}) == expected

    def test_missing_status_raises(self) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_retrieval({})

    @pytest.mark.parametrize("raw_status", ["bogus", "completed"])
    def test_unknown_or_unroutable_status_raises(self, raw_status: str) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_retrieval({"status": raw_status})

    @pytest.mark.parametrize(
        "status",
        [
            WorkflowStatus.READY_FOR_RETRIEVAL,
            WorkflowStatus.FILTERING,
            WorkflowStatus.NO_RESULTS,
        ],
    )
    def test_unexpected_statuses_raise(self, status: WorkflowStatus) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_retrieval({"status": status.value})

    def test_is_pure_and_does_not_mutate_state(self) -> None:
        state: dict[str, Any] = {
            "status": WorkflowStatus.RETRIEVING.value,
            "retrieved_count": 7,
        }
        before = dict(state)
        assert route_after_retrieval(state) == "filter_materials"
        assert state == before


class TestRouteAfterFilter:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (WorkflowStatus.FILTERING, "rank_materials"),
            (WorkflowStatus.NO_RESULTS, "finalize_no_results"),
            (WorkflowStatus.FAILED, "finalize_failure"),
        ],
    )
    def test_maps_each_status_to_single_edge(
        self, status: WorkflowStatus, expected: str
    ) -> None:
        assert route_after_filter({"status": status.value}) == expected

    def test_missing_status_raises(self) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_filter({})

    @pytest.mark.parametrize("raw_status", ["bogus", "completed"])
    def test_unknown_or_unroutable_status_raises(self, raw_status: str) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_filter({"status": raw_status})

    @pytest.mark.parametrize(
        "status",
        [
            WorkflowStatus.READY_FOR_RETRIEVAL,
            WorkflowStatus.RETRIEVING,
            WorkflowStatus.RANKING,
        ],
    )
    def test_unexpected_statuses_raise(self, status: WorkflowStatus) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_filter({"status": status.value})

    def test_is_pure_and_does_not_mutate_state(self) -> None:
        state: dict[str, Any] = {
            "status": WorkflowStatus.NO_RESULTS.value,
            "filtered_count": 0,
        }
        before = dict(state)
        assert route_after_filter(state) == "finalize_no_results"
        assert state == before


class TestRouteAfterRanking:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (WorkflowStatus.RANKING, "validate_results"),
            (WorkflowStatus.FAILED, "finalize_failure"),
        ],
    )
    def test_maps_each_status_to_single_edge(
        self, status: WorkflowStatus, expected: str
    ) -> None:
        assert route_after_ranking({"status": status.value}) == expected

    def test_missing_status_raises(self) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_ranking({})

    @pytest.mark.parametrize("raw_status", ["bogus", "completed"])
    def test_unknown_or_unroutable_status_raises(self, raw_status: str) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_ranking({"status": raw_status})

    @pytest.mark.parametrize(
        "status",
        [
            WorkflowStatus.READY_FOR_RETRIEVAL,
            WorkflowStatus.RETRIEVING,
            WorkflowStatus.FILTERING,
            WorkflowStatus.NO_RESULTS,
        ],
    )
    def test_unexpected_statuses_raise(self, status: WorkflowStatus) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_ranking({"status": status.value})

    def test_is_pure_and_does_not_mutate_state(self) -> None:
        state: dict[str, Any] = {
            "status": WorkflowStatus.RANKING.value,
            "returned_count": 5,
        }
        before = dict(state)
        assert route_after_ranking(state) == "validate_results"
        assert state == before


class TestRouteAfterValidation:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (WorkflowStatus.VALIDATING, "export_results"),
            (WorkflowStatus.FAILED, "finalize_failure"),
        ],
    )
    def test_maps_each_status_to_single_edge(
        self, status: WorkflowStatus, expected: str
    ) -> None:
        assert route_after_validation({"status": status.value}) == expected

    def test_missing_status_raises(self) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_validation({})

    @pytest.mark.parametrize("raw_status", ["bogus", "completed"])
    def test_unknown_or_unroutable_status_raises(self, raw_status: str) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_validation({"status": raw_status})

    @pytest.mark.parametrize(
        "status",
        [
            WorkflowStatus.READY_FOR_RETRIEVAL,
            WorkflowStatus.RETRIEVING,
            WorkflowStatus.FILTERING,
            WorkflowStatus.RANKING,
            WorkflowStatus.NO_RESULTS,
        ],
    )
    def test_unexpected_statuses_raise(self, status: WorkflowStatus) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_validation({"status": status.value})

    def test_is_pure_and_does_not_mutate_state(self) -> None:
        state: dict[str, Any] = {
            "status": WorkflowStatus.VALIDATING.value,
            "validation_passed": True,
        }
        before = dict(state)
        assert route_after_validation(state) == "export_results"
        assert state == before


class TestRouteAfterExport:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (WorkflowStatus.EXPORTING, "finalize_success"),
            (WorkflowStatus.FAILED, "finalize_failure"),
        ],
    )
    def test_maps_each_status_to_single_edge(
        self, status: WorkflowStatus, expected: str
    ) -> None:
        assert route_after_export({"status": status.value}) == expected

    def test_missing_status_raises(self) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_export({})

    @pytest.mark.parametrize("raw_status", ["bogus", "completed"])
    def test_unknown_or_unroutable_status_raises(self, raw_status: str) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_export({"status": raw_status})

    @pytest.mark.parametrize(
        "status",
        [
            WorkflowStatus.READY_FOR_RETRIEVAL,
            WorkflowStatus.RETRIEVING,
            WorkflowStatus.FILTERING,
            WorkflowStatus.RANKING,
            WorkflowStatus.VALIDATING,
            WorkflowStatus.NO_RESULTS,
        ],
    )
    def test_unexpected_statuses_raise(self, status: WorkflowStatus) -> None:
        with pytest.raises(WorkflowInvariantError):
            route_after_export({"status": status.value})

    def test_is_pure_and_does_not_mutate_state(self) -> None:
        state: dict[str, Any] = {
            "status": WorkflowStatus.EXPORTING.value,
            "exports": ["run-1/exports/request.json"],
        }
        before = dict(state)
        assert route_after_export(state) == "finalize_success"
        assert state == before
