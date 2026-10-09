"""Unit tests for the get_screening_result tool (S3.5-M3)."""

import pytest
from pydantic import ValidationError

from materials_screening.agent.errors import WorkflowResultReadError
from materials_screening.agent_tools.get_screening_result import (
    GetScreeningResultInput,
    GetScreeningResultTool,
)
from tests.unit.agent_tools.helpers import (
    FakeReader,
    FakeRunner,
    context,
    result_payload,
    state_view,
)


class TestGetScreeningResultTool:
    def test_completed_validated_reads_summary(self) -> None:
        payload = result_payload(n=3)
        reader = FakeReader(payload)
        runner = FakeRunner(
            view=state_view(
                status="completed",
                validation=True,
                retrieved=3,
            )
        )
        tool_context = context(runner=runner, reader=reader)
        output = GetScreeningResultTool().execute(
            GetScreeningResultInput(thread_id=None, top_n=2),
            tool_context,
        )
        assert output.workflow_status == "completed"
        assert output.error_code is None
        assert len(output.materials) == 2
        first = output.materials[0]
        assert first.rank == 1
        assert first.material_id == "mp-1"
        assert first.formula_pretty == "F1"
        assert first.band_gap_ev == 1.5
        assert first.is_gap_direct is True
        assert first.crystal_system == "Cubic"
        assert first.total_score == 0.8
        assert first.score_breakdown["stability"] == 0.45
        assert first.source == "mock"
        assert first.value_type == "dft_calculated"
        assert reader.calls == ["thread-1"]
        assert output.request_summary["limit"] == 10
        assert output.scientific_notice

    @pytest.mark.parametrize("top_n", [0, 21])
    def test_top_n_out_of_bounds_rejected(self, top_n: int) -> None:
        with pytest.raises(ValidationError):
            GetScreeningResultInput(thread_id=None, top_n=top_n)

    @pytest.mark.parametrize("top_n", [1, 20])
    def test_top_n_bounds_accepted(self, top_n: int) -> None:
        assert GetScreeningResultInput(top_n=top_n).top_n == top_n

    @pytest.mark.parametrize("status", ["retrieving", "filtering", "validating"])
    def test_running_status_returns_clear_state(self, status: str) -> None:
        runner = FakeRunner(view=state_view(status=status))
        output = GetScreeningResultTool().execute(
            GetScreeningResultInput(thread_id=None),
            context(runner=runner),
        )
        assert output.workflow_status == status
        assert output.materials == []
        assert output.error_code is None

    @pytest.mark.parametrize(
        "status",
        [
            "failed",
            "no_results",
            "needs_clarification",
            "invalid_request",
            "unsupported_request",
        ],
    )
    def test_non_result_status_returns_clear_state(self, status: str) -> None:
        runner = FakeRunner(view=state_view(status=status))
        output = GetScreeningResultTool().execute(
            GetScreeningResultInput(thread_id=None),
            context(runner=runner),
        )
        assert output.workflow_status == status
        assert output.materials == []

    def test_completed_without_validation(self) -> None:
        runner = FakeRunner(view=state_view(status="completed", validation=False))
        output = GetScreeningResultTool().execute(
            GetScreeningResultInput(thread_id=None),
            context(runner=runner),
        )
        assert output.error_code == "VALIDATION_NOT_PASSED"
        assert output.materials == []

    def test_no_active_thread(self) -> None:
        runner = FakeRunner(view=state_view())
        tool_context = context(runner=runner, active_thread=None)
        output = GetScreeningResultTool().execute(
            GetScreeningResultInput(thread_id=None),
            tool_context,
        )
        assert output.error_code == "NO_ACTIVE_WORKFLOW"

    def test_ownership_denied(self) -> None:
        runner = FakeRunner(view=state_view())
        output = GetScreeningResultTool().execute(
            GetScreeningResultInput(thread_id="thread-other"),
            context(runner=runner),
        )
        assert output.error_code == "OWNERSHIP_DENIED"

    def test_unknown_thread(self) -> None:
        runner = FakeRunner(view=state_view(status=""))
        output = GetScreeningResultTool().execute(
            GetScreeningResultInput(thread_id="thread-1"),
            context(runner=runner),
        )
        assert output.error_code == "THREAD_NOT_FOUND"

    def test_reader_error_maps_to_code(self) -> None:
        class _BrokenReader:
            def read(self, thread_id: str) -> dict[str, object]:
                raise WorkflowResultReadError("ARTIFACT_INTEGRITY", "corrupted")

        runner = FakeRunner(view=state_view(status="completed", validation=True))
        output = GetScreeningResultTool().execute(
            GetScreeningResultInput(thread_id=None),
            context(runner=runner, reader=_BrokenReader()),
        )
        assert output.error_code == "ARTIFACT_INTEGRITY"

    def test_malformed_result_maps_to_invalid(self) -> None:
        reader = FakeReader({"ranked_materials": [{"rank": 1}], "request": {}})
        runner = FakeRunner(view=state_view(status="completed", validation=True))
        output = GetScreeningResultTool().execute(
            GetScreeningResultInput(thread_id=None),
            context(runner=runner, reader=reader),
        )
        assert output.error_code == "RESULT_INVALID"
        assert output.materials == []

    def test_reader_unknown_bug_propagates(self) -> None:
        class _BoomReader:
            def read(self, thread_id: str) -> dict[str, object]:
                raise RuntimeError("unexpected")

        runner = FakeRunner(view=state_view(status="completed", validation=True))
        with pytest.raises(RuntimeError, match="unexpected"):
            GetScreeningResultTool().execute(
                GetScreeningResultInput(thread_id=None),
                context(runner=runner, reader=_BoomReader()),
            )

    def test_output_has_no_structure_or_provenance(self) -> None:
        reader = FakeReader(result_payload(n=1))
        runner = FakeRunner(view=state_view(status="completed", validation=True))
        output = GetScreeningResultTool().execute(
            GetScreeningResultInput(thread_id=None),
            context(runner=runner, reader=reader),
        )
        serialized = output.model_dump_json()
        assert "structure_dict" not in serialized
        assert "provenance" not in serialized
        assert "retrieved_at" not in serialized

    def test_evidence_recorded(self) -> None:
        reader = FakeReader(result_payload(n=1))
        runner = FakeRunner(view=state_view(status="completed", validation=True))
        tool_context = context(runner=runner, reader=reader)
        output = GetScreeningResultTool().execute(
            GetScreeningResultInput(thread_id=None),
            tool_context,
        )
        assert tool_context.ledger.evidence_ids() == (output.evidence_id,)
