"""Unit tests for the compare_ranked_materials tool (S3.5-M3)."""

from typing import Any

import pytest
from pydantic import ValidationError

from materials_screening.agent.errors import WorkflowResultReadError
from materials_screening.agent_tools.compare_ranked_materials import (
    CompareRankedMaterialsInput,
    CompareRankedMaterialsTool,
)
from tests.unit.agent_tools.helpers import (
    FakeReader,
    FakeRunner,
    context,
    state_view,
)


def _payload_with(
    scores: list[tuple[str, float, dict[str, float]]],
) -> dict[str, Any]:
    return {
        "request": {},
        "validation": {"passed": True},
        "ranked_materials": [
            {
                "rank": index,
                "record": {"material_id": material_id, "source": "mock"},
                "total_score": total,
                "score_breakdown": breakdown,
            }
            for index, (material_id, total, breakdown) in enumerate(scores, start=1)
        ],
    }


def _two_material_payload() -> dict[str, Any]:
    return _payload_with(
        [
            (
                "mp-1",
                0.8,
                {
                    "stability": 0.45,
                    "band_gap_match": 0.2,
                    "completeness": 0.1,
                    "direct_gap": 0.05,
                },
            ),
            (
                "mp-2",
                0.6,
                {
                    "stability": 0.3,
                    "band_gap_match": 0.2,
                    "completeness": 0.05,
                    "direct_gap": 0.05,
                },
            ),
        ]
    )


class TestCompareRankedMaterialsTool:
    def test_success_sorted_by_rank(self) -> None:
        reader = FakeReader(_two_material_payload())
        runner = FakeRunner(view=state_view(status="completed", validation=True))
        tool_context = context(runner=runner, reader=reader)
        output = CompareRankedMaterialsTool().execute(
            CompareRankedMaterialsInput(
                thread_id=None,
                material_ids=["mp-2", "mp-1"],
            ),
            tool_context,
        )
        assert output.error_code is None
        assert [row.material_id for row in output.rows] == ["mp-1", "mp-2"]
        assert [row.rank for row in output.rows] == [1, 2]
        assert output.rows[0].total_score == 0.8
        assert output.rows[0].stability_score == 0.45
        assert output.rows[0].band_gap_match_score == 0.2
        assert output.rows[0].completeness_score == 0.1
        assert output.rows[0].direct_gap_score == 0.05
        assert reader.calls == ["thread-1"]

    def test_deterministic_summary_content(self) -> None:
        reader = FakeReader(_two_material_payload())
        runner = FakeRunner(view=state_view(status="completed", validation=True))
        output = CompareRankedMaterialsTool().execute(
            CompareRankedMaterialsInput(
                thread_id=None,
                material_ids=["mp-1", "mp-2"],
            ),
            context(runner=runner, reader=reader),
        )
        assert "mp-1 的稳定性得分比mp-2 高 0.15。" in output.deterministic_summary
        assert "mp-1 的完整度得分比mp-2 高 0.05。" in output.deterministic_summary
        assert "mp-1 总分比mp-2 高 0.2，因此排名更高。" in output.deterministic_summary
        assert "带隙匹配得分" not in "".join(output.deterministic_summary)

    def test_summary_deterministic_across_runs(self) -> None:
        payload = _two_material_payload()
        runner = FakeRunner(view=state_view(status="completed", validation=True))
        first = CompareRankedMaterialsTool().execute(
            CompareRankedMaterialsInput(
                thread_id=None,
                material_ids=["mp-2", "mp-1"],
            ),
            context(runner=runner, reader=FakeReader(payload)),
        )
        second = CompareRankedMaterialsTool().execute(
            CompareRankedMaterialsInput(
                thread_id=None,
                material_ids=["mp-1", "mp-2"],
            ),
            context(runner=runner, reader=FakeReader(payload)),
        )
        assert first.deterministic_summary == second.deterministic_summary
        assert first.rows == second.rows

    def test_missing_breakdown_components_default_to_zero(self) -> None:
        payload = _payload_with(
            [
                ("mp-1", 0.5, {"stability": 0.4}),
                ("mp-2", 0.4, {}),
            ]
        )
        runner = FakeRunner(view=state_view(status="completed", validation=True))
        output = CompareRankedMaterialsTool().execute(
            CompareRankedMaterialsInput(
                thread_id=None,
                material_ids=["mp-1", "mp-2"],
            ),
            context(runner=runner, reader=FakeReader(payload)),
        )
        assert output.rows[1].stability_score == 0.0
        assert output.rows[1].band_gap_match_score == 0.0

    @pytest.mark.parametrize("count", [1, 6])
    def test_material_count_out_of_bounds_rejected(self, count: int) -> None:
        with pytest.raises(ValidationError):
            CompareRankedMaterialsInput(material_ids=[f"mp-{i}" for i in range(count)])

    @pytest.mark.parametrize("count", [2, 5])
    def test_material_count_bounds_accepted(self, count: int) -> None:
        input_model = CompareRankedMaterialsInput(
            material_ids=[f"mp-{i}" for i in range(count)]
        )
        assert len(input_model.material_ids) == count

    def test_duplicate_material_id_rejected(self) -> None:
        reader = FakeReader(_two_material_payload())
        runner = FakeRunner(view=state_view(status="completed", validation=True))
        output = CompareRankedMaterialsTool().execute(
            CompareRankedMaterialsInput(
                thread_id=None,
                material_ids=["mp-1", "mp-1"],
            ),
            context(runner=runner, reader=reader),
        )
        assert output.error_code == "DUPLICATE_MATERIAL_ID"

    def test_missing_material_clear_error(self) -> None:
        reader = FakeReader(_two_material_payload())
        runner = FakeRunner(view=state_view(status="completed", validation=True))
        output = CompareRankedMaterialsTool().execute(
            CompareRankedMaterialsInput(
                thread_id=None,
                material_ids=["mp-1", "mp-999"],
            ),
            context(runner=runner, reader=reader),
        )
        assert output.error_code == "MATERIAL_NOT_FOUND"
        assert "mp-999" in "".join(output.deterministic_summary)

    def test_no_active_thread(self) -> None:
        runner = FakeRunner(view=state_view())
        tool_context = context(runner=runner, active_thread=None)
        output = CompareRankedMaterialsTool().execute(
            CompareRankedMaterialsInput(
                thread_id=None,
                material_ids=["mp-1", "mp-2"],
            ),
            tool_context,
        )
        assert output.error_code == "NO_ACTIVE_WORKFLOW"

    def test_ownership_denied(self) -> None:
        runner = FakeRunner(view=state_view())
        output = CompareRankedMaterialsTool().execute(
            CompareRankedMaterialsInput(
                thread_id="thread-other",
                material_ids=["mp-1", "mp-2"],
            ),
            context(runner=runner),
        )
        assert output.error_code == "OWNERSHIP_DENIED"

    def test_unknown_thread(self) -> None:
        runner = FakeRunner(view=state_view(status=""))
        output = CompareRankedMaterialsTool().execute(
            CompareRankedMaterialsInput(
                thread_id="thread-1",
                material_ids=["mp-1", "mp-2"],
            ),
            context(runner=runner),
        )
        assert output.error_code == "THREAD_NOT_FOUND"

    @pytest.mark.parametrize("status", ["retrieving", "failed", "no_results"])
    def test_not_completed_status(self, status: str) -> None:
        runner = FakeRunner(view=state_view(status=status))
        output = CompareRankedMaterialsTool().execute(
            CompareRankedMaterialsInput(
                thread_id=None,
                material_ids=["mp-1", "mp-2"],
            ),
            context(runner=runner),
        )
        assert output.error_code == "WORKFLOW_NOT_COMPLETED"

    def test_validation_not_passed(self) -> None:
        runner = FakeRunner(view=state_view(status="completed", validation=False))
        output = CompareRankedMaterialsTool().execute(
            CompareRankedMaterialsInput(
                thread_id=None,
                material_ids=["mp-1", "mp-2"],
            ),
            context(runner=runner),
        )
        assert output.error_code == "VALIDATION_NOT_PASSED"

    def test_reader_error_maps_to_code(self) -> None:
        class _BrokenReader:
            def read(self, thread_id: str) -> dict[str, object]:
                raise WorkflowResultReadError("ARTIFACT_INTEGRITY", "corrupted")

        runner = FakeRunner(view=state_view(status="completed", validation=True))
        output = CompareRankedMaterialsTool().execute(
            CompareRankedMaterialsInput(
                thread_id=None,
                material_ids=["mp-1", "mp-2"],
            ),
            context(runner=runner, reader=_BrokenReader()),
        )
        assert output.error_code == "ARTIFACT_INTEGRITY"

    def test_evidence_recorded(self) -> None:
        reader = FakeReader(_two_material_payload())
        runner = FakeRunner(view=state_view(status="completed", validation=True))
        tool_context = context(runner=runner, reader=reader)
        output = CompareRankedMaterialsTool().execute(
            CompareRankedMaterialsInput(
                thread_id=None,
                material_ids=["mp-1", "mp-2"],
            ),
            tool_context,
        )
        assert tool_context.ledger.evidence_ids() == (output.evidence_id,)

    def test_unknown_bug_propagates(self) -> None:
        class _BoomRunner:
            def get_state(self, thread_id: str) -> object:
                raise RuntimeError("unexpected")

        with pytest.raises(RuntimeError, match="unexpected"):
            CompareRankedMaterialsTool().execute(
                CompareRankedMaterialsInput(
                    thread_id=None,
                    material_ids=["mp-1", "mp-2"],
                ),
                context(runner=_BoomRunner()),
            )
