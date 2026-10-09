"""Tests for per-case outcomes and redacted evaluation reports (D2-M8)."""

import json
from pathlib import Path

from materials_screening.evaluation.planner_evaluator import (
    PlannerEvalCase,
    PlannerEvaluator,
    PlannerStatus,
)
from materials_screening.evaluation.report import (
    failures_to_json,
    failures_to_markdown,
    metrics_to_markdown,
    write_eval_report,
)
from materials_screening.llm.mock_provider import (
    MockStructuredProvider,
    MockErrorKind,
)
from materials_screening.planner.models import (
    DraftStatus,
    EnergyUnit,
    PlannerDraft,
)
from materials_screening.planner.service import PlannerService
from materials_screening.planner.settings import Settings


def _draft() -> PlannerDraft:
    return PlannerDraft.model_validate(
        {
            "status": DraftStatus.EXTRACTED,
            "required_elements": [],
            "excluded_elements": [],
            "chemsys": None,
            "formula": None,
            "band_gap_min": None,
            "band_gap_max": None,
            "band_gap_unit": EnergyUnit.UNSPECIFIED,
            "hull_min": None,
            "hull_max": None,
            "hull_unit": "unspecified",
            "density_min": None,
            "density_max": None,
            "density_unit": "unspecified",
            "crystal_system": None,
            "spacegroup_numbers": [],
            "is_metal": None,
            "is_stable": None,
            "theoretical": None,
            "target_band_gap": None,
            "target_band_gap_unit": EnergyUnit.UNSPECIFIED,
            "limit": None,
            "ambiguities": [],
            "unsupported_requirements": [],
            "conflicts": [],
            "assumptions": [],
            "clarification_question": "",
            "evidence": [],
        }
    )


def _service(provider: MockStructuredProvider) -> PlannerService:
    return PlannerService(
        settings=Settings(
            _env_file=None,
            llm_provider="mock",
            planner_prompt_version="planner-v2",
            planner_schema_version="planner-draft-v1",
        ),
        provider=provider,
    )


def _case(case_id: str, query: str, status: PlannerStatus) -> PlannerEvalCase:
    return PlannerEvalCase(
        id=case_id,
        query=query,
        expected_status=status,
        expected_request={},
        expected_ambiguity_codes=(),
        expected_conflict_codes=(),
        tags=(),
    )


class TestEvaluateWithDetails:
    def test_schema_failure_outcome_is_redacted(self) -> None:
        provider = MockStructuredProvider(
            fixtures={"ok": _draft()},
            error_fixtures={"boom": MockErrorKind.SCHEMA_MISMATCH},
        )
        cases = [
            _case("ok", "ok", PlannerStatus.READY),
            _case("boom", "boom", PlannerStatus.READY),
        ]

        metrics, outcomes = PlannerEvaluator(_service(provider)).evaluate_with_details(
            cases
        )

        assert metrics.schema_success == 1
        by_id = {outcome.id: outcome for outcome in outcomes}
        assert by_id["ok"].schema_success is True
        assert by_id["ok"].actual_status is PlannerStatus.READY
        assert by_id["boom"].schema_success is False
        assert by_id["boom"].actual_status is None
        assert by_id["boom"].error_type == "LLMStructuredOutputError"


class TestReportRedaction:
    def _outcomes(self) -> tuple:
        provider = MockStructuredProvider(
            fixtures={"ok": _draft()},
            error_fixtures={"boom": MockErrorKind.INVALID_JSON},
        )
        cases = [
            _case("ok", "ok", PlannerStatus.READY),
            _case("boom", "boom", PlannerStatus.READY),
        ]
        _, outcomes = PlannerEvaluator(_service(provider)).evaluate_with_details(cases)
        return outcomes

    def test_failures_json_has_no_query_or_error_message(self) -> None:
        payload = json.loads(failures_to_json(self._outcomes()))
        assert len(payload) == 1
        assert payload[0]["id"] == "boom"
        assert payload[0]["error_type"] == "LLMStructuredOutputError"
        assert "query" not in payload[0]
        assert "message" not in payload[0]

    def test_failures_markdown_has_no_query_or_error_message(self) -> None:
        markdown = failures_to_markdown(self._outcomes())
        assert markdown.count("boom") == 1
        assert "| boom |" in markdown
        assert "no mock fixture" not in markdown
        assert "invalid JSON" not in markdown

    def test_metrics_markdown_contains_key_lines(self) -> None:
        provider = MockStructuredProvider(fixtures={"ok": _draft()})
        metrics, _ = PlannerEvaluator(_service(provider)).evaluate_with_details(
            [_case("ok", "ok", PlannerStatus.READY)]
        )
        markdown = metrics_to_markdown(metrics)
        assert "# Planner Evaluation Metrics" in markdown
        assert "- Total cases: 1" in markdown
        assert "Schema success rate: 1.000 (1/1)" in markdown

    def test_write_eval_report_creates_all_files(self, tmp_path: Path) -> None:
        provider = MockStructuredProvider(fixtures={"ok": _draft()})
        metrics, outcomes = PlannerEvaluator(_service(provider)).evaluate_with_details(
            [_case("ok", "ok", PlannerStatus.READY)]
        )
        run_dir = tmp_path / "run"

        write_eval_report(run_dir, metrics, outcomes)

        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "metrics.md").exists()
        assert (run_dir / "failures.json").exists()
        assert (run_dir / "failures.md").exists()
