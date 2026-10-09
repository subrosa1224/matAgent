"""Unit tests for the agent final validator (S3.5-M5)."""

import json
from typing import Any

import pytest

from materials_screening.agent.final_validator import (
    EvidenceRecord,
    FinalValidationContext,
    FinalValidationResult,
    FinalValidator,
)
from materials_screening.agent.models import AgentFinalDraft, AgentFinalStatus
from materials_screening.agent.policy import ConversationWorkflowLink


def _run_evidence(
    *,
    status: str = "completed",
    validation_passed: bool | None = True,
) -> EvidenceRecord:
    payload = {
        "status": status,
        "thread_id": "wf_t1",
        "planner_status": None,
        "clarification_question": None,
        "retrieved_count": 10,
        "filtered_count": 5,
        "returned_count": 5,
        "validation_passed": validation_passed,
        "exports": [],
        "warnings": [],
        "evidence_id": "ev-run",
    }
    return EvidenceRecord(
        evidence_id="ev-run",
        tool_name="run_screening_workflow",
        result_json=json.dumps(payload, ensure_ascii=False),
    )


def _result_evidence(material_ids: tuple[str, ...] = ("mp-1",)) -> EvidenceRecord:
    payload = {
        "status": "ok",
        "thread_id": "wf_t1",
        "materials": [
            {"material_id": material_id, "rank": index + 1}
            for index, material_id in enumerate(material_ids)
        ],
        "evidence_id": "ev-res",
    }
    return EvidenceRecord(
        evidence_id="ev-res",
        tool_name="get_screening_result",
        result_json=json.dumps(payload, ensure_ascii=False),
    )


def _context(**overrides: Any) -> FinalValidationContext:
    values: dict[str, Any] = {
        "conversation_id": "conv_1",
        "user_turn_id": "turn_1",
        "evidence": (_run_evidence(), _result_evidence()),
        "active_workflow_thread_id": "wf_t1",
        "conversation_links": (
            ConversationWorkflowLink(
                conversation_id="conv_1",
                thread_id="wf_t1",
                created_at="2026-08-06T00:00:00Z",
            ),
        ),
    }
    values.update(overrides)
    return FinalValidationContext(**values)


def _draft(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "status": AgentFinalStatus.COMPLETED.value,
        "answer": "推荐 BaTiO3（mp-1），带隙 3.2 eV。",
        "active_workflow_thread_id": "wf_t1",
        "referenced_material_ids": ["mp-1"],
        "evidence_ids": ["ev-res"],
        "warnings": [],
        "follow_up_question": None,
    }
    values.update(overrides)
    return values


def _validator(**overrides: Any) -> FinalValidator:
    return FinalValidator(**overrides)


class TestSchemaAndAnswer:
    def test_valid_completed_draft_passes(self) -> None:
        result = _validator().validate(_draft(), _context())

        assert result.ok is True
        assert result.errors == ()

    def test_draft_instance_accepted(self) -> None:
        draft = AgentFinalDraft.model_validate(_draft())

        result = _validator().validate(draft, _context())

        assert result.ok is True

    def test_invalid_schema_rejected(self) -> None:
        result = _validator().validate(_draft(status="bogus"), _context())

        assert result.ok is False
        assert "INVALID_FINAL_DRAFT" in result.codes

    def test_non_mapping_rejected(self) -> None:
        result = _validator().validate(42, _context())

        assert result.ok is False
        assert result.codes == ("INVALID_FINAL_DRAFT",)

    def test_empty_answer_rejected(self) -> None:
        result = _validator().validate(_draft(answer=" "), _context())

        assert "ANSWER_EMPTY" in result.codes

    def test_answer_too_long_rejected(self) -> None:
        validator = _validator(max_answer_chars=100)
        result = validator.validate(_draft(answer="x" * 200), _context())

        assert result.codes == ("ANSWER_TOO_LONG",)

    def test_answer_at_limit_ok(self) -> None:
        validator = _validator(max_answer_chars=100)
        draft = _draft(
            answer="x" * 100,
            active_workflow_thread_id=None,
            referenced_material_ids=[],
            evidence_ids=[],
        )

        result = validator.validate(draft, _context())

        assert result.ok is True

    def test_instance_draft_too_long_for_custom_limit(self) -> None:
        validator = _validator(max_answer_chars=10)
        draft = AgentFinalDraft.model_validate(_draft(answer="x" * 20))

        result = validator.validate(draft, _context())

        assert result.codes == ("ANSWER_TOO_LONG",)

    def test_needs_user_input_requires_follow_up_question(self) -> None:
        result = _validator().validate(
            _draft(
                status=AgentFinalStatus.NEEDS_USER_INPUT.value,
                follow_up_question=None,
            ),
            _context(),
        )

        assert "INVALID_FINAL_DRAFT" in result.codes

    def test_max_answer_chars_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _validator(max_answer_chars=0)


class TestEvidenceIds:
    def test_evidence_id_not_in_session_rejected(self) -> None:
        result = _validator().validate(
            _draft(evidence_ids=["ev-bogus"]),
            _context(),
        )

        assert "EVIDENCE_NOT_FOUND" in result.codes

    def test_only_invalid_evidence_ids_flagged(self) -> None:
        result = _validator().validate(
            _draft(evidence_ids=["ev-res", "ev-bogus"]),
            _context(),
        )

        assert "EVIDENCE_NOT_FOUND" in result.codes
        assert result.codes.count("EVIDENCE_NOT_FOUND") == 1


class TestReferencedMaterials:
    def test_referenced_material_not_in_evidence_rejected(self) -> None:
        result = _validator().validate(
            _draft(referenced_material_ids=["mp-999"]),
            _context(),
        )

        assert "MATERIAL_NOT_IN_EVIDENCE" in result.codes

    def test_material_ids_collected_from_nested_and_list_forms(self) -> None:
        payload = {
            "materials": [
                {"material_ids": ["mp-1", "mp-2"]},
                {"nested": {"material_id": "mp-3"}},
            ],
            "evidence_id": "ev-mats",
        }
        context = _context(
            evidence=(
                EvidenceRecord(
                    evidence_id="ev-mats",
                    tool_name="get_screening_result",
                    result_json=json.dumps(payload),
                ),
            ),
            active_workflow_thread_id=None,
            conversation_links=(),
        )
        draft = _draft(
            answer="推荐 mp-1、mp-2 与 mp-3。",
            referenced_material_ids=["mp-1", "mp-2", "mp-3"],
            evidence_ids=["ev-mats"],
            active_workflow_thread_id=None,
        )

        result = _validator().validate(draft, context)

        assert result.ok is True

    def test_invalid_evidence_json_is_ignored_for_material_facts(self) -> None:
        context = _context(
            evidence=(
                EvidenceRecord(
                    evidence_id="ev-bad",
                    tool_name="get_screening_result",
                    result_json="{not json",
                ),
            ),
            active_workflow_thread_id=None,
            conversation_links=(),
        )
        result = _validator().validate(
            _draft(
                answer="推荐 mp-1。",
                referenced_material_ids=["mp-1"],
                evidence_ids=["ev-bad"],
                active_workflow_thread_id=None,
            ),
            context,
        )

        assert "MATERIAL_NOT_IN_EVIDENCE" in result.codes


class TestAnswerMaterialIdConsistency:
    def test_answer_mentions_unreferenced_id_rejected(self) -> None:
        result = _validator().validate(
            _draft(answer="推荐 mp-1 与 mp-2。"),
            _context(),
        )

        assert "MATERIAL_ID_MISMATCH" in result.codes

    def test_answer_ids_all_referenced_ok(self) -> None:
        result = _validator().validate(
            _draft(
                answer="推荐 mp-1 与 mp-2。",
                referenced_material_ids=["mp-1", "mp-2"],
            ),
            _context(evidence=(_run_evidence(), _result_evidence(("mp-1", "mp-2")))),
        )

        assert result.ok is True


class TestEvidenceRequirement:
    def test_task_fact_marker_without_evidence_rejected(self) -> None:
        result = _validator().validate(
            _draft(
                answer="带隙 1.5 eV。",
                active_workflow_thread_id=None,
                referenced_material_ids=[],
                evidence_ids=[],
            ),
            _context(),
        )

        assert "TASK_FACT_WITHOUT_EVIDENCE" in result.codes

    def test_material_id_in_answer_without_evidence_rejected(self) -> None:
        result = _validator().validate(
            _draft(
                answer="mp-1 是候选材料。",
                active_workflow_thread_id=None,
                referenced_material_ids=[],
                evidence_ids=[],
            ),
            _context(),
        )

        assert "TASK_FACT_WITHOUT_EVIDENCE" in result.codes
        assert "MATERIAL_ID_MISMATCH" in result.codes

    def test_general_concept_without_evidence_ok(self) -> None:
        result = _validator().validate(
            _draft(
                answer="无机半导体是材料科学的重要领域。",
                active_workflow_thread_id=None,
                referenced_material_ids=[],
                evidence_ids=[],
            ),
            _context(),
        )

        assert result.ok is True

    def test_active_thread_alone_requires_evidence(self) -> None:
        result = _validator().validate(
            _draft(
                answer="筛选任务已完成。",
                referenced_material_ids=[],
                evidence_ids=[],
            ),
            _context(),
        )

        assert "TASK_FACT_WITHOUT_EVIDENCE" in result.codes


class TestThreadOwnership:
    def test_thread_not_owned_rejected(self) -> None:
        result = _validator().validate(
            _draft(active_workflow_thread_id="wf_other"),
            _context(),
        )

        assert "THREAD_OWNERSHIP_DENIED" in result.codes

    def test_thread_invalid_format_rejected(self) -> None:
        result = _validator().validate(
            _draft(active_workflow_thread_id="bad thread!"),
            _context(),
        )

        assert "THREAD_ID_INVALID" in result.codes

    def test_thread_owned_via_conversation_link_ok(self) -> None:
        context = _context(
            active_workflow_thread_id=None,
            conversation_links=(
                ConversationWorkflowLink(
                    conversation_id="conv_1",
                    thread_id="wf_old",
                    created_at="2026-08-01T00:00:00Z",
                ),
            ),
        )
        result = _validator().validate(
            _draft(active_workflow_thread_id="wf_old"),
            context,
        )

        assert result.ok is True

    def test_thread_none_ok(self) -> None:
        result = _validator().validate(
            _draft(active_workflow_thread_id=None),
            _context(),
        )

        assert result.ok is True


class TestWorkflowStatusMismatch:
    def _with_run_status(
        self,
        *,
        status: str,
        validation_passed: bool | None = True,
    ) -> FinalValidationContext:
        return _context(
            evidence=(
                _run_evidence(status=status, validation_passed=validation_passed),
                _result_evidence(),
            )
        )

    def test_failed_workflow_not_described_as_completed(self) -> None:
        result = _validator().validate(
            _draft(),
            self._with_run_status(status="failed"),
        )

        assert "WORKFLOW_STATUS_MISMATCH" in result.codes

    def test_no_results_workflow_can_be_reported_as_completed(self) -> None:
        result = _validator().validate(
            _draft(),
            self._with_run_status(status="no_results"),
        )

        assert "WORKFLOW_STATUS_MISMATCH" not in result.codes
        assert result.ok is True

    def test_no_results_honest_answer_ok(self) -> None:
        context = _context(
            evidence=(_run_evidence(status="no_results"),),
            active_workflow_thread_id="wf_t1",
        )
        draft = _draft(
            answer="没有找到符合条件的材料。",
            referenced_material_ids=[],
            evidence_ids=["ev-run"],
        )

        result = _validator().validate(draft, context)

        assert "WORKFLOW_STATUS_MISMATCH" not in result.codes
        assert result.ok is True

    def test_no_results_draft_citing_material_rejected(self) -> None:
        context = _context(
            evidence=(_run_evidence(status="no_results"),),
            active_workflow_thread_id="wf_t1",
        )
        draft = _draft(
            answer="推荐 mp-1，带隙 2.1 eV。",
            referenced_material_ids=["mp-1"],
            evidence_ids=["ev-run"],
        )

        result = _validator().validate(draft, context)

        assert "WORKFLOW_STATUS_MISMATCH" not in result.codes
        assert "MATERIAL_NOT_IN_EVIDENCE" in result.codes

    def test_clarification_not_described_as_completed(self) -> None:
        result = _validator().validate(
            _draft(),
            self._with_run_status(status="needs_clarification"),
        )

        assert "WORKFLOW_STATUS_MISMATCH" in result.codes

    def test_validation_failure_not_described_as_completed(self) -> None:
        result = _validator().validate(
            _draft(),
            self._with_run_status(status="completed", validation_passed=False),
        )

        assert "WORKFLOW_STATUS_MISMATCH" in result.codes

    def test_running_workflow_not_described_as_completed(self) -> None:
        result = _validator().validate(
            _draft(),
            self._with_run_status(status="initializing"),
        )

        assert "WORKFLOW_STATUS_MISMATCH" in result.codes

    def test_non_completed_draft_with_failed_workflow_ok(self) -> None:
        result = _validator().validate(
            _draft(status=AgentFinalStatus.ERROR.value),
            self._with_run_status(status="failed"),
        )

        assert "WORKFLOW_STATUS_MISMATCH" not in result.codes

    def test_completed_workflow_and_draft_ok(self) -> None:
        result = _validator().validate(_draft(), _context())

        assert result.ok is True

    def test_unparseable_workflow_evidence_is_skipped(self) -> None:
        context = _context(
            evidence=(
                EvidenceRecord(
                    evidence_id="ev-run-bad",
                    tool_name="run_screening_workflow",
                    result_json="{not json",
                ),
                _result_evidence(),
            )
        )

        result = _validator().validate(_draft(), context)

        assert "WORKFLOW_STATUS_MISMATCH" not in result.codes

    def test_status_check_error_does_not_block_honest_answer(self) -> None:
        status_payload = {
            "thread_id": "",
            "status": "error",
            "error_code": "NO_ACTIVE_WORKFLOW",
            "evidence_id": "ev-status",
        }
        context = _context(
            evidence=(
                EvidenceRecord(
                    evidence_id="ev-status",
                    tool_name="get_workflow_status",
                    result_json=json.dumps(status_payload),
                ),
            ),
            active_workflow_thread_id=None,
            conversation_links=(),
        )
        draft = _draft(
            answer="当前没有进行中的任务。",
            active_workflow_thread_id=None,
            referenced_material_ids=[],
            evidence_ids=[],
        )

        result = _validator().validate(draft, context)

        assert "WORKFLOW_STATUS_MISMATCH" not in result.codes
        assert result.ok is True


class TestSuspectedSecrets:
    def test_sk_key_in_answer_rejected(self) -> None:
        result = _validator().validate(
            _draft(answer="密钥 sk-abcdefghijklmnop 不应出现。"),
            _context(),
        )

        assert "SUSPECTED_SECRET" in result.codes

    def test_api_key_label_in_warning_rejected(self) -> None:
        result = _validator().validate(
            _draft(warnings=["api_key=sk-test-secret1234"]),
            _context(),
        )

        assert "SUSPECTED_SECRET" in result.codes

    def test_bearer_token_in_follow_up_rejected(self) -> None:
        result = _validator().validate(
            _draft(
                follow_up_question="Bearer AbCdEfGhIjKlMnOp123456 是否有效？",
            ),
            _context(),
        )

        assert "SUSPECTED_SECRET" in result.codes

    def test_authorization_header_rejected(self) -> None:
        result = _validator().validate(
            _draft(answer="Authorization: Bearer AbCdEfGhIjKlMnOp123456"),
            _context(),
        )

        assert "SUSPECTED_SECRET" in result.codes

    def test_normal_text_without_secret_ok(self) -> None:
        result = _validator().validate(_draft(), _context())

        assert "SUSPECTED_SECRET" not in result.codes


class TestAccumulationAndResult:
    def test_multiple_errors_accumulated(self) -> None:
        draft = _draft(
            answer="推荐 mp-2，带隙 1.0 eV。",
            referenced_material_ids=["mp-999"],
            evidence_ids=["ev-bogus"],
        )

        result = _validator().validate(draft, _context())

        assert "EVIDENCE_NOT_FOUND" in result.codes
        assert "MATERIAL_NOT_IN_EVIDENCE" in result.codes
        assert "MATERIAL_ID_MISMATCH" in result.codes
        assert "TASK_FACT_WITHOUT_EVIDENCE" in result.codes
        assert result.ok is False

    def test_result_is_frozen_and_codes_property(self) -> None:
        result = _validator().validate(
            _draft(evidence_ids=["ev-bogus"]),
            _context(),
        )

        assert isinstance(result, FinalValidationResult)
        assert result.model_config.get("frozen") is True
        assert result.codes == (
            "EVIDENCE_NOT_FOUND",
            "TASK_FACT_WITHOUT_EVIDENCE",
        )
