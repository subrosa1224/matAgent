"""Unit tests for the tool execution ledger (S3.5-M2)."""

import pytest

from materials_screening.agent.errors import AgentInvariantError
from materials_screening.agent.ledger import ToolExecutionLedger
from materials_screening.agent.tool_base import ToolSideEffect


class TestToolExecutionLedger:
    def test_record_and_get_result(self) -> None:
        ledger = ToolExecutionLedger()
        ledger.record(
            call_id="call_1",
            tool_name="get_workflow_status",
            evidence_id="evt_1",
            result_json='{"status":"ok"}',
            side_effect=ToolSideEffect.READ_ONLY,
        )
        assert ledger.has("call_1") is True
        assert ledger.get_result("call_1") == '{"status":"ok"}'
        assert ledger.has("call_2") is False
        assert ledger.get_result("call_2") is None

    def test_duplicate_call_id_rejected(self) -> None:
        ledger = ToolExecutionLedger()
        ledger.record(
            call_id="call_1",
            tool_name="get_workflow_status",
            evidence_id="evt_1",
            result_json="{}",
            side_effect=ToolSideEffect.READ_ONLY,
        )
        with pytest.raises(AgentInvariantError, match="duplicate call_id"):
            ledger.record(
                call_id="call_1",
                tool_name="get_workflow_status",
                evidence_id="evt_2",
                result_json="{}",
                side_effect=ToolSideEffect.READ_ONLY,
            )

    def test_query_hash_idempotency(self) -> None:
        ledger = ToolExecutionLedger()
        ledger.record(
            call_id="call_1",
            tool_name="run_screening_workflow",
            evidence_id="evt_1",
            result_json='{"status":"completed"}',
            side_effect=ToolSideEffect.CREATE_WORKFLOW_RUN,
            query_hash="hash-abc",
        )
        assert ledger.has_query_hash("hash-abc") is True
        assert ledger.get_by_query_hash("hash-abc") == '{"status":"completed"}'
        assert ledger.get_by_query_hash("hash-other") is None

    def test_duplicate_query_hash_rejected(self) -> None:
        ledger = ToolExecutionLedger()
        ledger.record(
            call_id="call_1",
            tool_name="run_screening_workflow",
            evidence_id="evt_1",
            result_json="{}",
            side_effect=ToolSideEffect.CREATE_WORKFLOW_RUN,
            query_hash="hash-abc",
        )
        with pytest.raises(AgentInvariantError, match="duplicate query hash"):
            ledger.record(
                call_id="call_2",
                tool_name="run_screening_workflow",
                evidence_id="evt_2",
                result_json="{}",
                side_effect=ToolSideEffect.CREATE_WORKFLOW_RUN,
                query_hash="hash-abc",
            )

    def test_evidence_and_call_ids_in_order(self) -> None:
        ledger = ToolExecutionLedger()
        ledger.record(
            call_id="c1",
            tool_name="get_workflow_status",
            evidence_id="e1",
            result_json="{}",
            side_effect=ToolSideEffect.READ_ONLY,
        )
        ledger.record(
            call_id="c2",
            tool_name="get_workflow_history",
            evidence_id="e2",
            result_json="{}",
            side_effect=ToolSideEffect.READ_ONLY,
        )
        assert ledger.evidence_ids() == ("e1", "e2")
        assert ledger.executed_call_ids() == ("c1", "c2")
        assert ledger.executed_tool_names() == (
            "get_workflow_status",
            "get_workflow_history",
        )

    def test_counts(self) -> None:
        ledger = ToolExecutionLedger()
        assert ledger.tool_call_count() == 0
        assert ledger.workflow_run_count() == 0
        ledger.record(
            call_id="c1",
            tool_name="get_workflow_status",
            evidence_id="e1",
            result_json="{}",
            side_effect=ToolSideEffect.READ_ONLY,
        )
        ledger.record(
            call_id="c2",
            tool_name="run_screening_workflow",
            evidence_id="e2",
            result_json="{}",
            side_effect=ToolSideEffect.CREATE_WORKFLOW_RUN,
        )
        ledger.record(
            call_id="c3",
            tool_name="run_screening_workflow",
            evidence_id="e3",
            result_json="{}",
            side_effect=ToolSideEffect.CREATE_WORKFLOW_RUN,
        )
        assert ledger.tool_call_count() == 3
        assert ledger.workflow_run_count() == 2

    def test_side_effect_executed(self) -> None:
        read_only = ToolExecutionLedger()
        read_only.record(
            call_id="c1",
            tool_name="get_workflow_status",
            evidence_id="e1",
            result_json="{}",
            side_effect=ToolSideEffect.READ_ONLY,
        )
        assert read_only.side_effect_executed() is False

        with_run = ToolExecutionLedger()
        with_run.record(
            call_id="c1",
            tool_name="run_screening_workflow",
            evidence_id="e1",
            result_json="{}",
            side_effect=ToolSideEffect.CREATE_WORKFLOW_RUN,
        )
        assert with_run.side_effect_executed() is True

    def test_empty_ledger(self) -> None:
        ledger = ToolExecutionLedger()
        assert ledger.evidence_ids() == ()
        assert ledger.executed_call_ids() == ()
        assert ledger.tool_call_count() == 0
        assert ledger.workflow_run_count() == 0
