"""Unit tests for the run_screening_workflow tool (S3.5-M3)."""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from materials_screening.agent.context import AgentToolContext
from materials_screening.agent.errors import WorkflowResultReadError
from materials_screening.agent.ledger import ToolExecutionLedger
from materials_screening.agent.tool_registry import AgentToolRegistry
from materials_screening.agent_tools.run_screening_workflow import (
    RunScreeningWorkflowInput,
    RunScreeningWorkflowTool,
)
from materials_screening.workflow.input_output import WorkflowInput, WorkflowOutput
from materials_screening.workflow.state import WorkflowStatus


class _SeqIdGenerator:
    def __init__(self) -> None:
        self.count = 0

    def new_id(self) -> str:
        self.count += 1
        return f"evt_{self.count}"


def _fixed_clock() -> datetime:
    return datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


class _FakeRunner:
    def __init__(self, output: WorkflowOutput) -> None:
        self._output = output
        self.calls: list[WorkflowInput] = []

    def run(
        self,
        workflow_input: WorkflowInput,
        *args: Any,
        **kwargs: Any,
    ) -> WorkflowOutput:
        self.calls.append(workflow_input)
        return self._output


def _output(
    status: WorkflowStatus,
    *,
    thread_id: str = "thread-1",
    **overrides: Any,
) -> WorkflowOutput:
    values: dict[str, Any] = {
        "run_id": "run-1",
        "thread_id": thread_id,
        "status": status,
        "planner_status": None,
        "request": None,
        "retrieved_count": 0,
        "filtered_count": 0,
        "returned_count": 0,
        "validation_passed": None,
        "exports": (),
        "warnings": (),
        "error": None,
        "clarification_question": None,
    }
    values.update(overrides)
    return WorkflowOutput.model_validate(values)


def _context(
    *,
    runner: _FakeRunner,
    ledger: ToolExecutionLedger | None = None,
    call_id: str = "call_1",
    user_turn_id: str = "turn_1",
    conversation_id: str = "c1",
    reader_payload: dict[str, Any] | None = None,
    reader_error: Exception | None = None,
) -> AgentToolContext:
    class _Reader:
        def read(self, thread_id: str) -> dict[str, Any]:
            if reader_error is not None:
                raise reader_error
            return (
                reader_payload
                if reader_payload is not None
                else {"ranked_materials": []}
            )

    return AgentToolContext(
        workflow_runner=runner,
        workflow_result_reader=_Reader(),
        clock=_fixed_clock,
        id_generator=_SeqIdGenerator(),
        ledger=ledger if ledger is not None else ToolExecutionLedger(),
        call_id=call_id,
        user_turn_id=user_turn_id,
        conversation_id=conversation_id,
    )


class TestRunScreeningWorkflowTool:
    def test_success_maps_safe_summary(self) -> None:
        runner = _FakeRunner(
            _output(
                WorkflowStatus.COMPLETED,
                planner_status="ready",
                retrieved_count=10,
                filtered_count=5,
                returned_count=3,
                validation_passed=True,
                exports=("run-1/exports/request.json",),
                warnings=("note",),
            )
        )
        context = _context(
            runner=runner,
            reader_payload={
                "ranked_materials": [
                    {
                        "rank": 1,
                        "record": {
                            "material_id": "mp-2",
                            "formula_pretty": "Fe2O3",
                            "band_gap_ev": 2.0,
                            "energy_above_hull_ev_atom": 0.02,
                            "density_g_cm3": 5.2,
                        },
                        "total_score": 0.93,
                    },
                    {
                        "rank": 2,
                        "record": {
                            "material_id": "mp-6",
                            "formula_pretty": "Si",
                            "band_gap_ev": 1.1,
                            "energy_above_hull_ev_atom": 0.03,
                            "density_g_cm3": 2.3,
                        },
                        "total_score": 0.87,
                    },
                ]
            },
        )
        output = RunScreeningWorkflowTool().execute(
            RunScreeningWorkflowInput(query="find materials"),
            context,
        )
        assert output.status == "completed"
        assert output.thread_id == "thread-1"
        assert output.planner_status == "ready"
        assert output.retrieved_count == 10
        assert output.filtered_count == 5
        assert output.returned_count == 3
        assert output.validation_passed is True
        assert output.exports == ["run-1/exports/request.json"]
        assert output.warnings == ["note"]
        assert output.evidence_id == "evt_1"
        assert len(runner.calls) == 1
        assert runner.calls[0].query == "find materials"
        assert [c.material_id for c in output.top_candidates] == ["mp-2", "mp-6"]
        assert output.top_candidates[0].formula_pretty == "Fe2O3"
        assert output.top_candidates[0].total_score == 0.93

    def test_result_read_error_yields_empty_candidates(self) -> None:
        runner = _FakeRunner(
            _output(
                WorkflowStatus.COMPLETED,
                returned_count=3,
            )
        )
        output = RunScreeningWorkflowTool().execute(
            RunScreeningWorkflowInput(query="q"),
            _context(
                runner=runner,
                reader_error=WorkflowResultReadError("RESULT_NOT_FOUND", "missing"),
            ),
        )
        assert output.top_candidates == []

    def test_no_results_maps_status(self) -> None:
        runner = _FakeRunner(
            _output(
                WorkflowStatus.NO_RESULTS,
                retrieved_count=7,
                filtered_count=0,
                returned_count=0,
            )
        )
        output = RunScreeningWorkflowTool().execute(
            RunScreeningWorkflowInput(query="none"),
            _context(runner=runner),
        )
        assert output.status == "no_results"
        assert output.retrieved_count == 7
        assert output.filtered_count == 0

    def test_clarification_maps_safe_fields(self) -> None:
        runner = _FakeRunner(
            _output(
                WorkflowStatus.NEEDS_CLARIFICATION,
                planner_status="needs_clarification",
                clarification_question="please clarify",
            )
        )
        output = RunScreeningWorkflowTool().execute(
            RunScreeningWorkflowInput(query="unclear"),
            _context(runner=runner),
        )
        assert output.status == "needs_clarification"
        assert output.planner_status == "needs_clarification"
        assert output.clarification_question == "please clarify"
        assert output.thread_id == "thread-1"

    def test_invalid_maps_status(self) -> None:
        runner = _FakeRunner(
            _output(
                WorkflowStatus.INVALID_REQUEST,
                planner_status="invalid",
            )
        )
        output = RunScreeningWorkflowTool().execute(
            RunScreeningWorkflowInput(query="invalid"),
            _context(runner=runner),
        )
        assert output.status == "invalid_request"
        assert output.planner_status == "invalid"

    def test_unsupported_maps_status(self) -> None:
        runner = _FakeRunner(
            _output(
                WorkflowStatus.UNSUPPORTED_REQUEST,
                planner_status="unsupported",
            )
        )
        output = RunScreeningWorkflowTool().execute(
            RunScreeningWorkflowInput(query="unsupported"),
            _context(runner=runner),
        )
        assert output.status == "unsupported_request"
        assert output.planner_status == "unsupported"

    def test_failure_maps_safe_warning_without_traceback(self) -> None:
        runner = _FakeRunner(
            _output(
                WorkflowStatus.FAILED,
                error={
                    "code": "RETRIEVAL_FAILED",
                    "message": "safe summary",
                    "retryable": False,
                },
            )
        )
        output = RunScreeningWorkflowTool().execute(
            RunScreeningWorkflowInput(query="boom"),
            _context(runner=runner),
        )
        assert output.status == "failed"
        assert output.warnings == ["RETRIEVAL_FAILED: safe summary"]
        assert "traceback" not in output.model_dump_json()

    def test_same_turn_same_query_is_idempotent(self) -> None:
        runner = _FakeRunner(_output(WorkflowStatus.COMPLETED))
        context = _context(runner=runner)
        tool = RunScreeningWorkflowTool()
        first = tool.execute(
            RunScreeningWorkflowInput(query="same query"),
            context,
        )
        second = tool.execute(
            RunScreeningWorkflowInput(query="same query"),
            context,
        )
        assert first.evidence_id == second.evidence_id
        assert first == second
        assert len(runner.calls) == 1

    def test_different_query_runs_again(self) -> None:
        runner = _FakeRunner(_output(WorkflowStatus.COMPLETED))
        ledger = ToolExecutionLedger()
        first = _context(runner=runner, ledger=ledger, call_id="call_1")
        second = _context(runner=runner, ledger=ledger, call_id="call_2")
        tool = RunScreeningWorkflowTool()
        tool.execute(RunScreeningWorkflowInput(query="query a"), first)
        tool.execute(RunScreeningWorkflowInput(query="query b"), second)
        assert len(runner.calls) == 2
        assert ledger.tool_call_count() == 2

    def test_same_query_different_turn_runs_again(self) -> None:
        runner = _FakeRunner(_output(WorkflowStatus.COMPLETED))
        first_turn = _context(runner=runner, user_turn_id="turn_1")
        second_turn = _context(runner=runner, user_turn_id="turn_2")
        tool = RunScreeningWorkflowTool()
        tool.execute(RunScreeningWorkflowInput(query="same"), first_turn)
        tool.execute(RunScreeningWorkflowInput(query="same"), second_turn)
        assert len(runner.calls) == 2

    def test_ledger_records_evidence_and_query_hash(self) -> None:
        ledger = ToolExecutionLedger()
        runner = _FakeRunner(_output(WorkflowStatus.COMPLETED))
        context = _context(runner=runner, ledger=ledger, call_id="call_1")
        RunScreeningWorkflowTool().execute(
            RunScreeningWorkflowInput(query="hash me"),
            context,
        )
        assert ledger.evidence_ids() == ("evt_1",)
        assert ledger.executed_call_ids() == ("call_1",)
        assert ledger.tool_call_count() == 1
        assert ledger.side_effect_executed() is True
        assert ledger.workflow_run_count() == 1

    def test_query_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RunScreeningWorkflowInput(query="x" * 4001)

    @pytest.mark.parametrize(
        "extra",
        [
            {"thread_id": "t"},
            {"provider": "intern"},
            {"repository": "materials-project"},
            {"output_path": "/tmp"},
            {"checkpoint": "x"},
            {"replay": True},
            {"retry": 2},
            {"api_key": "sk-secret"},
        ],
    )
    def test_model_controlled_fields_rejected(self, extra: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            RunScreeningWorkflowInput.model_validate({"query": "q", **extra})

    def test_input_frozen(self) -> None:
        assert RunScreeningWorkflowInput.model_config.get("frozen") is True

    def test_unknown_bug_propagates(self) -> None:
        class _BoomRunner:
            def run(
                self, workflow_input: WorkflowInput, *args: Any, **kwargs: Any
            ) -> WorkflowOutput:
                raise RuntimeError("unexpected bug")

        context = _context(runner=_BoomRunner())
        with pytest.raises(RuntimeError, match="unexpected bug"):
            RunScreeningWorkflowTool().execute(
                RunScreeningWorkflowInput(query="boom"),
                context,
            )

    def test_output_exposes_only_safe_fields(self) -> None:
        runner = _FakeRunner(_output(WorkflowStatus.COMPLETED))
        output = RunScreeningWorkflowTool().execute(
            RunScreeningWorkflowInput(query="q"),
            _context(runner=runner),
        )
        allowed = {
            "status",
            "thread_id",
            "planner_status",
            "clarification_question",
            "retrieved_count",
            "filtered_count",
            "returned_count",
            "validation_passed",
            "exports",
            "warnings",
            "evidence_id",
            "top_candidates",
        }
        assert set(output.model_dump()) == allowed

    def test_tool_registers_in_whitelist(self) -> None:
        registry = AgentToolRegistry([RunScreeningWorkflowTool()])
        assert registry.get("run_screening_workflow").name == ("run_screening_workflow")
