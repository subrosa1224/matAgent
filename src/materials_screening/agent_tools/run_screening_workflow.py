"""Agent tool: run the screening workflow (S3.5-M3)."""

import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.agent.context import AgentToolContext
from materials_screening.agent.errors import WorkflowResultReadError
from materials_screening.agent.tool_base import (
    ToolSideEffect,
)
from materials_screening.workflow.input_output import (
    WorkflowInput,
    WorkflowOutput,
)
from materials_screening.workflow.state import WorkflowStatus

_QUERY_MAX_LENGTH = 4000


class RunScreeningWorkflowInput(BaseModel):
    """Only a natural language query; no provider/path/thread control."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=_QUERY_MAX_LENGTH)


class CandidateSummary(BaseModel):
    """A safe, minimal candidate row the agent may cite in its answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    material_id: str
    formula_pretty: str
    band_gap_ev: float | None = None
    energy_above_hull_ev_atom: float | None = None
    density_g_cm3: float | None = None
    total_score: float = Field(ge=0, le=1)


class RunScreeningWorkflowOutput(BaseModel):
    """Safe run summary; never internal state, artifacts or tracebacks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    thread_id: str
    planner_status: str | None
    clarification_question: str | None
    retrieved_count: int
    filtered_count: int
    returned_count: int
    validation_passed: bool | None
    exports: list[str]
    warnings: list[str]
    evidence_id: str
    top_candidates: list[CandidateSummary] = Field(default_factory=list)


def _idempotency_key(user_turn_id: str, query: str) -> str:
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return f"{user_turn_id}:{query_hash}"


class RunScreeningWorkflowTool:
    """Run the deterministic screening workflow via WorkflowRunner only."""

    name = "run_screening_workflow"
    description = (
        "Run a materials screening workflow from a natural language query "
        "and return the validated result summary including the top candidate "
        "materials."
    )
    input_model = RunScreeningWorkflowInput
    output_model = RunScreeningWorkflowOutput
    side_effect = ToolSideEffect.CREATE_WORKFLOW_RUN

    def execute(
        self,
        arguments: RunScreeningWorkflowInput,
        context: AgentToolContext,
    ) -> RunScreeningWorkflowOutput:
        """Run once per (user_turn, query); reuse the first result otherwise."""
        idem_key = _idempotency_key(context.user_turn_id, arguments.query)
        cached = context.ledger.get_by_query_hash(idem_key)
        if cached is not None:
            return RunScreeningWorkflowOutput.model_validate_json(cached)

        result = context.workflow_runner.run(WorkflowInput(query=arguments.query))
        evidence_id = context.id_generator.new_id()
        safe = self._to_safe_output(
            result,
            evidence_id,
            context.workflow_result_reader,
        )
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=evidence_id,
            result_json=safe.model_dump_json(),
            side_effect=self.side_effect,
            query_hash=idem_key,
        )
        return safe

    @staticmethod
    def _to_safe_output(
        result: WorkflowOutput,
        evidence_id: str,
        reader: Any,
    ) -> RunScreeningWorkflowOutput:
        status = result.status.value
        warnings = list(result.warnings)
        if result.status is WorkflowStatus.FAILED and result.error is not None:
            warnings.append(
                f"{result.error.get('code', 'ERROR')}: "
                f"{result.error.get('message', 'workflow failed')}"
            )
        return RunScreeningWorkflowOutput(
            status=status,
            thread_id=result.thread_id,
            planner_status=result.planner_status,
            clarification_question=result.clarification_question,
            retrieved_count=result.retrieved_count,
            filtered_count=result.filtered_count,
            returned_count=result.returned_count,
            validation_passed=result.validation_passed,
            exports=list(result.exports),
            warnings=warnings,
            evidence_id=evidence_id,
            top_candidates=RunScreeningWorkflowTool._top_candidates(result, reader),
        )

    @staticmethod
    def _top_candidates(
        result: WorkflowOutput,
        reader: Any,
    ) -> list[CandidateSummary]:
        """Return up to 8 concrete candidates from the validated result."""
        if result.status is not WorkflowStatus.COMPLETED or result.returned_count <= 0:
            return []
        try:
            payload = reader.read(result.thread_id)
        except WorkflowResultReadError:
            return []
        ranked = payload.get("ranked_materials") if isinstance(payload, dict) else None
        if not isinstance(ranked, list):
            return []
        candidates: list[CandidateSummary] = []
        for item in ranked[:8]:
            if not isinstance(item, dict):
                continue
            record = item.get("record")
            if not isinstance(record, dict):
                continue
            candidates.append(
                CandidateSummary(
                    rank=int(item.get("rank", 0)),
                    material_id=str(record.get("material_id", "")),
                    formula_pretty=str(record.get("formula_pretty", "")),
                    band_gap_ev=record.get("band_gap_ev"),
                    energy_above_hull_ev_atom=record.get("energy_above_hull_ev_atom"),
                    density_g_cm3=record.get("density_g_cm3"),
                    total_score=float(item.get("total_score", 0.0) or 0.0),
                )
            )
        return candidates
