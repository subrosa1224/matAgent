"""Agent tool: get_screening_result (S3.5-M3)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.agent.context import AgentToolContext
from materials_screening.agent.errors import WorkflowResultReadError
from materials_screening.agent.tool_base import ToolSideEffect
from materials_screening.agent_tools.common import resolve_owned_thread

_SCIENTIFIC_NOTICE = (
    "数据库属性主要为计算值，不等同于实验值；"
    "排名不代表材料必然可合成、无毒或适用于实际器件。"
)
_RUNNING_STATUSES = frozenset(
    {
        "initializing",
        "resolving_request",
        "ready_for_retrieval",
        "retrieving",
        "filtering",
        "ranking",
        "validating",
        "exporting",
    }
)
_NON_RESULT_STATUSES = frozenset(
    {
        "failed",
        "no_results",
        "needs_clarification",
        "invalid_request",
        "unsupported_request",
    }
)


class GetScreeningResultInput(BaseModel):
    """Read a validated screening result; None uses the active thread."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str | None = None
    top_n: int = Field(default=5, ge=1, le=20)


class RankedMaterialSummary(BaseModel):
    """Safe per-material summary; no structure or full provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int
    material_id: str
    formula_pretty: str
    band_gap_ev: float | None = None
    energy_above_hull_ev_atom: float | None = None
    formation_energy_ev_atom: float | None = None
    is_gap_direct: bool | None = None
    crystal_system: str | None = None
    total_score: float
    score_breakdown: dict[str, float]
    source: str
    value_type: str
    warnings: list[str] = Field(default_factory=list)


class GetScreeningResultOutput(BaseModel):
    """Safe validated result summary with a fixed scientific notice."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str
    workflow_status: str
    request_summary: dict[str, Any] = Field(default_factory=dict)
    materials: list[RankedMaterialSummary] = Field(default_factory=list)
    scientific_notice: str = _SCIENTIFIC_NOTICE
    error_code: str | None = None
    evidence_id: str = Field(min_length=1)


def _request_summary(request: object) -> dict[str, Any]:
    if not isinstance(request, dict):
        return {}
    keys = (
        "limit",
        "band_gap_ev",
        "energy_above_hull_ev_atom",
        "density_g_cm3",
        "required_elements",
        "excluded_elements",
        "is_metal",
        "target_band_gap_ev",
        "crystal_system",
    )
    return {key: request[key] for key in keys if key in request}


class GetScreeningResultTool:
    """Read the validated screening result; never re-queries or re-ranks."""

    name = "get_screening_result"
    description = "Read the validated screening result summary for an owned thread."
    input_model = GetScreeningResultInput
    output_model = GetScreeningResultOutput
    side_effect = ToolSideEffect.READ_ONLY

    def execute(
        self,
        arguments: GetScreeningResultInput,
        context: AgentToolContext,
    ) -> GetScreeningResultOutput:
        thread_id, error_code = resolve_owned_thread(
            requested_thread_id=arguments.thread_id,
            context=context,
        )
        evidence_id = context.id_generator.new_id()
        if error_code is not None or thread_id is None:
            output = GetScreeningResultOutput(
                thread_id=thread_id or "",
                workflow_status="error",
                error_code=error_code,
                evidence_id=evidence_id,
            )
        else:
            view = context.workflow_runner.get_state(thread_id)
            if view.status == "":
                output = GetScreeningResultOutput(
                    thread_id=thread_id,
                    workflow_status="error",
                    error_code="THREAD_NOT_FOUND",
                    evidence_id=evidence_id,
                )
            elif (
                view.status in _RUNNING_STATUSES or view.status in _NON_RESULT_STATUSES
            ):
                output = GetScreeningResultOutput(
                    thread_id=thread_id,
                    workflow_status=view.status,
                    error_code=None,
                    evidence_id=evidence_id,
                )
            elif view.validation_passed is not True:
                output = GetScreeningResultOutput(
                    thread_id=thread_id,
                    workflow_status=view.status,
                    error_code="VALIDATION_NOT_PASSED",
                    evidence_id=evidence_id,
                )
            else:
                output = self._read_validated(
                    thread_id=thread_id,
                    workflow_status=view.status,
                    top_n=arguments.top_n,
                    context=context,
                    evidence_id=evidence_id,
                )
        self._record(context, output)
        return output

    def _read_validated(
        self,
        *,
        thread_id: str,
        workflow_status: str,
        top_n: int,
        context: AgentToolContext,
        evidence_id: str,
    ) -> GetScreeningResultOutput:
        try:
            payload = context.workflow_result_reader.read(thread_id)
        except WorkflowResultReadError as exc:
            return GetScreeningResultOutput(
                thread_id=thread_id,
                workflow_status=workflow_status,
                error_code=exc.code,
                evidence_id=evidence_id,
            )
        ranked = payload.get("ranked_materials", [])
        try:
            materials = [
                self._summarize(item)
                for item in ranked[:top_n]
                if isinstance(item, dict)
            ]
        except WorkflowResultReadError as exc:
            return GetScreeningResultOutput(
                thread_id=thread_id,
                workflow_status=workflow_status,
                error_code=exc.code,
                evidence_id=evidence_id,
            )
        return GetScreeningResultOutput(
            thread_id=thread_id,
            workflow_status=workflow_status,
            request_summary=_request_summary(payload.get("request")),
            materials=materials,
            error_code=None,
            evidence_id=evidence_id,
        )

    @staticmethod
    def _summarize(item: dict[str, Any]) -> RankedMaterialSummary:
        record = item.get("record")
        if not isinstance(record, dict):
            raise WorkflowResultReadError(
                "RESULT_INVALID",
                "ranked material record is missing",
            )
        symmetry = record.get("symmetry")
        crystal_system = None
        if isinstance(symmetry, dict):
            crystal_system = symmetry.get("crystal_system")
        provenance = record.get("provenance")
        value_type = "unknown"
        if isinstance(provenance, list) and provenance:
            first = provenance[0]
            if isinstance(first, dict):
                value_type = str(first.get("value_type", "unknown"))
        breakdown = item.get("score_breakdown")
        score_breakdown: dict[str, float] = {}
        if isinstance(breakdown, dict):
            score_breakdown = {
                str(key): float(value)
                for key, value in breakdown.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        return RankedMaterialSummary(
            rank=int(item.get("rank", 0)),
            material_id=str(record.get("material_id", "")),
            formula_pretty=str(record.get("formula_pretty", "")),
            band_gap_ev=record.get("band_gap_ev"),
            energy_above_hull_ev_atom=record.get("energy_above_hull_ev_atom"),
            formation_energy_ev_atom=record.get("formation_energy_ev_atom"),
            is_gap_direct=record.get("is_gap_direct"),
            crystal_system=crystal_system,
            total_score=float(item.get("total_score", 0.0)),
            score_breakdown=score_breakdown,
            source=str(record.get("source", "")),
            value_type=value_type,
            warnings=[],
        )

    def _record(
        self,
        context: AgentToolContext,
        output: GetScreeningResultOutput,
    ) -> None:
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=output.evidence_id,
            result_json=output.model_dump_json(),
            side_effect=ToolSideEffect.READ_ONLY,
        )
