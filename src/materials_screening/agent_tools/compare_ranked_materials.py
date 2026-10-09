"""Agent tool: compare_ranked_materials (S3.5-M3)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.agent.context import AgentToolContext
from materials_screening.agent.errors import WorkflowResultReadError
from materials_screening.agent.tool_base import ToolSideEffect
from materials_screening.agent_tools.common import resolve_owned_thread

_COMPONENT_LABELS = (
    ("stability", "稳定性得分"),
    ("band_gap_match", "带隙匹配得分"),
    ("completeness", "完整度得分"),
    ("direct_gap", "直接带隙得分"),
)
_COMPONENT_KEYS = {
    "stability": "stability_score",
    "band_gap_match": "band_gap_match_score",
    "completeness": "completeness_score",
    "direct_gap": "direct_gap_score",
}
_EPSILON = 1e-9


class CompareRankedMaterialsInput(BaseModel):
    """Compare 2-5 material ids from the same validated result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str | None = None
    material_ids: list[str] = Field(min_length=2, max_length=5)


class MaterialComparisonRow(BaseModel):
    """One deterministic comparison row; never LLM-computed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    material_id: str
    rank: int
    total_score: float
    stability_score: float
    band_gap_match_score: float
    completeness_score: float
    direct_gap_score: float


class CompareRankedMaterialsOutput(BaseModel):
    """Safe comparison output with a deterministic summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str
    rows: list[MaterialComparisonRow]
    deterministic_summary: list[str]
    error_code: str | None = None
    evidence_id: str = Field(min_length=1)


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


class CompareRankedMaterialsTool:
    """Deterministically compare materials from one validated result."""

    name = "compare_ranked_materials"
    description = (
        "Compare 2-5 ranked materials from the same validated screening result."
    )
    input_model = CompareRankedMaterialsInput
    output_model = CompareRankedMaterialsOutput
    side_effect = ToolSideEffect.READ_ONLY

    def execute(
        self,
        arguments: CompareRankedMaterialsInput,
        context: AgentToolContext,
    ) -> CompareRankedMaterialsOutput:
        thread_id, error_code = resolve_owned_thread(
            requested_thread_id=arguments.thread_id,
            context=context,
        )
        evidence_id = context.id_generator.new_id()
        if error_code is not None or thread_id is None:
            output = CompareRankedMaterialsOutput(
                thread_id=thread_id or "",
                rows=[],
                deterministic_summary=[],
                error_code=error_code,
                evidence_id=evidence_id,
            )
        else:
            output = self._read_and_compare(
                thread_id=thread_id,
                material_ids=arguments.material_ids,
                context=context,
                evidence_id=evidence_id,
            )
        self._record(context, output)
        return output

    def _read_and_compare(
        self,
        *,
        thread_id: str,
        material_ids: list[str],
        context: AgentToolContext,
        evidence_id: str,
    ) -> CompareRankedMaterialsOutput:
        view = context.workflow_runner.get_state(thread_id)
        if view.status == "":
            return self._error(thread_id, "THREAD_NOT_FOUND", evidence_id)
        if view.status != "completed":
            return self._error(thread_id, "WORKFLOW_NOT_COMPLETED", evidence_id)
        if view.validation_passed is not True:
            return self._error(thread_id, "VALIDATION_NOT_PASSED", evidence_id)
        try:
            payload = context.workflow_result_reader.read(thread_id)
        except WorkflowResultReadError as exc:
            return self._error(thread_id, exc.code, evidence_id)

        by_id = self._index_result(payload.get("ranked_materials", []))
        if len(set(material_ids)) != len(material_ids):
            return self._error(thread_id, "DUPLICATE_MATERIAL_ID", evidence_id)
        missing = [
            material_id for material_id in material_ids if material_id not in by_id
        ]
        if missing:
            return self._error(
                thread_id,
                "MATERIAL_NOT_FOUND",
                evidence_id,
                message=f"materials not in result: {sorted(missing)}",
            )
        rows = [
            self._build_row(material_id, by_id[material_id])
            for material_id in material_ids
        ]
        rows.sort(key=lambda row: (row.rank, row.material_id))
        summary = self._deterministic_summary(rows)
        return CompareRankedMaterialsOutput(
            thread_id=thread_id,
            rows=rows,
            deterministic_summary=summary,
            error_code=None,
            evidence_id=evidence_id,
        )

    @staticmethod
    def _error(
        thread_id: str,
        error_code: str,
        evidence_id: str,
        message: str | None = None,
    ) -> CompareRankedMaterialsOutput:
        summary = [message] if message is not None else []
        return CompareRankedMaterialsOutput(
            thread_id=thread_id,
            rows=[],
            deterministic_summary=summary,
            error_code=error_code,
            evidence_id=evidence_id,
        )

    @staticmethod
    def _index_result(ranked: object) -> dict[str, dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        if not isinstance(ranked, list):
            return by_id
        for item in ranked:
            if not isinstance(item, dict):
                continue
            record = item.get("record")
            if not isinstance(record, dict):
                continue
            material_id = str(record.get("material_id", ""))
            if material_id:
                by_id[material_id] = item
        return by_id

    @staticmethod
    def _build_row(
        material_id: str,
        item: dict[str, Any],
    ) -> MaterialComparisonRow:
        breakdown = item.get("score_breakdown")
        if not isinstance(breakdown, dict):
            breakdown = {}

        def component(key: str) -> float:
            value = breakdown.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            return 0.0

        total = item.get("total_score")
        return MaterialComparisonRow(
            material_id=material_id,
            rank=int(item.get("rank", 0)),
            total_score=float(total) if isinstance(total, (int, float)) else 0.0,
            stability_score=component("stability"),
            band_gap_match_score=component("band_gap_match"),
            completeness_score=component("completeness"),
            direct_gap_score=component("direct_gap"),
        )

    @staticmethod
    def _deterministic_summary(
        rows: list[MaterialComparisonRow],
    ) -> list[str]:
        lines: list[str] = []
        for previous, current in zip(rows, rows[1:], strict=False):
            for component, label in _COMPONENT_LABELS:
                key = _COMPONENT_KEYS[component]
                difference = getattr(previous, key) - getattr(current, key)
                if abs(difference) < _EPSILON:
                    continue
                better = previous if difference > 0 else current
                worse = current if difference > 0 else previous
                lines.append(
                    f"{better.material_id} 的{label}比"
                    f"{worse.material_id} 高 {_fmt(abs(difference))}。"
                )
            total_difference = previous.total_score - current.total_score
            if abs(total_difference) >= _EPSILON:
                better = previous if total_difference > 0 else current
                lines.append(
                    f"{better.material_id} 总分比"
                    f"{current.material_id} 高 {_fmt(abs(total_difference))}，"
                    "因此排名更高。"
                )
        return lines

    def _record(
        self,
        context: AgentToolContext,
        output: CompareRankedMaterialsOutput,
    ) -> None:
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=output.evidence_id,
            result_json=output.model_dump_json(),
            side_effect=ToolSideEffect.READ_ONLY,
        )
