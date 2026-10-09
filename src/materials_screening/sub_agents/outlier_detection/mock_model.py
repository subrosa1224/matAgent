"""Offline mock model for the Outlier Detection sub-agent.

Mirrors the pattern of ``WorkflowDrivenMockAgentModel`` but drives the two
outlier-detection tools (``detect_property_outliers`` /
``detect_multivariate_outliers``) against the built-in mock material set so
``master ask --llm-provider mock`` returns genuine outlier reports instead of
the generic concept-answer string.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentRequest,
    MaterialAgentResponse,
)
from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
)

# Formulas from tests/fixtures/mp_documents.json — the bundled offline material set.
_OUTLIER_INTENT_MARKERS: tuple[str, ...] = (
    "离群", "异常值", "异常", "outlier", "anomaly",
    "异常检测", "离群检测", "离群值", "abnormal",
)

# Mapping from query tokens to MaterialRecord property fields.
_PROPERTY_HINTS: dict[str, str] = {
    "band_gap": "band_gap_ev",
    "bandgap": "band_gap_ev",
    "带隙": "band_gap_ev",
    "禁带": "band_gap_ev",
    "density": "density_g_cm3",
    "密度": "density_g_cm3",
    "formation_energy": "formation_energy_ev_atom",
    "形成能": "formation_energy_ev_atom",
    "hull": "energy_above_hull_ev_atom",
    "凸包": "energy_above_hull_ev_atom",
    "energy_above_hull": "energy_above_hull_ev_atom",
}


def _extract_properties(query: str) -> list[str]:
    """Return unique property fields mentioned in *query*, default band_gap_ev."""
    lowered = query.lower()
    found: list[str] = []
    seen: set[str] = set()
    for token, field in _PROPERTY_HINTS.items():
        if token in lowered and field not in seen:
            found.append(field)
            seen.add(field)
    return found if found else ["band_gap_ev"]


class OutlierDetectionMockAgentModel:
    """Offline mock that drives the outlier-detection tools against built-in data.

    First call: emit ``detect_property_outliers`` (single property) or
    ``detect_multivariate_outliers`` (multiple properties) with the bundled
    material formulas when the query sounds like an outlier-detection request;
    otherwise answer conversationally.

    Second call: parse the tool's ``ToolResultEnvelope`` and compose a
    deterministic final draft from the actual service output.
    """

    _CONCEPT_ANSWER = (
        "（离线演示）当前为离线演示模式。离群检测查询会基于内置的 7 种示例材料"
        "（LiFeO2、Fe2O3、TiO2、Cu、PbS、Si、GaAs）执行 Z-score / IQR / "
        "Mahalanobis / Isolation Forest 分析并给出真实离群报告；"
        "切换到 intern 提供商后可获得完整对话能力。"
    )

    def __init__(
        self,
        *,
        provider: str = "mock",
        model: str = "mock-outlier-detection-agent",
    ) -> None:
        self._provider = provider
        self._model = model
        self._call_count = 0

    # -- generate ---------------------------------------------------------------

    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        self._call_count += 1
        if self._turn_has_tool_result(request):
            return self._draft_turn(request)
        query = self._last_user_message(request)
        if not query or not self._is_outlier_query(query):
            return self._draft(status="completed", answer=self._CONCEPT_ANSWER)
        return self._tool_call_turn(query)

    # -- intent detection -------------------------------------------------------

    @staticmethod
    def _is_outlier_query(query: str) -> bool:
        lowered = query.lower()
        return any(marker in lowered for marker in _OUTLIER_INTENT_MARKERS)

    # -- tool call emission -----------------------------------------------------

    def _tool_call_turn(self, query: str) -> MaterialAgentResponse:
        item = AgentFunctionCallItem(
            call_id="mock_od_call_1",
            name="run_outlier_detection",
            arguments=json.dumps(
                {"query": query}, ensure_ascii=False, separators=(",", ":")
            ),
        )
        return self._response((item,))

    # -- draft (after tool execution) -------------------------------------------

    def _draft_turn(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        envelope = self._last_tool_envelope(request)
        if envelope is None or envelope.get("status") != "ok":
            detail = self._error_detail(envelope)
            return self._draft(
                status="error",
                answer=f"（离线演示）离群检测失败：{detail}",
            )
        output = envelope.get("output")
        if not isinstance(output, dict):
            return self._draft(
                status="error",
                answer="（离线演示）离群检测工具返回了无法解析的结果。",
            )
        evidence_id = str(envelope.get("evidence_id", ""))
        evidence = (evidence_id,) if evidence_id else ()
        warnings = tuple(str(value) for value in output.get("warnings", []))
        referenced_material_ids = tuple(
            str(record["material_id"])
            for record in output.get("records", [])
            if isinstance(record, dict) and record.get("material_id")
        )

        if "property_name" in output:
            answer = self._property_outlier_answer(output)
        else:
            answer = self._multivariate_outlier_answer(output)

        return self._draft(
            status="completed",
            answer=answer,
            evidence=evidence,
            referenced_material_ids=referenced_material_ids,
            warnings=warnings,
        )

    @staticmethod
    def _property_outlier_answer(output: dict[str, Any]) -> str:
        prop = output.get("property_name", "?")
        method = output.get("method", "?")
        threshold = output.get("threshold", 2.0)
        distribution = output.get("distribution", {})
        records: list[dict[str, Any]] = output.get("records", [])

        lines = [
            f"（离线演示）单属性离群检测完成 — "
            f"属性: {prop}  |  方法: {method}  |  阈值: {threshold}",
            f"样本数: {distribution.get('count', '?')}  |  "
            f"均值: {distribution.get('mean', '?'):.3f}  |  "
            f"标准差: {distribution.get('std', '?'):.3f}",
            "",
        ]

        outliers = [r for r in records if r.get("is_outlier")]
        normal = [r for r in records if not r.get("is_outlier")]

        if outliers:
            lines.append(f"⚠ 发现 {len(outliers)} 个离群值：")
            for r in outliers:
                label = r.get("material_label", "?")
                z_val = r.get("z_score")
                direction = r.get("direction", "")
                z_str = f"z={z_val:.2f}" if isinstance(z_val, (int, float)) else ""
                dir_str = {"high": "↑偏高", "low": "↓偏低"}.get(str(direction), "")
                lines.append(f"  • {label}  {z_str}  {dir_str}")
        else:
            lines.append("✓ 未发现离群值（全部在正常范围内）")

        if normal:
            lines.append(f"\n正常样本 ({len(normal)} 个):")
            labels = [r.get("material_label", "?") for r in normal]
            lines.append("  " + ", ".join(labels))

        return "\n".join(lines)

    @staticmethod
    def _multivariate_outlier_answer(output: dict[str, Any]) -> str:
        props = output.get("properties", [])
        method = output.get("method", "?")
        threshold = output.get("threshold")
        records: list[dict[str, Any]] = output.get("records", [])

        prop_str = ", ".join(props) if props else "?"
        lines = [
            f"（离线演示）多变量离群检测完成 — "
            f"属性: {prop_str}  |  方法: {method}"
            + (
                f"  |  离群阈值: {float(threshold):.3f}"
                if isinstance(threshold, (int, float))
                else ""
            ),
            f"样本数: {len(records)}",
            "",
        ]

        # Sort by anomaly score descending
        sorted_records = sorted(
            records,
            key=lambda r: float(r.get("anomaly_score", 0) or 0),
            reverse=True,
        )
        outliers = [record for record in sorted_records if record.get("is_outlier")]
        if outliers:
            lines.append(f"结论：发现 {len(outliers)} 个多变量离群材料。")
        else:
            lines.append("结论：未发现多变量离群材料。")
        lines.append("")
        for r in sorted_records:
            label = r.get("material_label", "?")
            score = r.get("anomaly_score", 0)
            score = float(score) if isinstance(score, (int, float)) else 0.0
            is_outlier = bool(r.get("is_outlier", False))
            abnormal = r.get("abnormal_properties", [])
            reason = r.get("outlier_reason", "")
            flag = " ⚠" if is_outlier else ""
            lines.append(f"  {label}:  score={score:.3f}{flag}")
            if abnormal and is_outlier:
                ab_str = ", ".join(abnormal)
                lines.append(f"    异常属性: {ab_str}  ({reason})")

        return "\n".join(lines)

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _error_detail(envelope: dict[str, Any] | None) -> str:
        if envelope is None:
            return "未获得工具结果"
        err = envelope.get("error")
        if isinstance(err, dict):
            return f"{err.get('code', 'ERROR')}: {err.get('message', '工具执行失败')}"
        return "工具执行失败"

    def _draft(
        self,
        *,
        status: str,
        answer: str,
        evidence: tuple[str, ...] = (),
        referenced_material_ids: tuple[str, ...] = (),
        warnings: tuple[str, ...] = (),
    ) -> MaterialAgentResponse:
        draft = {
            "status": status,
            "answer": answer,
            "active_workflow_thread_id": None,
            "referenced_material_ids": list(referenced_material_ids),
            "evidence_ids": list(evidence),
            "warnings": list(warnings),
            "follow_up_question": None,
        }
        item = AgentMessageItem(
            role="assistant",
            content=json.dumps(draft, ensure_ascii=False, separators=(",", ":")),
        )
        return self._response((item,))

    def _response(
        self,
        output_items: tuple[AgentMessageItem | AgentFunctionCallItem, ...],
    ) -> MaterialAgentResponse:
        payload = json.dumps(
            {
                "call": self._call_count,
                "provider": self._provider,
                "model": self._model,
                "output": [item.model_dump(mode="json") for item in output_items],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return MaterialAgentResponse(
            status=AgentModelStatus.COMPLETED,
            output_items=output_items,
            request_id=f"mock-{digest}",
            provider=self._provider,
            model=self._model,
        )

    # -- transcript helpers (mirror WorkflowDrivenMockAgentModel) ----------------

    @staticmethod
    def _turn_has_tool_result(request: MaterialAgentRequest) -> bool:
        last_user_index = -1
        for index, item in enumerate(request.input_items):
            if isinstance(item, AgentMessageItem) and item.role == "user":
                last_user_index = index
        if last_user_index < 0:
            return False
        for item in request.input_items[last_user_index + 1:]:
            if isinstance(item, AgentFunctionOutputItem):
                return True
        return False

    @staticmethod
    def _last_user_message(request: MaterialAgentRequest) -> str | None:
        for item in reversed(request.input_items):
            if isinstance(item, AgentMessageItem) and item.role == "user":
                return item.content
        return None

    @staticmethod
    def _last_tool_envelope(
        request: MaterialAgentRequest,
    ) -> dict[str, Any] | None:
        last_user_index = -1
        for index, item in enumerate(request.input_items):
            if isinstance(item, AgentMessageItem) and item.role == "user":
                last_user_index = index
        for item in reversed(request.input_items[last_user_index + 1:]):
            if not isinstance(item, AgentFunctionOutputItem):
                continue
            try:
                payload = json.loads(item.output)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict):
                return payload
        return None
