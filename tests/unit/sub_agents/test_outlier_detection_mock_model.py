"""Unit tests for OutlierDetectionMockAgentModel."""

from __future__ import annotations

import json

from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentRequest,
)
from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
)
from materials_screening.agent.tool_base import AgentToolDefinition, ToolSideEffect
from materials_screening.sub_agents.outlier_detection.mock_model import (
    OutlierDetectionMockAgentModel,
    _extract_properties,
)

# ── Helpers ─────────────────────────────────────────────────────────────────

_MOCK_TOOLS = (
    AgentToolDefinition(
        name="run_outlier_detection",
        description="Detect outliers named in a natural-language query.",
        parameters={"type": "object", "additionalProperties": False},
        side_effect=ToolSideEffect.READ_ONLY,
        version="1.0.0",
    ),
)

_FINAL_DRAFT_SCHEMA: dict = {"type": "object"}


def _user_request(query: str) -> MaterialAgentRequest:
    """Request with a single user message, no prior tool results."""
    return MaterialAgentRequest(
        input_items=(AgentMessageItem(role="user", content=query),),
        instructions="You are an outlier detection agent.",
        tool_definitions=_MOCK_TOOLS,
        final_draft_schema=_FINAL_DRAFT_SCHEMA,
    )


def _request_after_tool(query: str, envelope: dict) -> MaterialAgentRequest:
    """Request where the model already emitted a tool call and got a result."""
    tool_output = json.dumps(envelope, ensure_ascii=False)
    return MaterialAgentRequest(
        input_items=(
            AgentMessageItem(role="user", content=query),
            AgentFunctionCallItem(
                call_id="mock_od_call_1",
                name="detect_property_outliers",
                arguments='{"source":{"material_formulas":["TiO2"]},"property":"band_gap_ev"}',
            ),
            AgentFunctionOutputItem(call_id="mock_od_call_1", output=tool_output),
        ),
        instructions="You are an outlier detection agent.",
        tool_definitions=_MOCK_TOOLS,
        final_draft_schema=_FINAL_DRAFT_SCHEMA,
    )


# ── Intent Detection ────────────────────────────────────────────────────────


class TestIntentDetection:
    def test_outlier_query_detected(self) -> None:
        model = OutlierDetectionMockAgentModel()
        for q in ("检测离群值", "查找异常材料", "find outliers", "anomaly detection"):
            assert model._is_outlier_query(q), f"should match: {q!r}"

    def test_non_outlier_query_not_matched(self) -> None:
        model = OutlierDetectionMockAgentModel()
        for q in ("帮我写代码", "hello world", "今天天气怎么样"):
            assert not model._is_outlier_query(q), f"should NOT match: {q!r}"


# ── Property extraction ─────────────────────────────────────────────────────


class TestPropertyExtraction:
    def test_default_band_gap(self) -> None:
        assert _extract_properties("检测材料的离群值") == ["band_gap_ev"]

    def test_single_hint(self) -> None:
        assert _extract_properties("检测密度的异常值") == ["density_g_cm3"]

    def test_multiple_hints(self) -> None:
        props = _extract_properties("检测带隙和hull的异常值")
        assert "band_gap_ev" in props
        assert "energy_above_hull_ev_atom" in props

    def test_formation_energy_hint(self) -> None:
        assert _extract_properties("形成能的离群检测") == ["formation_energy_ev_atom"]


# ── Generate: concept question ──────────────────────────────────────────────


class TestConceptQuestion:
    def test_returns_concept_answer(self) -> None:
        model = OutlierDetectionMockAgentModel()
        resp = model.generate(_user_request("你好，介绍一下自己"))
        assert resp.status == AgentModelStatus.COMPLETED
        content = resp.output_items[0].content  # type: ignore[union-attr]
        assert "离线演示" in str(content)


# ── Generate: tool call emission ────────────────────────────────────────────


class TestToolCallEmission:
    def test_single_property_emits_unified_tool(self) -> None:
        model = OutlierDetectionMockAgentModel()
        resp = model.generate(_user_request("检测带隙的离群值"))
        items = resp.output_items
        assert len(items) == 1
        call = items[0]
        assert isinstance(call, AgentFunctionCallItem)
        assert call.name == "run_outlier_detection"
        assert "query" in json.loads(call.arguments)

    def test_multi_property_emits_unified_tool(self) -> None:
        model = OutlierDetectionMockAgentModel()
        resp = model.generate(_user_request("检测带隙和密度的多变量异常"))
        items = resp.output_items
        call = items[0]
        assert isinstance(call, AgentFunctionCallItem)
        assert call.name == "run_outlier_detection"

    def test_default_single_property(self) -> None:
        model = OutlierDetectionMockAgentModel()
        resp = model.generate(_user_request("找一下离群点"))
        call = resp.output_items[0]
        assert isinstance(call, AgentFunctionCallItem)
        assert call.name == "run_outlier_detection"
        args = json.loads(call.arguments)
        assert list(args) == ["query"]


# ── Generate: draft from tool result ────────────────────────────────────────

_PROPERTY_ENVELOPE: dict = {
    "status": "ok",
    "tool_name": "detect_property_outliers",
    "call_id": "mock_od_call_1",
    "evidence_id": "evt_1",
    "output": {
        "property_name": "band_gap_ev",
        "method": "zscore",
        "threshold": 2.0,
        "records": [
            {
                "material_id": "mp-1",
                "material_label": "LiFeO2 (mp-1)",
                "value": 2.1,
                "z_score": 0.5,
                "direction": "none",
                "is_outlier": False,
            },
            {
                "material_id": "mp-5",
                "material_label": "GaAs (mp-5)",
                "value": 5.0,
                "z_score": 3.2,
                "direction": "high",
                "is_outlier": True,
            },
        ],
        "distribution": {
            "count": 7,
            "mean": 2.5,
            "std": 1.2,
            "min_value": 1.0,
            "max_value": 5.0,
        },
        "warnings": [],
        "evidence_id": "evt_1",
    },
    "error": None,
}

_MULTIVARIATE_ENVELOPE: dict = {
    "status": "ok",
    "tool_name": "detect_multivariate_outliers",
    "call_id": "mock_od_call_1",
    "evidence_id": "evt_2",
    "output": {
        "properties": ["band_gap_ev", "density_g_cm3"],
        "method": "mahalanobis",
        "records": [
            {
                "material_id": "mp-1",
                "material_label": "LiFeO2 (mp-1)",
                "anomaly_score": 0.5,
                "is_outlier": False,
                "abnormal_properties": [],
                "outlier_reason": "b=0.1+d=0.2",
            },
            {
                "material_id": "mp-5",
                "material_label": "GaAs (mp-5)",
                "anomaly_score": 5.2,
                "is_outlier": True,
                "abnormal_properties": ["band_gap_ev"],
                "outlier_reason": "b=3.0+d=-2.0",
            },
        ],
        "warnings": [],
        "evidence_id": "evt_2",
    },
    "error": None,
}


class TestDraftFromToolResult:
    def test_property_outlier_draft(self) -> None:
        model = OutlierDetectionMockAgentModel()
        resp = model.generate(_request_after_tool("检测离群值", _PROPERTY_ENVELOPE))
        assert resp.status == AgentModelStatus.COMPLETED
        msg = resp.output_items[0]
        assert isinstance(msg, AgentMessageItem)
        content = str(msg.content)
        assert "单属性离群检测" in content
        assert "GaAs" in content
        assert "⚠" in content

    def test_multivariate_outlier_draft(self) -> None:
        model = OutlierDetectionMockAgentModel()
        resp = model.generate(_request_after_tool("检测离群值", _MULTIVARIATE_ENVELOPE))
        msg = resp.output_items[0]
        assert isinstance(msg, AgentMessageItem)
        content = str(msg.content)
        assert "多变量离群检测" in content
        assert "结论：发现 1 个多变量离群材料" in content

    def test_multivariate_no_outlier_has_explicit_conclusion(self) -> None:
        envelope = json.loads(json.dumps(_MULTIVARIATE_ENVELOPE))
        for record in envelope["output"]["records"]:
            record["is_outlier"] = False
        envelope["output"]["threshold"] = 2.448
        model = OutlierDetectionMockAgentModel()

        response = model.generate(_request_after_tool("检测离群值", envelope))

        message = response.output_items[0]
        assert isinstance(message, AgentMessageItem)
        content = str(message.content)
        assert "结论：未发现多变量离群材料" in content
        assert "离群阈值: 2.448" in content

    def test_draft_contains_evidence_id(self) -> None:
        model = OutlierDetectionMockAgentModel()
        resp = model.generate(_request_after_tool("检测离群值", _PROPERTY_ENVELOPE))
        msg = resp.output_items[0]
        content = json.loads(str(msg.content))
        assert "evt_1" in content.get("evidence_ids", [])

    def test_error_envelope_handled(self) -> None:
        model = OutlierDetectionMockAgentModel()
        err_env = {
            "status": "error",
            "tool_name": "detect_property_outliers",
            "call_id": "x",
            "evidence_id": "",
            "error": {"code": "TEST", "message": "模拟失败"},
        }
        resp = model.generate(_request_after_tool("检测离群值", err_env))
        content = str(resp.output_items[0].content)  # type: ignore[union-attr]
        assert "离群检测失败" in content
        assert "模拟失败" in content

    def test_missing_envelope_handled(self) -> None:
        model = OutlierDetectionMockAgentModel()
        req = MaterialAgentRequest(
            input_items=(
                AgentMessageItem(role="user", content="检测离群值"),
                AgentFunctionCallItem(
                    call_id="x",
                    name="detect_property_outliers",
                    arguments='{"source":{"material_formulas":["TiO2"]}}',
                ),
                AgentFunctionOutputItem(call_id="x", output=""),
            ),
            instructions="You are an outlier detection agent.",
            tool_definitions=_MOCK_TOOLS,
            final_draft_schema=_FINAL_DRAFT_SCHEMA,
        )
        resp = model.generate(req)
        content = str(resp.output_items[0].content)  # type: ignore[union-attr]
        assert "离群检测失败" in content
