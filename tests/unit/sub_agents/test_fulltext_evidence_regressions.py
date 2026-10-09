"""Regression cases for sample binding, attribution and bounded delivery."""

import hashlib
from typing import Any

import pytest

from materials_screening.llm.base import StructuredProviderResponse
from materials_screening.master.master_nodes import (
    _literature_analysis_passthrough,
    _materials_database_passthrough,
)
from materials_screening.sub_agents.literature.automation import (
    AutomatedDossierExtractor,
)
from materials_screening.sub_agents.literature.dossier import (
    DossierExtractionBatch,
    DossierItemCandidate,
    DossierSummaryTranslation,
    DossierTranslationBatch,
)
from materials_screening.sub_agents.literature.matrix_automation import (
    sanitize_measurements,
)
from materials_screening.sub_agents.literature.models import (
    ExperimentalGroup,
    ExperimentalMeasurement,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord

TABLE = """Table 1. Photocatalytic results of all samples.
Sample Name
Degradation (%)
K (min−1)
R2
ZnO
25.3%
0.00166
0.99093
10% ZnO/g-C3N4
63.2%
0.00503
0.96199
20% ZnO/g-C3N4
79.1%
0.00717
0.94071
"""


def test_short_quote_cannot_hide_a_wrong_table_row() -> None:
    row = measurement("63.2", "10% ZnO/g-C3N4\n63.2%")
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        text=TABLE,
        page_from=9,
        page_to=9,
        text_sha256=hashlib.sha256(TABLE.encode()).hexdigest(),
    )
    assert not sanitize_measurements(
        [row],
        groups=[group("20% ZnO/g-C3N4")],
        warnings=[],
        chunks_by_id={"chunk-1": chunk},
    )


def test_capacity_prose_is_not_rejected_by_unrelated_same_chunk_table() -> None:
    quote = (
        "Among these samples, the sample prepared at 150 C delivered a discharge "
        "capacity of 194.5 mAh/g after 50 cycles."
    )
    text = "Table 1. Surface area.\nSample\nSBET\n100 C\n10\n150 C\n20\n" + quote
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        text=text,
        page_from=6,
        page_to=6,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    row = measurement("194.5", quote).model_copy(
        update={"metric": "discharge capacity", "unit": "mAh/g"}
    )
    assert (
        len(
            sanitize_measurements(
                [row],
                groups=[group("150 C")],
                warnings=[],
                chunks_by_id={chunk.chunk_id: chunk},
            )
        )
        == 1
    )


@pytest.mark.parametrize("quote", ["194.5", "194.5 mAh/g"])
def test_bare_capacity_number_cannot_bypass_sample_context(quote: str) -> None:
    text = "Table 1. Surface area.\nSample\nSBET\n100 C\n10\n150 C\n20\n"
    text += "A different sample delivered a capacity of 194.5 mAh/g."
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        text=text,
        page_from=6,
        page_to=6,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    row = measurement("194.5", quote).model_copy(
        update={"metric": "discharge capacity", "unit": "mAh/g"}
    )
    assert not sanitize_measurements(
        [row],
        groups=[group("150 C")],
        warnings=[],
        chunks_by_id={chunk.chunk_id: chunk},
    )


def group(label: str, variables: dict[str, str] | None = None) -> ExperimentalGroup:
    return ExperimentalGroup(
        group_id="g-1",
        document_id="doc-1",
        label=label,
        role="unknown",
        material=label,
        variables=variables or {},
        source_quote=TABLE,
        chunk_id="chunk-1",
        page_from=9,
        page_to=9,
        source_text_sha256=hashlib.sha256(TABLE.encode()).hexdigest(),
    )


def measurement(value: str, quote: str = TABLE) -> ExperimentalMeasurement:
    return ExperimentalMeasurement(
        measurement_id="m-1",
        group_id="g-1",
        document_id="doc-1",
        metric="photocatalytic degradation efficiency",
        value_text=value,
        numeric_value=float(value),
        unit="%",
        source_quote=quote,
        chunk_id="chunk-1",
        page_from=9,
        page_to=9,
        source_text_sha256=hashlib.sha256(TABLE.encode()).hexdigest(),
    )


@pytest.mark.parametrize(
    "label,value",
    [
        ("ZnO/g-C3N4 composite", "25.3"),
        ("ZnO/g-C3N4 composite", "63.2"),
        ("20% ZnO/g-C3N4", "63.2"),
        ("ZnO", "63.2"),
    ],
)
def test_multisample_table_rejects_unbound_or_wrong_sample(
    label: str, value: str
) -> None:
    warnings: list[str] = []
    rows = sanitize_measurements(
        (measurement(value),), groups=(group(label),), warnings=warnings
    )
    assert not rows
    assert warnings


def test_table_binding_keeps_exact_sample_and_value() -> None:
    rows = sanitize_measurements(
        (measurement("63.2"),), groups=(group("10% ZnO/g-C3N4"),), warnings=[]
    )
    assert len(rows) == 1


def test_malformed_table_does_not_fall_back_to_page_presence() -> None:
    quote = TABLE.replace("0.00503\n", "")
    assert not sanitize_measurements(
        (measurement("25.3", quote),), groups=(group("20% ZnO/g-C3N4"),), warnings=[]
    )


def test_table_binding_does_not_use_wrong_metric_column() -> None:
    row = measurement("0.00503")
    assert not sanitize_measurements(
        (row,), groups=(group("10% ZnO/g-C3N4"),), warnings=[]
    )


def test_table_binding_does_not_use_a_percentage_under_the_wrong_metric() -> None:
    row = measurement("63.2").model_copy(update={"metric": "porosity"})
    assert not sanitize_measurements(
        (row,), groups=(group("10% ZnO/g-C3N4"),), warnings=[]
    )


def test_local_table_extraction_keeps_all_rows_without_llm_guesses() -> None:
    from materials_screening.sub_agents.literature.matrix_automation import (
        AutomatedMatrixExtractor,
        MatrixExtractionBatch,
    )

    chunk = ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        page_from=9,
        page_to=9,
        text=TABLE,
        text_sha256=hashlib.sha256(TABLE.encode()).hexdigest(),
    )

    class Store:
        def get_chunks(self, _: Any) -> list[ChunkRecord]:
            return [chunk]

    class EmptyLlm:
        def generate_structured(self, **_: Any) -> StructuredProviderResponse[Any]:
            return StructuredProviderResponse(
                parsed=MatrixExtractionBatch(),
                provider="fake",
                model="fake",
                request_id="r-1",
                latency_ms=0,
                input_tokens=1,
                output_tokens=1,
                reasoning_tokens=0,
                raw_output_sha256="a" * 64,
            )

    result = AutomatedMatrixExtractor(EmptyLlm(), Store()).extract(
        document_id="doc-1", chunks=(chunk,)
    )
    labels = {g.group_id: g.label for g in result.groups}
    assert {(labels[m.group_id], m.numeric_value) for m in result.measurements} == {
        ("ZnO", 25.3),
        ("10% ZnO/g-C3N4", 63.2),
        ("20% ZnO/g-C3N4", 79.1),
    }
    assert all(m.review_status == "pending" for m in result.measurements)
    assert result.diagnostics.llm_measurements == 0
    assert result.diagnostics.accepted_measurements == 3
    assert result.diagnostics.table_measurements == 3


def test_coordinated_prose_does_not_swap_sample_outcomes() -> None:
    quote = "Sample A reached 10 MPa; Sample B reached 20 MPa."
    row = measurement("20", quote).model_copy(
        update={"unit": "MPa", "metric": "strength"}
    )
    assert not sanitize_measurements((row,), groups=(group("Sample A"),), warnings=[])


class DossierLlm:
    def generate_structured(self, **kwargs: Any) -> StructuredProviderResponse[Any]:
        if kwargs["output_model"] is DossierTranslationBatch:
            parsed = DossierTranslationBatch(
                items=(
                    DossierSummaryTranslation(
                        item_key="item-0", chinese_summary="样品在900°C煅烧2 h。"
                    ),
                    DossierSummaryTranslation(
                        item_key="item-1", chinese_summary="本文样品在450°C煅烧15 min。"
                    ),
                )
            )
        else:
            category = "preparation"
            parsed = DossierExtractionBatch(
                items=(
                    DossierItemCandidate(
                        category=category,
                        summary="样品在900°C煅烧2 h。",
                        source_quote=(
                            "Here, calcination at 900°C for 2 h produced the samples."
                        ),
                        chunk_id="chunk-1",
                    ),
                    DossierItemCandidate(
                        category=category,
                        summary="本文样品在450°C煅烧15 min。",
                        source_quote=(
                            "In this work, the samples were calcined "
                            "at 450°C for 15 min."
                        ),
                        chunk_id="chunk-1",
                    ),
                )
                if "Target category: preparation" in kwargs["user_text"]
                else ()
            )
        return StructuredProviderResponse(
            parsed=parsed,
            provider="fake",
            model="fake",
            request_id="r-1",
            latency_ms=0,
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            raw_output_sha256="a" * 64,
        )


def test_preceding_attribution_quarantines_previous_work_not_current_method() -> None:
    text = (
        "Smith et al. [23] prepared oxide nanoparticles in 2017. "
        "Here, calcination at 900°C for 2 h produced the samples.\n"
        "2 Experimental\n"
        "In this work, the samples were calcined at 450°C for 15 min."
    )
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        page_from=2,
        page_to=2,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    dossier, warnings = AutomatedDossierExtractor(DossierLlm()).extract(
        document_id="doc-1", title="Paper", chunks=(chunk,)
    )
    prior = next(item for item in dossier.items if "900" in item.source_quote)
    own = next(item for item in dossier.items if "450" in item.source_quote)
    assert prior.risk_level == "high"
    assert own.risk_level != "high"
    assert any("prior work" in warning for warning in warnings)
    assert "900" not in (dossier.narrative_summary or "")


def test_long_literature_response_keeps_warning_and_does_not_crash() -> None:
    warning = "零安全数值；未执行数据分析，不能排名。"
    draft = _materials_database_passthrough(
        {
            "model_call_count": 1,
            "executed_call_ids": ["call-1"],
            "sub_agent_results": [
                {
                    "status": "ok",
                    "sub_agent_name": "literature",
                    "call_id": "call-1",
                    "response_text": "证据报告\n" + "正文证据\n" * 1900,
                    "warnings": [warning],
                }
            ],
        }
    )
    assert draft is not None
    assert len(draft["answer"]) <= 8000
    assert warning in draft["answer"]
    assert warning in draft["warnings"]
    assert "完整" in draft["answer"]


def test_combined_long_report_is_bounded_and_keeps_critical_warning() -> None:
    warning = "不可跨样品排名。"
    draft = _literature_analysis_passthrough(
        {
            "executed_call_ids": ["lit-1", "analysis-1"],
            "sub_agent_results": [
                {
                    "status": "ok",
                    "sub_agent_name": "literature",
                    "call_id": "lit-1",
                    "response_text": "LITERATURE_DATASET_HANDOFF: "
                    "dataset_id=dataset-1\n" + "证据\n" * 3000,
                    "warnings": [warning, "待审核。" * 500],
                },
                {
                    "status": "ok",
                    "sub_agent_name": "data_analysis",
                    "call_id": "analysis-1",
                    "response_text": "分析\n" * 700,
                    "warnings": [],
                },
            ],
        }
    )
    assert draft is not None
    assert len(draft["answer"]) <= 8000
    assert warning in draft["warnings"]
    assert warning in draft["answer"]
