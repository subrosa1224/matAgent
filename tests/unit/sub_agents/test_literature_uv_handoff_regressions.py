from __future__ import annotations

import hashlib

import pytest

from materials_screening.llm.base import StructuredProviderResponse
from materials_screening.sub_agents.literature.matrix import validate_matrix_evidence
from materials_screening.sub_agents.literature.matrix_automation import (
    AutomatedMatrixExtractor,
    GroupCandidate,
    MatrixExtractionBatch,
    MeasurementCandidate,
    _consolidate_semantic_groups,
    _ground_mapping,
    _has_group_binding,
    _infer_group_id,
    _measurement_evidence,
)
from materials_screening.sub_agents.literature.models import (
    ExperimentalGroup,
    ExperimentalMeasurement,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord
from materials_screening.sub_agents.literature.user_report import (
    LiteratureUserReport,
    UserPaperReport,
    UserReportEvidence,
    UserReportNarrative,
    _validate_narrative,
)


def _chunk(text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id="chunk-uv",
        document_id="doc-uv",
        paper_id=None,
        page_from=4,
        page_to=4,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def _group(chunk: ChunkRecord, group_id: str = "group-uv") -> ExperimentalGroup:
    return ExperimentalGroup(
        group_id=group_id,
        document_id=chunk.document_id,
        label="MLG/ZnONRs device",
        role="treatment",
        material="MLG/ZnONRs",
        variables={"MLG": "MLG", "ZnONRs": "ZnONRs"},
        source_quote=chunk.text,
        chunk_id=chunk.chunk_id,
        page_from=chunk.page_from,
        page_to=chunk.page_to,
        source_text_sha256=chunk.text_sha256,
    )


def test_duplicate_group_identity_is_not_ambiguous() -> None:
    chunk = _chunk("MLG/ZnONRs device responsivity was 113 AW − 1.")
    group = _group(chunk)
    assert _infer_group_id(chunk.text, (group, group)) == group.group_id


def test_distinct_group_identities_remain_ambiguous() -> None:
    chunk = _chunk("MLG/ZnONRs device responsivity was 113 AW − 1.")
    assert (
        _infer_group_id(chunk.text, (_group(chunk), _group(chunk, "group-other")))
        is None
    )


def test_negative_variable_is_not_grounded_by_a_material_word_fragment() -> None:
    assert (
        _ground_mapping(
            {"Schottky Junction": "No"},
            "The ZnONRs form a Schottky junction with graphene.",
            numeric_fallback=True,
        )
        == {}
    )


def test_negative_variable_does_not_bind_the_actual_device_measurement() -> None:
    chunk = _chunk("Graphene/ZnONR array Schottky junction 365 113 AW − 1")
    control = _group(chunk).model_copy(
        update={"label": "ZnONR/Si Control", "variables": {"Schottky Junction": "No"}}
    )
    assert not _has_group_binding(control, chunk.text)
    assert _infer_group_id(chunk.text, (control,)) is None


@pytest.mark.parametrize(
    "change",
    [
        {"material": "GaN"},
        {"role": "reference"},
        {"conditions": {"bias": "-5 V"}},
        {"variables": {"different_variable": "365"}},
    ],
)
def test_equal_variable_values_do_not_merge_distinct_experimental_groups(
    change,
) -> None:
    chunk = _chunk("Graphene/ZnONR array 365 nm")
    first = _group(chunk).model_copy(update={"variables": {"wavelength": "365"}})
    second = first.model_copy(update={"group_id": "group-other", **change})
    groups, _ = _consolidate_semantic_groups((first, second), ())
    assert len(groups) == 2


def test_unit_typography_can_locate_original_measurement_evidence() -> None:
    chunk = _chunk("MLG/ZnONRs device responsivity was 113 AW − 1.")
    candidate = MeasurementCandidate(
        group_key="uv",
        metric="responsivity",
        value_text="113",
        numeric_value=113,
        unit="AW⁻¹",
        source_quote="MLG/ZnONRs device responsivity was 113 AW⁻¹.",
        chunk_id=chunk.chunk_id,
    )
    assert _measurement_evidence(candidate, chunk) == chunk.text


class _Store:
    def __init__(self, chunk: ChunkRecord) -> None:
        self.chunk = chunk

    def get_chunks(self, chunk_ids: tuple[str, ...]) -> list[ChunkRecord]:
        return [self.chunk] if self.chunk.chunk_id in chunk_ids else []


def _validate_unit(unit: str, text: str) -> None:
    chunk = _chunk(text)
    group = _group(chunk)
    measurement = ExperimentalMeasurement(
        measurement_id="measurement-uv",
        document_id=chunk.document_id,
        group_id=group.group_id,
        metric="responsivity",
        value_text="113",
        numeric_value=113,
        unit=unit,
        source_quote=chunk.text,
        chunk_id=chunk.chunk_id,
        page_from=chunk.page_from,
        page_to=chunk.page_to,
        source_text_sha256=chunk.text_sha256,
    )
    validate_matrix_evidence(
        _Store(chunk),
        document_id=chunk.document_id,
        groups=(group,),
        measurements=(measurement,),
    )


def test_unit_typography_passes_without_changing_source_quote() -> None:
    _validate_unit("AW⁻¹", "MLG/ZnONRs responsivity was 113 AW − 1.")


def test_duplicate_groups_and_typographic_units_survive_the_full_extractor() -> None:
    chunk = _chunk("MLG/ZnONRs responsivity was 113 AW − 1 and gain was 385.")
    candidate_group = GroupCandidate(
        group_key="uv-device",
        label="MLG/ZnONRs",
        material="MLG/ZnONRs",
        variables={"MLG": "MLG", "ZnONRs": "ZnONRs"},
        source_quote=chunk.text,
        chunk_id=chunk.chunk_id,
    )
    batch = MatrixExtractionBatch(
        groups=(candidate_group, candidate_group),
        measurements=(
            MeasurementCandidate(
                metric="responsivity",
                value_text="113",
                numeric_value=113,
                unit="AW⁻¹",
                source_quote=chunk.text,
                chunk_id=chunk.chunk_id,
            ),
            MeasurementCandidate(
                metric="gain",
                value_text="385",
                numeric_value=385,
                source_quote=chunk.text,
                chunk_id=chunk.chunk_id,
            ),
        ),
    )

    class Llm:
        def generate_structured(self, **kwargs: object) -> StructuredProviderResponse:
            return StructuredProviderResponse(
                parsed=batch,
                provider="fake",
                model="fake",
                request_id="test",
                latency_ms=0,
                input_tokens=1,
                output_tokens=1,
                reasoning_tokens=0,
                raw_output_sha256="a" * 64,
            )

    result = AutomatedMatrixExtractor(Llm(), _Store(chunk)).extract(
        document_id=chunk.document_id, chunks=(chunk,)
    )
    assert len(result.groups) == 1
    assert {row.numeric_value for row in result.measurements} == {113, 385}
    assert all(row.source_quote == chunk.text for row in result.measurements)
    assert all(row.review_status == "pending" for row in result.measurements)


@pytest.mark.parametrize(
    ("unit", "source_unit"),
    [("AW⁻¹", "mAW − 1"), ("AW⁻¹", "AW − 2"), ("mPa", "MPa"), ("ms", "m s")],
)
def test_unit_matching_never_changes_scale_dimension_or_case(
    unit: str, source_unit: str
) -> None:
    with pytest.raises(ValueError, match="unit"):
        _validate_unit(unit, f"MLG/ZnONRs measurement was 113 {source_unit}.")


def _report() -> LiteratureUserReport:
    evidence = UserReportEvidence(
        evidence_id="E-1234567890",
        category="innovation",
        summary="研究构建了石墨烯与氧化锌纳米棒紫外探测器。",
        page=2,
        source_quote="A graphene/ZnO nanorod ultraviolet photodetector was fabricated.",
        risk_level="low",
        display_status="reliable",
    )
    return LiteratureUserReport(
        report_id="user-lit-test",
        topic="紫外探测",
        papers=(
            UserPaperReport(
                document_id="doc-uv",
                title="UV device",
                dossier_status="pending",
                evidence=(evidence,),
                low_risk_count=1,
                medium_risk_count=0,
                excluded_high_risk_count=0,
                excluded_unsafe_count=0,
            ),
        ),
    )


def _narrative(statement: str) -> UserReportNarrative:
    return UserReportNarrative(
        markdown=(
            "## 主题概述\n研究构建了石墨烯与氧化锌纳米棒紫外探测器。 [E-1234567890]\n"
            "## 逐篇解读\n## 共同结论\n## 关键差异\n## 设计启示\n## 适用边界\n"
            f"{statement} [E-1234567890]"
        )
    )


def test_report_rejects_paper_absence_claim_from_filtered_evidence() -> None:
    report = _report()
    with pytest.raises(ValueError, match="absence"):
        _validate_narrative(
            _narrative("本文未提供响应度和增益，因而无法量化探测效率。"),
            report,
            {item.evidence_id: item for item in report.papers[0].evidence},
        )


def test_report_can_state_extraction_scope_without_claiming_paper_absence() -> None:
    report = _report()
    _validate_narrative(
        _narrative("当前已提取证据未提供响应度，需继续核对全文。"),
        report,
        {item.evidence_id: item for item in report.papers[0].evidence},
    )
