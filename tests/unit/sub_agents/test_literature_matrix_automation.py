from __future__ import annotations

import hashlib
from typing import Any

from materials_screening.llm.base import StructuredProviderResponse
from materials_screening.sub_agents.literature.matrix_automation import (
    AutomatedMatrixExtractor,
    ClaimCandidate,
    GroupCandidate,
    MatrixExtractionBatch,
    MeasurementCandidate,
    _extract_factorial_groups,
    _extract_series_groups,
    detect_matrix_warnings,
    expand_measurement_group_evidence,
    sanitize_measurements,
)
from materials_screening.sub_agents.literature.models import (
    ExperimentalGroup,
    ExperimentalMeasurement,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord


def _chunk() -> ChunkRecord:
    text = (
        "Abstract: Sample B improved compressive strength. "
        "Sample A was the control at 25 °C with compressive strength 10.0 MPa. "
        "Sample B contained 1.5 mol% Nb "
        "and reached compressive strength 12.0 ± 0.4 MPa."
    )
    return ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        page_from=2,
        page_to=2,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


class Store:
    def get_chunks(self, chunk_ids: tuple[str, ...]) -> list[ChunkRecord]:
        return [_chunk()] if "chunk-1" in chunk_ids else []


class FakeLlm:
    def __init__(self, batch: MatrixExtractionBatch) -> None:
        self.batch = batch

    def generate_structured(self, **_: Any) -> StructuredProviderResponse[Any]:
        return StructuredProviderResponse(
            parsed=self.batch,
            provider="fake",
            model="fake",
            request_id="matrix-1",
            latency_ms=0,
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            raw_output_sha256="a" * 64,
        )


def test_generic_extractor_builds_only_pending_grounded_records() -> None:
    text = _chunk().text
    batch = MatrixExtractionBatch(
        groups=(
            GroupCandidate(
                group_key="sample-a",
                label="Sample A",
                role="control",
                material="control sample",
                conditions={"temperature": "25 °C"},
                source_quote=text,
                chunk_id="chunk-1",
            ),
            GroupCandidate(
                group_key="sample-b",
                label="Sample B",
                role="treatment",
                material="Nb sample",
                variables={"Nb": "1.5 mol%"},
                conditions={},
                source_quote=text,
                chunk_id="chunk-1",
            ),
        ),
        measurements=(
            MeasurementCandidate(
                group_key="sample-a",
                metric="compressive strength",
                value_text="10.0",
                numeric_value=10.0,
                unit="MPa",
                source_quote=text,
                chunk_id="chunk-1",
            ),
            MeasurementCandidate(
                group_key="sample-b",
                metric="compressive strength",
                value_text="12.0",
                numeric_value=12.0,
                unit="MPa",
                uncertainty_text="± 0.4",
                source_quote=text,
                chunk_id="chunk-1",
            ),
        ),
        abstract_claims=(
            ClaimCandidate(
                claim_text="Sample B improved compressive strength.",
                source_quote=text,
                chunk_id="chunk-1",
            ),
        ),
    )
    result = AutomatedMatrixExtractor(FakeLlm(batch), Store()).extract(
        document_id="doc-1", chunks=(_chunk(),)
    )

    assert len(result.groups) == 2
    assert len(result.measurements) == 2
    assert len(result.claims) == 1
    assert result.groups[0].review_status == "pending"
    assert result.measurements[0].unit == "MPa"
    assert result.measurements[1].uncertainty_text == "± 0.4"
    assert result.claims[0].source_section == "abstract"
    assert len(result.comparisons) == 1
    assert result.comparisons[0].direction == "increase"
    assert result.claim_evidence_links[0].assessment == "supported"


def test_checkpointed_batch_replays_through_evidence_gates_without_llm() -> None:
    from unittest.mock import Mock

    llm = Mock()
    cached = MatrixExtractionBatch(groups=(), measurements=())
    result = AutomatedMatrixExtractor(llm, Store()).extract(
        document_id="doc-1",
        chunks=(_chunk(),),
        cached_batches={1: cached},
    )
    llm.generate_structured.assert_not_called()
    assert result.diagnostics.successful_batches == 1
    assert result.diagnostics.batch_attempts[0].status == "reused"


def test_numeric_mismatch_and_nonverbatim_quote_are_rejected() -> None:
    batch = MatrixExtractionBatch(
        groups=(
            GroupCandidate(
                group_key="sample-b",
                label="Sample B",
                material="Nb sample",
                variables={"Nb": "1.5 mol%"},
                source_quote=_chunk().text,
                chunk_id="chunk-1",
            ),
        ),
        measurements=(
            MeasurementCandidate(
                group_key="sample-b",
                metric="compressive strength",
                value_text="12.0",
                numeric_value=99.0,
                unit="MPa",
                source_quote=_chunk().text,
                chunk_id="chunk-1",
            ),
        ),
        abstract_claims=(
            ClaimCandidate(
                claim_text="Invented claim.",
                source_quote="Invented claim.",
                chunk_id="chunk-1",
            ),
        ),
    )
    result = AutomatedMatrixExtractor(FakeLlm(batch), Store()).extract(
        document_id="doc-1", chunks=(_chunk(),)
    )

    assert not result.measurements
    assert not result.claims
    assert any("numeric value mismatch" in item for item in result.warnings)
    assert any("abstract claim rejected" in item for item in result.warnings)


def test_layout_warnings_are_explicit() -> None:
    text = "Table 2 is presented in Figure 3; values are shown in Fig. 3."
    chunk = ChunkRecord(
        chunk_id="chunk-layout",
        document_id="doc-1",
        paper_id=None,
        page_from=4,
        page_to=5,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    warnings = detect_matrix_warnings((chunk,))

    assert any(item.startswith("CROSS_PAGE_TABLE") for item in warnings)
    assert any(item.startswith("FIGURE_ONLY_VALUES") for item in warnings)
    assert any(item.startswith("MISSING_TABLE_DATA") for item in warnings)


def test_group_candidate_flattens_model_list_conditions() -> None:
    candidate = GroupCandidate.model_validate(
        {
            "group_key": "g1",
            "label": "G1",
            "material": "ceramic",
            "variables": {"porosity": ["60 %", "70 %"]},
            "conditions": {"assays": ["CCK-8", "qRT-PCR"]},
            "source_quote": "60 %, 70 %, CCK-8 and qRT-PCR",
            "chunk_id": "chunk-1",
        }
    )

    assert candidate.variables == {"porosity[1]": "60 %", "porosity[2]": "70 %"}
    assert candidate.conditions == {"assays[1]": "CCK-8", "assays[2]": "qRT-PCR"}


def test_unbound_measurement_is_rejected_without_losing_batch() -> None:
    batch = MatrixExtractionBatch(
        groups=(
            GroupCandidate(
                group_key="sample-b",
                label="Sample B",
                material="Nb sample",
                variables={"Nb": "1.5 mol%"},
                source_quote=_chunk().text,
                chunk_id="chunk-1",
            ),
        ),
        measurements=(
            MeasurementCandidate(
                metric="film thickness",
                value_text="8",
                unit="µm",
                source_quote=_chunk().text,
                chunk_id="chunk-1",
            ),
        ),
    )
    result = AutomatedMatrixExtractor(FakeLlm(batch), Store()).extract(
        document_id="doc-1", chunks=(_chunk(),)
    )

    assert len(result.groups) == 1
    assert not result.measurements
    assert any("group_key" in warning for warning in result.warnings)


def test_coordinated_evidence_keeps_only_explicit_measurement_assignments() -> None:
    quote = (
        "Elevated COL-I expression was observed in the polyhedral structure "
        "with 65 % porosity and square structure with 70 % porosity after 7 days."
    )
    chunk = ChunkRecord(
        chunk_id="chunk-coordinated",
        document_id="doc-1",
        paper_id=None,
        page_from=6,
        page_to=6,
        text=quote,
        text_sha256=hashlib.sha256(quote.encode()).hexdigest(),
    )
    groups = (
        ExperimentalGroup(
            group_id="g-poly",
            document_id="doc-1",
            label="Polyhedral 65%",
            role="treatment",
            material="CaP",
            variables={"structure": "polyhedral", "porosity": "65"},
            source_quote=quote,
            chunk_id=chunk.chunk_id,
            page_from=6,
            page_to=6,
            source_text_sha256=chunk.text_sha256,
        ),
        ExperimentalGroup(
            group_id="g-square",
            document_id="doc-1",
            label="Square 70%",
            role="treatment",
            material="CaP",
            variables={"structure": "square", "porosity": "70"},
            source_quote=quote,
            chunk_id=chunk.chunk_id,
            page_from=6,
            page_to=6,
            source_text_sha256=chunk.text_sha256,
        ),
    )
    measurement = ExperimentalMeasurement(
        measurement_id="m-poly",
        group_id="g-poly",
        document_id="doc-1",
        metric="COL-I expression timing",
        value_text="7 days",
        numeric_value=7,
        unit="days",
        source_quote=quote,
        chunk_id=chunk.chunk_id,
        page_from=6,
        page_to=6,
        source_text_sha256=chunk.text_sha256,
    )

    result = expand_measurement_group_evidence(
        (measurement,), groups=groups, chunks_by_id={chunk.chunk_id: chunk}
    )

    assert result == (measurement,)

    # A shared time can still be represented for both groups, but the second
    # assignment must be explicit rather than inferred from co-occurrence.
    square = measurement.model_copy(
        update={"measurement_id": "m-square", "group_id": "g-square"}
    )
    assert expand_measurement_group_evidence(
        (measurement, square), groups=groups, chunks_by_id={chunk.chunk_id: chunk}
    ) == (measurement, square)


def test_measurement_sanitizer_maps_pdf_header_and_removes_claim_range() -> None:
    common = {
        "group_id": "g1",
        "document_id": "doc-1",
        "unit": "%",
        "source_quote": "Photoconversion devices: H (%) 1.3",
        "chunk_id": "chunk-1",
        "page_from": 1,
        "page_to": 1,
        "source_text_sha256": "a" * 64,
    }
    rows = (
        ExperimentalMeasurement(
            measurement_id="m-h",
            metric="H",
            value_text="1.3",
            numeric_value=1.3,
            **common,
        ),
        ExperimentalMeasurement(
            measurement_id="m-duplicate",
            metric="photoconversion efficiency",
            value_text="1.3",
            numeric_value=1.3,
            **common,
        ),
        ExperimentalMeasurement(
            measurement_id="m-range",
            metric="photovoltaic conversion efficiency improvement",
            value_text="1.1% to 1.3%",
            numeric_value=None,
            **common,
        ),
    )
    warnings: list[str] = []
    result = sanitize_measurements(rows, warnings=warnings)

    assert len(result) == 1
    assert result[0].metric == "photoconversion efficiency"
    assert len(warnings) == 2


def test_group_keeps_numeric_variable_and_drops_ungrounded_condition() -> None:
    text = "Devices used RPbI2=FAI ratios of 0.85, 1.05 and 1.16."
    chunk = ChunkRecord(
        chunk_id="chunk-ratio",
        document_id="doc-1",
        paper_id=None,
        page_from=1,
        page_to=1,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )

    class RatioStore:
        def get_chunks(self, chunk_ids: tuple[str, ...]) -> list[ChunkRecord]:
            return [chunk] if "chunk-ratio" in chunk_ids else []

    batch = MatrixExtractionBatch(
        groups=(
            GroupCandidate(
                group_key="ratio-1.05",
                label="RPbI2=FAI 1.05",
                material="perovskite",
                variables={"precursor ratio": "RPbI2=FAI = 1.05"},
                conditions={"annealing": "100 °C for 30 min"},
                source_quote="Devices with ratio 1.05 were prepared.",
                chunk_id="chunk-ratio",
            ),
        )
    )
    result = AutomatedMatrixExtractor(FakeLlm(batch), RatioStore()).extract(
        document_id="doc-1", chunks=(chunk,)
    )

    assert len(result.groups) == 1
    assert result.groups[0].variables == {"precursor ratio": "1.05"}
    assert not result.groups[0].conditions
    assert any("dropped 1" in warning for warning in result.warnings)


def test_generic_series_parser_recovers_explicit_ratio_groups() -> None:
    text = (
        "We obtained different molar ratios for PbI2/FAI varying from 1.16, "
        "1.05, 1.00, and 0.85."
    )
    chunk = ChunkRecord(
        chunk_id="chunk-series",
        document_id="doc-1",
        paper_id=None,
        page_from=5,
        page_to=5,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    groups = _extract_series_groups("doc-1", (chunk,), ())

    assert {next(iter(group.variables.values())) for group in groups} == {
        "0.85",
        "1.00",
        "1.05",
        "1.16",
    }


def test_factorial_parser_expands_explicit_factor_level_design() -> None:
    text = (
        "The scaffolds included triangular (T), diamond (D), square (S), and "
        "polyhedral (P) structures, each featuring six porosity levels ranging "
        "from 50 % to 75 %."
    )
    chunk = ChunkRecord(
        chunk_id="chunk-factorial",
        document_id="doc-1",
        paper_id=None,
        page_from=3,
        page_to=3,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    groups = _extract_factorial_groups("doc-1", (chunk,), ())

    assert len(groups) == 25
    assert {group.label for group in groups} >= {
        "triangular 50 %",
        "diamond 75 %",
        "square 70 %",
        "polyhedral 65 %",
        "all 24 scaffold configurations",
    }
