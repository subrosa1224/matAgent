"""Digit-attached percentages must not relax other unit or evidence checks."""

import hashlib

import pytest
from tests.unit.sub_agents.test_literature_matrix_automation import FakeLlm

from materials_screening.sub_agents.literature.matrix import (
    unit_present_in_evidence,
    validate_matrix_evidence,
)
from materials_screening.sub_agents.literature.matrix_automation import (
    AutomatedMatrixExtractor,
    GroupCandidate,
    MatrixExtractionBatch,
    MeasurementCandidate,
    _locate_measurement_evidence,
)
from materials_screening.sub_agents.literature.models import (
    ExperimentalGroup,
    ExperimentalMeasurement,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord


@pytest.mark.parametrize(
    "text",
    [
        "retention 87.1%.",
        "retention 87.1 %.",
        "87.1\u00a0%",
        "87.1\n%",
        "45.2%, 87.1%, 75.3%, 70%, respectively.",
        "0%",
        "100%",
        "0.5%",
        ".5%",
        "-2%",
        "−2%",
        "+2%",
        "8.71e1%",
        "8.71E+1%",
        "87.1％",
        "８７．１％",
        "retention (%)",
    ],
)
def test_percent_typography_with_or_without_number_spacing(text):
    assert unit_present_in_evidence("%", text)


@pytest.mark.parametrize(
    "text",
    [
        "87.1",
        "87.1‰",
        "87.1‱",
        "87.1 percent",
        "87.1wt%",
        "87.1 wt%",
        "87.1mol%",
        "87.1 mol%",
        "SampleA87.1%",
        "sample87%",
        "87.1%wt",
        "87.1%2",
    ],
)
def test_percent_does_not_match_missing_unit_or_compound_tokens(text):
    assert not unit_present_in_evidence("%", text)


@pytest.mark.parametrize(
    "unit,text,expected",
    [
        ("A", "3 mA", False),
        ("mA", "3 A", False),
        ("mPa", "3 MPa", False),
        ("AW⁻¹", "3 mAW − 1", False),
        ("AW⁻¹", "3 AW − 2", False),
        ("ms", "3 m s", False),
        ("wt%", "3 mol%", False),
        ("mol%", "3 wt%", False),
        ("wt%", "3wt%", False),
        ("mAh/g", "3 Ah/g", False),
        ("Ah/g", "3 mAh/g", False),
        ("mAh/g", "3 mAh/g", True),
        ("AW⁻¹", "3 AW − 1", True),
        ("wt%", "3 wt%", True),
        ("mol%", "3 mol%", True),
        ("Ω", "3 Ω", True),
    ],
)
def test_all_non_percent_unit_rules_remain_unchanged(unit, text, expected):
    assert unit_present_in_evidence(unit, text) is expected


def chunk(text):
    return ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        page_from=6,
        page_to=6,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


class Store:
    def __init__(self, source):
        self.source = source

    def get_chunks(self, ids):
        return [self.source] if self.source.chunk_id in ids else []


def parent(source):
    return ExperimentalGroup(
        group_id="g-1",
        document_id="doc-1",
        label="Sample A",
        role="treatment",
        material="oxide",
        source_quote=source.text,
        chunk_id=source.chunk_id,
        page_from=6,
        page_to=6,
        source_text_sha256=source.text_sha256,
    )


def measurement(source, **updates):
    return ExperimentalMeasurement(
        measurement_id="m-1",
        group_id="g-1",
        document_id="doc-1",
        metric="capacity retention after 50 cycles",
        value_text="87.1",
        numeric_value=87.1,
        unit="%",
        source_quote=source.text,
        chunk_id=source.chunk_id,
        page_from=6,
        page_to=6,
        source_text_sha256=source.text_sha256,
    ).model_copy(update=updates)


def test_attached_percent_passes_provenance_validation_without_rewriting_source():
    source = chunk("Sample A retained 87.1% after 50 cycles.")
    group, row = parent(source), measurement(source)
    before = (group.model_dump(), row.model_dump())
    validate_matrix_evidence(
        Store(source), document_id="doc-1", groups=(group,), measurements=(row,)
    )
    assert before == (group.model_dump(), row.model_dump())
    assert row.source_quote == source.text
    assert row.source_text_sha256 == hashlib.sha256(source.text.encode()).hexdigest()


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"source_text_sha256": "0" * 64}, "hash"),
        ({"page_from": 5}, "pages"),
        ({"value_text": "99.9"}, "value"),
        ({"unit": "mAh/g"}, "unit"),
        ({"uncertainty_text": "± 1.0"}, "uncertainty"),
    ],
)
def test_percent_recognition_does_not_bypass_provenance_or_value_checks(
    updates, reason
):
    source = chunk("Sample A retained 87.1% after 50 cycles.")
    with pytest.raises(ValueError, match=reason):
        validate_matrix_evidence(
            Store(source),
            document_id="doc-1",
            groups=(parent(source),),
            measurements=(measurement(source, **updates),),
        )


def test_candidate_location_recovers_actual_retention_values_but_not_missing_capacity():
    source = chunk(
        "The first capacities are 223.2 mAh/g, 223.3 mAh/g, 205 mAh/g, "
        "and 202.3 mAh/g with capacity retention "
        "45.2%, 87.1%, 75.3%, 70%, respectively."
    )
    for value in ("45.2", "87.1", "75.3", "70"):
        row = MeasurementCandidate(
            group_key="sample",
            metric="capacity retention",
            value_text=value,
            numeric_value=float(value),
            unit="%",
            source_quote=source.text,
            chunk_id=source.chunk_id,
        )
        located, quote = _locate_measurement_evidence(
            row, {source.chunk_id: source}, (source,)
        )
        assert located == source and quote == source.text
    missing = row.model_copy(
        update={
            "metric": "discharge capacity",
            "value_text": "194.5",
            "numeric_value": 194.5,
            "unit": "mAh/g",
        }
    )
    assert _locate_measurement_evidence(
        missing, {source.chunk_id: source}, (source,)
    ) == (None, None)


def test_full_extractor_keeps_explicit_percent_pending_without_new_measurements():
    source = chunk("Sample A retained 87.1% after 50 cycles.")
    batch = MatrixExtractionBatch(
        groups=(
            GroupCandidate(
                group_key="sample-a",
                label="Sample A",
                material="oxide",
                source_quote=source.text,
                chunk_id=source.chunk_id,
            ),
        ),
        measurements=(
            MeasurementCandidate(
                group_key="sample-a",
                metric="capacity retention after 50 cycles",
                value_text="87.1",
                numeric_value=87.1,
                unit="%",
                source_quote=source.text,
                chunk_id=source.chunk_id,
            ),
        ),
    )
    before = batch.model_dump()
    result = AutomatedMatrixExtractor(FakeLlm(batch), Store(source)).extract(
        document_id="doc-1", chunks=(source,)
    )
    assert batch.model_dump() == before
    assert len(result.measurements) == 1
    assert result.measurements[0].value_text == "87.1"
    assert result.measurements[0].unit == "%"
    assert result.measurements[0].source_quote == source.text
    assert result.measurements[0].review_status == "pending"
    assert result.diagnostics.measurement_candidates[0].status == "constructed"


def test_percent_candidates_still_pass_through_sample_binding_rejection():
    # Use the existing gate's supported clause boundary, not an unsupported
    # comma/while construction. This patch does not change clause parsing.
    source = chunk("Sample A retained 87.1%; Sample B retained 70%.")
    batch = MatrixExtractionBatch(
        groups=(
            GroupCandidate(
                group_key="sample-a",
                label="Sample A",
                material="oxide",
                source_quote=source.text,
                chunk_id=source.chunk_id,
            ),
        ),
        measurements=(
            MeasurementCandidate(
                group_key="sample-a",
                metric="capacity retention",
                value_text="70",
                numeric_value=70.0,
                unit="%",
                source_quote=source.text,
                chunk_id=source.chunk_id,
            ),
        ),
    )
    result = AutomatedMatrixExtractor(FakeLlm(batch), Store(source)).extract(
        document_id="doc-1", chunks=(source,)
    )
    assert result.diagnostics.before_validation_measurements == 1
    assert result.measurements == ()
    assert any("sample/value binding" in warning for warning in result.warnings)
