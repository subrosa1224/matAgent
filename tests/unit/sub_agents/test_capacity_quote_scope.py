"""Narrow an unrelated-table quote only for explicit, unique capacity evidence."""

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
from tests.unit.master.test_literature_evidence_trial import _report, _ReportStore
from tests.unit.sub_agents.test_literature_matrix_automation import FakeLlm

from materials_screening.master.literature_evidence_trial import (
    LiteratureEvidenceTrialService,
)
from materials_screening.sub_agents.literature.matrix_automation import (
    AutomatedMatrixExtractor,
    GroupCandidate,
    MatrixExtractionBatch,
    MeasurementCandidate,
    expand_measurement_group_evidence,
    sanitize_measurements,
)
from materials_screening.sub_agents.literature.models import (
    ExperimentalGroup,
    ExperimentalMeasurement,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord

TABLE = (
    "Table 1. Surface area.\nSample\nSBET/m2·g−1\nPore Diameter/nm\n"
    "100 ◦C\n82.11\n8.46\n150 ◦C\n194.48\n3.27\n"
)
EXPLICIT = (
    "Among these samples, the sample obtained at 150 ◦C exhibits the highest "
    "initial specific capacity of 223.3 mAh/g."
)


def source(text):
    return ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        page_from=6,
        page_to=6,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def parent(chunk, temperature="150 ◦C", **updates):
    return ExperimentalGroup(
        group_id="group-1",
        document_id="doc-1",
        label=temperature,
        role="treatment",
        material="oxide",
        variables={"calcination_temperature": temperature},
        source_quote=chunk.text,
        chunk_id=chunk.chunk_id,
        page_from=6,
        page_to=6,
        source_text_sha256=chunk.text_sha256,
    ).model_copy(update=updates)


def row(chunk, **updates):
    return ExperimentalMeasurement(
        measurement_id="measurement-1",
        group_id="group-1",
        document_id="doc-1",
        metric="initial specific capacity",
        value_text="223.3",
        numeric_value=223.3,
        unit="mAh/g",
        source_quote=chunk.text,
        chunk_id=chunk.chunk_id,
        page_from=6,
        page_to=6,
        source_text_sha256=chunk.text_sha256,
    ).model_copy(update=updates)


def normalize_and_sanitize(chunk, group, measurement, *, extra_chunks=()):
    chunks = {c.chunk_id: c for c in (chunk, *extra_chunks)}
    prepared = expand_measurement_group_evidence(
        (measurement,),
        groups=(group,),
        chunks_by_id=chunks,
    )
    accepted = sanitize_measurements(
        prepared,
        groups=(group,),
        chunks_by_id=chunks,
        warnings=[],
    )
    return prepared, accepted


def test_unique_explicit_capacity_sentence_is_narrowed_without_changing_data():
    chunk = source(TABLE + EXPLICIT)
    group, measurement = parent(chunk), row(chunk)
    before = (group.model_dump_json(), measurement.model_dump_json())
    prepared, accepted = normalize_and_sanitize(chunk, group, measurement)
    assert len(accepted) == 1
    assert accepted[0].source_quote == EXPLICIT
    assert prepared[0].source_quote == EXPLICIT
    assert accepted[0].model_copy(update={"source_quote": chunk.text}) == measurement
    assert before == (group.model_dump_json(), measurement.model_dump_json())
    assert accepted[0].source_quote in chunk.text
    assert accepted[0].review_status == "pending"


@pytest.mark.parametrize(
    "text,temperature,updates",
    [
        (TABLE + EXPLICIT, "100 ◦C", {}),
        (TABLE + EXPLICIT, "150 ◦F", {}),
        (TABLE + EXPLICIT, "150 ◦C", {"value_text": "22.3", "numeric_value": 22.3}),
        (TABLE + EXPLICIT, "150 ◦C", {"unit": "Ah/g"}),
        (TABLE + EXPLICIT, "150 ◦C", {"metric": "charge transfer resistance"}),
        (TABLE + EXPLICIT, "150 ◦C", {"uncertainty_text": "± 0.4"}),
        (
            TABLE + "The initial capacities were 223.3 and 205 mAh/g, respectively.",
            "150 ◦C",
            {},
        ),
        (TABLE + EXPLICIT + " " + EXPLICIT, "150 ◦C", {}),
        (
            TABLE + EXPLICIT + " The sample obtained at 150 ◦C had an initial "
            "specific capacity of 225 mAh/g at a different rate.",
            "150 ◦C",
            {},
        ),
        (TABLE + "Previous studies reported: " + EXPLICIT, "150 ◦C", {}),
        (TABLE + "References\n" + EXPLICIT, "150 ◦C", {}),
        (TABLE + "not exact " + EXPLICIT, "150 ◦C", {"source_quote": TABLE + EXPLICIT}),
        (
            TABLE + "The 100 ◦C and 150 ◦C samples had a capacity of 223.3 mAh/g.",
            "150 ◦C",
            {},
        ),
        (
            TABLE
            + "The 100 ◦C sample had 223.3 mAh/g and the 150 ◦C sample had 225 mAh/g.",
            "150 ◦C",
            {},
        ),
        (
            TABLE
            + "The 150 ◦C sample had 225 mAh/g and 223.3 mAh/g at two different rates.",
            "150 ◦C",
            {},
        ),
        (TABLE + "The 150 ◦C sample had 223.3 MAh/g.", "150 ◦C", {}),
        (
            TABLE
            + (
                "The sample was prepared at 150 ◦C. "
                "It had an initial capacity of 223.3 mAh/g."
            ),
            "150 ◦C",
            {},
        ),
        (
            TABLE
            + (
                "The 150 ◦C sample had a discharge capacity "
                "of 223.3 mAh/g after 50 cycles."
            ),
            "150 ◦C",
            {},
        ),
    ],
    ids=[
        "wrong-sample",
        "wrong-scale",
        "wrong-value",
        "wrong-unit",
        "wrong-metric",
        "missing-uncertainty",
        "unbound-list",
        "repeated-quote",
        "different-condition",
        "prior-work",
        "reference-section",
        "unlocated-original-quote",
        "shared-samples",
        "wrong-pair-in-one-sentence",
        "multiple-quantities",
        "wrong-unit-case",
        "cross-sentence-attribution",
        "wrong-capacity-baseline",
    ],
)
def test_unsafe_or_ambiguous_capacity_evidence_is_not_rescued(
    text, temperature, updates
):
    chunk = source(text)
    measurement = row(chunk, **updates)
    prepared, accepted = normalize_and_sanitize(
        chunk, parent(chunk, temperature), measurement
    )
    assert not accepted
    assert prepared[0].source_quote == measurement.source_quote


def test_all_group_variables_must_be_in_the_same_explicit_sentence():
    chunk = source(TABLE + "Samples used 1 % doping. " + EXPLICIT)
    group = parent(chunk, variables={"temperature": "150 ◦C", "doping": "1 %"})
    prepared, accepted = normalize_and_sanitize(chunk, group, row(chunk))
    assert not accepted
    assert prepared[0].source_quote == chunk.text


def test_unknown_or_nonverbatim_source_and_existing_short_quotes_stay_unchanged():
    chunk = source(TABLE + EXPLICIT)
    group = parent(chunk)
    for measurement, parents, chunks in (
        (row(chunk), (), {chunk.chunk_id: chunk}),
        (row(chunk), (group,), {}),
        (row(chunk, source_quote=EXPLICIT), (group,), {chunk.chunk_id: chunk}),
        (
            row(chunk, source_quote="150 ◦C had 223.3 mAh/g."),
            (group,),
            {chunk.chunk_id: chunk},
        ),
        (row(chunk, source_text_sha256="a" * 64), (group,), {chunk.chunk_id: chunk}),
        (row(chunk, page_from=5), (group,), {chunk.chunk_id: chunk}),
    ):
        result = expand_measurement_group_evidence(
            (measurement,),
            groups=parents,
            chunks_by_id=chunks,
        )
        assert result == (measurement,)


def test_reference_continuation_cannot_be_narrowed_to_own_measurement():
    chunk = source(TABLE + EXPLICIT)
    header = replace(
        source("References\n"), chunk_id="references", page_from=5, page_to=5
    )
    prepared, accepted = normalize_and_sanitize(
        chunk, parent(chunk), row(chunk), extra_chunks=(header,)
    )
    assert not accepted
    assert prepared[0].source_quote == chunk.text


def test_missing_cycle_capacity_is_not_added_and_explicit_initial_capacity_hands_off():
    """Fake candidate input, not a replay or an additional real model request."""
    chunk = source(
        TABLE + EXPLICIT + " More importantly, the discharge capacity retains "
        "194.5 mAh/g after 50 cycles."
    )
    batch = MatrixExtractionBatch(
        groups=(
            GroupCandidate(
                group_key="150C",
                label="150°C",
                role="treatment",
                material="oxide",
                variables={"calcination_temperature": "150°C"},
                source_quote=chunk.text,
                chunk_id=chunk.chunk_id,
            ),
        ),
        measurements=(
            MeasurementCandidate(
                group_key="150C",
                metric="initial specific capacity",
                value_text="223.3",
                numeric_value=223.3,
                unit="mAh/g",
                source_quote=chunk.text,
                chunk_id=chunk.chunk_id,
            ),
        ),
    )
    before = batch.model_dump_json()

    class MemoryStore:
        def get_chunks(self, ids):
            return [chunk] if chunk.chunk_id in ids else []

    result = AutomatedMatrixExtractor(FakeLlm(batch), MemoryStore()).extract(
        document_id=chunk.document_id,
        chunks=(chunk,),
    )
    assert len(result.measurements) == 1
    assert result.measurements[0].source_quote == EXPLICIT
    assert result.measurements[0].value_text == "223.3"
    assert batch.model_dump_json() == before
    audit = next(
        warning for warning in result.warnings if "quote scope narrowed" in warning
    )
    assert hashlib.sha256(chunk.text.encode()).hexdigest() in audit
    assert hashlib.sha256(EXPLICIT.encode()).hexdigest() in audit
    assert "source chunk chunk-1 unchanged" in audit

    class MemoryMatrices:
        def load_matrix(self, document_id, *, status="approved"):
            return (
                (list(result.groups), list(result.measurements), [], [], [])
                if status == "pending"
                else ([], [], [], [], [])
            )

        def get_document_chunks(self, document_id):
            return [chunk]

    class MemoryDatasets:
        records = None

        def register_records(self, records, **kwargs):
            self.records = records
            return SimpleNamespace(dataset_id="memory-only-dataset")

    datasets = MemoryDatasets()
    handoff = LiteratureEvidenceTrialService(
        report_store=_ReportStore(_report()),
        matrix_store=MemoryMatrices(),
        dataset_store=datasets,
    ).prepare(topic=_report().topic)
    assert handoff.record_count == 1
    assert len(datasets.records) == 1
    assert datasets.records[0]["value_text"] == "223.3"
    assert datasets.records[0]["review_status"] == "pending"


def test_other_capacity_unit_is_matched_without_scale_conversion():
    explicit = EXPLICIT.replace("223.3 mAh/g", "0.2233 Ah/g")
    chunk = source(TABLE + explicit)
    prepared, accepted = normalize_and_sanitize(
        chunk,
        parent(chunk),
        row(chunk, value_text="0.2233", numeric_value=0.2233, unit="Ah/g"),
    )
    assert len(accepted) == 1
    assert prepared[0].source_quote == explicit
    assert prepared[0].numeric_value == 0.2233 and prepared[0].unit == "Ah/g"
