"""Extraction and pending handoff must agree without altering source records."""

import hashlib
from types import SimpleNamespace

import pytest

from materials_screening.master.literature_evidence_trial import (
    LiteratureEvidenceTrialService,
)
from materials_screening.sub_agents.literature.matrix_automation import (
    sanitize_measurements,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord
from tests.unit.master.test_literature_evidence_trial import (
    _MatrixStore,
    _report,
    _ReportStore,
)
from tests.unit.sub_agents.test_fulltext_evidence_regressions import (
    TABLE,
    group,
    measurement,
)

CAPACITY = (
    "Among these samples, the sample prepared at 150 C delivered a discharge "
    "capacity of 194.5 mAh/g after 50 cycles."
)
SURFACE_TABLE = "Table 1. Surface area.\nSample\nSBET\n100 C\n10\n150 C\n20\n"


@pytest.mark.parametrize(
    "text,quote,label,value,metric,unit,expected,reference_header",
    [
        (
            SURFACE_TABLE + CAPACITY,
            CAPACITY,
            "150 C",
            "194.5",
            "discharge capacity",
            "mAh/g",
            1,
            False,
        ),
        (
            TABLE,
            TABLE,
            "10% ZnO/g-C3N4",
            "63.2",
            "photocatalytic degradation efficiency",
            "%",
            1,
            False,
        ),
        (
            TABLE,
            "10% ZnO/g-C3N4\n63.2%",
            "20% ZnO/g-C3N4",
            "63.2",
            "photocatalytic degradation efficiency",
            "%",
            0,
            False,
        ),
        (
            TABLE,
            TABLE,
            "ZnO/g-C3N4 composite",
            "63.2",
            "photocatalytic degradation efficiency",
            "%",
            0,
            False,
        ),
        (
            SURFACE_TABLE + CAPACITY,
            "194.5 mAh/g",
            "150 C",
            "194.5",
            "discharge capacity",
            "mAh/g",
            0,
            False,
        ),
        (
            SURFACE_TABLE + CAPACITY,
            CAPACITY,
            "100 C",
            "194.5",
            "discharge capacity",
            "mAh/g",
            0,
            False,
        ),
        (
            "References\n" + CAPACITY,
            CAPACITY,
            "150 C",
            "194.5",
            "discharge capacity",
            "mAh/g",
            0,
            False,
        ),
        (CAPACITY, CAPACITY, "150 C", "194.5", "discharge capacity", "mAh/g", 0, True),
        (
            "Previous studies reported: " + CAPACITY,
            "Previous studies reported: " + CAPACITY,
            "150 C",
            "194.5",
            "discharge capacity",
            "mAh/g",
            0,
            False,
        ),
    ],
    ids=[
        "unrelated-table",
        "exact-table-row",
        "wrong-table-row",
        "broad-label",
        "bare-number",
        "wrong-prose-sample",
        "reference-section",
        "reference-continuation",
        "prior-work",
    ],
)
def test_extraction_and_handoff_share_binding_rules(
    text, quote, label, value, metric, unit, expected, reference_header, variables=None
) -> None:
    digest = hashlib.sha256(text.encode()).hexdigest()
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        text=text,
        page_from=9,
        page_to=9,
        text_sha256=digest,
    )
    chunks = [chunk]
    if reference_header:
        chunks.insert(
            0,
            ChunkRecord(
                chunk_id="ref-header",
                document_id="doc-1",
                paper_id=None,
                text="References\n",
                page_from=8,
                page_to=8,
                text_sha256=hashlib.sha256(b"References\n").hexdigest(),
            ),
        )
    parent = group(label, variables).model_copy(
        update={"source_quote": quote, "source_text_sha256": digest}
    )
    row = measurement(value, quote).model_copy(
        update={
            "metric": metric,
            "unit": unit,
            "source_text_sha256": digest,
        }
    )
    original = (parent.model_dump_json(), row.model_dump_json())
    accepted = sanitize_measurements(
        [row],
        groups=[parent],
        warnings=[],
        chunks_by_id={item.chunk_id: item for item in chunks},
    )
    assert len(accepted) == expected

    class MatrixStore(_MatrixStore):
        def get_document_chunks(self, document_id):
            assert document_id == "doc-1"
            return chunks

    class MemoryDatasets:
        records = None

        def register_records(self, records, **kwargs):
            self.records = records
            return SimpleNamespace(dataset_id="memory-only-dataset")

    datasets = MemoryDatasets()
    handoff = LiteratureEvidenceTrialService(
        report_store=_ReportStore(_report()),
        matrix_store=MatrixStore([parent], [row]),
        dataset_store=datasets,
    ).prepare(topic="任务5")
    assert handoff.record_count == expected
    assert handoff.dataset_id == ("memory-only-dataset" if expected else None)
    if not expected:
        assert datasets.records is None
        assert any("已隔离" in warning for warning in handoff.warnings)
    assert (parent.model_dump_json(), row.model_dump_json()) == original


@pytest.mark.parametrize("variable,expected", [("9", 1), ("9.0", 0), ("8", 0)])
def test_handoff_does_not_relax_verbatim_group_variables(variable, expected):
    quote = "pH 9 produced rods 60 nm long."
    test_extraction_and_handoff_share_binding_rules(
        quote,
        quote,
        "pH 9",
        "60",
        "length",
        "nm",
        expected,
        False,
        variables={"pH": variable},
    )
