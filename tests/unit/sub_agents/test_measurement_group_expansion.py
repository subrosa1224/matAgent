"""Group co-occurrence cannot create a new sample/value assignment."""

import hashlib
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
)
from materials_screening.sub_agents.literature.models import (
    ExperimentalGroup,
    ExperimentalMeasurement,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord

SURFACE = (
    "The specific surface areas of samples calcined at 100 ◦C, 150 ◦C, "
    "250 ◦C and 350 ◦C are 82.11 m2·g−1, 194.48 m2·g−1, "
    "159.30 m2·g−1 and 103.42 m2·g−1, respectively."
)
PORES = (
    "Furthermore, the corresponding desorption average pore diameter of "
    "these samples are 8.46 nm, 3.27 nm, 3.61 nm and 8.28 nm, respectively."
)
RESISTANCE = (
    "The Rct value of electrodes calcined at 100 ◦C and 350 ◦C are "
    "152.8 Ω and 121.7 Ω, respectively. The values are higher than electrodes "
    "calcined at 150 ◦C and 250 ◦C, i.e., 94.4 Ω and 94.59 Ω, respectively."
)
TEMPERATURES = ("100", "150", "250", "350")
EXPECTED = {
    "specific surface area": ("82.11", "194.48", "159.30", "103.42"),
    "pore diameter": ("8.46", "3.27", "3.61", "8.28"),
    "charge transfer resistance (Rct)": ("152.8", "94.4", "94.59", "121.7"),
}


def chunk(text, *, page=5):
    return ChunkRecord(
        chunk_id=f"chunk-{page}",
        document_id="doc-1",
        paper_id=None,
        page_from=page,
        page_to=page,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def groups(source):
    return tuple(
        ExperimentalGroup(
            group_id=f"group-{temperature}",
            document_id="doc-1",
            label=f"{temperature}°C",
            role="treatment",
            material="LiFeO2",
            variables={"calcination_temperature": f"{temperature} ◦C"},
            source_quote=source.text,
            chunk_id=source.chunk_id,
            page_from=source.page_from,
            page_to=source.page_to,
            source_text_sha256=source.text_sha256,
        )
        for temperature in TEMPERATURES
    )


def measurement(
    source, *, quote=None, value="82.11", metric="specific surface area", unit="m2·g−1"
):
    return ExperimentalMeasurement(
        measurement_id="measurement-original",
        document_id="doc-1",
        group_id="group-100",
        metric=metric,
        value_text=value,
        numeric_value=float(value),
        unit=unit,
        source_quote=quote or source.text,
        chunk_id=source.chunk_id,
        page_from=source.page_from,
        page_to=source.page_to,
        source_text_sha256=source.text_sha256,
    )


@pytest.mark.parametrize(
    "quote,value,metric,unit",
    [
        (SURFACE, "82.11", "specific surface area", "m2·g−1"),
        (RESISTANCE, "152.8", "charge transfer resistance (Rct)", "Ω"),
        (
            "100 ◦C and 150 ◦C samples were both tested after 7 days.",
            "7",
            "measurement time",
            "days",
        ),
        (
            "100 ◦C and 150 ◦C samples both reached 10 MPa.",
            "10",
            "strength",
            "MPa",
        ),
        (
            "Sample A at 100 ◦C reached 10 MPa; Sample B at 150 ◦C reached 20 MPa.",
            "10",
            "strength",
            "MPa",
        ),
    ],
    ids=[
        "surface-respectively",
        "resistance-respectively",
        "shared-time",
        "shared-value",
        "different-values",
    ],
)
def test_cooccurring_groups_never_clone_an_existing_measurement(
    quote, value, metric, unit
):
    source = chunk(quote)
    row = measurement(source, value=value, metric=metric, unit=unit)
    before = row.model_dump_json()
    parents = groups(source)
    parent_before = [parent.model_dump_json() for parent in parents]
    result = expand_measurement_group_evidence(
        (row,),
        groups=parents,
        chunks_by_id={source.chunk_id: source},
    )
    assert len(result) == 1
    assert result[0].group_id == row.group_id
    assert result[0].measurement_id == row.measurement_id
    assert row.model_dump_json() == before
    assert [parent.model_dump_json() for parent in parents] == parent_before


def test_context_expansion_preserves_the_original_sample_and_value():
    source = chunk(SURFACE + " " + PORES)
    row = measurement(source, quote="82.11 m2·g−1")
    result = expand_measurement_group_evidence(
        (row,),
        groups=groups(source),
        chunks_by_id={source.chunk_id: source},
    )
    assert len(result) == 1
    assert result[0].group_id == "group-100"
    assert result[0].source_quote == source.text
    assert result[0].value_text == "82.11"
    assert result[0].unit == row.unit
    assert result[0].source_text_sha256 == source.text_sha256
    assert row.source_quote == "82.11 m2·g−1"


def test_missing_group_or_chunk_does_not_create_or_modify_assignments():
    source = chunk(SURFACE)
    row = measurement(source)
    for parents, sources in (
        ((), {}),
        (groups(source), {}),
        ((), {source.chunk_id: source}),
    ):
        assert expand_measurement_group_evidence(
            (row,),
            groups=parents,
            chunks_by_id=sources,
        ) == (row,)


def test_explicit_shared_measurements_remain_separate_records():
    source = chunk("100 ◦C and 150 ◦C samples were both tested after 7 days.")
    first = measurement(source, value="7", metric="measurement time", unit="days")
    second = first.model_copy(
        update={
            "measurement_id": "measurement-second",
            "group_id": "group-150",
        }
    )
    result = expand_measurement_group_evidence(
        (first, second),
        groups=groups(source),
        chunks_by_id={source.chunk_id: source},
    )
    assert result == (first, second)


def test_source_shaped_fixture_extracts_and_hands_off_twelve_exact_pairs():
    """Local candidate fixture, not replay of the unsaved live model response."""
    surface = chunk(SURFACE + " " + PORES)
    resistance = chunk(RESISTANCE, page=9)
    sources = {source.chunk_id: source for source in (surface, resistance)}
    batch = MatrixExtractionBatch(
        groups=tuple(
            GroupCandidate(
                group_key=temperature,
                label=f"{temperature}°C",
                role="treatment",
                material="LiFeO2",
                variables={"calcination_temperature": f"{temperature}°C"},
                source_quote=surface.text,
                chunk_id=surface.chunk_id,
            )
            for temperature in TEMPERATURES
        ),
        measurements=tuple(
            MeasurementCandidate(
                group_key=temperature,
                metric=metric,
                value_text=value,
                numeric_value=float(value),
                unit=unit,
                source_quote=quote,
                chunk_id=source.chunk_id,
            )
            for metric, unit, quote, source in (
                ("specific surface area", "m2·g−1", SURFACE, surface),
                ("pore diameter", "nm", PORES, surface),
                ("charge transfer resistance (Rct)", "Ω", RESISTANCE, resistance),
            )
            for temperature, value in zip(TEMPERATURES, EXPECTED[metric], strict=True)
        ),
    )

    class MemoryStore:
        def get_chunks(self, ids):
            return [sources[key] for key in ids if key in sources]

    result = AutomatedMatrixExtractor(FakeLlm(batch), MemoryStore()).extract(
        document_id="doc-1",
        chunks=tuple(sources.values()),
    )
    assert result.diagnostics.before_validation_measurements == 12
    assert len(result.measurements) == 12
    labels = {parent.group_id: parent.label for parent in result.groups}
    expected = {
        (f"{temperature}°C", metric, value)
        for metric, values in EXPECTED.items()
        for temperature, value in zip(TEMPERATURES, values, strict=True)
    }
    assert {
        (labels[row.group_id], row.metric, row.value_text)
        for row in result.measurements
    } == expected
    for row in result.measurements:
        source = sources[row.chunk_id]
        assert row.source_text_sha256 == source.text_sha256
        assert row.source_quote in source.text
        assert row.review_status == "pending"

    class MemoryMatrices:
        def load_matrix(self, document_id, *, status="approved"):
            assert document_id == "doc-1"
            return (
                (list(result.groups), list(result.measurements), [], [], [])
                if status == "pending"
                else ([], [], [], [], [])
            )

        def get_document_chunks(self, document_id):
            assert document_id == "doc-1"
            return list(sources.values())

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
    assert handoff.record_count == 12
    assert handoff.dataset_id == "memory-only-dataset"
    assert not any("冲突" in warning for warning in handoff.warnings)
    assert {
        (record["group_label"], record["metric"], str(record["value_text"]))
        for record in datasets.records
    } == expected
