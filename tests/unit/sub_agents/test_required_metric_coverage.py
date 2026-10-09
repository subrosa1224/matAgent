"""Specified metric coverage is separate from JSON success and expert review."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from tests.unit.sub_agents.test_literature_matrix_automation import (
    FakeLlm,
    Store,
    _chunk,
)
from tests.unit.sub_agents.test_matrix_candidate_diagnostics import candidate, group

from materials_screening.sub_agents.literature.matrix_automation import (
    AutomatedMatrixExtractor,
    MatrixExtractionBatch,
    MatrixExtractionDiagnostics,
)
from materials_screening.sub_agents.literature.metric_coverage import (
    RequiredMetric,
    check_required_metric_coverage,
)
from materials_screening.sub_agents.literature.models import ExperimentalMeasurement


def measurement(**updates):
    return ExperimentalMeasurement(
        measurement_id="m-1",
        group_id="g-1",
        document_id="doc-1",
        metric="discharge capacity after 50 cycles",
        value_text="100",
        numeric_value=100,
        unit="mAh/g",
        source_quote="DO NOT LOG THIS QUOTE",
        chunk_id="chunk-1",
        page_from=1,
        page_to=1,
        source_text_sha256="a" * 64,
        review_status="pending",
    ).model_copy(update=updates)


def requirement(**updates):
    return RequiredMetric(metric="discharge capacity after 50 cycles", **updates)


def extract(*, row=None, requirements=()):
    batch = MatrixExtractionBatch(
        groups=(group(),), measurements=(candidate() if row is None else row,)
    )
    return AutomatedMatrixExtractor(FakeLlm(batch), Store()).extract(
        document_id="doc-1", chunks=(_chunk(),), required_metrics=requirements
    )


def test_no_requirements_is_not_checked_even_with_many_rows():
    result = check_required_metric_coverage([measurement()] * 30)
    assert result.status == "not_checked"
    assert result.checks == ()


def test_missing_metric_is_incomplete_not_paper_absence():
    result = check_required_metric_coverage([], [requirement()])
    assert result.status == "incomplete"
    assert result.checks[0].status == "missing"
    assert result.checks[0].measurement_ids == ()
    assert result.scope == "specified_metrics_in_validated_measurements"


@pytest.mark.parametrize(
    "wrong_metric",
    [
        "initial discharge capacity",
        "discharge capacity",
        "discharge capacity after 5 cycles",
        "discharge capacity after 100 cycles",
        "capacity retention after 50 cycles",
        "discharge capacity after 50 cycles extra",
    ],
)
def test_initial_capacity_retention_and_other_cycles_do_not_fill_requirement(
    wrong_metric,
):
    result = check_required_metric_coverage(
        [measurement(metric=wrong_metric)], [requirement()]
    )
    assert result.status == "incomplete"


def test_only_explicit_aliases_match_and_no_values_are_needed():
    row = measurement(metric="specific discharge capacity at cycle 50")
    assert check_required_metric_coverage([row], [requirement()]).status == "incomplete"
    result = check_required_metric_coverage(
        [row], [requirement(aliases=(row.metric,), unit="mAh/g")]
    )
    assert result.status == "covered"
    assert result.checks[0].measurement_ids == ("m-1",)
    assert "100" not in result.model_dump_json()
    assert row.source_quote not in result.model_dump_json()
    assert row.review_status == "pending"  # covered does not mean expert approved


def test_case_and_whitespace_only_normalization_does_not_mutate_rows():
    row = measurement(metric="  Discharge  capacity AFTER\n50 cycles  ")
    before = row.model_dump()
    result = check_required_metric_coverage([row], [requirement()])
    assert result.status == "covered"
    assert row.model_dump() == before


@pytest.mark.parametrize("unit", ["Ah/g", "mah/g", "mAh g-1", "%", None])
def test_units_are_not_converted_or_case_folded(unit):
    result = check_required_metric_coverage(
        [measurement(unit=unit)], [requirement(unit="mAh/g")]
    )
    assert result.status == "incomplete"


def test_group_requirement_does_not_use_other_sample():
    result = check_required_metric_coverage(
        [measurement()], [requirement(group_id="g-2"), requirement(group_id="g-1")]
    )
    assert result.status == "incomplete"
    assert [check.status for check in result.checks] == ["missing", "present"]


def test_unspecified_group_means_at_least_one_row_not_all_samples():
    result = check_required_metric_coverage([measurement()], [requirement()])
    assert result.status == "covered"
    assert result.checks[0].requirement.group_id is None


def test_all_matching_measurement_ids_are_preserved_in_input_order():
    result = check_required_metric_coverage(
        [measurement(), measurement(measurement_id="m-2", group_id="g-2")],
        [requirement()],
    )
    assert result.checks[0].measurement_ids == ("m-1", "m-2")


@pytest.mark.parametrize(
    "fields",
    [
        {"metric": ""},
        {"metric": " \n "},
        {"metric": "x", "aliases": (" ",)},
        {"metric": "x", "unit": " "},
        {"metric": "x", "group_id": " "},
        {"metric": "x" * 121},
        {"metric": "x", "aliases": ("y" * 121,)},
        {"metric": "x", "aliases": ("alias",) * 21},
        {"metric": "x", "expected_value": 194.5},
    ],
)
def test_invalid_or_answer_leaking_requirements_fail_explicitly(fields):
    with pytest.raises(ValidationError):
        RequiredMetric(**fields)


def test_extractor_checks_final_measurements_without_changing_extraction_or_prompt():
    old = extract()
    checked = extract(requirements=(RequiredMetric(metric="compressive strength"),))
    assert checked.diagnostics.required_metric_coverage.status == "covered"
    assert replace(checked, diagnostics=old.diagnostics) == old
    assert checked.diagnostics.required_metric_coverage.checks[0].measurement_ids == (
        checked.measurements[0].measurement_id,
    )


@pytest.mark.parametrize("row", [candidate(group_key="unknown"), candidate(unit="GPa")])
def test_rejected_candidate_is_missing_even_if_its_metric_matches(row):
    result = extract(row=row, requirements=(RequiredMetric(metric=row.metric),))
    assert result.measurements == ()
    assert result.diagnostics.required_metric_coverage.status == "incomplete"
    assert any("REQUIRED_METRIC_COVERAGE_INCOMPLETE" in w for w in result.warnings)


def test_constructed_candidate_removed_by_final_binding_gate_is_missing():
    result = extract(
        row=candidate(value_text="10.0", numeric_value=10.0),
        requirements=(RequiredMetric(metric="compressive strength"),),
    )
    assert result.diagnostics.measurement_candidates[0].status == "constructed"
    assert result.measurements == ()  # Sample A value incorrectly attributed to B
    assert result.diagnostics.required_metric_coverage.status == "incomplete"


def test_legacy_diagnostics_default_to_not_checked_and_round_trip():
    old = MatrixExtractionDiagnostics.model_validate({"accepted_measurements": 12})
    assert old.required_metric_coverage.status == "not_checked"
    current = extract(
        requirements=(RequiredMetric(metric="missing metric"),)
    ).diagnostics
    assert (
        MatrixExtractionDiagnostics.model_validate_json(current.model_dump_json())
        == current
    )
    saved = json.loads(current.model_dump_json())
    assert saved["required_metric_coverage"]["checks"][0]["status"] == "missing"


def test_empty_document_cannot_be_covered():
    result = AutomatedMatrixExtractor(
        FakeLlm(MatrixExtractionBatch()), Store()
    ).extract(document_id="doc-1", chunks=(), required_metrics=(requirement(),))
    assert result.diagnostics.required_metric_coverage.status == "incomplete"
    assert result.diagnostics.attempted_batches == 0


def test_requirements_do_not_change_model_input_or_add_calls():
    class RecordingLlm(FakeLlm):
        def __init__(self, batch):
            super().__init__(batch)
            self.requests = []

        def generate_structured(self, **kwargs):
            self.requests.append(kwargs)
            return super().generate_structured(**kwargs)

    batch = MatrixExtractionBatch(groups=(group(),), measurements=(candidate(),))
    plain_llm, checked_llm = RecordingLlm(batch), RecordingLlm(batch)
    AutomatedMatrixExtractor(plain_llm, Store()).extract(
        document_id="doc-1", chunks=(_chunk(),)
    )
    result = AutomatedMatrixExtractor(checked_llm, Store()).extract(
        document_id="doc-1",
        chunks=(_chunk(),),
        required_metrics=(requirement(),),
    )
    assert checked_llm.requests == plain_llm.requests
    assert len(checked_llm.requests) == 1
    assert result.diagnostics.required_metric_coverage.status == "incomplete"


def test_cli_persists_missing_requirements_even_when_warnings_are_hidden(
    monkeypatch, tmp_path: Path
):
    from materials_screening import cli

    result = extract(requirements=(requirement(),))
    monkeypatch.chdir(tmp_path)
    store = SimpleNamespace(
        get_document_chunks=lambda _: [_chunk()],
        save_experiment_matrix=lambda *args: None,
        save_comparisons=lambda *args: None,
        save_claims_and_links=lambda *args: None,
    )
    settings = SimpleNamespace(
        llm_provider="intern", intern_model="fake", llm_max_output_tokens=8192
    )
    settings.model_copy = lambda **kwargs: settings
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(cli, "_literature_pgvector_store", lambda: store)
    monkeypatch.setattr(cli, "create_llm_provider", lambda _: object())
    monkeypatch.setattr(
        AutomatedMatrixExtractor, "extract", lambda *args, **kwargs: result
    )
    cli.literature_matrix_extract_command("doc-1", show_warnings=False)
    files = list((tmp_path / "data/literature_extractions").glob("*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    coverage = saved["diagnostics"]["required_metric_coverage"]
    assert coverage["status"] == "incomplete"
    assert coverage["checks"][0]["requirement"]["metric"] == requirement().metric
    assert coverage["checks"][0]["measurement_ids"] == []
    assert saved["warnings"] == list(result.warnings)
