"""Observe parsed candidates without altering construction or evidence gates."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.unit.sub_agents.test_literature_matrix_automation import (
    FakeLlm,
    Store,
    _chunk,
)

from materials_screening.llm.errors import LLMStructuredOutputError
from materials_screening.sub_agents.literature.matrix_automation import (
    AutomatedMatrixExtractor,
    GroupCandidate,
    MatrixExtractionBatch,
    MatrixExtractionDiagnostics,
    MeasurementCandidate,
)


def group(**updates):
    return GroupCandidate(
        group_key="sample-b",
        label="Sample B",
        material="Nb sample",
        variables={"Nb": "1.5 mol%"},
        source_quote=_chunk().text,
        chunk_id="chunk-1",
    ).model_copy(update=updates)


def candidate(**updates):
    return MeasurementCandidate(
        group_key="sample-b",
        metric="compressive strength",
        value_text="12.0",
        numeric_value=12.0,
        unit="MPa",
        source_quote=_chunk().text,
        chunk_id="chunk-1",
    ).model_copy(update=updates)


def extract(row, *, groups=None):
    batch = MatrixExtractionBatch(
        groups=(group(),) if groups is None else groups, measurements=(row,)
    )
    original = batch.model_dump()
    result = AutomatedMatrixExtractor(FakeLlm(batch), Store()).extract(
        document_id="doc-1", chunks=(_chunk(),)
    )
    assert batch.model_dump() == original
    assert result.diagnostics.llm_measurements == 1
    assert len(result.diagnostics.measurement_candidates) == 1
    return result, result.diagnostics.measurement_candidates[0]


@pytest.mark.parametrize(
    "updates,reasons,warning",
    [
        (
            {"group_key": "unknown"},
            ("unknown_group_key",),
            "unknown group or quote",
        ),
        (
            {"value_text": "999.99", "numeric_value": 999.99},
            ("unlocated_measurement_evidence",),
            "unknown group or quote",
        ),
        (
            {"unit": "mAh/g"},
            ("unlocated_measurement_evidence",),
            "unknown group or quote",
        ),
        (
            {"group_key": "unknown", "value_text": "999.99"},
            ("unknown_group_key", "unlocated_measurement_evidence"),
            "unknown group or quote",
        ),
        (
            {"numeric_value": 99.0},
            ("numeric_value_mismatch",),
            "numeric value mismatch",
        ),
    ],
)
def test_preconstruction_rejection_keeps_value_and_exact_failure(
    updates, reasons, warning
):
    row = candidate(**updates)
    result, diagnostic = extract(row)
    assert result.measurements == ()
    assert diagnostic.status == "rejected"
    assert diagnostic.rejection_reasons == reasons
    assert diagnostic.value_text == row.value_text
    assert diagnostic.numeric_value == row.numeric_value
    assert diagnostic.unit == row.unit
    assert diagnostic.group_key == row.group_key
    assert diagnostic.chunk_id == row.chunk_id
    assert (
        diagnostic.source_quote_sha256
        == hashlib.sha256(row.source_quote.encode()).hexdigest()
    )
    assert result.warnings == (f"measurement compressive strength rejected: {warning}",)


def test_constructed_is_not_a_claim_of_final_gate_acceptance():
    # Grounded numbers, but a quote naming Sample A does not bind Sample B.
    row = candidate(value_text="10.0", numeric_value=10.0)
    result, diagnostic = extract(row)
    assert diagnostic.status == "constructed"
    assert diagnostic.rejection_reasons == ()
    assert result.diagnostics.before_validation_measurements == 1
    assert result.measurements == ()
    assert any("rejected:" in warning for warning in result.warnings)


def test_constructed_summary_preserves_final_pending_row_without_exposing_quote():
    result, diagnostic = extract(candidate())
    assert len(result.measurements) == 1
    assert result.measurements[0].review_status == "pending"
    assert diagnostic.status == "constructed"
    assert diagnostic.batch_index == diagnostic.candidate_index == 1
    assert diagnostic.resolved_group_id == result.measurements[0].group_id
    assert diagnostic.resolved_chunk_id == "chunk-1"
    assert diagnostic.resolved_quote_sha256 == diagnostic.source_quote_sha256
    assert diagnostic.source_quote_chars == len(_chunk().text)
    assert _chunk().text not in result.diagnostics.model_dump_json()
    assert "source_quote" not in diagnostic.model_dump()
    assert diagnostic.truncated_fields == ()


@pytest.mark.parametrize("grounded", [True, False])
def test_missing_group_and_evidence_are_independently_observable(grounded):
    row = candidate(group_key=None, **({} if grounded else {"value_text": "999.99"}))
    result, diagnostic = extract(row, groups=())
    reasons = ("missing_or_ambiguous_group_key",)
    if not grounded:
        reasons += ("unlocated_measurement_evidence",)
    assert diagnostic.rejection_reasons == reasons
    assert diagnostic.resolved_group_id is None
    assert result.measurements == ()
    assert "missing or ambiguous group_key" in result.warnings[0]


def test_fallback_evidence_location_is_observed_without_changing_it():
    row = candidate(chunk_id="missing-chunk", source_quote="no such quote")
    result, diagnostic = extract(row)
    assert len(result.measurements) == 1
    assert diagnostic.chunk_id == "missing-chunk"
    assert diagnostic.resolved_chunk_id == "chunk-1"
    assert diagnostic.resolved_quote_sha256 == _chunk().text_sha256
    assert diagnostic.resolved_quote_sha256 != diagnostic.source_quote_sha256


def test_long_candidate_fields_are_bounded_and_quote_content_is_never_logged():
    secret = "quote-content-must-not-be-logged"
    row = candidate(
        metric="x" * 5000,
        value_text="y" * 5000,
        unit="z" * 5000,
        group_key="g" * 5000,
        chunk_id="c" * 5000,
        source_quote=secret * 500,
    )
    _, diagnostic = extract(row)
    assert set(diagnostic.truncated_fields) == {
        "metric",
        "value_text",
        "unit",
        "group_key",
        "chunk_id",
    }
    for name in diagnostic.truncated_fields:
        assert len(getattr(diagnostic, name)) == 120
    assert secret not in diagnostic.model_dump_json()
    assert diagnostic.source_quote_chars == len(row.source_quote)


def test_every_parsed_candidate_is_recorded_even_above_twenty():
    batch = MatrixExtractionBatch(
        measurements=tuple(candidate(group_key="unknown") for _ in range(200))
    )
    result = AutomatedMatrixExtractor(FakeLlm(batch), Store()).extract(
        document_id="doc-1", chunks=(_chunk(),)
    )
    diagnostics = result.diagnostics.measurement_candidates
    assert len(diagnostics) == result.diagnostics.llm_measurements == 200
    assert [d.candidate_index for d in diagnostics] == list(range(1, 201))
    assert all(d.batch_index == 1 for d in diagnostics)
    assert result.measurements == ()


def test_successful_later_batch_keeps_its_actual_batch_number():
    class FailFirstBatch(FakeLlm):
        calls = 0

        def generate_structured(self, **kwargs):
            self.calls += 1
            if self.calls <= 2:
                raise LLMStructuredOutputError("Intern response exceeded max_tokens")
            return super().generate_structured(**kwargs)

    second = replace(_chunk(), chunk_id="chunk-2", page_from=3, page_to=3)
    llm = FailFirstBatch(
        MatrixExtractionBatch(measurements=(candidate(group_key="unknown"),))
    )
    result = AutomatedMatrixExtractor(llm, Store()).extract(
        document_id="doc-1", chunks=(_chunk(), second), batch_chars=1
    )
    assert llm.calls == 3
    assert len(result.diagnostics.measurement_candidates) == 1
    assert result.diagnostics.measurement_candidates[0].batch_index == 2
    assert result.diagnostics.measurement_candidates[0].candidate_index == 1


def test_old_diagnostic_json_defaults_to_no_candidate_observations():
    old = MatrixExtractionDiagnostics.model_validate({"llm_measurements": 20})
    assert old.measurement_candidates == ()
    assert old.llm_measurements == 20
    restored = MatrixExtractionDiagnostics.model_validate_json(old.model_dump_json())
    assert restored == old


def test_diagnostic_addition_does_not_change_the_model_output_schema():
    digest = hashlib.sha256(
        json.dumps(
            MatrixExtractionBatch.model_json_schema(), ensure_ascii=False
        ).encode()
    ).hexdigest()
    # Schema fingerprint from the existing v7/v8 extraction contract.
    assert digest == "2018b84094d7cb6f11bc3c1338932d38d7c5402dd464eca86ffc5b703a6828e3"


def test_recovered_response_is_observed_once_without_an_extra_model_call():
    class Recover(FakeLlm):
        calls = 0

        def generate_structured(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise LLMStructuredOutputError("Intern response exceeded max_tokens")
            return super().generate_structured(**kwargs)

    llm = Recover(MatrixExtractionBatch(groups=(group(),), measurements=(candidate(),)))
    result = AutomatedMatrixExtractor(llm, Store()).extract(
        document_id="doc-1", chunks=(_chunk(),)
    )
    assert llm.calls == 2
    assert len(result.diagnostics.measurement_candidates) == 1
    assert [a.status for a in result.diagnostics.batch_attempts] == ["error", "ok"]


def test_a_returned_194_5_candidate_is_visible_even_when_evidence_fails():
    row = candidate(
        metric="capacity retention after 50 cycles",
        value_text="194.5",
        numeric_value=194.5,
        unit="%",
    )
    result, diagnostic = extract(row)
    assert diagnostic.value_text == "194.5"
    assert diagnostic.rejection_reasons == ("unlocated_measurement_evidence",)
    assert result.measurements == ()


def test_an_omitted_capacity_is_not_fabricated_in_candidate_diagnostics():
    text = "Sample B retained a discharge capacity of 194.5 mAh/g after 50 cycles."
    chunk = replace(
        _chunk(), text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest()
    )
    result = AutomatedMatrixExtractor(
        FakeLlm(MatrixExtractionBatch()), Store()
    ).extract(document_id="doc-1", chunks=(chunk,))
    assert result.diagnostics.llm_measurements == 0
    assert result.diagnostics.measurement_candidates == ()
    assert result.measurements == ()


def test_diagnostic_hashing_does_not_break_existing_quote_fallback():
    row = candidate(source_quote="\ud800")
    result, diagnostic = extract(row)
    assert len(result.measurements) == 1
    assert (
        diagnostic.source_quote_sha256
        == hashlib.sha256(row.source_quote.encode(errors="surrogatepass")).hexdigest()
    )


def test_cli_persists_candidate_details_without_full_quotes(
    monkeypatch, tmp_path: Path
):
    from materials_screening import cli

    result, _ = extract(candidate(group_key="unknown"))
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
    text = files[0].read_text(encoding="utf-8")
    saved = json.loads(text)["diagnostics"]["measurement_candidates"]
    assert len(saved) == 1
    assert saved[0]["value_text"] == "12.0"
    assert saved[0]["rejection_reasons"] == ["unknown_group_key"]
    assert _chunk().text not in text
