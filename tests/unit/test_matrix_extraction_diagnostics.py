import json
from pathlib import Path
from types import SimpleNamespace

from materials_screening import cli
from materials_screening.sub_agents.literature import matrix_automation
from materials_screening.sub_agents.literature.matrix_automation import (
    MatrixBatchAttemptDiagnostic,
    MatrixExtractionDiagnostics,
    PendingMatrixExtraction,
)


def test_cli_persists_gate_reasons_even_when_display_is_hidden(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    result = PendingMatrixExtraction(
        groups=(),
        measurements=(),
        comparisons=(),
        claims=(),
        claim_evidence_links=(),
        warnings=("measurement capacity rejected: unknown group or quote",),
        diagnostics=MatrixExtractionDiagnostics(
            llm_measurements=2,
            accepted_measurements=0,
            rejection_reasons={"unknown group or quote": 2},
            batch_attempts=(
                MatrixBatchAttemptDiagnostic(
                    batch_index=1,
                    attempt=1,
                    status="error",
                    chunk_ids=("chunk-test",),
                    page_ranges=((2, 3),),
                    evidence_chars=100,
                    output_token_budget=8192,
                    error_type="LLMStructuredOutputError",
                    error_category="invalid_json",
                ),
            ),
        ),
    )
    store = SimpleNamespace(
        get_document_chunks=lambda _: [object()],
        save_experiment_matrix=lambda *args: None,
        save_comparisons=lambda *args: None,
        save_claims_and_links=lambda *args: None,
    )
    settings = SimpleNamespace(
        llm_provider="intern", intern_model="test", llm_max_output_tokens=8192
    )
    settings.model_copy = lambda **kwargs: settings
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(cli, "_literature_pgvector_store", lambda: store)
    monkeypatch.setattr(cli, "create_llm_provider", lambda _: object())
    monkeypatch.setattr(
        matrix_automation.AutomatedMatrixExtractor,
        "extract",
        lambda *args, **kwargs: result,
    )
    for _ in range(2):
        cli.literature_matrix_extract_command("doc-test", show_warnings=False)
    files = list((tmp_path / "data/literature_extractions").glob("*.json"))
    assert len(files) == 2  # Never overwrite an earlier diagnostic attempt.
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["diagnostics"]["llm_measurements"] == 2
    assert saved["diagnostics"]["accepted_measurements"] == 0
    assert saved["diagnostics"]["rejection_reasons"] == {"unknown group or quote": 2}
    assert saved["warnings"] == list(result.warnings)
    assert saved["diagnostics"]["batch_attempts"][0]["error_category"] == "invalid_json"
    assert saved["diagnostics"]["batch_attempts"][0]["page_ranges"] == [[2, 3]]
    assert "Extraction diagnostics:" in capsys.readouterr().out
