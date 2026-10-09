from __future__ import annotations

from pathlib import Path

from materials_screening.sub_agents.literature.batch import (
    BatchPaperResult,
    LiteratureBatchStore,
    document_id_for_pdf,
    matrix_analysis_complete,
    missing_index_paths,
    should_extract_matrix,
)


def _row(document_id: str, status: str = "completed") -> BatchPaperResult:
    return BatchPaperResult.model_validate(
        {
            "document_id": document_id,
            "file_name": f"{document_id}.pdf",
            "title": f"Paper {document_id}",
            "status": status,
            "dossier_status": "pending",
            "dossier_items": 10,
            "completeness_score": 0.8,
            "matrix_groups": 4,
            "matrix_measurements": 12,
            "pending_measurements": 2,
        }
    )


def test_batch_report_is_stable_persistent_and_counts_attention(tmp_path: Path) -> None:
    store = LiteratureBatchStore(tmp_path)
    report = store.save((_row("doc-a"), _row("doc-b", "partial")))

    assert report.completed_count == 1
    assert report.attention_count == 1
    assert report.review_required_count == 2
    assert store.load(report.batch_id) == report
    assert store.save(tuple(reversed(report.papers))).batch_id == report.batch_id


def test_document_id_matches_pdf_content_not_file_name(tmp_path: Path) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "renamed.pdf"
    first.write_bytes(b"%PDF-1.7 same-content")
    second.write_bytes(b"%PDF-1.7 same-content")

    assert document_id_for_pdf(first) == document_id_for_pdf(second)


def test_batch_indexes_only_missing_documents(tmp_path: Path) -> None:
    indexed = tmp_path / "indexed.pdf"
    missing = tmp_path / "missing.pdf"
    indexed.write_bytes(b"indexed")
    missing.write_bytes(b"missing")
    indexed_id = document_id_for_pdf(indexed)

    result = missing_index_paths(
        (indexed, missing), lambda document_id: document_id == indexed_id
    )

    assert result == (missing,)


def test_incomplete_matrix_retry_is_explicit() -> None:
    incomplete = {"groups": 3, "measurements": 0}
    assert should_extract_matrix(incomplete, retry_incomplete=False) is False
    assert should_extract_matrix(incomplete, retry_incomplete=True) is True
    qualitative = {
        "groups": 2,
        "measurements": 0,
        "claims": 7,
        "claim_evidence_links": 7,
        "verified_claim_checks": 7,
    }
    assert should_extract_matrix(qualitative, retry_incomplete=True) is False
    assert should_extract_matrix({}, retry_incomplete=False) is True


def test_matrix_analysis_accepts_checked_qualitative_evidence() -> None:
    assert matrix_analysis_complete({"measurements": 9}) is True
    assert matrix_analysis_complete({"verified_claim_checks": 7}) is True
    assert matrix_analysis_complete({"claim_evidence_links": 7}) is False
    assert matrix_analysis_complete({"groups": 3}) is False
