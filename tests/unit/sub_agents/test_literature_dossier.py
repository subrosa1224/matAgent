from __future__ import annotations

from pathlib import Path

from materials_screening.sub_agents.literature.dossier import (
    DossierItem,
    PaperDossier,
    PaperDossierStore,
)


def test_dossier_review_updates_status_and_preserves_audit_event(
    tmp_path: Path,
) -> None:
    root = tmp_path / "literature_dossiers"
    root.mkdir()
    dossier = PaperDossier(
        document_id="doc-1",
        title="Test paper",
        extraction_method="manual_pilot",
        items=(
            DossierItem(
                category="research_problem",
                summary="A grounded problem.",
                source_quote="original evidence",
                chunk_id="chunk-1",
                page=1,
            ),
        ),
    )
    (root / "doc-1.json").write_text(
        dossier.model_dump_json(indent=2), encoding="utf-8"
    )

    review = PaperDossierStore(root).review(
        document_id="doc-1",
        decision="approved",
        reviewer="human",
        reason="checked",
    )

    assert review.previous_status == "pending"
    loaded = PaperDossierStore(root).load("doc-1")
    assert loaded is not None
    assert loaded.review_status == "approved"
    event = tmp_path / "literature_dossier_reviews" / "doc-1"
    assert list(event.glob("*.json"))


def test_approved_candidate_can_be_published_to_trusted_store(
    tmp_path: Path,
) -> None:
    candidate_store = PaperDossierStore(tmp_path / "candidates")
    approved_store = PaperDossierStore(tmp_path / "approved")
    dossier = PaperDossier(
        document_id="doc-1",
        title="Test paper",
        extraction_method="llm",
        items=(
            DossierItem(
                category="research_problem",
                summary="Grounded summary.",
                source_quote="original evidence",
                chunk_id="chunk-1",
                page=1,
            ),
        ),
    )
    candidate_store.save_pending(dossier)
    candidate_store.review(
        document_id="doc-1", decision="approved", reviewer="human"
    )
    reviewed = candidate_store.load("doc-1")

    assert reviewed is not None
    approved_store.save_reviewed(reviewed)
    assert approved_store.load("doc-1") == reviewed
