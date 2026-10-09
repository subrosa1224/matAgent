"""Attachment persistence and conversation authorization, not PDF analysis."""

from pathlib import Path

import pytest

from materials_screening.master.artifact_registry import ArtifactRegistry


def pdf(tmp_path: Path) -> Path:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.7\noriginal text")
    return source


def test_registration_survives_registry_restart(tmp_path: Path) -> None:
    first = ArtifactRegistry(tmp_path / "artifacts")
    artifact = first.register_pdf(pdf(tmp_path), conversation_id="conv-a")
    restored = ArtifactRegistry(tmp_path / "artifacts")
    assert restored.get(artifact.artifact_id) == artifact
    assert (
        restored.resolve_pdf(artifact.artifact_id, conversation_id="conv-a")
        .read_bytes()
        .startswith(b"%PDF-")
    )


def test_cross_conversation_reference_is_rejected(tmp_path: Path) -> None:
    first = ArtifactRegistry(tmp_path / "artifacts")
    artifact = first.register_pdf(pdf(tmp_path), conversation_id="conv-a")
    with pytest.raises(ValueError, match="会话"):
        first.resolve_pdf(artifact.artifact_id, conversation_id="conv-b")


def test_no_implicit_grant_for_legacy_standalone_attachment(tmp_path: Path) -> None:
    first = ArtifactRegistry(tmp_path / "artifacts")
    artifact = first.register_pdf(pdf(tmp_path))
    assert first.resolve_path(artifact.artifact_id).is_file()
    with pytest.raises(ValueError, match="会话"):
        first.resolve_pdf(artifact.artifact_id, conversation_id="conv-a")


def test_same_pdf_can_be_explicitly_uploaded_in_two_conversations(
    tmp_path: Path,
) -> None:
    first = ArtifactRegistry(tmp_path / "artifacts")
    source = pdf(tmp_path)
    artifact = first.register_pdf(source, conversation_id="conv-a")
    duplicate = first.register_pdf(source, conversation_id="conv-b")
    assert duplicate.artifact_id == artifact.artifact_id
    restored = ArtifactRegistry(tmp_path / "artifacts")
    assert restored.resolve_pdf(
        artifact.artifact_id, conversation_id="conv-a"
    ) == restored.resolve_pdf(artifact.artifact_id, conversation_id="conv-b")
    assert restored.get(artifact.artifact_id).created_at == artifact.created_at


def test_changed_pdf_fails_after_restart(tmp_path: Path) -> None:
    first = ArtifactRegistry(tmp_path / "artifacts")
    artifact = first.register_pdf(pdf(tmp_path), conversation_id="conv-a")
    first.resolve_path(artifact.artifact_id).write_bytes(b"%PDF-1.7\nchanged text")
    restored = ArtifactRegistry(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="内容"):
        restored.resolve_pdf(artifact.artifact_id, conversation_id="conv-a")
    with pytest.raises(ValueError, match="内容"):
        first.resolve_path(artifact.artifact_id)


def test_reregister_does_not_repair_tampered_stored_pdf(tmp_path: Path) -> None:
    first = ArtifactRegistry(tmp_path / "artifacts")
    source = pdf(tmp_path)
    artifact = first.register_pdf(source, conversation_id="conv-a")
    first.resolve_path(artifact.artifact_id).write_bytes(b"%PDF-1.7\nchanged text")
    with pytest.raises(ValueError, match="内容"):
        first.register_pdf(source, conversation_id="conv-b")


@pytest.mark.parametrize("conversation", ["", "../outside", "with space", "x" * 256])
def test_unsafe_conversation_names_rejected(tmp_path: Path, conversation: str) -> None:
    first = ArtifactRegistry(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="会话"):
        first.register_pdf(pdf(tmp_path), conversation_id=conversation)


def test_malicious_catalog_path_is_rejected(tmp_path: Path) -> None:
    import sqlite3

    first = ArtifactRegistry(tmp_path / "artifacts")
    artifact = first.register_pdf(pdf(tmp_path), conversation_id="conv-a")
    with sqlite3.connect(tmp_path / "artifacts/artifacts.sqlite") as connection:
        connection.execute(
            "UPDATE artifacts SET relative_path='../paper.pdf' WHERE artifact_id=?",
            (artifact.artifact_id,),
        )
    restored = ArtifactRegistry(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="路径"):
        restored.resolve_pdf(artifact.artifact_id, conversation_id="conv-a")


def test_catalog_missing_never_scans_for_matching_pdf(tmp_path: Path) -> None:
    first = ArtifactRegistry(tmp_path / "artifacts")
    artifact = first.register_pdf(pdf(tmp_path), conversation_id="conv-a")
    import sqlite3

    with sqlite3.connect(tmp_path / "artifacts/artifacts.sqlite") as connection:
        connection.execute(
            "DELETE FROM artifacts WHERE artifact_id=?", (artifact.artifact_id,)
        )
    with pytest.raises(KeyError, match="unknown artifact"):
        ArtifactRegistry(tmp_path / "artifacts").resolve_pdf(
            artifact.artifact_id, conversation_id="conv-a"
        )
