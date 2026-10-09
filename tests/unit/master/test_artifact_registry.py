"""Security and reference tests for the unified Artifact Registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.master import ArtifactRegistry


def _pdf(path: Path, body: bytes = b"test") -> Path:
    path.write_bytes(b"%PDF-1.7\n" + body)
    return path


def test_pdf_registration_exposes_metadata_but_not_private_path(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path / "artifacts")
    artifact = registry.register_pdf(_pdf(tmp_path / "paper.pdf"))

    assert artifact.artifact_id.startswith("artifact-pdf-")
    assert artifact.owner_agent == "literature"
    assert artifact.display_name == "paper.pdf"
    assert "path" not in artifact.metadata
    assert registry.resolve_path(artifact.artifact_id).is_file()


def test_registry_rejects_fake_pdf_and_unknown_artifact(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path / "artifacts")
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"not a pdf")
    with pytest.raises(ValueError, match="有效PDF"):
        registry.register_pdf(fake)
    with pytest.raises(KeyError, match="unknown artifact"):
        registry.resolve_path("artifact-pdf-missing")


def test_registry_rejects_oversized_pdf(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path / "artifacts", max_pdf_bytes=8)
    with pytest.raises(ValueError, match="100 MB"):
        registry.register_pdf(_pdf(tmp_path / "large.pdf", b"too-large"))


def test_explicit_attachment_reference_resolves_owner(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path / "artifacts")
    artifact = registry.register_pdf(_pdf(tmp_path / "paper.pdf"))
    ids = [artifact.artifact_id]

    assert registry.resolve_owner_reference(ids, "深度分析第1篇") == "literature"
    assert registry.resolve_owner_reference(ids, "综合这些论文") == "literature"
    assert registry.resolve_owner_reference(ids, "查询 mp-149") is None


def test_data_registration_creates_dataset_without_exposing_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "experiment.csv"
    source.write_text("group,value\nA,1\nB,2\n", encoding="utf-8")
    store = DatasetStore(tmp_path / "datasets")
    registry = ArtifactRegistry(tmp_path / "artifacts", dataset_store=store)

    artifact = registry.register_data_file(source)

    assert artifact.artifact_id.startswith("artifact-data-")
    assert artifact.owner_agent == "data_analysis"
    assert artifact.artifact_type == "tabular_dataset"
    assert artifact.domain_id.startswith("dataset-")
    assert artifact.metadata["row_count"] == 2
    assert "path" not in artifact.metadata
    assert store.get(artifact.domain_id).display_name == "experiment.csv"
    assert registry.resolve_path(artifact.artifact_id).is_file()


def test_data_registration_rejects_invalid_and_resolves_owner(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path / "artifacts")
    invalid = tmp_path / "nested.json"
    invalid.write_text('[{"sample": {"name": "A"}}]', encoding="utf-8")
    with pytest.raises(ValueError, match="nested data"):
        registry.register_data_file(invalid)

    source = tmp_path / "valid.json"
    source.write_text('[{"sample": "A", "value": 1}]', encoding="utf-8")
    artifact = registry.register_data_file(source)
    assert (
        registry.resolve_owner_reference(
            [artifact.artifact_id], "分析我上传的数据集"
        )
        == "data_analysis"
    )
