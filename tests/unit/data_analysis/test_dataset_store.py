from __future__ import annotations

from pathlib import Path

import pytest

from materials_screening.data_analysis.dataset_store import DatasetStore


def _write_dataset(path: Path) -> bytes:
    content = b"sample_id,value\nA,1.2\nB,2.4\n"
    path.write_bytes(content)
    return content


def test_store_copies_dataset_and_does_not_expose_source_path(tmp_path: Path) -> None:
    source = tmp_path / "upload" / "experiment.csv"
    source.parent.mkdir()
    original = _write_dataset(source)
    store = DatasetStore(tmp_path / "private")

    reference = store.register_file(
        source,
        source_artifact_id="artifact-data-upload-1",
    )

    stored_path = store.resolve_path(reference.dataset_id)
    assert stored_path.read_bytes() == original
    assert stored_path.parent == (tmp_path / "private" / "datasets").resolve()
    assert str(source) not in reference.model_dump_json()
    source.write_text("sample_id,value\nCHANGED,99\n", encoding="utf-8")
    assert store.load_dataframe(reference.dataset_id).iloc[0]["sample_id"] == "A"


def test_store_recognizes_duplicate_content_and_reloads_metadata(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    _write_dataset(first_path)
    _write_dataset(second_path)
    root = tmp_path / "private"
    store = DatasetStore(root)

    first = store.register_file(
        first_path, source_artifact_id="artifact-data-same-content"
    )
    second = store.register_file(
        second_path, source_artifact_id="artifact-data-same-content"
    )
    reloaded = DatasetStore(root)

    assert second == first
    assert reloaded.get(first.dataset_id) == first
    assert reloaded.load_dataframe(first.dataset_id).shape == (2, 2)
    assert len(tuple((root / "datasets").glob("dataset-*.csv"))) == 1


def test_long_lived_reader_discovers_dataset_registered_by_another_instance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "experiment.csv"
    _write_dataset(source)
    root = tmp_path / "private"
    reader = DatasetStore(root)
    writer = DatasetStore(root)

    reference = writer.register_file(
        source, source_artifact_id="artifact-data-shared-store"
    )

    assert reader.get(reference.dataset_id) == reference
    assert reader.load_dataframe(reference.dataset_id).shape == (2, 2)


def test_store_rejects_invalid_artifact_id_without_creating_dataset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "data.csv"
    _write_dataset(source)
    root = tmp_path / "private"

    with pytest.raises(ValueError):
        DatasetStore(root).register_file(
            source, source_artifact_id="../../artifact"
        )

    assert not (root / "datasets").exists()


def test_store_unknown_or_missing_dataset_is_rejected(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "private")
    with pytest.raises(KeyError, match="unknown dataset"):
        store.get("dataset-unknown")

    source = tmp_path / "data.csv"
    _write_dataset(source)
    reference = store.register_file(
        source, source_artifact_id="artifact-data-upload-2"
    )
    store.resolve_path(reference.dataset_id).unlink()
    with pytest.raises(ValueError, match="missing"):
        store.resolve_path(reference.dataset_id)


def test_store_detects_tampered_dataset_content(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    _write_dataset(source)
    store = DatasetStore(tmp_path / "private")
    reference = store.register_file(
        source, source_artifact_id="artifact-data-tamper"
    )
    store.resolve_path(reference.dataset_id).write_text(
        "sample_id,value\nX,999\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="fingerprint"):
        store.load_dataframe(reference.dataset_id)


def test_store_ignores_corrupt_metadata_on_reload(tmp_path: Path) -> None:
    metadata_root = tmp_path / "private" / "metadata"
    metadata_root.mkdir(parents=True)
    (metadata_root / "dataset-bad.json").write_text("not json", encoding="utf-8")

    store = DatasetStore(tmp_path / "private")

    with pytest.raises(KeyError):
        store.get("dataset-bad")
