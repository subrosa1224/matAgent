"""Unit tests for the filesystem run artifact store (S3-M3)."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from materials_screening.workflow.artifact_store import (
    ArtifactName,
    FileRunArtifactStore,
)
from materials_screening.workflow.errors import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactStoreError,
)
from materials_screening.workflow.state import ArtifactRef


def _tmp_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.name.endswith(".tmp")]


class TestArtifactName:
    def test_fixed_names(self) -> None:
        assert [name.value for name in ArtifactName] == [
            "retrieval",
            "filtered",
            "filter_trace",
            "ranked",
            "validation",
            "screening_result",
            "export_manifest",
        ]

    def test_custom_name_rejected(self) -> None:
        with pytest.raises(ValueError):
            ArtifactName("user_custom")


class TestInitializeRun:
    def test_creates_run_dir_and_manifest(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={"version": "workflow-v1"})
        manifest_path = tmp_path / "runs" / "run-1" / "manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == "artifact-manifest-v1"
        assert manifest["run_id"] == "run-1"
        assert manifest["metadata"] == {"version": "workflow-v1"}
        assert manifest["artifacts"] == {}

    def test_idempotent(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        store.initialize_run(run_id="run-1", metadata={})
        manifest_path = tmp_path / "runs" / "run-1" / "manifest.json"
        assert manifest_path.is_file()

    def test_manifest_run_id_conflict(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "run-1"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "artifact-manifest-v1",
                    "run_id": "other-run",
                    "artifacts": {},
                }
            ),
            encoding="utf-8",
        )
        store = FileRunArtifactStore(tmp_path / "runs")
        with pytest.raises(ArtifactConflictError):
            store.initialize_run(run_id="run-1", metadata={})

    @pytest.mark.parametrize(
        "run_id",
        ["", ".", "..", "../evil", "a/b", "a\\b", "x" * 255, " a"],
    )
    def test_unsafe_run_ids_rejected(self, tmp_path: Path, run_id: str) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        with pytest.raises(ArtifactStoreError):
            store.initialize_run(run_id=run_id, metadata={})

    def test_no_checkpoint_database_written(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        store.put_json(run_id="run-1", name="retrieval", value={"ok": True})
        suffixes = {path.suffix.lower() for path in tmp_path.rglob("*")}
        assert ".sqlite" not in suffixes
        assert ".db" not in suffixes


class TestPutJson:
    def _store(self, tmp_path: Path) -> FileRunArtifactStore:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        return store

    def test_writes_artifact_and_returns_ref(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        ref = store.put_json(
            run_id="run-1",
            name="retrieval",
            value={"band_gap": 1.2},
            schema_name="retrieval-v1",
            schema_version="1",
            item_count=3,
        )
        assert isinstance(ref, ArtifactRef)
        assert ref.name == "retrieval"
        assert ref.relative_path == "run-1/artifacts/retrieval.json"
        assert len(ref.sha256) == 64
        assert ref.size_bytes > 0
        assert ref.item_count == 3
        assert ref.schema_name == "retrieval-v1"
        artifact_path = tmp_path / "runs" / ref.relative_path
        assert artifact_path.is_file()
        assert json.loads(artifact_path.read_text(encoding="utf-8")) == {
            "band_gap": 1.2
        }
        assert _tmp_files(tmp_path / "runs") == []

    def test_stable_serialization_independent_of_key_order(
        self, tmp_path: Path
    ) -> None:
        store = self._store(tmp_path)
        first = store.put_json(
            run_id="run-1",
            name="filtered",
            value={"z": 1, "a": {"y": 2, "b": 3}},
        )
        second = store.put_json(
            run_id="run-1",
            name="filtered",
            value={"a": {"b": 3, "y": 2}, "z": 1},
        )
        assert first.sha256 == second.sha256
        assert first == second

    def test_same_content_is_reused(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        first = store.put_json(run_id="run-1", name="ranked", value={"a": 1})
        second = store.put_json(run_id="run-1", name="ranked", value={"a": 1})
        assert first == second
        manifest = json.loads(
            (tmp_path / "runs" / "run-1" / "manifest.json").read_text(encoding="utf-8")
        )
        assert list(manifest["artifacts"]) == ["ranked"]

    def test_same_name_different_content_conflicts(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        store.put_json(run_id="run-1", name="validation", value={"passed": True})
        with pytest.raises(ArtifactConflictError):
            store.put_json(run_id="run-1", name="validation", value={"passed": False})
        artifact_path = tmp_path / "runs" / "run-1" / "artifacts" / "validation.json"
        assert json.loads(artifact_path.read_text(encoding="utf-8")) == {"passed": True}

    def test_unknown_artifact_name_rejected(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        with pytest.raises(ArtifactStoreError):
            store.put_json(run_id="run-1", name="custom", value={})

    def test_requires_initialize_run(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        with pytest.raises(ArtifactStoreError):
            store.put_json(run_id="run-1", name="retrieval", value={})

    def test_manifest_tracks_multiple_artifacts(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        store.put_json(run_id="run-1", name="retrieval", value={"n": 1})
        store.put_json(run_id="run-1", name="filter_trace", value={"n": 2})
        manifest = json.loads(
            (tmp_path / "runs" / "run-1" / "manifest.json").read_text(encoding="utf-8")
        )
        assert set(manifest["artifacts"]) == {"retrieval", "filter_trace"}
        assert manifest["artifacts"]["retrieval"]["sha256"]

    def test_refs_never_contain_full_content(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        ref = store.put_json(
            run_id="run-1",
            name="retrieval",
            value={"records": [{"id": i} for i in range(100)]},
        )
        dumped = ref.model_dump(mode="json")
        assert "records" not in dumped
        assert "value" not in dumped
        assert "content" not in dumped


class TestGetJson:
    def test_round_trip(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        ref = store.put_json(
            run_id="run-1", name="screening_result", value={"ok": True, "n": 5}
        )
        assert store.get_json(ref) == {"ok": True, "n": 5}

    def test_tampered_file_rejected(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        ref = store.put_json(run_id="run-1", name="retrieval", value={"ok": True})
        artifact_path = tmp_path / "runs" / ref.relative_path
        artifact_path.write_text('{"ok": false}', encoding="utf-8")
        assert store.verify(ref) is False
        with pytest.raises(ArtifactIntegrityError):
            store.get_json(ref)

    def test_missing_file_rejected(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        ref = store.put_json(run_id="run-1", name="retrieval", value={"ok": True})
        (tmp_path / "runs" / ref.relative_path).unlink()
        assert store.exists(ref) is False
        assert store.verify(ref) is False
        with pytest.raises(ArtifactIntegrityError):
            store.get_json(ref)

    def test_forged_ref_with_parent_traversal_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRef.model_validate(
                {
                    "name": "retrieval",
                    "relative_path": "../outside.json",
                    "sha256": "a" * 64,
                    "media_type": "application/json",
                    "size_bytes": 1,
                }
            )


class TestVerifyAndExists:
    def test_verify_true_for_valid_ref(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        ref = store.put_json(run_id="run-1", name="ranked", value=[1, 2, 3])
        assert store.exists(ref) is True
        assert store.verify(ref) is True

    def test_verify_false_for_ref_to_other_run(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        store.initialize_run(run_id="run-2", metadata={})
        ref = store.put_json(run_id="run-1", name="ranked", value=[1])
        forged = ref.model_copy(update={"relative_path": "run-2/artifacts/ranked.json"})
        assert store.exists(forged) is False
        assert store.verify(forged) is False


class TestTmpCleanup:
    def test_tmp_removed_on_replace_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})

        def _fail_replace(self: Path, target: Path) -> None:
            raise OSError("simulated replace failure")

        monkeypatch.setattr(Path, "replace", _fail_replace)
        with pytest.raises(ArtifactStoreError):
            store.put_json(run_id="run-1", name="retrieval", value={"ok": True})
        run_dir = tmp_path / "runs" / "run-1"
        assert _tmp_files(run_dir) == []
        assert not (run_dir / "artifacts" / "retrieval.json").exists()


def _forged_ref(relative_path: str) -> ArtifactRef:
    """Build an ArtifactRef that bypasses model validation."""
    return ArtifactRef.model_construct(
        name="retrieval",
        relative_path=relative_path,
        sha256="0" * 64,
        media_type="application/json",
        size_bytes=1,
    )


class TestErrorBranches:
    def test_put_json_accepts_pydantic_model(self, tmp_path: Path) -> None:
        from pydantic import BaseModel as PydanticModel

        class Sample(PydanticModel):
            value: int

        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        ref = store.put_json(run_id="run-1", name="retrieval", value=Sample(value=1))
        assert store.get_json(ref) == {"value": 1}

    def test_put_json_read_failure_wrapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        store.put_json(run_id="run-1", name="retrieval", value={"ok": True})

        def _fail_read_bytes(self: Path) -> bytes:
            raise OSError("simulated read failure")

        monkeypatch.setattr(Path, "read_bytes", _fail_read_bytes)
        with pytest.raises(ArtifactStoreError):
            store.put_json(run_id="run-1", name="retrieval", value={"ok": True})

    def test_get_json_read_failure_wrapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        ref = store.put_json(run_id="run-1", name="retrieval", value={"ok": True})

        def _fail_read_text(self: Path, **kwargs: object) -> str:
            raise OSError("simulated read failure")

        monkeypatch.setattr(Path, "read_text", _fail_read_text)
        with pytest.raises(ArtifactIntegrityError):
            store.get_json(ref)

    def test_verify_false_for_forged_traversal_ref(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        forged = _forged_ref("../outside.json")
        assert store.verify(forged) is False
        with pytest.raises(ArtifactStoreError):
            store.exists(forged)

    def test_verify_false_on_read_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        ref = store.put_json(run_id="run-1", name="retrieval", value={"ok": True})

        def _fail_read_bytes(self: Path) -> bytes:
            raise OSError("simulated read failure")

        monkeypatch.setattr(Path, "read_bytes", _fail_read_bytes)
        assert store.verify(ref) is False

    def test_run_dir_escapes_root_rejected(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        with pytest.raises(ArtifactStoreError):
            store._run_dir("../evil")

    def test_assert_contained_rejects_escape(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        root = (tmp_path / "runs").resolve()
        outside = tmp_path / "outside"
        with pytest.raises(ArtifactStoreError):
            store._assert_contained(outside, root)

    def test_corrupted_manifest_rejected(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "run-1"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text("{not json", encoding="utf-8")
        store = FileRunArtifactStore(tmp_path / "runs")
        with pytest.raises(ArtifactIntegrityError):
            store.initialize_run(run_id="run-1", metadata={})

    def test_non_object_manifest_rejected(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "run-1"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text("[]", encoding="utf-8")
        store = FileRunArtifactStore(tmp_path / "runs")
        with pytest.raises(ArtifactIntegrityError):
            store.initialize_run(run_id="run-1", metadata={})

    def test_manifest_write_failure_wrapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        real_replace = Path.replace

        def _selective_replace(self: Path, target: Path) -> None:
            if self.name == ".manifest.json.tmp":
                raise OSError("simulated manifest write failure")
            return real_replace(self, target)

        monkeypatch.setattr(Path, "replace", _selective_replace)
        with pytest.raises(ArtifactStoreError):
            store.put_json(run_id="run-1", name="retrieval", value={"ok": True})
        artifact_path = tmp_path / "runs" / "run-1" / "artifacts" / "retrieval.json"
        assert artifact_path.exists()
        assert _tmp_files(tmp_path / "runs") == []


class TestAudit:
    """Audit-focused tests for atomicity, hashing, idempotency and safety."""

    def test_hash_is_based_on_final_file_bytes(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        value = {"a": 1, "b": [True, None, "x"]}
        ref = store.put_json(run_id="run-1", name="retrieval", value=value)
        artifact_path = tmp_path / "runs" / ref.relative_path
        file_bytes = artifact_path.read_bytes()
        serialized = json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2
        ).encode("utf-8")
        assert file_bytes == serialized
        assert ref.sha256 == hashlib.sha256(file_bytes).hexdigest()

    def test_replay_is_no_op_for_same_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        real_replace = Path.replace
        replace_count = 0

        def _counting_replace(self: Path, target: Path) -> None:
            nonlocal replace_count
            replace_count += 1
            return real_replace(self, target)

        monkeypatch.setattr(Path, "replace", _counting_replace)
        store.put_json(run_id="run-1", name="ranked", value={"a": 1})
        first_count = replace_count
        assert first_count > 0
        store.put_json(run_id="run-1", name="ranked", value={"a": 1})
        assert replace_count == first_count

    def test_conflict_does_not_update_manifest(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        first = store.put_json(
            run_id="run-1", name="validation", value={"passed": True}
        )
        with pytest.raises(ArtifactConflictError):
            store.put_json(run_id="run-1", name="validation", value={"passed": False})
        manifest = json.loads(
            (tmp_path / "runs" / "run-1" / "manifest.json").read_text(encoding="utf-8")
        )
        entry = manifest["artifacts"]["validation"]
        assert entry["sha256"] == first.sha256
        assert entry["relative_path"] == first.relative_path

    def test_manifest_self_heals_missing_entry(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        store.put_json(run_id="run-1", name="retrieval", value={"ok": True})
        manifest_path = tmp_path / "runs" / "run-1" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["artifacts"]["retrieval"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        store.put_json(run_id="run-1", name="retrieval", value={"ok": True})
        restored = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert restored["artifacts"]["retrieval"]["name"] == "retrieval"

    def test_symlinked_artifact_dir_cannot_escape(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        run_dir = tmp_path / "runs" / "run-1"
        run_dir.mkdir(parents=True)
        try:
            (run_dir / "artifacts").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not available on this platform")

        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        with pytest.raises(ArtifactStoreError):
            store.put_json(run_id="run-1", name="retrieval", value={"ok": True})
        assert list(outside.iterdir()) == []

    def test_tmp_cleaned_on_fsync_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})

        def _fail_fsync(fd: int) -> None:
            raise OSError("simulated fsync failure")

        monkeypatch.setattr(os, "fsync", _fail_fsync)
        with pytest.raises(ArtifactStoreError):
            store.put_json(run_id="run-1", name="retrieval", value={"ok": True})
        assert _tmp_files(tmp_path / "runs") == []

    def test_non_serializable_value_rejected_without_content(
        self, tmp_path: Path
    ) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        secret = "secret-content-should-not-leak"
        with pytest.raises(ArtifactStoreError) as excinfo:
            store.put_json(
                run_id="run-1",
                name="retrieval",
                value={"path": tmp_path / secret},
            )
        assert secret not in str(excinfo.value)

    def test_unicode_serialization_is_stable(self, tmp_path: Path) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        value = {"材料": "钙钛矿", "ok": True}
        first = store.put_json(run_id="run-1", name="retrieval", value=value)
        second = store.put_json(run_id="run-1", name="retrieval", value=value)
        assert first == second
        assert store.get_json(first) == value
