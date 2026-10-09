"""Unit tests for the FileWorkflowResultReader (S3.5-M3)."""

import json
from pathlib import Path
from typing import Any

import pytest

from materials_screening.agent.errors import WorkflowResultReadError
from materials_screening.agent_tools.result_reader import (
    FileWorkflowResultReader,
)
from materials_screening.workflow.artifact_store import (
    FileRunArtifactStore,
)


class TestFileWorkflowResultReader:
    def test_reads_validated_result(self, tmp_path: Path) -> None:
        run_root = tmp_path / "runs"
        store = FileRunArtifactStore(run_root)
        store.initialize_run(run_id="thread-1", metadata={})
        payload: dict[str, Any] = {
            "request": {"limit": 5},
            "ranked_materials": [],
            "validation": {"passed": True},
        }
        store.put_json(
            run_id="thread-1",
            name="screening_result",
            value=payload,
        )
        reader = FileWorkflowResultReader(run_root)
        assert reader.read("thread-1") == payload

    def test_missing_thread(self, tmp_path: Path) -> None:
        reader = FileWorkflowResultReader(tmp_path / "runs")
        with pytest.raises(WorkflowResultReadError) as excinfo:
            reader.read("thread-missing")
        assert excinfo.value.code == "THREAD_NOT_FOUND"

    def test_corrupted_artifact(self, tmp_path: Path) -> None:
        run_root = tmp_path / "runs"
        store = FileRunArtifactStore(run_root)
        store.initialize_run(run_id="thread-1", metadata={})
        store.put_json(
            run_id="thread-1",
            name="screening_result",
            value={"ok": True},
        )
        artifact = run_root / "thread-1" / "artifacts" / "screening_result.json"
        artifact.write_text('{"tampered": true}', encoding="utf-8")
        reader = FileWorkflowResultReader(run_root)
        with pytest.raises(WorkflowResultReadError) as excinfo:
            reader.read("thread-1")
        assert excinfo.value.code == "ARTIFACT_INTEGRITY"

    def test_manifest_without_result_artifact(self, tmp_path: Path) -> None:
        run_root = tmp_path / "runs"
        run_dir = run_root / "thread-1"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"run_id": "thread-1", "artifacts": {}}),
            encoding="utf-8",
        )
        reader = FileWorkflowResultReader(run_root)
        with pytest.raises(WorkflowResultReadError) as excinfo:
            reader.read("thread-1")
        assert excinfo.value.code == "RESULT_NOT_FOUND"
