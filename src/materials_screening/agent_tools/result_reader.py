"""Concrete WorkflowResultReader backed by the run artifact store (S3.5-M3)."""

import json
from pathlib import Path
from typing import Any

from materials_screening.agent.errors import WorkflowResultReadError
from materials_screening.workflow.artifact_store import (
    FileRunArtifactStore,
)
from materials_screening.workflow.errors import (
    ArtifactIntegrityError,
    ArtifactStoreError,
)
from materials_screening.workflow.state import ArtifactRef


class FileWorkflowResultReader:
    """Read and hash-verify a validated screening_result artifact."""

    def __init__(self, run_root: Path) -> None:
        self._run_root = run_root.expanduser().resolve()
        self._store = FileRunArtifactStore(self._run_root)

    def read(self, thread_id: str) -> dict[str, Any]:
        manifest_path = self._run_root / thread_id / "manifest.json"
        if not manifest_path.is_file():
            raise WorkflowResultReadError(
                "THREAD_NOT_FOUND",
                f"thread {thread_id!r} has no workflow run",
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ref_dict = manifest.get("artifacts", {}).get("screening_result")
        if not isinstance(ref_dict, dict):
            raise WorkflowResultReadError(
                "RESULT_NOT_FOUND",
                "screening_result artifact missing",
            )
        try:
            ref = ArtifactRef.model_validate(ref_dict)
            payload = self._store.get_json(ref)
        except ArtifactIntegrityError as exc:
            raise WorkflowResultReadError(
                "ARTIFACT_INTEGRITY",
                "screening_result artifact corrupted",
            ) from exc
        except ArtifactStoreError as exc:
            raise WorkflowResultReadError(
                "RESULT_NOT_FOUND",
                "screening_result artifact unavailable",
            ) from exc
        if not isinstance(payload, dict):
            raise WorkflowResultReadError(
                "RESULT_INVALID",
                "screening_result artifact is invalid",
            )
        return payload
