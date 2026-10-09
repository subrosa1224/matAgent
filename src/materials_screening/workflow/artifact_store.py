"""Filesystem-backed run artifact store (S3-M3)."""

import hashlib
import json
import os
import re
from contextlib import suppress
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from materials_screening.workflow.errors import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactStoreError,
)
from materials_screening.workflow.state import ArtifactRef

MANIFEST_SCHEMA_VERSION = "artifact-manifest-v1"
_ARTIFACT_MEDIA_TYPE = "application/json"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,253}$")
_JSON_INDENT = 2


class ArtifactName(StrEnum):
    """Fixed artifact names; users cannot define custom artifact filenames."""

    RETRIEVAL = "retrieval"
    FILTERED = "filtered"
    FILTER_TRACE = "filter_trace"
    RANKED = "ranked"
    VALIDATION = "validation"
    SCREENING_RESULT = "screening_result"
    EXPORT_MANIFEST = "export_manifest"


def _validate_run_id(run_id: str) -> str:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ArtifactStoreError(f"unsafe run_id: {run_id!r}")
    return run_id


def _to_json_text(value: object) -> str:
    """Serialize JSON-compatible data with deterministic key ordering."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=_JSON_INDENT)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


class FileRunArtifactStore:
    """Atomic, hash-verified artifact store rooted at one run directory tree.

    Layout (per run):
        <run_root>/<run_id>/manifest.json
        <run_root>/<run_id>/artifacts/<name>.json
    """

    def __init__(self, run_root: Path) -> None:
        self._run_root = run_root.expanduser().resolve()

    def initialize_run(self, *, run_id: str, metadata: dict[str, Any]) -> None:
        """Create the run directory and manifest idempotently."""
        safe_run_id = _validate_run_id(run_id)
        run_dir = self._run_dir(safe_run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            existing = self._read_manifest(run_dir)
            if existing.get("run_id") != run_id:
                raise ArtifactConflictError(
                    f"manifest run_id {existing.get('run_id')!r} "
                    f"does not match {run_id!r}"
                )
            return
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": _iso_now(),
            "metadata": metadata,
            "artifacts": {},
        }
        self._write_manifest(run_dir, manifest)

    def put_json(
        self,
        *,
        run_id: str,
        name: str,
        value: object,
        schema_name: str | None = None,
        schema_version: str | None = None,
        item_count: int | None = None,
    ) -> ArtifactRef:
        """Write one artifact atomically; same content is reused, different
        content under the same name conflicts."""
        safe_run_id = _validate_run_id(run_id)
        artifact_name = self._resolve_artifact_name(name)
        run_dir = self._run_dir(safe_run_id)
        manifest = self._read_manifest(run_dir)

        try:
            text = _to_json_text(value)
        except (TypeError, ValueError) as exc:
            raise ArtifactStoreError(
                f"artifact {artifact_name.value!r} value is not JSON serializable"
            ) from exc
        content = text.encode("utf-8")
        content_hash = _sha256_bytes(content)
        target = run_dir / "artifacts" / f"{artifact_name.value}.json"
        self._assert_contained(target, run_dir)

        if target.exists():
            try:
                existing_hash = _sha256_bytes(target.read_bytes())
            except OSError as exc:
                raise ArtifactStoreError(
                    f"failed to read existing artifact {artifact_name.value!r}: {exc}"
                ) from exc
            if existing_hash != content_hash:
                raise ArtifactConflictError(
                    f"artifact {artifact_name.value!r} already exists "
                    "with different content"
                )
        else:
            try:
                self._atomic_write_bytes(target, content)
            except OSError as exc:
                raise ArtifactStoreError(
                    f"failed to write artifact {artifact_name.value!r}: {exc}"
                ) from exc

        ref = ArtifactRef(
            name=artifact_name.value,
            relative_path=f"{safe_run_id}/artifacts/{artifact_name.value}.json",
            sha256=content_hash,
            media_type=_ARTIFACT_MEDIA_TYPE,
            size_bytes=len(content),
            item_count=item_count,
            schema_name=schema_name,
            schema_version=schema_version,
        )
        entry = ref.model_dump(mode="json")
        if manifest["artifacts"].get(artifact_name.value) != entry:
            manifest["artifacts"][artifact_name.value] = entry
            self._write_manifest(run_dir, manifest)
        return ref

    def get_json(self, ref: ArtifactRef) -> object:
        """Read an artifact after verifying its SHA-256."""
        path = self._resolve_ref_path(ref)
        if not path.is_file():
            raise ArtifactIntegrityError(f"artifact file missing: {ref.name}")
        if not self.verify(ref):
            raise ArtifactIntegrityError(f"artifact hash mismatch: {ref.name}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                f"artifact unreadable or corrupted: {ref.name}"
            ) from exc

    def exists(self, ref: ArtifactRef) -> bool:
        return self._resolve_ref_path(ref).is_file()

    def verify(self, ref: ArtifactRef) -> bool:
        """Return True only when the artifact file matches ref.sha256."""
        try:
            path = self._resolve_ref_path(ref)
        except ArtifactStoreError:
            return False
        if not path.is_file():
            return False
        try:
            actual = _sha256_bytes(path.read_bytes())
        except OSError:
            return False
        return actual == ref.sha256

    def _run_dir(self, run_id: str) -> Path:
        run_dir = (self._run_root / run_id).resolve()
        if run_dir != self._run_root and self._run_root not in run_dir.parents:
            raise ArtifactStoreError(f"run directory escapes run root: {run_id!r}")
        return run_dir

    def _resolve_ref_path(self, ref: ArtifactRef) -> Path:
        relative = Path(ref.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactStoreError(f"unsafe artifact path: {ref.relative_path!r}")
        path = (self._run_root / ref.relative_path).resolve()
        self._assert_contained(path, self._run_root)
        return path

    @staticmethod
    def _assert_contained(path: Path, root: Path) -> None:
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ArtifactStoreError(f"path escapes root: {path}")

    @staticmethod
    def _resolve_artifact_name(name: str) -> ArtifactName:
        try:
            return ArtifactName(name)
        except ValueError as exc:
            raise ArtifactStoreError(f"unknown artifact name: {name!r}") from exc

    def _read_manifest(self, run_dir: Path) -> dict[str, Any]:
        manifest_path = run_dir / "manifest.json"
        try:
            raw = manifest_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ArtifactStoreError(
                "run not initialized; call initialize_run first"
            ) from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ArtifactIntegrityError("manifest is corrupted") from exc
        if not isinstance(data, dict):
            raise ArtifactIntegrityError("manifest must be a JSON object")
        return data

    def _write_manifest(self, run_dir: Path, manifest: dict[str, Any]) -> None:
        manifest_path = run_dir / "manifest.json"
        try:
            self._atomic_write_bytes(
                manifest_path,
                _to_json_text(manifest).encode("utf-8"),
            )
        except OSError as exc:
            raise ArtifactStoreError(f"failed to write manifest: {exc}") from exc

    @staticmethod
    def _atomic_write_bytes(path: Path, content: bytes) -> None:
        """Write via a same-directory temp file plus atomic replace.

        ``Path.replace`` delegates to ``os.replace``; on Windows this maps to
        ``MoveFileEx`` with ``MOVEFILE_REPLACE_EXISTING``, which is atomic when
        source and destination are on the same volume (same directory here).
        """
        temp_path = path.with_name(f".{path.name}.tmp")
        try:
            with temp_path.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temp_path.replace(path)
        except Exception:
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)
            raise
