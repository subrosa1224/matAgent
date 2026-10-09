"""Idempotent workflow export adapter (S3-M4)."""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from materials_screening.errors import ExportConflictError, ExportError
from materials_screening.formatting.export_files import (
    atomic_write_text,
    build_csv,
    build_provenance_document,
    json_text,
    write_cif_files,
)
from materials_screening.formatting.markdown_report import build_markdown_report
from materials_screening.models import ScreeningResult
from materials_screening.services.export_service import ExportResult

EXPORT_MANIFEST_SCHEMA_VERSION = "workflow-export-manifest-v1"
_EXPORT_DIR_NAME = "exports"
_MANIFEST_FILE = "export_manifest.json"
_SAFE_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,253}$")


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


class WorkflowExportAdapter:
    """Export a validated ScreeningResult into a fixed run exports directory.

    Writes to ``<run_root>/<run_id>/exports/`` (never a new timestamp
    directory). Re-exporting the same request/result reuses the existing
    files; different content conflicts; missing or corrupted files fail.
    """

    def __init__(self, run_root: Path) -> None:
        self._run_root = run_root.expanduser().resolve()

    def export(self, result: ScreeningResult) -> ExportResult:
        run_id = result.metadata.run_id
        if not _SAFE_RUN_ID_PATTERN.fullmatch(run_id):
            raise ExportError(f"unsafe run_id: {run_id!r}")
        exports_dir = self._run_root / run_id / _EXPORT_DIR_NAME
        self._assert_contained(exports_dir, self._run_root)

        request_text = json_text(result.request.model_dump(mode="json"))
        result_text = json_text(result.model_dump(mode="json"))
        request_hash = _sha256_hex(request_text)
        result_hash = _sha256_hex(result_text)
        manifest_path = exports_dir / _MANIFEST_FILE
        if manifest_path.exists():
            return self._reuse(
                exports_dir=exports_dir,
                manifest_path=manifest_path,
                request_hash=request_hash,
                result_hash=result_hash,
            )
        return self._fresh_export(
            exports_dir=exports_dir,
            run_id=run_id,
            result=result,
            request_text=request_text,
            result_text=result_text,
            request_hash=request_hash,
            result_hash=result_hash,
        )

    def _fresh_export(
        self,
        *,
        exports_dir: Path,
        run_id: str,
        result: ScreeningResult,
        request_text: str,
        result_text: str,
        request_hash: str,
        result_hash: str,
    ) -> ExportResult:
        exports_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        try:
            atomic_write_text(exports_dir / "request.json", request_text)
            atomic_write_text(exports_dir / "result.json", result_text)
            atomic_write_text(
                exports_dir / "provenance.json",
                json_text(build_provenance_document(result)),
            )
            atomic_write_text(
                exports_dir / "candidates.csv",
                build_csv(result),
                encoding="utf-8-sig",
            )
            atomic_write_text(
                exports_dir / "report.md",
                build_markdown_report(result),
            )
            cif_dir = exports_dir / "cif"
            write_cif_files(result, cif_dir, warnings)
        except OSError as exc:
            raise ExportError(f"export failed: {exc}") from exc

        manifest = {
            "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": _iso_now(),
            "request_hash": request_hash,
            "result_hash": result_hash,
            "warnings": list(warnings),
            "files": self._collect_files(exports_dir),
        }
        atomic_write_text(
            exports_dir / _MANIFEST_FILE,
            json_text(manifest),
        )
        files = tuple(
            path
            for path in sorted(exports_dir.rglob("*"))
            if path.is_file() and path.name != _MANIFEST_FILE
        )
        return ExportResult(
            run_id=run_id,
            run_dir=exports_dir,
            files=files,
            warnings=tuple(warnings),
        )

    def _reuse(
        self,
        *,
        exports_dir: Path,
        manifest_path: Path,
        request_hash: str,
        result_hash: str,
    ) -> ExportResult:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExportError("export manifest is unreadable") from exc
        if not isinstance(manifest, dict):
            raise ExportError("export manifest must be a JSON object")
        if (
            manifest.get("request_hash") != request_hash
            or manifest.get("result_hash") != result_hash
        ):
            raise ExportConflictError(
                "export manifest exists with different request/result hashes; "
                "refusing to overwrite"
            )
        files = self._verify_files(exports_dir, manifest)
        return ExportResult(
            run_id=str(manifest.get("run_id")),
            run_dir=exports_dir,
            files=files,
            warnings=tuple(manifest.get("warnings", ())),
        )

    def _verify_files(
        self, exports_dir: Path, manifest: dict[str, Any]
    ) -> tuple[Path, ...]:
        files: list[Path] = []
        recorded = manifest.get("files")
        if not isinstance(recorded, dict):
            raise ExportError("export manifest has no files map")
        for relative, expected_sha in recorded.items():
            if not isinstance(relative, str) or not isinstance(expected_sha, str):
                raise ExportError("export manifest files map is invalid")
            path = self._resolve_export_path(exports_dir, relative)
            if not path.is_file():
                raise ExportError(f"export file missing: {relative}")
            try:
                actual_sha = _sha256_hex(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError) as exc:
                raise ExportError(f"export file unreadable: {relative}") from exc
            if actual_sha != expected_sha:
                raise ExportError(f"export file hash mismatch: {relative}")
            files.append(path)
        return tuple(files)

    @staticmethod
    def _collect_files(exports_dir: Path) -> dict[str, str]:
        files: dict[str, str] = {}
        for path in sorted(exports_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(exports_dir).as_posix()
            files[relative] = _sha256_hex(path.read_text(encoding="utf-8"))
        return files

    def _resolve_export_path(self, exports_dir: Path, relative: str) -> Path:
        path = (exports_dir / relative).resolve()
        self._assert_contained(path, exports_dir)
        return path

    @staticmethod
    def _assert_contained(path: Path, root: Path) -> None:
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ExportError(f"path escapes root: {path}")
