"""File exports for screening results (M5)."""

import re
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from materials_screening.errors import ExportError
from materials_screening.formatting.export_files import (
    CSV_COLUMNS,  # noqa: F401 - re-exported for stage-1 compatibility
    build_csv,
    build_provenance_document,
    json_text,
    write_cif_files,
)
from materials_screening.formatting.export_files import (
    atomic_write_text as _atomic_write_text,
)
from materials_screening.formatting.markdown_report import build_markdown_report
from materials_screening.models import ScreeningResult

_RUN_ID_PATTERN = re.compile(r"^run_[A-Za-z0-9._-]+$")


class ExportResult(BaseModel):
    """Result of one export run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    run_dir: Path
    files: tuple[Path, ...]
    warnings: tuple[str, ...] = ()


class ExportService:
    """Write all run outputs atomically into a run directory."""

    def export(
        self,
        result: ScreeningResult,
        output_root: Path,
        *,
        include_cif: bool = True,
    ) -> ExportResult:
        run_id = result.metadata.run_id
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ExportError(f"unsafe run_id: {run_id!r}")
        run_dir = output_root / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            files: list[Path] = []
            warnings: list[str] = []

            request_path = run_dir / "request.json"
            _atomic_write_text(
                request_path,
                json_text(result.request.model_dump(mode="json")),
            )
            files.append(request_path)

            result_path = run_dir / "result.json"
            _atomic_write_text(
                result_path,
                json_text(result.model_dump(mode="json")),
            )
            files.append(result_path)

            provenance_path = run_dir / "provenance.json"
            _atomic_write_text(
                provenance_path,
                json_text(build_provenance_document(result)),
            )
            files.append(provenance_path)

            csv_path = run_dir / "candidates.csv"
            _atomic_write_text(csv_path, build_csv(result), encoding="utf-8-sig")
            files.append(csv_path)

            report_path = run_dir / "report.md"
            _atomic_write_text(report_path, build_markdown_report(result))
            files.append(report_path)

            if include_cif:
                cif_dir = run_dir / "cif"
                cif_files = write_cif_files(result, cif_dir, warnings)
                files.extend(cif_files)

            return ExportResult(
                run_id=run_id,
                run_dir=run_dir,
                files=tuple(files),
                warnings=tuple(warnings),
            )
        except Exception as exc:
            shutil.rmtree(run_dir, ignore_errors=True)
            if isinstance(exc, ExportError):
                raise
            raise ExportError(f"export failed: {exc}") from exc
