"""Unit tests for WorkflowExportAdapter (S3-M4)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pymatgen.core.structure import Structure

from materials_screening.errors import ExportConflictError, ExportError
from materials_screening.formatting.export_files import CSV_COLUMNS
from materials_screening.models import (
    FilterTrace,
    FloatRange,
    MaterialRecord,
    RunMetadata,
    ScreeningRequest,
    ScreeningResult,
    ValidationReport,
)
from materials_screening.services.ranking_service import RankingService
from materials_screening.workflow import export_adapter
from materials_screening.workflow.export_adapter import (
    EXPORT_MANIFEST_SCHEMA_VERSION,
    WorkflowExportAdapter,
)

RETRIEVED_AT = datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


def _structure_dict() -> dict[str, object]:
    structure = Structure(
        lattice=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Fe", "O"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    return structure.as_dict()


def _record(
    structure: dict[str, object] | None = None,
) -> MaterialRecord:
    return MaterialRecord(
        source="mock",
        material_id="mp-1",
        formula_pretty="Fe2O3",
        elements=("Fe", "O"),
        band_gap_ev=2.0,
        is_metal=False,
        structure_dict=structure,
    )


def _build_result(
    *,
    request: ScreeningRequest | None = None,
    run_id: str = "run-1",
    records: tuple[MaterialRecord, ...] | None = None,
) -> ScreeningResult:
    screening_request = request or ScreeningRequest(
        band_gap_ev=FloatRange(min=1.0, max=3.0),
        limit=10,
    )
    items = records if records is not None else (_record(),)
    ranked = RankingService().rank(items, screening_request)
    metadata = RunMetadata(
        run_id=run_id,
        started_at=RETRIEVED_AT,
        finished_at=RETRIEVED_AT,
        source="mock",
        database_version="fixture-v1",
        mp_api_version=None,
        pymatgen_version=None,
        application_version="test",
        query_fingerprint="abc",
    )
    return ScreeningResult(
        request=screening_request,
        metadata=metadata,
        retrieved_count=len(items),
        passed_filter_count=len(items),
        ranked_materials=ranked,
        filter_trace=FilterTrace(steps=(), rejections=()),
        validation=ValidationReport(
            passed=True,
            checked_material_ids=tuple(item.record.material_id for item in ranked),
        ),
    )


def _exports_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs" / "run-1" / "exports"


class TestWorkflowExportAdapter:
    def test_exports_all_files_and_manifest(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        result = _build_result()
        exported = adapter.export(result)

        exports_dir = _exports_dir(tmp_path)
        for name in (
            "request.json",
            "result.json",
            "provenance.json",
            "candidates.csv",
            "report.md",
            "export_manifest.json",
        ):
            assert (exports_dir / name).is_file()
        manifest = json.loads(
            (exports_dir / "export_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["schema_version"] == EXPORT_MANIFEST_SCHEMA_VERSION
        assert manifest["run_id"] == "run-1"
        assert len(manifest["request_hash"]) == 64
        assert len(manifest["result_hash"]) == 64
        assert "candidates.csv" in manifest["files"]
        assert exported.run_dir == exports_dir
        assert exported.files

    def test_csv_has_utf8_bom_and_columns(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        adapter.export(_build_result())
        csv_bytes = (_exports_dir(tmp_path) / "candidates.csv").read_bytes()
        assert csv_bytes.startswith(b"\xef\xbb\xbf")
        header = csv_bytes.decode("utf-8-sig").splitlines()[0]
        assert header.split(",") == list(CSV_COLUMNS)

    def test_same_input_reuses_existing_export(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        result = _build_result()
        first = adapter.export(result)
        manifest_path = _exports_dir(tmp_path) / "export_manifest.json"
        manifest_before = manifest_path.read_bytes()

        second = adapter.export(result)
        assert second.run_dir == first.run_dir
        assert manifest_path.read_bytes() == manifest_before
        assert second.warnings == first.warnings
        assert {path.name for path in second.files} == {
            path.name for path in first.files
        }

    def test_different_request_hash_conflicts(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        adapter.export(_build_result())
        changed_request = ScreeningRequest(
            band_gap_ev=FloatRange(min=1.0, max=4.0),
            limit=10,
        )
        with pytest.raises(ExportConflictError):
            adapter.export(_build_result(request=changed_request))

    def test_different_result_hash_conflicts(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        first = _build_result()
        adapter.export(first)
        changed = first.model_copy(
            update={
                "metadata": first.metadata.model_copy(
                    update={"database_version": "fixture-v2"}
                )
            }
        )
        with pytest.raises(ExportConflictError):
            adapter.export(changed)

    def test_missing_file_fails(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        adapter.export(_build_result())
        (_exports_dir(tmp_path) / "candidates.csv").unlink()
        with pytest.raises(ExportError, match="missing"):
            adapter.export(_build_result())

    def test_corrupted_file_fails(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        adapter.export(_build_result())
        report_path = _exports_dir(tmp_path) / "report.md"
        report_path.write_text("tampered", encoding="utf-8")
        with pytest.raises(ExportError, match="hash mismatch"):
            adapter.export(_build_result())

    def test_fixed_directory_no_timestamp_dirs(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        result = _build_result()
        first = adapter.export(result)
        second = adapter.export(result)
        assert first.run_dir == second.run_dir == _exports_dir(tmp_path)
        assert sorted(
            path.name for path in (tmp_path / "runs" / "run-1").iterdir()
        ) == ["exports"]

    def test_no_structure_only_warns(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        result = _build_result(records=(_record(structure=None),))
        exported = adapter.export(result)
        assert any("no structure" in warning for warning in exported.warnings)
        assert not list((_exports_dir(tmp_path) / "cif").glob("*.cif"))
        manifest = json.loads(
            (_exports_dir(tmp_path) / "export_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["warnings"] == list(exported.warnings)

    def test_cif_exported(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        result = _build_result(records=(_record(structure=_structure_dict()),))
        exported = adapter.export(result)
        cif_path = _exports_dir(tmp_path) / "cif" / "mp-1.cif"
        assert cif_path.is_file()
        assert cif_path in exported.files
        manifest = json.loads(
            (_exports_dir(tmp_path) / "export_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert "cif/mp-1.cif" in manifest["files"]

    def test_unsafe_run_id_rejected(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        result = _build_result(run_id="../evil")
        with pytest.raises(ExportError, match="unsafe run_id"):
            adapter.export(result)

    def test_corrupted_manifest_fails(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        adapter.export(_build_result())
        manifest_path = _exports_dir(tmp_path) / "export_manifest.json"
        manifest_path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ExportError, match="unreadable"):
            adapter.export(_build_result())

    def test_non_object_manifest_fails(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        adapter.export(_build_result())
        manifest_path = _exports_dir(tmp_path) / "export_manifest.json"
        manifest_path.write_text("[]", encoding="utf-8")
        with pytest.raises(ExportError, match="JSON object"):
            adapter.export(_build_result())

    def test_invalid_files_map_fails(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        adapter.export(_build_result())
        manifest_path = _exports_dir(tmp_path) / "export_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = "not-a-map"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ExportError, match="no files map"):
            adapter.export(_build_result())

    def test_invalid_file_entry_type_fails(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        adapter.export(_build_result())
        manifest_path = _exports_dir(tmp_path) / "export_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["report.md"] = 123
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ExportError, match="invalid"):
            adapter.export(_build_result())

    def test_manifest_relative_path_escape_fails(self, tmp_path: Path) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")
        adapter.export(_build_result())
        manifest_path = _exports_dir(tmp_path) / "export_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["../outside.json"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ExportError, match="escapes root"):
            adapter.export(_build_result())

    def test_write_failure_wrapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = WorkflowExportAdapter(tmp_path / "runs")

        def _fail_write(path: Path, content: str, encoding: str = "utf-8") -> None:
            raise OSError("simulated write failure")

        monkeypatch.setattr(export_adapter, "atomic_write_text", _fail_write)
        with pytest.raises(ExportError, match="export failed"):
            adapter.export(_build_result())
