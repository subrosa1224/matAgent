"""Unit tests for ExportService and the Markdown report (M5)."""

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pymatgen.core.structure import Structure

from materials_screening.errors import ExportError
from materials_screening.formatting.markdown_report import (
    DISCLAIMER,
    build_markdown_report,
)
from materials_screening.models import (
    CrystalSystem,
    FilterTrace,
    MaterialRecord,
    PropertyProvenance,
    PropertyValueType,
    RunMetadata,
    ScreeningRequest,
    ScreeningResult,
    SymmetryInfo,
    ValidationReport,
)
from materials_screening.services import export_service
from materials_screening.services.export_service import (
    CSV_COLUMNS,
    ExportService,
)
from materials_screening.services.ranking_service import RankingService

RUN_ID = "run_20260805_120000_a1b2c3d4"
DATABASE_VERSION = "v2026.08"
RETRIEVED_AT = datetime(2026, 8, 5, 0, 0, tzinfo=UTC)


def _structure_dict() -> dict[str, object]:
    structure = Structure(
        lattice=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Fe", "O"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    return structure.as_dict()


def _record(**overrides: object) -> MaterialRecord:
    values: dict[str, object] = {
        "source": "materials_project",
        "material_id": "mp-1",
        "formula_pretty": "Fe2O3",
        "elements": ("Fe", "O"),
        "chemsys": "Fe-O",
        "band_gap_ev": 2.0,
        "energy_above_hull_ev_atom": 0.01,
        "formation_energy_ev_atom": -1.5,
        "density_g_cm3": 5.2,
        "is_metal": False,
        "is_gap_direct": True,
        "is_stable": True,
        "theoretical": False,
        "deprecated": False,
        "symmetry": SymmetryInfo(
            crystal_system=CrystalSystem.TRIGONAL,
            symbol="R-3c",
            number=167,
        ),
        "structure_dict": _structure_dict(),
        "provenance": (),
    }
    values.update(overrides)
    return MaterialRecord.model_validate(values)


def _with_provenance(record: MaterialRecord) -> MaterialRecord:
    provenance = tuple(
        PropertyProvenance(
            property_name=name,
            source="materials_project",
            source_material_id=record.material_id,
            value_type=PropertyValueType.DFT_CALCULATED,
            database_version=DATABASE_VERSION,
            retrieved_at=RETRIEVED_AT,
        )
        for name in (
            "band_gap_ev",
            "energy_above_hull_ev_atom",
            "formation_energy_ev_atom",
            "density_g_cm3",
            "is_metal",
            "is_gap_direct",
            "is_stable",
            "theoretical",
            "symmetry",
            "structure_dict",
        )
        if getattr(record, name) is not None
    )
    return record.model_copy(update={"provenance": provenance})


def _build_result(
    records: tuple[MaterialRecord, ...],
    request: ScreeningRequest | None = None,
    run_id: str = RUN_ID,
) -> ScreeningResult:
    screening_request = request or ScreeningRequest()
    ranked = RankingService().rank(records, screening_request)
    metadata = RunMetadata(
        run_id=run_id,
        started_at=RETRIEVED_AT,
        finished_at=RETRIEVED_AT,
        source="mock",
        database_version=DATABASE_VERSION,
        mp_api_version="0.46.4",
        pymatgen_version="2026.5.4",
        application_version="0.1.0",
        query_fingerprint="abc",
    )
    return ScreeningResult(
        request=screening_request,
        metadata=metadata,
        retrieved_count=len(records),
        passed_filter_count=len(records),
        ranked_materials=ranked,
        filter_trace=FilterTrace(steps=(), rejections=()),
        validation=ValidationReport(
            passed=True,
            errors=(),
            warnings=(),
            checked_material_ids=tuple(item.record.material_id for item in ranked),
        ),
    )


class TestExportOutputs:
    def test_exports_all_files_without_temp_leftovers(self, tmp_path: Path) -> None:
        result = _build_result((_record(),))
        exported = ExportService().export(result, tmp_path)
        names = {path.name for path in exported.run_dir.iterdir()}
        assert {
            "request.json",
            "result.json",
            "provenance.json",
            "candidates.csv",
            "report.md",
        } <= names
        assert (exported.run_dir / "cif" / "mp-1.cif").exists()
        assert list(tmp_path.rglob("*.tmp")) == []
        assert exported.run_id == RUN_ID
        assert exported.run_dir == tmp_path / RUN_ID

    def test_result_json_round_trip(self, tmp_path: Path) -> None:
        result = _build_result((_record(), _record(material_id="mp-2")))
        exported = ExportService().export(result, tmp_path)
        raw = (exported.run_dir / "result.json").read_text(encoding="utf-8")
        restored = ScreeningResult.model_validate(json.loads(raw))
        assert restored.model_dump(mode="json") == result.model_dump(mode="json")

    def test_request_json_matches(self, tmp_path: Path) -> None:
        request = ScreeningRequest(required_elements=("Fe",), limit=5)
        result = _build_result((_record(),), request)
        exported = ExportService().export(result, tmp_path)
        raw = (exported.run_dir / "request.json").read_text(encoding="utf-8")
        restored = ScreeningRequest.model_validate(json.loads(raw))
        assert restored == request

    def test_provenance_json_content(self, tmp_path: Path) -> None:
        result = _build_result((_with_provenance(_record()),))
        exported = ExportService().export(result, tmp_path)
        raw = (exported.run_dir / "provenance.json").read_text(encoding="utf-8")
        document = json.loads(raw)
        material = document["materials"][0]
        assert material["material_id"] == "mp-1"
        names = {entry["property_name"] for entry in material["provenance"]}
        assert "band_gap_ev" in names


class TestCsv:
    def test_columns_match_document(self, tmp_path: Path) -> None:
        result = _build_result((_record(),))
        exported = ExportService().export(result, tmp_path)
        text = (exported.run_dir / "candidates.csv").read_text(encoding="utf-8-sig")
        rows = list(csv.reader(io.StringIO(text)))
        assert rows[0] == list(CSV_COLUMNS)
        assert len(rows[0]) == 23

    def test_csv_has_utf8_bom(self, tmp_path: Path) -> None:
        result = _build_result((_record(),))
        exported = ExportService().export(result, tmp_path)
        raw = (exported.run_dir / "candidates.csv").read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")

    def test_csv_row_values(self, tmp_path: Path) -> None:
        result = _build_result((_record(),))
        exported = ExportService().export(result, tmp_path)
        text = (exported.run_dir / "candidates.csv").read_text(encoding="utf-8-sig")
        rows = list(csv.reader(io.StringIO(text)))
        row = rows[1]
        assert row[0] == "1"
        assert row[1] == "mp-1"
        assert row[2] == "Fe2O3"
        assert row[3] == "Fe O"
        assert row[4] == "Fe-O"
        assert row[5] == "2.0"
        assert row[6] == "0.01"
        assert row[7] == "-1.5"
        assert row[8] == "5.2"
        assert row[9] == "Trigonal"
        assert row[10] == "R-3c"
        assert row[11] == "167"
        assert row[12] == "False"
        assert row[13] == "True"
        assert row[14] == "True"
        assert row[15] == "False"
        assert row[16] == str(result.ranked_materials[0].total_score)
        assert row[21] == "materials_project"
        assert row[22] == DATABASE_VERSION


class TestCif:
    def test_no_cif_skips_cif_dir(self, tmp_path: Path) -> None:
        result = _build_result((_record(),))
        exported = ExportService().export(result, tmp_path, include_cif=False)
        assert not (exported.run_dir / "cif").exists()

    def test_cif_exported(self, tmp_path: Path) -> None:
        result = _build_result((_record(),))
        exported = ExportService().export(result, tmp_path)
        cif_path = exported.run_dir / "cif" / "mp-1.cif"
        assert cif_path.exists()
        text = cif_path.read_text(encoding="utf-8")
        assert "data_" in text

    def test_no_structure_skips_cif_with_warning(self, tmp_path: Path) -> None:
        record = _record(material_id="mp-1", structure_dict=None)
        result = _build_result((record,))
        exported = ExportService().export(result, tmp_path)
        assert not (exported.run_dir / "cif" / "mp-1.cif").exists()
        assert any("no structure" in warning for warning in exported.warnings)

    def test_material_id_sanitized(self, tmp_path: Path) -> None:
        record = _record(material_id="mp:1/2")
        result = _build_result((record,))
        exported = ExportService().export(result, tmp_path)
        cif_dir = exported.run_dir / "cif"
        names = {path.name for path in cif_dir.iterdir()}
        assert "mp12.cif" in names

    def test_traversal_material_id_cannot_escape_cif_dir(self, tmp_path: Path) -> None:
        record = _record(material_id="../escape")
        result = _build_result((record,))
        exported = ExportService().export(result, tmp_path)
        cif_dir = exported.run_dir / "cif"
        names = {path.name for path in cif_dir.iterdir()}
        assert "..escape.cif" in names
        assert not (tmp_path / "escape.cif").exists()
        for path in cif_dir.iterdir():
            assert cif_dir.resolve() in path.resolve().parents

    def test_cif_filename_collision_raises_and_cleans(self, tmp_path: Path) -> None:
        records = (
            _record(material_id="ab"),
            _record(material_id="a/b"),
        )
        result = _build_result(records)
        run_dir = tmp_path / RUN_ID
        with pytest.raises(ExportError, match="collision"):
            ExportService().export(result, tmp_path)
        assert not run_dir.exists()


class TestAtomicity:
    def test_atomic_write_removes_temp_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "out.json"

        def broken_replace(self: Path, other: Path) -> Path:
            raise OSError("simulated replace failure")

        monkeypatch.setattr(Path, "replace", broken_replace)
        with pytest.raises(OSError):
            export_service._atomic_write_text(target, "{}")
        assert not target.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_partial_failure_cleans_run_dir_and_temp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = export_service._atomic_write_text

        def failing_write(path: Path, content: str, encoding: str = "utf-8") -> None:
            if path.name == "result.json":
                temp = path.with_name(f".{path.name}.tmp")
                temp.write_text("partial", encoding=encoding)
                raise OSError("simulated write failure")
            original(path, content, encoding)

        monkeypatch.setattr(export_service, "_atomic_write_text", failing_write)
        result = _build_result((_record(),))
        run_dir = tmp_path / RUN_ID
        with pytest.raises(ExportError):
            ExportService().export(result, tmp_path)
        assert not run_dir.exists()
        assert list(tmp_path.rglob("*.tmp")) == []

    def test_unsafe_run_id_raises(self, tmp_path: Path) -> None:
        result = _build_result((_record(),))
        tampered = result.model_copy(
            update={
                "metadata": result.metadata.model_copy(update={"run_id": "run/../x"})
            }
        )
        with pytest.raises(ExportError, match="unsafe run_id"):
            ExportService().export(tampered, tmp_path)
        assert not (tmp_path / "run").exists()

    @pytest.mark.parametrize("run_id", [".", ".."])
    def test_dot_run_id_cannot_escape_output_root(
        self, tmp_path: Path, run_id: str
    ) -> None:
        result = _build_result((_record(),), run_id=run_id)
        with pytest.raises(ExportError, match="unsafe run_id"):
            ExportService().export(result, tmp_path)
        assert list(tmp_path.iterdir()) == []


class TestMarkdownReport:
    def test_report_contains_fixed_sections_and_disclaimer(
        self, tmp_path: Path
    ) -> None:
        result = _build_result((_record(),))
        exported = ExportService().export(result, tmp_path)
        text = (exported.run_dir / "report.md").read_text(encoding="utf-8")
        for section in (
            "# 材料筛选结果",
            "## 筛选条件",
            "## 数据来源",
            "## 筛选统计",
            "## 候选材料",
            "## 排名方法",
            "## 验证结果",
            "## 警告",
            "## 科学说明",
        ):
            assert section in text
        assert DISCLAIMER in text
        assert "LLM" not in text

    def test_report_uses_only_result_data(self) -> None:
        result = _build_result((_record(material_id="mp-1"),))
        text = build_markdown_report(result)
        assert "mp-1" in text
        assert "Fe2O3" in text
        assert "mock" in text
