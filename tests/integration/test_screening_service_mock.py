"""End-to-end ScreeningService tests with the mock repository (M6)."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pymatgen.core.structure import Structure

from materials_screening.errors import ExportError, ValidationFailedError
from materials_screening.fingerprints import request_fingerprint
from materials_screening.models import (
    CrystalSystem,
    FloatRange,
    MaterialRecord,
    PropertyProvenance,
    PropertyValueType,
    ScreeningRequest,
    ScreeningResult,
    SymmetryInfo,
)
from materials_screening.repositories.mock import MockMaterialsRepository
from materials_screening.services.export_service import ExportResult, ExportService
from materials_screening.services.filter_service import FilterService
from materials_screening.services.ranking_service import RankingService
from materials_screening.services.screening_service import ScreeningService
from materials_screening.services.validation_service import ValidationService

FIXED_TIME = datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC)
DATABASE_VERSION = "v2026.08"


def _structure_dict() -> dict[str, object]:
    structure = Structure(
        lattice=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Fe", "O"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    return structure.as_dict()


def _record(**overrides: object) -> MaterialRecord:
    values: dict[str, object] = {
        "source": "mock",
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
            source="mock",
            source_material_id=record.material_id,
            value_type=PropertyValueType.DFT_CALCULATED,
            database_version=DATABASE_VERSION,
            retrieved_at=FIXED_TIME,
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


def _make_service(
    records: tuple[MaterialRecord, ...],
    export_service: ExportService | None = None,
) -> ScreeningService:
    return ScreeningService(
        repository=MockMaterialsRepository(records=records),
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=export_service or ExportService(),
        clock=lambda: FIXED_TIME,
    )


class TestScreeningServiceMock:
    def test_full_pipeline(self, tmp_path: Path) -> None:
        records = (
            _with_provenance(_record(material_id="mp-1")),
            _with_provenance(_record(material_id="mp-2")),
            _with_provenance(_record(material_id="mp-3", deprecated=True)),
        )
        request = ScreeningRequest()
        output = _make_service(records).run(request, tmp_path)

        assert output.result.request == request
        assert output.result.retrieved_count == 3
        assert output.result.passed_filter_count == 2
        assert [item.record.material_id for item in output.result.ranked_materials] == [
            "mp-1",
            "mp-2",
        ]
        assert [item.rank for item in output.result.ranked_materials] == [1, 2]
        assert output.result.validation.passed is True
        assert output.result.metadata.started_at == FIXED_TIME
        assert output.result.metadata.finished_at == FIXED_TIME
        assert output.result.metadata.source == "mock"
        assert output.result.metadata.database_version == "fixture-v1"
        assert output.result.metadata.query_fingerprint == request_fingerprint(request)
        assert output.result.metadata.run_id.startswith("run_20260805_120000_")
        assert len(output.result.metadata.run_id.split("_")[-1]) == 8

        names = {path.name for path in output.exports.run_dir.iterdir()}
        assert {
            "request.json",
            "result.json",
            "provenance.json",
            "candidates.csv",
            "report.md",
        } <= names
        assert (output.exports.run_dir / "cif" / "mp-1.cif").exists()
        report = (output.exports.run_dir / "report.md").read_text(encoding="utf-8")
        assert "本结果主要基于 Materials Project 中的计算数据" in report

    def test_zero_candidates(self, tmp_path: Path) -> None:
        output = _make_service(()).run(ScreeningRequest(), tmp_path)
        assert output.result.retrieved_count == 0
        assert output.result.passed_filter_count == 0
        assert output.result.ranked_materials == ()
        assert output.result.validation.passed is True
        csv_text = (output.exports.run_dir / "candidates.csv").read_text(
            encoding="utf-8-sig"
        )
        assert csv_text.count("\n") == 1

    def test_limit_truncates_after_ranking(self, tmp_path: Path) -> None:
        records = (
            _with_provenance(_record(material_id="mp-1", band_gap_ev=2.0)),
            _with_provenance(_record(material_id="mp-2", band_gap_ev=1.0)),
        )
        request = ScreeningRequest(
            band_gap_ev=FloatRange(min=1.0, max=3.0),
            target_band_gap_ev=2.0,
            limit=1,
        )
        output = _make_service(records).run(request, tmp_path)
        assert len(output.result.ranked_materials) == 1
        assert output.result.ranked_materials[0].record.material_id == "mp-1"

    def test_validation_failure_raises_and_does_not_export(
        self, tmp_path: Path
    ) -> None:
        service = _make_service((_record(material_id="mp-1"),))
        with pytest.raises(ValidationFailedError):
            service.run(ScreeningRequest(), tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_export_failure_raises_export_error(self, tmp_path: Path) -> None:
        class FailingExportService:
            def export(
                self,
                result: ScreeningResult,
                output_root: Path,
                *,
                include_cif: bool = True,
            ) -> ExportResult:
                raise ExportError("disk full")

        service = _make_service(
            (_with_provenance(_record()),),
            export_service=FailingExportService(),
        )
        with pytest.raises(ExportError, match="disk full"):
            service.run(ScreeningRequest(), tmp_path)
