"""Unit tests for ValidationService (M5)."""

from datetime import UTC, datetime

from materials_screening.models import (
    CrystalSystem,
    FilterTrace,
    FloatRange,
    MaterialRecord,
    PropertyProvenance,
    PropertyValueType,
    RunMetadata,
    ScreeningRequest,
    ScreeningResult,
    SymmetryInfo,
    ValidationReport,
)
from materials_screening.services.ranking_service import RankingService
from materials_screening.services.validation_service import ValidationService

RETRIEVED_AT = datetime(2026, 8, 5, 0, 0, tzinfo=UTC)
DATABASE_VERSION = "v2026.08"

KEY_PROPERTIES: tuple[str, ...] = (
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
        "structure_dict": {"lattice": {"a": 5.0}, "sites": []},
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
        for name in KEY_PROPERTIES
        if getattr(record, name) is not None
    )
    return record.model_copy(update={"provenance": provenance})


def _build_result(
    records: tuple[MaterialRecord, ...],
    request: ScreeningRequest | None = None,
    database_version: str | None = DATABASE_VERSION,
) -> ScreeningResult:
    screening_request = request or ScreeningRequest()
    ranked = RankingService().rank(records, screening_request)
    metadata = RunMetadata(
        run_id="run-1",
        started_at=RETRIEVED_AT,
        finished_at=RETRIEVED_AT,
        source="mock",
        database_version=database_version,
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
            passed=False, errors=(), warnings=(), checked_material_ids=()
        ),
    )


class TestValidResult:
    def test_valid_result_passes(self) -> None:
        records = (
            _with_provenance(_record(material_id="mp-1")),
            _with_provenance(_record(material_id="mp-2")),
        )
        report = ValidationService().validate(_build_result(records))
        assert report.passed is True
        assert report.errors == ()
        assert report.warnings == ()
        assert report.checked_material_ids == ("mp-1", "mp-2")

    def test_validator_does_not_modify_result(self) -> None:
        result = _build_result((_with_provenance(_record()),))
        before = result.model_dump(mode="json")
        ValidationService().validate(result)
        assert result.model_dump(mode="json") == before


class TestTamperedResults:
    def test_duplicate_id_is_error(self) -> None:
        records = (
            _with_provenance(_record(material_id="mp-1")),
            _with_provenance(_record(material_id="mp-1")),
        )
        report = ValidationService().validate(_build_result(records))
        assert report.passed is False
        assert any("duplicate material id" in error for error in report.errors)

    def test_broken_rank_sequence_is_error(self) -> None:
        result = _build_result((_with_provenance(_record()),))
        tampered = result.model_copy(
            update={
                "ranked_materials": (
                    result.ranked_materials[0].model_copy(update={"rank": 3}),
                )
            }
        )
        report = ValidationService().validate(tampered)
        assert report.passed is False
        assert any("rank sequence" in error for error in report.errors)

    def test_non_monotonic_total_is_error(self) -> None:
        high = _with_provenance(_record(material_id="mp-a", band_gap_ev=2.0))
        low = _with_provenance(_record(material_id="mp-b", band_gap_ev=1.0))
        request = ScreeningRequest(
            band_gap_ev=FloatRange(min=1.0, max=3.0),
            target_band_gap_ev=2.0,
        )
        result = _build_result((high, low), request)
        ranked = result.ranked_materials
        assert ranked[0].record.material_id == "mp-a"
        tampered = result.model_copy(
            update={
                "ranked_materials": (
                    ranked[1].model_copy(update={"rank": 1}),
                    ranked[0].model_copy(update={"rank": 2}),
                )
            }
        )
        report = ValidationService().validate(tampered)
        assert report.passed is False
        assert any("not non-increasing" in error for error in report.errors)

    def test_total_score_tamper_is_error(self) -> None:
        result = _build_result((_with_provenance(_record()),))
        item = result.ranked_materials[0]
        tampered = result.model_copy(
            update={
                "ranked_materials": (
                    item.model_copy(update={"total_score": item.total_score + 0.1}),
                )
            }
        )
        report = ValidationService().validate(tampered)
        assert report.passed is False
        assert any("total_score mismatch" in error for error in report.errors)

    def test_breakdown_tamper_is_error(self) -> None:
        result = _build_result((_with_provenance(_record()),))
        item = result.ranked_materials[0]
        tampered_breakdown = item.score_breakdown.model_copy(update={"stability": 1.0})
        tampered = result.model_copy(
            update={
                "ranked_materials": (
                    item.model_copy(update={"score_breakdown": tampered_breakdown}),
                )
            }
        )
        report = ValidationService().validate(tampered)
        assert report.passed is False
        assert any("score breakdown mismatch" in error for error in report.errors)

    def test_out_of_range_total_is_error(self) -> None:
        result = _build_result((_with_provenance(_record()),))
        item = result.ranked_materials[0]
        tampered = result.model_copy(
            update={"ranked_materials": (item.model_copy(update={"total_score": 1.5}),)}
        )
        report = ValidationService().validate(tampered)
        assert report.passed is False
        assert any("total_score out of range" in error for error in report.errors)

    def test_missing_provenance_is_error(self) -> None:
        record = _record(material_id="mp-1")
        report = ValidationService().validate(_build_result((record,)))
        assert report.passed is False
        assert any("missing provenance" in error for error in report.errors)
        assert any("band_gap_ev" in error for error in report.errors)

    def test_limit_exceeded_is_error(self) -> None:
        records = (
            _with_provenance(_record(material_id="mp-1")),
            _with_provenance(_record(material_id="mp-2")),
        )
        result = _build_result(records, ScreeningRequest(limit=1))
        report = ValidationService().validate(result)
        assert report.passed is False
        assert any("exceeds limit" in error for error in report.errors)

    def test_excluded_element_violation_is_error(self) -> None:
        record = _with_provenance(_record(material_id="mp-1", elements=("Pb",)))
        result = _build_result((record,), ScreeningRequest(excluded_elements=("Pb",)))
        report = ValidationService().validate(result)
        assert report.passed is False
        assert any("violates hard constraints" in error for error in report.errors)
        assert any("contains_excluded_element" in error for error in report.errors)

    def test_numeric_range_violation_is_error(self) -> None:
        record = _with_provenance(_record(material_id="mp-1", band_gap_ev=0.5))
        result = _build_result(
            (record,), ScreeningRequest(band_gap_ev=FloatRange(min=1.0, max=2.0))
        )
        report = ValidationService().validate(result)
        assert report.passed is False
        assert any("band_gap_below_min" in error for error in report.errors)

    def test_metallicity_violation_is_error(self) -> None:
        record = _with_provenance(_record(material_id="mp-1", is_metal=True))
        result = _build_result((record,), ScreeningRequest(is_metal=False))
        report = ValidationService().validate(result)
        assert report.passed is False
        assert any("metallicity_mismatch" in error for error in report.errors)

    def test_deprecated_candidate_is_error(self) -> None:
        record = _with_provenance(_record(material_id="mp-1", deprecated=True))
        result = _build_result((record,))
        report = ValidationService().validate(result)
        assert report.passed is False
        assert any("violates hard constraints" in error for error in report.errors)


class TestWarnings:
    def test_warnings_do_not_fail_validation(self) -> None:
        record = _with_provenance(
            _record(
                material_id="mp-1",
                structure_dict=None,
                formation_energy_ev_atom=None,
                density_g_cm3=None,
                is_gap_direct=False,
                theoretical=True,
            )
        )
        result = _build_result((record,), database_version=None)
        report = ValidationService().validate(result)
        assert report.passed is True
        assert report.errors == ()
        warning_text = "\n".join(report.warnings)
        assert "no structure" in warning_text
        assert "missing formation energy" in warning_text
        assert "missing density" in warning_text
        assert "non-direct band gap" in warning_text
        assert "theoretical material" in warning_text
        assert "database version unavailable" in warning_text
