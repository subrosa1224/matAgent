"""Unit tests for domain models (M2)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from materials_screening.models import (
    CrystalSystem,
    FilterStep,
    FilterTrace,
    FloatRange,
    MaterialRecord,
    PropertyProvenance,
    PropertyValueType,
    RankedMaterial,
    Rejection,
    RunMetadata,
    ScoreBreakdown,
    ScreeningRequest,
    ScreeningResult,
    SymmetryInfo,
    ValidationReport,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

ALL_MODEL_CLASSES: tuple[type[BaseModel], ...] = (
    FloatRange,
    ScreeningRequest,
    PropertyProvenance,
    SymmetryInfo,
    MaterialRecord,
    Rejection,
    FilterStep,
    FilterTrace,
    ScoreBreakdown,
    RankedMaterial,
    ValidationReport,
    RunMetadata,
    ScreeningResult,
)


@pytest.mark.parametrize("model_cls", ALL_MODEL_CLASSES)
def test_all_models_extra_forbid_and_frozen(model_cls: type[BaseModel]) -> None:
    assert model_cls.model_config.get("extra") == "forbid"
    assert model_cls.model_config.get("frozen") is True


class TestFloatRange:
    def test_min_only(self) -> None:
        assert FloatRange(min=1.0) == FloatRange(min=1.0)

    def test_max_only(self) -> None:
        assert FloatRange(max=2.0) == FloatRange(max=2.0)

    def test_both_bounds_valid(self) -> None:
        assert FloatRange(min=1.0, max=2.0) == FloatRange(min=1.0, max=2.0)

    def test_min_greater_than_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FloatRange(min=2.0, max=1.0)

    def test_no_bounds_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FloatRange()

    def test_min_equals_max_allowed(self) -> None:
        assert FloatRange(min=1.0, max=1.0) == FloatRange(min=1.0, max=1.0)

    def test_nan_bound_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FloatRange(min=float("nan"))

    def test_infinite_bound_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FloatRange(min=float("-inf"))
        with pytest.raises(ValidationError):
            FloatRange(max=float("inf"))

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FloatRange.model_validate({"min": 0.0, "unexpected": 1.0})


class TestScreeningRequest:
    def test_defaults(self) -> None:
        request = ScreeningRequest()
        assert request.required_elements == ()
        assert request.excluded_elements == ()
        assert request.limit == 10
        assert request.is_metal is False

    def test_negative_band_gap_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest(band_gap_ev=FloatRange(min=-0.1, max=2.0))

    def test_negative_band_gap_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest(band_gap_ev=FloatRange(min=0.0, max=-0.1))

    def test_negative_hull_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest(energy_above_hull_ev_atom=FloatRange(min=0.0, max=-0.01))

    def test_negative_hull_min_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest(energy_above_hull_ev_atom=FloatRange(min=-0.01, max=0.1))

    def test_zero_band_gap_range_allowed(self) -> None:
        request = ScreeningRequest(band_gap_ev=FloatRange(min=0.0, max=0.0))
        assert request.band_gap_ev == FloatRange(min=0.0, max=0.0)

    def test_invalid_element_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest(required_elements=("Xx",))

    def test_required_excluded_conflict_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest(required_elements=("Pb",), excluded_elements=("Pb",))

    def test_conflict_detected_after_normalization(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest(required_elements=("pb",), excluded_elements=("PB",))

    def test_required_excluded_disjoint_ok(self) -> None:
        request = ScreeningRequest(required_elements=("Li",), excluded_elements=("Pb",))
        assert request.required_elements == ("Li",)
        assert request.excluded_elements == ("Pb",)

    def test_elements_normalized_deduped_sorted(self) -> None:
        request = ScreeningRequest(required_elements=("O", "Li", "O", "Fe", "FE"))
        assert request.required_elements == ("Fe", "Li", "O")

    def test_chemsys_normalized(self) -> None:
        assert ScreeningRequest(chemsys="O-Li-Fe").chemsys == "Fe-Li-O"

    def test_empty_strings_become_none(self) -> None:
        request = ScreeningRequest(chemsys="", formula="")
        assert request.chemsys is None
        assert request.formula is None

    def test_whitespace_only_chemsys_becomes_none(self) -> None:
        assert ScreeningRequest(chemsys="   ").chemsys is None

    def test_spacegroup_number_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest(spacegroup_numbers=(0,))
        with pytest.raises(ValidationError):
            ScreeningRequest(spacegroup_numbers=(231,))

    def test_spacegroup_boundaries_allowed(self) -> None:
        request = ScreeningRequest(spacegroup_numbers=(1, 230))
        assert request.spacegroup_numbers == (1, 230)

    def test_spacegroup_numbers_deduped_sorted(self) -> None:
        request = ScreeningRequest(spacegroup_numbers=(230, 1, 2, 2))
        assert request.spacegroup_numbers == (1, 2, 230)

    def test_target_band_gap_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest(target_band_gap_ev=-0.1)

    def test_target_within_range_ok(self) -> None:
        request = ScreeningRequest(
            band_gap_ev=FloatRange(min=1.2, max=2.0),
            target_band_gap_ev=1.6,
        )
        assert request.target_band_gap_ev == 1.6

    def test_target_below_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest(
                band_gap_ev=FloatRange(min=1.2, max=2.0),
                target_band_gap_ev=1.0,
            )

    def test_target_above_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest(
                band_gap_ev=FloatRange(min=1.2, max=2.0),
                target_band_gap_ev=2.5,
            )

    def test_target_at_range_boundaries_allowed(self) -> None:
        at_min = ScreeningRequest(
            band_gap_ev=FloatRange(min=1.2, max=2.0),
            target_band_gap_ev=1.2,
        )
        at_max = ScreeningRequest(
            band_gap_ev=FloatRange(min=1.2, max=2.0),
            target_band_gap_ev=2.0,
        )
        assert at_min.target_band_gap_ev == 1.2
        assert at_max.target_band_gap_ev == 2.0

    def test_nan_target_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest(target_band_gap_ev=float("nan"))

    def test_infinite_band_gap_bound_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest(band_gap_ev=FloatRange(max=float("inf")))

    def test_target_with_min_bound_only_ok(self) -> None:
        request = ScreeningRequest(
            band_gap_ev=FloatRange(min=1.2),
            target_band_gap_ev=1.5,
        )
        assert request.target_band_gap_ev == 1.5

    def test_target_with_max_bound_only_ok(self) -> None:
        request = ScreeningRequest(
            band_gap_ev=FloatRange(max=2.0),
            target_band_gap_ev=1.5,
        )
        assert request.target_band_gap_ev == 1.5

    def test_target_without_range_ok(self) -> None:
        assert ScreeningRequest(target_band_gap_ev=1.6).target_band_gap_ev == 1.6

    def test_crystal_system_accepted(self) -> None:
        assert (
            ScreeningRequest(crystal_system="Cubic").crystal_system
            == CrystalSystem.CUBIC
        )

    def test_crystal_system_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest(crystal_system="cubic")
        with pytest.raises(ValidationError):
            ScreeningRequest(crystal_system="Nope")

    def test_limit_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest(limit=0)
        with pytest.raises(ValidationError):
            ScreeningRequest(limit=101)

    def test_limit_boundaries_allowed(self) -> None:
        assert ScreeningRequest(limit=1).limit == 1
        assert ScreeningRequest(limit=100).limit == 100

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest.model_validate({"limit": 10, "junk": 1})

    def test_unknown_field_in_json_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreeningRequest.model_validate_json('{"junk": 1}')


def _make_record(**overrides: object) -> MaterialRecord:
    values: dict[str, object] = {
        "source": "materials_project",
        "material_id": "mp-1",
        "formula_pretty": "Fe2O3",
        "elements": ("Fe", "O"),
        "band_gap_ev": 2.1,
        "energy_above_hull_ev_atom": 0.01,
        "density_g_cm3": 5.2,
        "symmetry": SymmetryInfo(
            crystal_system=CrystalSystem.TRIGONAL,
            symbol="R-3c",
            number=167,
        ),
        "provenance": (),
    }
    values.update(overrides)
    return MaterialRecord.model_validate(values)


class TestMaterialRecord:
    def test_valid_record(self) -> None:
        record = _make_record()
        assert record.material_id == "mp-1"
        assert record.symmetry is not None
        assert record.symmetry.number == 167

    def test_negative_band_gap_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_record(band_gap_ev=-1.0)

    def test_negative_hull_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_record(energy_above_hull_ev_atom=-0.1)

    def test_non_positive_density_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_record(density_g_cm3=0.0)

    def test_nan_band_gap_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_record(band_gap_ev=float("nan"))

    def test_infinite_density_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_record(density_g_cm3=float("inf"))

    def test_spacegroup_number_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SymmetryInfo(number=231)

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_record(raw_fields={"x": 1})


class TestPropertyProvenance:
    def test_valid_provenance(self) -> None:
        provenance = PropertyProvenance(
            property_name="band_gap_ev",
            source="materials_project",
            source_material_id="mp-1",
            value_type=PropertyValueType.DFT_CALCULATED,
            database_version="v2026.08",
            retrieved_at=datetime(2026, 8, 5, tzinfo=UTC),
        )
        assert provenance.value_type is PropertyValueType.DFT_CALCULATED

    def test_invalid_value_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PropertyProvenance(
                property_name="band_gap_ev",
                source="materials_project",
                source_material_id="mp-1",
                value_type="guess",
                database_version=None,
                retrieved_at=datetime(2026, 8, 5, tzinfo=UTC),
            )

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PropertyProvenance(
                property_name="band_gap_ev",
                source="materials_project",
                source_material_id="mp-1",
                value_type=PropertyValueType.DFT_CALCULATED,
                database_version=None,
                retrieved_at=datetime(2026, 8, 5, tzinfo=UTC),
                method="GGA",
                junk=True,
            )


def _make_full_result() -> ScreeningResult:
    record = _make_record()
    ranked = RankedMaterial(
        record=record,
        rank=1,
        total_score=0.85,
        score_breakdown=ScoreBreakdown(
            stability=0.9,
            band_gap_match=0.8,
            completeness=1.0,
            direct_gap=0.0,
            weighted_stability=0.36,
            weighted_band_gap_match=0.32,
            weighted_completeness=0.1,
            weighted_direct_gap=0.0,
        ),
    )
    trace = FilterTrace(
        steps=(
            FilterStep(
                name="band_gap",
                before_count=10,
                after_count=5,
                rejection_count=5,
                reason_counts={"band_gap_out_of_range": 5},
            ),
        ),
        rejections=(Rejection(material_id="mp-2", reasons=("band_gap_out_of_range",)),),
    )
    validation = ValidationReport(
        passed=True,
        errors=(),
        warnings=(),
        checked_material_ids=("mp-1",),
    )
    metadata = RunMetadata(
        run_id="run-1",
        started_at=datetime(2026, 8, 5, 0, 0, tzinfo=UTC),
        finished_at=datetime(2026, 8, 5, 0, 0, 1, tzinfo=UTC),
        source="mock",
        database_version="fixture-v1",
        mp_api_version="0.46.4",
        pymatgen_version="2026.5.4",
        application_version="0.1.0",
        query_fingerprint="abc",
    )
    return ScreeningResult(
        request=ScreeningRequest(
            required_elements=("Fe", "O"),
            band_gap_ev=FloatRange(min=1.0, max=2.5),
            target_band_gap_ev=1.8,
        ),
        metadata=metadata,
        retrieved_count=10,
        passed_filter_count=1,
        ranked_materials=(ranked,),
        filter_trace=trace,
        validation=validation,
    )


def test_full_result_json_roundtrip() -> None:
    result = _make_full_result()
    restored = ScreeningResult.model_validate(result.model_dump(mode="json"))
    assert restored == result


class TestExampleRequests:
    @pytest.mark.parametrize(
        "filename",
        ["semiconductor_request.json", "li_fe_o_request.json"],
    )
    def test_valid_example_loads(self, filename: str) -> None:
        raw = (EXAMPLES_DIR / filename).read_text(encoding="utf-8")
        request = ScreeningRequest.model_validate(json.loads(raw))
        assert request is not None

    def test_invalid_example_rejected(self) -> None:
        raw = (EXAMPLES_DIR / "invalid_request.json").read_text(encoding="utf-8")
        with pytest.raises(ValidationError):
            ScreeningRequest.model_validate(json.loads(raw))
