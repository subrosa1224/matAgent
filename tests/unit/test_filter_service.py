"""Unit tests for FilterService hard constraints and trace (M4)."""

import pytest

from materials_screening.models import (
    CrystalSystem,
    FloatRange,
    MaterialRecord,
    ScreeningRequest,
    SymmetryInfo,
)
from materials_screening.services.filter_service import FilterService, RejectionCode

EXPECTED_STEP_NAMES = (
    "deprecated",
    "duplicate_id",
    "excluded_elements",
    "required_elements",
    "chemsys",
    "formula",
    "is_metal",
    "is_stable",
    "theoretical",
    "crystal_system",
    "spacegroup",
    "band_gap",
    "energy_above_hull",
    "density",
)

EXPECTED_REJECTION_CODES = {
    "deprecated_material",
    "duplicate_material_id",
    "contains_excluded_element",
    "missing_required_element",
    "element_count_mismatch",
    "chemsys_mismatch",
    "formula_mismatch",
    "metallicity_missing",
    "metallicity_mismatch",
    "stability_missing",
    "stability_mismatch",
    "theoretical_flag_missing",
    "theoretical_flag_mismatch",
    "crystal_system_missing",
    "crystal_system_mismatch",
    "spacegroup_missing",
    "spacegroup_mismatch",
    "band_gap_missing",
    "band_gap_below_min",
    "band_gap_above_max",
    "energy_above_hull_missing",
    "energy_above_hull_below_min",
    "energy_above_hull_above_max",
    "density_missing",
    "density_below_min",
    "density_above_max",
}


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
        "structure_dict": None,
        "provenance": (),
    }
    values.update(overrides)
    return MaterialRecord.model_validate(values)


def _full_request() -> ScreeningRequest:
    return ScreeningRequest(
        required_elements=("Fe", "O"),
        excluded_elements=("Pb",),
        chemsys="Fe-O",
        formula="Fe2O3",
        band_gap_ev=FloatRange(min=1.0, max=3.0),
        energy_above_hull_ev_atom=FloatRange(min=0.0, max=0.1),
        density_g_cm3=FloatRange(min=2.0, max=8.0),
        crystal_system=CrystalSystem.TRIGONAL,
        spacegroup_numbers=(167,),
        is_metal=False,
        is_stable=True,
        theoretical=False,
    )


REJECTION_CASES: list[tuple[RejectionCode, dict[str, object], ScreeningRequest]] = [
    (
        RejectionCode.ELEMENT_COUNT_MISMATCH,
        {"elements": ("Ba", "Ti", "O")},
        ScreeningRequest(num_elements=2),
    ),
    (RejectionCode.DEPRECATED_MATERIAL, {"deprecated": True}, ScreeningRequest()),
    (
        RejectionCode.CONTAINS_EXCLUDED_ELEMENT,
        {"elements": ("Pb", "S")},
        ScreeningRequest(excluded_elements=("Pb",)),
    ),
    (
        RejectionCode.MISSING_REQUIRED_ELEMENT,
        {"elements": ("Fe",)},
        ScreeningRequest(required_elements=("Fe", "O")),
    ),
    (
        RejectionCode.MISSING_REQUIRED_ELEMENT,
        {"elements": ()},
        ScreeningRequest(required_elements=("Fe",)),
    ),
    (
        RejectionCode.CHEMSYS_MISMATCH,
        {"chemsys": "Li-Fe-O"},
        ScreeningRequest(chemsys="Fe-O"),
    ),
    (
        RejectionCode.CHEMSYS_MISMATCH,
        {"chemsys": None},
        ScreeningRequest(chemsys="Fe-O"),
    ),
    (
        RejectionCode.FORMULA_MISMATCH,
        {"formula_pretty": "FeO"},
        ScreeningRequest(formula="Fe2O3"),
    ),
    (
        RejectionCode.METALLICITY_MISSING,
        {"is_metal": None},
        ScreeningRequest(is_metal=False),
    ),
    (
        RejectionCode.METALLICITY_MISMATCH,
        {"is_metal": True},
        ScreeningRequest(is_metal=False),
    ),
    (
        RejectionCode.STABILITY_MISSING,
        {"is_stable": None},
        ScreeningRequest(is_stable=True),
    ),
    (
        RejectionCode.STABILITY_MISMATCH,
        {"is_stable": False},
        ScreeningRequest(is_stable=True),
    ),
    (
        RejectionCode.THEORETICAL_FLAG_MISSING,
        {"theoretical": None},
        ScreeningRequest(theoretical=False),
    ),
    (
        RejectionCode.THEORETICAL_FLAG_MISMATCH,
        {"theoretical": True},
        ScreeningRequest(theoretical=False),
    ),
    (
        RejectionCode.CRYSTAL_SYSTEM_MISSING,
        {"symmetry": None},
        ScreeningRequest(crystal_system="Cubic"),
    ),
    (
        RejectionCode.CRYSTAL_SYSTEM_MISSING,
        {"symmetry": SymmetryInfo(crystal_system=None, number=225)},
        ScreeningRequest(crystal_system="Cubic"),
    ),
    (
        RejectionCode.CRYSTAL_SYSTEM_MISMATCH,
        {"symmetry": SymmetryInfo(crystal_system=CrystalSystem.CUBIC, number=225)},
        ScreeningRequest(crystal_system="Trigonal"),
    ),
    (
        RejectionCode.SPACEGROUP_MISSING,
        {"symmetry": None},
        ScreeningRequest(spacegroup_numbers=(225,)),
    ),
    (
        RejectionCode.SPACEGROUP_MISSING,
        {"symmetry": SymmetryInfo(crystal_system=CrystalSystem.CUBIC, number=None)},
        ScreeningRequest(spacegroup_numbers=(225,)),
    ),
    (
        RejectionCode.SPACEGROUP_MISMATCH,
        {"symmetry": SymmetryInfo(crystal_system=CrystalSystem.CUBIC, number=227)},
        ScreeningRequest(spacegroup_numbers=(225,)),
    ),
    (
        RejectionCode.BAND_GAP_MISSING,
        {"band_gap_ev": None},
        ScreeningRequest(band_gap_ev=FloatRange(min=1.0, max=2.0)),
    ),
    (
        RejectionCode.BAND_GAP_BELOW_MIN,
        {"band_gap_ev": 0.5},
        ScreeningRequest(band_gap_ev=FloatRange(min=1.0, max=2.0)),
    ),
    (
        RejectionCode.BAND_GAP_ABOVE_MAX,
        {"band_gap_ev": 2.5},
        ScreeningRequest(band_gap_ev=FloatRange(min=1.0, max=2.0)),
    ),
    (
        RejectionCode.ENERGY_ABOVE_HULL_MISSING,
        {"energy_above_hull_ev_atom": None},
        ScreeningRequest(energy_above_hull_ev_atom=FloatRange(min=0.0, max=0.1)),
    ),
    (
        RejectionCode.ENERGY_ABOVE_HULL_BELOW_MIN,
        {"energy_above_hull_ev_atom": 0.0},
        ScreeningRequest(energy_above_hull_ev_atom=FloatRange(min=0.05, max=0.1)),
    ),
    (
        RejectionCode.ENERGY_ABOVE_HULL_ABOVE_MAX,
        {"energy_above_hull_ev_atom": 0.2},
        ScreeningRequest(energy_above_hull_ev_atom=FloatRange(min=0.0, max=0.1)),
    ),
    (
        RejectionCode.DENSITY_MISSING,
        {"density_g_cm3": None},
        ScreeningRequest(density_g_cm3=FloatRange(min=2.0, max=8.0)),
    ),
    (
        RejectionCode.DENSITY_BELOW_MIN,
        {"density_g_cm3": 1.0},
        ScreeningRequest(density_g_cm3=FloatRange(min=2.0, max=8.0)),
    ),
    (
        RejectionCode.DENSITY_ABOVE_MAX,
        {"density_g_cm3": 9.0},
        ScreeningRequest(density_g_cm3=FloatRange(min=2.0, max=8.0)),
    ),
]


class TestRejectionCodeStability:
    def test_values_match_documented_codes(self) -> None:
        assert {code.value for code in RejectionCode} == EXPECTED_REJECTION_CODES

    def test_rejection_code_is_string_enum(self) -> None:
        assert issubclass(RejectionCode, str)
        assert RejectionCode.BAND_GAP_MISSING == "band_gap_missing"


class TestFilterServicePass:
    def test_full_constraints_pass(self) -> None:
        records = (_record(), _record(material_id="mp-2"))
        passed, trace = FilterService().apply(records, _full_request())
        assert [record.material_id for record in passed] == ["mp-1", "mp-2"]
        assert trace.rejections == ()
        assert all(step.rejection_count == 0 for step in trace.steps)

    def test_empty_input_returns_empty_trace(self) -> None:
        passed, trace = FilterService().apply((), ScreeningRequest())
        assert passed == ()
        assert [step.name for step in trace.steps] == list(EXPECTED_STEP_NAMES)
        assert all(step.rejection_count == 0 for step in trace.steps)

    def test_no_constraints_pass_everything(self) -> None:
        records = (
            _record(material_id="mp-1"),
            _record(material_id="mp-2"),
        )
        passed, trace = FilterService().apply(records, ScreeningRequest())
        assert len(passed) == 2
        assert trace.rejections == ()

    def test_input_order_is_preserved(self) -> None:
        records = (
            _record(material_id="mp-3"),
            _record(material_id="mp-1"),
            _record(material_id="mp-2"),
        )
        passed, _ = FilterService().apply(records, ScreeningRequest())
        assert [record.material_id for record in passed] == [
            "mp-3",
            "mp-1",
            "mp-2",
        ]

    def test_input_records_are_not_mutated(self) -> None:
        records = (_record(material_id="mp-1"),)
        original = records[0]
        passed, _ = FilterService().apply(records, _full_request())
        assert passed[0] is original
        assert records == (original,)


@pytest.mark.parametrize(
    ("code", "overrides", "screening_request"),
    REJECTION_CASES,
    ids=[f"{code.value}" for code, _, _ in REJECTION_CASES],
)
def test_each_rejection_code(
    code: RejectionCode,
    overrides: dict[str, object],
    screening_request: ScreeningRequest,
) -> None:
    record = _record(**overrides)
    passed, trace = FilterService().apply((record,), screening_request)
    assert passed == ()
    assert len(trace.rejections) == 1
    assert trace.rejections[0].material_id == "mp-1"
    assert trace.rejections[0].reasons == (code.value,)
    rejecting_steps = [
        step
        for step in trace.steps
        if step.rejection_count == 1 and step.reason_counts.get(code.value) == 1
    ]
    assert len(rejecting_steps) == 1


class TestDeduplication:
    def test_duplicate_key_keeps_first_and_rejects_second(self) -> None:
        records = (
            _record(material_id="mp-1"),
            _record(material_id="mp-1"),
        )
        passed, trace = FilterService().apply(records, ScreeningRequest())
        assert len(passed) == 1
        assert passed[0] is records[0]
        assert trace.rejections[0].reasons == (
            RejectionCode.DUPLICATE_MATERIAL_ID.value,
        )

    def test_same_id_different_source_is_not_duplicate(self) -> None:
        records = (
            _record(source="materials_project", material_id="mp-1"),
            _record(source="mock", material_id="mp-1"),
        )
        passed, trace = FilterService().apply(records, ScreeningRequest())
        assert len(passed) == 2
        assert trace.rejections == ()

    def test_same_formula_different_ids_is_not_duplicate(self) -> None:
        records = (
            _record(material_id="mp-1"),
            _record(material_id="mp-2"),
        )
        passed, trace = FilterService().apply(records, ScreeningRequest())
        assert len(passed) == 2
        assert trace.rejections == ()

    def test_duplicate_deprecated_records_are_rejected_as_deprecated(self) -> None:
        records = (
            _record(material_id="mp-1", deprecated=True),
            _record(material_id="mp-1", deprecated=True),
        )
        passed, trace = FilterService().apply(records, ScreeningRequest())
        assert passed == ()
        assert [rejection.reasons for rejection in trace.rejections] == [
            (RejectionCode.DEPRECATED_MATERIAL.value,),
            (RejectionCode.DEPRECATED_MATERIAL.value,),
        ]


class TestExcludedElements:
    @pytest.mark.parametrize("element", ["Pb", "Cd", "Hg"])
    def test_contains_any_excluded_element_rejected(self, element: str) -> None:
        request = ScreeningRequest(excluded_elements=("Pb", "Cd", "Hg"))
        record = _record(material_id="mp-x", elements=(element,))
        passed, trace = FilterService().apply((record,), request)
        assert passed == ()
        assert trace.rejections[0].reasons == (
            RejectionCode.CONTAINS_EXCLUDED_ELEMENT.value,
        )

    def test_record_without_excluded_elements_passes(self) -> None:
        request = ScreeningRequest(excluded_elements=("Pb", "Cd", "Hg"))
        record = _record(material_id="mp-x", elements=("Fe", "O"))
        passed, _ = FilterService().apply((record,), request)
        assert [item.material_id for item in passed] == ["mp-x"]


class TestFilterTrace:
    def test_trace_counts_across_steps(self) -> None:
        records = (
            _record(material_id="mp-1", band_gap_ev=0.5),
            _record(material_id="mp-2"),
            _record(material_id="mp-3", deprecated=True),
        )
        request = ScreeningRequest(band_gap_ev=FloatRange(min=1.0, max=2.0))
        passed, trace = FilterService().apply(records, request)
        assert [record.material_id for record in passed] == ["mp-2"]
        assert [step.name for step in trace.steps] == list(EXPECTED_STEP_NAMES)

        deprecated_step = trace.steps[0]
        assert deprecated_step.before_count == 3
        assert deprecated_step.after_count == 2
        assert deprecated_step.rejection_count == 1
        assert deprecated_step.reason_counts == {
            RejectionCode.DEPRECATED_MATERIAL.value: 1
        }

        band_gap_step = trace.steps[11]
        assert band_gap_step.before_count == 2
        assert band_gap_step.after_count == 1
        assert band_gap_step.rejection_count == 1
        assert band_gap_step.reason_counts == {
            RejectionCode.BAND_GAP_BELOW_MIN.value: 1
        }

        assert trace.steps[1].before_count == 2
        assert trace.steps[13].after_count == 1
        assert {rejection.reasons for rejection in trace.rejections} == {
            (RejectionCode.DEPRECATED_MATERIAL.value,),
            (RejectionCode.BAND_GAP_BELOW_MIN.value,),
        }

    def test_rejections_follow_input_order(self) -> None:
        records = (
            _record(material_id="mp-1", band_gap_ev=0.5),
            _record(material_id="mp-2", deprecated=True),
        )
        request = ScreeningRequest(band_gap_ev=FloatRange(min=1.0, max=2.0))
        _, trace = FilterService().apply(records, request)
        assert [rejection.material_id for rejection in trace.rejections] == [
            "mp-1",
            "mp-2",
        ]
