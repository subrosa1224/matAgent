"""Hard constraint filtering with a reproducible trace (M4)."""

from collections.abc import Sequence
from enum import StrEnum

from materials_screening.models import (
    FilterStep,
    FilterTrace,
    MaterialRecord,
    Rejection,
    ScreeningRequest,
)


class RejectionCode(StrEnum):
    """Stable machine-readable rejection reason codes (document §9.3)."""

    DEPRECATED_MATERIAL = "deprecated_material"
    DUPLICATE_MATERIAL_ID = "duplicate_material_id"
    CONTAINS_EXCLUDED_ELEMENT = "contains_excluded_element"
    MISSING_REQUIRED_ELEMENT = "missing_required_element"
    ELEMENT_COUNT_MISMATCH = "element_count_mismatch"
    CHEMSYS_MISMATCH = "chemsys_mismatch"
    FORMULA_MISMATCH = "formula_mismatch"
    METALLICITY_MISSING = "metallicity_missing"
    METALLICITY_MISMATCH = "metallicity_mismatch"
    STABILITY_MISSING = "stability_missing"
    STABILITY_MISMATCH = "stability_mismatch"
    THEORETICAL_FLAG_MISSING = "theoretical_flag_missing"
    THEORETICAL_FLAG_MISMATCH = "theoretical_flag_mismatch"
    CRYSTAL_SYSTEM_MISSING = "crystal_system_missing"
    CRYSTAL_SYSTEM_MISMATCH = "crystal_system_mismatch"
    SPACEGROUP_MISSING = "spacegroup_missing"
    SPACEGROUP_MISMATCH = "spacegroup_mismatch"
    BAND_GAP_MISSING = "band_gap_missing"
    BAND_GAP_BELOW_MIN = "band_gap_below_min"
    BAND_GAP_ABOVE_MAX = "band_gap_above_max"
    ENERGY_ABOVE_HULL_MISSING = "energy_above_hull_missing"
    ENERGY_ABOVE_HULL_BELOW_MIN = "energy_above_hull_below_min"
    ENERGY_ABOVE_HULL_ABOVE_MAX = "energy_above_hull_above_max"
    DENSITY_MISSING = "density_missing"
    DENSITY_BELOW_MIN = "density_below_min"
    DENSITY_ABOVE_MAX = "density_above_max"


_STEP_NAMES: tuple[str, ...] = (
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

_CODE_TO_STEP: dict[RejectionCode, str] = {
    RejectionCode.DEPRECATED_MATERIAL: "deprecated",
    RejectionCode.DUPLICATE_MATERIAL_ID: "duplicate_id",
    RejectionCode.CONTAINS_EXCLUDED_ELEMENT: "excluded_elements",
    RejectionCode.MISSING_REQUIRED_ELEMENT: "required_elements",
    RejectionCode.ELEMENT_COUNT_MISMATCH: "required_elements",
    RejectionCode.CHEMSYS_MISMATCH: "chemsys",
    RejectionCode.FORMULA_MISMATCH: "formula",
    RejectionCode.METALLICITY_MISSING: "is_metal",
    RejectionCode.METALLICITY_MISMATCH: "is_metal",
    RejectionCode.STABILITY_MISSING: "is_stable",
    RejectionCode.STABILITY_MISMATCH: "is_stable",
    RejectionCode.THEORETICAL_FLAG_MISSING: "theoretical",
    RejectionCode.THEORETICAL_FLAG_MISMATCH: "theoretical",
    RejectionCode.CRYSTAL_SYSTEM_MISSING: "crystal_system",
    RejectionCode.CRYSTAL_SYSTEM_MISMATCH: "crystal_system",
    RejectionCode.SPACEGROUP_MISSING: "spacegroup",
    RejectionCode.SPACEGROUP_MISMATCH: "spacegroup",
    RejectionCode.BAND_GAP_MISSING: "band_gap",
    RejectionCode.BAND_GAP_BELOW_MIN: "band_gap",
    RejectionCode.BAND_GAP_ABOVE_MAX: "band_gap",
    RejectionCode.ENERGY_ABOVE_HULL_MISSING: "energy_above_hull",
    RejectionCode.ENERGY_ABOVE_HULL_BELOW_MIN: "energy_above_hull",
    RejectionCode.ENERGY_ABOVE_HULL_ABOVE_MAX: "energy_above_hull",
    RejectionCode.DENSITY_MISSING: "density",
    RejectionCode.DENSITY_BELOW_MIN: "density",
    RejectionCode.DENSITY_ABOVE_MAX: "density",
}


def _first_failure(
    record: MaterialRecord,
    request: ScreeningRequest,
    seen_keys: set[tuple[str, str]],
) -> RejectionCode | None:
    """Return the first rejection code in fixed filter order, or None."""
    if record.deprecated is True:
        return RejectionCode.DEPRECATED_MATERIAL

    key = (record.source, record.material_id)
    if key in seen_keys:
        return RejectionCode.DUPLICATE_MATERIAL_ID

    if request.excluded_elements and (
        set(record.elements) & set(request.excluded_elements)
    ):
        return RejectionCode.CONTAINS_EXCLUDED_ELEMENT

    if request.required_elements and not set(request.required_elements).issubset(
        set(record.elements)
    ):
        return RejectionCode.MISSING_REQUIRED_ELEMENT

    if request.chemsys is not None and record.chemsys != request.chemsys:
        return RejectionCode.CHEMSYS_MISMATCH

    if (
        request.num_elements is not None
        and len(set(record.elements)) != request.num_elements
    ):
        return RejectionCode.ELEMENT_COUNT_MISMATCH

    if request.formula is not None and record.formula_pretty != request.formula:
        return RejectionCode.FORMULA_MISMATCH

    if request.is_metal is not None:
        if record.is_metal is None:
            return RejectionCode.METALLICITY_MISSING
        if record.is_metal != request.is_metal:
            return RejectionCode.METALLICITY_MISMATCH

    if request.is_stable is not None:
        if record.is_stable is None:
            return RejectionCode.STABILITY_MISSING
        if record.is_stable != request.is_stable:
            return RejectionCode.STABILITY_MISMATCH

    if request.theoretical is not None:
        if record.theoretical is None:
            return RejectionCode.THEORETICAL_FLAG_MISSING
        if record.theoretical != request.theoretical:
            return RejectionCode.THEORETICAL_FLAG_MISMATCH

    if request.crystal_system is not None:
        if record.symmetry is None or record.symmetry.crystal_system is None:
            return RejectionCode.CRYSTAL_SYSTEM_MISSING
        if record.symmetry.crystal_system != request.crystal_system:
            return RejectionCode.CRYSTAL_SYSTEM_MISMATCH

    if request.spacegroup_numbers:
        if record.symmetry is None or record.symmetry.number is None:
            return RejectionCode.SPACEGROUP_MISSING
        if record.symmetry.number not in request.spacegroup_numbers:
            return RejectionCode.SPACEGROUP_MISMATCH

    band_range = request.band_gap_ev
    if band_range is not None:
        if record.band_gap_ev is None:
            return RejectionCode.BAND_GAP_MISSING
        if band_range.min is not None and record.band_gap_ev < band_range.min:
            return RejectionCode.BAND_GAP_BELOW_MIN
        if band_range.max is not None and record.band_gap_ev > band_range.max:
            return RejectionCode.BAND_GAP_ABOVE_MAX

    hull_range = request.energy_above_hull_ev_atom
    if hull_range is not None:
        if record.energy_above_hull_ev_atom is None:
            return RejectionCode.ENERGY_ABOVE_HULL_MISSING
        if (
            hull_range.min is not None
            and record.energy_above_hull_ev_atom < hull_range.min
        ):
            return RejectionCode.ENERGY_ABOVE_HULL_BELOW_MIN
        if (
            hull_range.max is not None
            and record.energy_above_hull_ev_atom > hull_range.max
        ):
            return RejectionCode.ENERGY_ABOVE_HULL_ABOVE_MAX

    density_range = request.density_g_cm3
    if density_range is not None:
        if record.density_g_cm3 is None:
            return RejectionCode.DENSITY_MISSING
        if density_range.min is not None and record.density_g_cm3 < density_range.min:
            return RejectionCode.DENSITY_BELOW_MIN
        if density_range.max is not None and record.density_g_cm3 > density_range.max:
            return RejectionCode.DENSITY_ABOVE_MAX

    return None


class FilterService:
    """Pure, stateless hard filter service with a reproducible trace."""

    def apply(
        self,
        records: Sequence[MaterialRecord],
        request: ScreeningRequest,
    ) -> tuple[tuple[MaterialRecord, ...], FilterTrace]:
        """Return records passing all hard constraints and the full trace."""
        passed: list[MaterialRecord] = []
        rejections: list[Rejection] = []
        seen_keys: set[tuple[str, str]] = set()
        step_rejections: dict[str, int] = {name: 0 for name in _STEP_NAMES}
        step_reason_counts: dict[str, dict[str, int]] = {
            name: {} for name in _STEP_NAMES
        }

        for record in records:
            code = _first_failure(record, request, seen_keys)
            if code is None:
                seen_keys.add((record.source, record.material_id))
                passed.append(record)
                continue
            step = _CODE_TO_STEP[code]
            code_value = code.value
            step_rejections[step] += 1
            step_reason_counts[step][code_value] = (
                step_reason_counts[step].get(code_value, 0) + 1
            )
            rejections.append(
                Rejection(material_id=record.material_id, reasons=(code_value,))
            )

        steps: list[FilterStep] = []
        before_count = len(records)
        for name in _STEP_NAMES:
            after_count = before_count - step_rejections[name]
            steps.append(
                FilterStep(
                    name=name,
                    before_count=before_count,
                    after_count=after_count,
                    rejection_count=step_rejections[name],
                    reason_counts=dict(step_reason_counts[name]),
                )
            )
            before_count = after_count

        trace = FilterTrace(steps=tuple(steps), rejections=tuple(rejections))
        return tuple(passed), trace
