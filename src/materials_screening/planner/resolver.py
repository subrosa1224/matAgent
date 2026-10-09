"""Deterministic planner resolver (D2-M2).

The resolver never accesses Intern or Materials Project. It converts
units, validates elements and ranges, detects conflicts, ambiguities and
unsupported tasks, and builds a ScreeningRequest only for READY results.
"""

import unicodedata
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from materials_screening.chemistry import normalize_chemsys, normalize_element_list
from materials_screening.llm.metadata import ProviderMetadata
from materials_screening.models import CrystalSystem, FloatRange, ScreeningRequest
from materials_screening.planner.models import (
    DensityUnit,
    EnergyUnit,
    HullUnit,
    PlannerDraft,
    PlannerResult,
    PlannerStatus,
)
from materials_screening.planner.rules import (
    AmbiguityCode,
    ConflictCode,
    InvalidCode,
    StabilityRule,
    UnsupportedCode,
)
from materials_screening.planner.unit_conversion import (
    density_to_g_per_cm3,
    energy_to_ev,
    hull_to_ev_per_atom,
)

_BAND_GAP_DEFAULT_UNIT = EnergyUnit.EV
_HULL_DEFAULT_UNIT = HullUnit.EV_PER_ATOM
_DENSITY_DEFAULT_UNIT = DensityUnit.G_PER_CM3

_BAND_GAP_EV_PLAUSIBLE = (Decimal("0"), Decimal("20"))
_HULL_EV_ATOM_PLAUSIBLE = (Decimal("0"), Decimal("2"))
_DENSITY_G_CM3_PLAUSIBLE = (Decimal("0"), Decimal("25"))

_APPLICATION_GOAL_KEYWORDS = (
    "光伏",
    "太阳能",
    "电池",
    "光催化",
    "热电",
    "发光",
    "透明导电",
    "solar",
    "photovoltaic",
    "battery",
    "thermoelectric",
)

_TOXIC_OR_RARE_KEYWORDS = ("有毒", "毒性", "稀有", "稀缺", "稀土", "toxic", "rare")

_HULL_EVIDENCE_KEYWORDS = (
    "hull",
    "凸包",
    "能量高于凸包",
    "ev/atom",
    "mev/atom",
)

_UNSUPPORTED_KEYWORDS: dict[UnsupportedCode, tuple[str, ...]] = {
    UnsupportedCode.PREDICT_MATERIALS: (
        "预测",
        "生成候选",
        "发现新材料",
        "设计新材料",
        "predict materials",
    ),
    UnsupportedCode.PREDICT_PROPERTIES: (
        "预测属性",
        "预测带隙",
        "预测稳定性",
        "predict properties",
    ),
    UnsupportedCode.DFT_COMPUTATION: (
        "dft",
        "第一性原理",
        "从头计算",
        "计算任务",
        "跑计算",
    ),
    UnsupportedCode.SYNTHESIS_ADVICE: (
        "合成方法",
        "如何合成",
        "制备方法",
        "synthesis",
    ),
    UnsupportedCode.EXPERIMENTAL_DATA: (
        "实验数据",
        "实验测量",
        "experimental",
    ),
}


def normalize_query(query: str) -> str:
    """Trim whitespace and normalize Unicode for deterministic matching."""
    return unicodedata.normalize("NFC", query.strip())


def _plausible(value: Decimal, bounds: tuple[Decimal, Decimal]) -> bool:
    return bounds[0] <= value <= bounds[1]


def _unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _range_for(
    min_value: Decimal | None, max_value: Decimal | None
) -> FloatRange | None:
    if min_value is None and max_value is None:
        return None
    return FloatRange(
        min=float(min_value) if min_value is not None else None,
        max=float(max_value) if max_value is not None else None,
    )


def _has_hull_evidence(draft: PlannerDraft) -> bool:
    lowered = " ".join(item.quote for item in draft.evidence).lower()
    return any(keyword in lowered for keyword in _HULL_EVIDENCE_KEYWORDS)


def _build_clarification_question(codes: list[str]) -> str:
    if AmbiguityCode.UNIT_UNSPECIFIED_IMPLAUSIBLE.value in codes:
        return "请明确数值及其单位（例如 1.2 eV、50 meV/atom、5 g/cm3）。"
    if AmbiguityCode.LLM_FLAGGED.value in codes:
        return "请求包含无法确定的条件，请补充说明。"
    return "请补充或明确筛选条件后重试。"


class PlannerResolver:
    """Pure, deterministic resolver from a PlannerDraft to a PlannerResult."""

    def resolve(
        self,
        query: str,
        draft: PlannerDraft,
        provider_metadata: ProviderMetadata | None = None,
    ) -> PlannerResult:
        normalized = normalize_query(query)
        invalid: list[str] = []
        conflicts: list[str] = []
        ambiguities: list[str] = []
        unsupported: list[str] = []
        assumptions: list[str] = []
        clarifications: list[str] = []

        if not normalized:
            invalid.append(InvalidCode.EMPTY_QUERY.value)

        required = self._normalize_elements(draft.required_elements, invalid)
        excluded = self._normalize_elements(draft.excluded_elements, invalid)

        if set(required) & set(excluded):
            conflicts.append(ConflictCode.REQUIRED_AND_EXCLUDED_ELEMENT.value)

        chemsys = self._normalize_chemsys(draft.chemsys, invalid)

        band_min, band_max = self._convert_band(
            draft, assumptions, ambiguities, clarifications, invalid
        )
        hull_min, hull_max = self._convert_hull(
            draft, assumptions, ambiguities, clarifications, invalid
        )
        density_min, density_max = self._convert_density(
            draft, assumptions, ambiguities, clarifications, invalid
        )
        target = self._convert_target(
            draft, assumptions, ambiguities, clarifications, invalid
        )

        self._check_range_order(
            band_min, band_max, ConflictCode.BAND_GAP_MIN_ABOVE_MAX, conflicts
        )
        self._check_range_order(
            hull_min, hull_max, ConflictCode.HULL_MIN_ABOVE_MAX, conflicts
        )
        self._check_range_order(
            density_min, density_max, ConflictCode.DENSITY_MIN_ABOVE_MAX, conflicts
        )

        if target is not None and (
            (band_min is not None and target < band_min)
            or (band_max is not None and target > band_max)
        ):
            conflicts.append(ConflictCode.TARGET_OUTSIDE_BAND_GAP.value)

        spacegroups = self._normalize_spacegroups(draft.spacegroup_numbers, invalid)
        crystal_system = self._normalize_crystal_system(draft.crystal_system, invalid)

        limit = draft.limit
        if limit is not None and not (1 <= limit <= 100):
            invalid.append(InvalidCode.INVALID_LIMIT.value)
            limit = None

        if draft.is_stable is True:
            assumptions.append(StabilityRule.STABLE_AS_IS_STABLE.value)
            if (
                draft.hull_max is not None
                and draft.hull_max == 0
                and not _has_hull_evidence(draft)
            ):
                ambiguities.append(AmbiguityCode.STABLE_NOT_HULL_ZERO.value)
                hull_max = None

        lowered = normalized.lower()
        if any(keyword in lowered for keyword in _APPLICATION_GOAL_KEYWORDS):
            ambiguities.append(AmbiguityCode.APPLICATION_GOAL.value)
        if any(keyword in lowered for keyword in _TOXIC_OR_RARE_KEYWORDS):
            ambiguities.append(AmbiguityCode.TOXIC_OR_RARE_ELEMENTS.value)

        if draft.unsupported_requirements:
            unsupported.append(UnsupportedCode.LLM_FLAGGED.value)
        for code, keywords in _UNSUPPORTED_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                unsupported.append(code.value)
        if (
            any(keyword in lowered for keyword in _APPLICATION_GOAL_KEYWORDS)
            and UnsupportedCode.LLM_FLAGGED.value in unsupported
        ):
            # Application goals are ambiguities by design (e.g. 光伏材料);
            # an LLM-flagged "unsupported" without a deterministic
            # unsupported keyword is downgraded so the query resolves as an
            # ambiguous READY plan instead of UNSUPPORTED.
            unsupported = [
                code
                for code in unsupported
                if code != UnsupportedCode.LLM_FLAGGED.value
            ]

        if draft.conflicts:
            conflicts.append(ConflictCode.LLM_FLAGGED.value)
        if draft.ambiguities:
            clarifications.append(AmbiguityCode.LLM_FLAGGED.value)
            ambiguities.append(AmbiguityCode.LLM_FLAGGED.value)

        status = PlannerStatus.READY
        if invalid or conflicts:
            status = PlannerStatus.INVALID
        elif unsupported:
            status = PlannerStatus.UNSUPPORTED
        elif clarifications:
            status = PlannerStatus.NEEDS_CLARIFICATION

        if status is PlannerStatus.INVALID and not invalid:
            invalid.append(InvalidCode.CONFLICTING_REQUIREMENTS.value)

        request = None
        if status is PlannerStatus.READY:
            request = ScreeningRequest(
                required_elements=required,
                excluded_elements=excluded,
                chemsys=chemsys,
                formula=draft.formula,
                band_gap_ev=_range_for(band_min, band_max),
                energy_above_hull_ev_atom=_range_for(hull_min, hull_max),
                density_g_cm3=_range_for(density_min, density_max),
                crystal_system=crystal_system,
                spacegroup_numbers=spacegroups,
                is_metal=draft.is_metal,
                is_stable=draft.is_stable,
                theoretical=draft.theoretical,
                target_band_gap_ev=(float(target) if target is not None else None),
                limit=limit if limit is not None else 10,
            )

        clarification_question = ""
        if status is PlannerStatus.NEEDS_CLARIFICATION:
            clarification_question = _build_clarification_question(clarifications)

        return PlannerResult(
            status=status,
            query=normalized,
            request=request,
            clarification_question=clarification_question,
            invalid_reasons=_unique(invalid),
            conflicts=_unique(conflicts),
            assumptions=_unique(assumptions),
            ambiguities=_unique(ambiguities),
            unsupported_requirements=_unique(unsupported),
            provider_metadata=provider_metadata,
        )

    def _normalize_elements(
        self, elements: list[str], invalid: list[str]
    ) -> tuple[str, ...]:
        try:
            return normalize_element_list(elements)
        except ValueError:
            invalid.append(InvalidCode.INVALID_ELEMENT.value)
            return ()

    def _normalize_chemsys(self, chemsys: str | None, invalid: list[str]) -> str | None:
        if chemsys is None:
            return None
        try:
            return normalize_chemsys(chemsys)
        except ValueError:
            invalid.append(InvalidCode.INVALID_CHEMSYS.value)
            return None

    def _normalize_spacegroups(
        self, numbers: list[int], invalid: list[str]
    ) -> tuple[int, ...]:
        normalized = tuple(sorted(set(numbers)))
        if any(number < 1 or number > 230 for number in normalized):
            invalid.append(InvalidCode.INVALID_SPACEGROUP.value)
            return ()
        return normalized

    def _normalize_crystal_system(
        self, value: str | None, invalid: list[str]
    ) -> CrystalSystem | None:
        if value is None:
            return None
        try:
            return CrystalSystem(value)
        except ValueError:
            invalid.append(InvalidCode.INVALID_CRYSTAL_SYSTEM.value)
            return None

    def _convert_band(
        self,
        draft: PlannerDraft,
        assumptions: list[str],
        ambiguities: list[str],
        clarifications: list[str],
        invalid: list[str],
    ) -> tuple[Decimal | None, Decimal | None]:
        band_min = self._convert_single(
            draft.band_gap_min,
            draft.band_gap_unit,
            _BAND_GAP_DEFAULT_UNIT,
            _BAND_GAP_EV_PLAUSIBLE,
            energy_to_ev,
            InvalidCode.NEGATIVE_BAND_GAP.value,
            assumptions,
            ambiguities,
            clarifications,
            invalid,
        )
        band_max = self._convert_single(
            draft.band_gap_max,
            draft.band_gap_unit,
            _BAND_GAP_DEFAULT_UNIT,
            _BAND_GAP_EV_PLAUSIBLE,
            energy_to_ev,
            InvalidCode.NEGATIVE_BAND_GAP.value,
            assumptions,
            ambiguities,
            clarifications,
            invalid,
        )
        return band_min, band_max

    def _convert_hull(
        self,
        draft: PlannerDraft,
        assumptions: list[str],
        ambiguities: list[str],
        clarifications: list[str],
        invalid: list[str],
    ) -> tuple[Decimal | None, Decimal | None]:
        hull_min = self._convert_single(
            draft.hull_min,
            draft.hull_unit,
            _HULL_DEFAULT_UNIT,
            _HULL_EV_ATOM_PLAUSIBLE,
            hull_to_ev_per_atom,
            InvalidCode.NEGATIVE_HULL.value,
            assumptions,
            ambiguities,
            clarifications,
            invalid,
        )
        hull_max = self._convert_single(
            draft.hull_max,
            draft.hull_unit,
            _HULL_DEFAULT_UNIT,
            _HULL_EV_ATOM_PLAUSIBLE,
            hull_to_ev_per_atom,
            InvalidCode.NEGATIVE_HULL.value,
            assumptions,
            ambiguities,
            clarifications,
            invalid,
        )
        return hull_min, hull_max

    def _convert_density(
        self,
        draft: PlannerDraft,
        assumptions: list[str],
        ambiguities: list[str],
        clarifications: list[str],
        invalid: list[str],
    ) -> tuple[Decimal | None, Decimal | None]:
        density_min = self._convert_single(
            draft.density_min,
            draft.density_unit,
            _DENSITY_DEFAULT_UNIT,
            _DENSITY_G_CM3_PLAUSIBLE,
            density_to_g_per_cm3,
            InvalidCode.NON_POSITIVE_DENSITY.value,
            assumptions,
            ambiguities,
            clarifications,
            invalid,
            non_positive=True,
        )
        density_max = self._convert_single(
            draft.density_max,
            draft.density_unit,
            _DENSITY_DEFAULT_UNIT,
            _DENSITY_G_CM3_PLAUSIBLE,
            density_to_g_per_cm3,
            InvalidCode.NON_POSITIVE_DENSITY.value,
            assumptions,
            ambiguities,
            clarifications,
            invalid,
            non_positive=True,
        )
        return density_min, density_max

    def _convert_target(
        self,
        draft: PlannerDraft,
        assumptions: list[str],
        ambiguities: list[str],
        clarifications: list[str],
        invalid: list[str],
    ) -> Decimal | None:
        return self._convert_single(
            draft.target_band_gap,
            draft.target_band_gap_unit,
            _BAND_GAP_DEFAULT_UNIT,
            _BAND_GAP_EV_PLAUSIBLE,
            energy_to_ev,
            InvalidCode.NEGATIVE_TARGET_BAND_GAP.value,
            assumptions,
            ambiguities,
            clarifications,
            invalid,
        )

    def _convert_single(
        self,
        value: float | None,
        unit: EnergyUnit | HullUnit | DensityUnit,
        default_unit: EnergyUnit | HullUnit | DensityUnit,
        plausible_bounds: tuple[Decimal, Decimal],
        convert: Callable[[Decimal, Any], Decimal],
        negative_code: str,
        assumptions: list[str],
        ambiguities: list[str],
        clarifications: list[str],
        invalid: list[str],
        *,
        non_positive: bool = False,
    ) -> Decimal | None:
        if value is None:
            return None
        decimal_value = Decimal(str(value))
        if decimal_value < 0 or (non_positive and decimal_value == 0):
            invalid.append(negative_code)
            return None
        if unit.value != "unspecified":
            return convert(decimal_value, unit)
        if _plausible(decimal_value, plausible_bounds):
            assumptions.append(AmbiguityCode.UNIT_UNSPECIFIED_ASSUMED.value)
            return convert(decimal_value, default_unit)
        clarifications.append(AmbiguityCode.UNIT_UNSPECIFIED_IMPLAUSIBLE.value)
        ambiguities.append(AmbiguityCode.UNIT_UNSPECIFIED_IMPLAUSIBLE.value)
        return None

    def _check_range_order(
        self,
        min_value: Decimal | None,
        max_value: Decimal | None,
        conflict_code: ConflictCode,
        conflicts: list[str],
    ) -> None:
        if min_value is not None and max_value is not None and min_value > max_value:
            conflicts.append(conflict_code.value)
