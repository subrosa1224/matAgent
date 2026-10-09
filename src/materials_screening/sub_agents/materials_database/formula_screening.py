"""Conservative parsing of an explicit formula/nonmetal/hull screening clause.

This is not a general natural-language constraint parser. Unsupported extra
conditions retain the ordinary model path, rather than silently losing a filter.
"""

from __future__ import annotations

import math
import re

from pymatgen.core.composition import Composition

from .models import SearchMaterialsInput

_CLAUSE = re.compile(
    r"(?:请)?从\s*Materials?\s+Project\s*(?:筛选|查询)\s*"
    r"(?:化学式\s*(?:严格)?为\s*)?"
    r"(?P<formula>(?:[A-Z][a-z]?\d*){2,10})\s*[、，,]\s*"
    r"(?P<conditions>[^。；;\n]{1,180}?)\s*的(?:结构|候选|材料)"
    r"(?=\s*[，,]\s*(?:分析|并分析|按))"
)
_CONDITIONS = re.compile(
    r"(?:(?P<crystal>正交|立方|四方|六方|三方|单斜|三斜)\s*晶系\s*[、，,]\s*)?"
    r"非金属\s*(?:且|[、，,])\s*凸包能\s*"
    r"(?:不高于|不大于|小于等于|≤|<=)\s*"
    r"(?P<hull>\d+(?:\.\d+)?)\s*eV\s*/\s*atom"
)
_CRYSTALS = {
    "正交": "Orthorhombic",
    "立方": "Cubic",
    "四方": "Tetragonal",
    "六方": "Hexagonal",
    "三方": "Trigonal",
    "单斜": "Monoclinic",
    "三斜": "Triclinic",
}
_OTHER_WORK = re.compile(
    r"详情|导出|排序|离群|异常检测|top[- ]?\d|前\s*\d+|BAND_GAP_EXPLORATION_POOL",
    re.I,
)
_UNPARSED_CONDITION = re.compile(
    r"不低于|不高于|不大于|小于|大于|高于|低于|等于|介于|至少|至多|范围为|[<>≤≥]"
    r"|要求|限定|必须|不含|排除|仅含|只要|仅限|同时满足|(?:exclude|without)\b",
    re.I,
)
FORMULA_SCREENING_FIELDS = (
    "material_id",
    "formula_pretty",
    "elements",
    "crystal_system",
    "band_gap_ev",
    "density_g_cm3",
    "energy_above_hull_ev_atom",
    "is_metal",
    "is_stable",
    "formation_energy_ev_atom",
)


def parse_formula_screening(message: str) -> SearchMaterialsInput | None:
    """Return complete supported constraints or None; never infer missing ones."""
    matches = list(_CLAUSE.finditer(message))
    if len(matches) != 1 or _OTHER_WORK.search(message):
        return None
    match = matches[0]
    remaining = message[: match.start()] + message[match.end() :]
    if _UNPARSED_CONDITION.search(remaining) or re.search(
        r"(?:筛选|查询|查找|寻找)", remaining
    ):
        return None
    conditions = _CONDITIONS.fullmatch(match["conditions"].strip())
    if conditions is None:
        return None
    formula = match["formula"]
    try:
        composition = Composition(formula, strict=True)
        if composition.is_element or composition.reduced_formula != formula:
            return None
    except (ValueError, TypeError):
        return None
    hull = float(conditions["hull"])
    if not math.isfinite(hull):
        return None
    filters = [
        {"field": "is_metal", "operator": "eq", "value": False},
        {"field": "energy_above_hull_ev_atom", "operator": "lte", "value": hull},
    ]
    if conditions["crystal"]:
        filters.append(
            {
                "field": "crystal_system",
                "operator": "eq",
                "value": _CRYSTALS[conditions["crystal"]],
            }
        )
    return SearchMaterialsInput(
        formula=formula,
        filters=filters,
        fields=FORMULA_SCREENING_FIELDS,
        limit=100,
    )


def formula_request_matches(
    actual: SearchMaterialsInput, expected: SearchMaterialsInput
) -> bool:
    """Compare the complete filter scope; display fields may include extra data."""
    if (
        actual.formula != expected.formula
        or actual.chemsys
        or actual.required_elements
        or actual.excluded_elements
        or actual.material_ids
        or actual.num_elements is not None
        or actual.sort
    ):
        return False
    wanted = {(f.field, f.operator.value): f.value for f in expected.filters}
    given = {(f.field, f.operator.value): f.value for f in actual.filters}
    return (
        len(actual.filters) == len(wanted)
        and given == wanted
        and given.get(("is_metal", "eq")) is False
        and type(given.get(("energy_above_hull_ev_atom", "lte"))) in (int, float)
        and set(FORMULA_SCREENING_FIELDS).issubset(actual.fields)
    )
