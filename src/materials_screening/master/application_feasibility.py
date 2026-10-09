"""Transparent application-priority rules for database screening candidates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from materials_screening.models import MaterialRecord

CandidateRiskLevel = Literal["standard", "caution"]

_RADIOACTIVE_ACTINIDES = frozenset(
    {
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
    }
)
_HIGH_CONCERN_ELEMENTS = frozenset({"As", "Tl", "Be"})
_RARE_EARTH_ELEMENTS = frozenset(
    {
        "Sc",
        "Y",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
    }
)


class ApplicationCandidate(BaseModel):
    """One candidate in the UV-photodetector validation queue."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    priority: int
    material_id: str
    formula_pretty: str
    elements: tuple[str, ...]
    band_gap_ev: float | None
    energy_above_hull_ev_atom: float | None
    density_g_cm3: float | None
    theoretical: bool | None
    is_gap_direct: bool | None
    risk_level: CandidateRiskLevel
    risk_notes: tuple[str, ...] = ()
    ranking_notes: tuple[str, ...] = ()


def prioritize_uv_candidates(
    records: Sequence[MaterialRecord], *, limit: int = 5, unique_formulas: bool = False
) -> tuple[ApplicationCandidate, ...]:
    """Build a practical validation queue without altering the raw query."""

    if not 1 <= limit <= 100:
        raise ValueError("candidate limit must be between 1 and 100")
    eligible = [
        record
        for record in records
        if not (set(record.elements) & _RADIOACTIVE_ACTINIDES)
    ]
    ordered = sorted(eligible, key=_uv_priority_key)
    if unique_formulas:
        first_by_formula: dict[str, MaterialRecord] = {}
        for record in ordered:
            first_by_formula.setdefault(record.formula_pretty, record)
        ordered = list(first_by_formula.values())
    output: list[ApplicationCandidate] = []
    for priority, record in enumerate(ordered[:limit], 1):
        risk_notes = _risk_notes(record)
        ranking_notes = (
            (
                "Materials Project theoretical=false；仅表示数据库来源不是纯理论条目，"
                "不等同于紫外探测器实验验证"
            )
            if record.theoretical is False
            else "Materials Project 标记为理论条目"
            if record.theoretical is True
            else "Materials Project 未提供 theoretical 标记",
            "直接带隙" if record.is_gap_direct is True else "非直接带隙或类型未知",
            (
                f"E_hull={record.energy_above_hull_ev_atom:.4g} eV/atom"
                if record.energy_above_hull_ev_atom is not None
                else "E_hull 缺失"
            ),
        )
        output.append(
            ApplicationCandidate(
                priority=priority,
                material_id=record.material_id,
                formula_pretty=record.formula_pretty,
                elements=record.elements,
                band_gap_ev=record.band_gap_ev,
                energy_above_hull_ev_atom=record.energy_above_hull_ev_atom,
                density_g_cm3=record.density_g_cm3,
                theoretical=record.theoretical,
                is_gap_direct=record.is_gap_direct,
                risk_level="caution" if risk_notes else "standard",
                risk_notes=risk_notes,
                ranking_notes=ranking_notes,
            )
        )
    return tuple(output)


def render_uv_candidate_queue(candidates: Sequence[ApplicationCandidate]) -> str:
    """Render the transparent priority queue used for literature validation."""

    lines = [
        "| 优先级 | Material ID | 化学式 | 风险 | 数据来源 | 带隙类型 | E hull |",
        "|---:|---|---|---|---|---|---:|",
    ]
    for row in candidates:
        risk = "常规" if row.risk_level == "standard" else "需注意"
        if row.risk_notes:
            risk += "（" + "；".join(row.risk_notes) + "）"
        origin = (
            "非纯理论条目"
            if row.theoretical is False
            else "理论条目"
            if row.theoretical is True
            else "未知"
        )
        gap_kind = "直接" if row.is_gap_direct is True else "非直接/未知"
        hull = (
            f"{row.energy_above_hull_ev_atom:.4g}"
            if row.energy_above_hull_ev_atom is not None
            else "—"
        )
        lines.append(
            f"| {row.priority} | {row.material_id} | {row.formula_pretty} | "
            f"{risk} | {origin} | {gap_kind} | {hull} |"
        )
    return "\n".join(lines)


def _risk_notes(record: MaterialRecord) -> tuple[str, ...]:
    elements = set(record.elements)
    notes: list[str] = []
    high_concern = sorted(elements & _HIGH_CONCERN_ELEMENTS)
    rare_earth = sorted(elements & _RARE_EARTH_ELEMENTS)
    if high_concern:
        notes.append("高关注元素 " + "/".join(high_concern))
    if rare_earth:
        notes.append("稀土供应关注 " + "/".join(rare_earth))
    return tuple(notes)


def _uv_priority_key(record: MaterialRecord) -> tuple[int, int, int, float, str]:
    risk_rank = 1 if _risk_notes(record) else 0
    source_rank = (
        0 if record.theoretical is False else 1 if record.theoretical is None else 2
    )
    direct_rank = (
        0 if record.is_gap_direct is True else 1 if record.is_gap_direct is None else 2
    )
    hull_rank = (
        record.energy_above_hull_ev_atom
        if record.energy_above_hull_ev_atom is not None
        else math.inf
    )
    return (risk_rank, source_rank, direct_rank, hull_rank, record.material_id)
