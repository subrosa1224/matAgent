from __future__ import annotations

from materials_screening.master.application_feasibility import (
    prioritize_uv_candidates,
    render_uv_candidate_queue,
)
from materials_screening.models import MaterialRecord


def _record(
    material_id: str,
    formula: str,
    elements: tuple[str, ...],
    *,
    theoretical: bool | None,
    direct: bool | None,
    hull: float,
) -> MaterialRecord:
    return MaterialRecord(
        source="materials_project",
        material_id=material_id,
        formula_pretty=formula,
        elements=elements,
        band_gap_ev=3.4,
        energy_above_hull_ev_atom=hull,
        density_g_cm3=5.0,
        theoretical=theoretical,
        is_gap_direct=direct,
    )


def test_uv_queue_excludes_actinides_and_uses_transparent_priority() -> None:
    rows = (
        _record("mp-act", "UO2", ("U", "O"), theoretical=False, direct=True, hull=0),
        _record(
            "mp-exp-direct",
            "ZnO",
            ("Zn", "O"),
            theoretical=False,
            direct=True,
            hull=0.04,
        ),
        _record(
            "mp-exp-indirect",
            "SnO2",
            ("Sn", "O"),
            theoretical=False,
            direct=False,
            hull=0.0,
        ),
        _record(
            "mp-theory",
            "Ga2O3",
            ("Ga", "O"),
            theoretical=True,
            direct=True,
            hull=0.001,
        ),
        _record(
            "mp-caution",
            "YAsO4",
            ("Y", "As", "O"),
            theoretical=False,
            direct=True,
            hull=0.0,
        ),
    )

    queue = prioritize_uv_candidates(rows, limit=5)

    assert [row.material_id for row in queue] == [
        "mp-exp-direct",
        "mp-exp-indirect",
        "mp-theory",
        "mp-caution",
    ]
    assert queue[-1].risk_level == "caution"
    assert queue[-1].risk_notes == (
        "高关注元素 As",
        "稀土供应关注 Y",
    )
    assert "不等同于紫外探测器实验验证" in queue[0].ranking_notes[0]


def test_uv_queue_markdown_discloses_risk_and_origin() -> None:
    queue = prioritize_uv_candidates(
        (
            _record(
                "mp-1",
                "BeO",
                ("Be", "O"),
                theoretical=False,
                direct=True,
                hull=0.0,
            ),
        )
    )

    rendered = render_uv_candidate_queue(queue)

    assert "mp-1" in rendered
    assert "高关注元素 Be" in rendered
    assert "非纯理论条目" in rendered
    assert "直接" in rendered
