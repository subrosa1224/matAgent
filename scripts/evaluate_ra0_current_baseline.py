"""Measure the current screening core against the frozen RA-0 filter cases.

This records what the existing implementation can do before RA-1. Unsupported
task-specific constraints and ranking semantics are reported, never approximated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from materials_screening.models import (
    FloatRange,
    MaterialRecord,
    ScreeningRequest,
    SymmetryInfo,
)
from materials_screening.services.filter_service import FilterService

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "tests" / "fixtures" / "research" / "benchmark_v1"
SUPPORTED_CONSTRAINTS = {
    "required_elements",
    "excluded_elements",
    "formula_pretty",
    "band_gap_ev",
    "energy_above_hull_ev_atom",
    "density_g_cm3",
    "is_metal",
    "is_stable",
    "theoretical",
    "deprecated",
    "crystal_system",
    "spacegroup_number",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _range(value: list[float | None] | None) -> FloatRange | None:
    if value is None:
        return None
    return FloatRange(min=value[0], max=value[1])


def _request(constraints: dict[str, Any], limit: int) -> ScreeningRequest:
    return ScreeningRequest(
        required_elements=tuple(constraints.get("required_elements", ())),
        excluded_elements=tuple(constraints.get("excluded_elements", ())),
        formula=constraints.get("formula_pretty"),
        band_gap_ev=_range(constraints.get("band_gap_ev")),
        energy_above_hull_ev_atom=_range(constraints.get("energy_above_hull_ev_atom")),
        density_g_cm3=_range(constraints.get("density_g_cm3")),
        crystal_system=constraints.get("crystal_system"),
        spacegroup_numbers=(constraints["spacegroup_number"],)
        if "spacegroup_number" in constraints
        else (),
        is_metal=constraints.get("is_metal"),
        is_stable=constraints.get("is_stable"),
        theoretical=constraints.get("theoretical"),
        limit=min(limit, 100),
    )


def _record(row: dict[str, Any]) -> MaterialRecord:
    symmetry = None
    if (
        row.get("crystal_system") is not None
        or row.get("spacegroup_number") is not None
    ):
        symmetry = SymmetryInfo(
            crystal_system=row.get("crystal_system"),
            symbol=row.get("spacegroup_symbol"),
            number=row.get("spacegroup_number"),
        )
    return MaterialRecord(
        source=row["source"],
        material_id=row["material_id"],
        formula_pretty=row["formula_pretty"],
        elements=tuple(row["elements"]),
        chemsys=row.get("chemsys"),
        band_gap_ev=row.get("band_gap_ev"),
        energy_above_hull_ev_atom=row.get("energy_above_hull_ev_atom"),
        formation_energy_ev_atom=row.get("formation_energy_ev_atom"),
        density_g_cm3=row.get("density_g_cm3"),
        is_metal=row.get("is_metal"),
        is_gap_direct=row.get("is_gap_direct"),
        is_stable=row.get("is_stable"),
        theoretical=row.get("theoretical"),
        deprecated=row.get("deprecated"),
        symmetry=symmetry,
        structure_hash=row.get("structure_hash"),
    )


def main() -> None:
    manifest = _json(BENCHMARK / "manifest.json")
    snapshot = ROOT / manifest["assets"]["database_snapshot"]["path"]
    records = tuple(
        _record(json.loads(line))
        for line in snapshot.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    cases = _json(BENCHMARK / "screening_cases" / "cases.json")
    expected_rows = _json(BENCHMARK / "expected" / "screening_expected.json")
    expected = {row["case_id"]: row for row in expected_rows}

    results: list[dict[str, Any]] = []
    for case in cases:
        unsupported = sorted(set(case["constraints"]) - SUPPORTED_CONSTRAINTS)
        if unsupported:
            results.append(
                {
                    "case_id": case["case_id"],
                    "status": "unsupported",
                    "unsupported_constraints": unsupported,
                    "filter_exact": False,
                    "task_specific_sort_supported": False,
                }
            )
            continue
        request = _request(case["constraints"], case["limit"])
        passed, _trace = FilterService().apply(records, request)
        actual_ids = sorted(record.material_id for record in passed)
        expected_ids = sorted(expected[case["case_id"]]["expected_all_material_ids"])
        results.append(
            {
                "case_id": case["case_id"],
                "status": "evaluated",
                "actual_match_count": len(actual_ids),
                "expected_match_count": len(expected_ids),
                "filter_exact": actual_ids == expected_ids,
                "task_specific_sort_supported": False,
            }
        )

    evaluated = [row for row in results if row["status"] == "evaluated"]
    exact = [row for row in evaluated if row["filter_exact"]]
    payload = {
        "benchmark_id": manifest["benchmark_id"],
        "baseline_kind": "existing_implementation_before_ra1",
        "screening_case_count": len(cases),
        "filter_cases_evaluated": len(evaluated),
        "filter_cases_exact": len(exact),
        "filter_exact_rate": len(exact) / len(evaluated) if evaluated else 0.0,
        "unsupported_filter_cases": len(cases) - len(evaluated),
        "task_specific_ranking_cases_supported": 0,
        "identity_relationship_cases_supported": 0,
        "end_to_end_tasks_completed": 0,
        "end_to_end_task_count": manifest["end_to_end_task_count"],
        "explanations": [
            (
                "The current FilterService is evaluated only for constraints it "
                "represents exactly."
            ),
            (
                "is_gap_direct is present on records but is not an existing "
                "ScreeningRequest hard filter."
            ),
            (
                "The current RankingService uses a fixed semiconductor score and "
                "does not implement each benchmark case's explicit sort contract."
            ),
            (
                "No project-level identity resolver or end-to-end "
                "ScreeningOrchestrator exists before RA-1/RA-3."
            ),
        ],
        "cases": results,
    }
    output = BENCHMARK / "expected" / "current_baseline.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
