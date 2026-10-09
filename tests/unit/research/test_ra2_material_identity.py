from __future__ import annotations

import json
from pathlib import Path

from materials_screening.research import (
    IdentityRelation,
    IdentityUse,
    MaterialIdentityRecord,
    MaterialIdentityResolver,
)

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "tests" / "fixtures" / "research" / "benchmark_v1"


def _records() -> dict[str, dict[str, object]]:
    path = BENCHMARK / "database_snapshots" / "materials_project_oxide_300.jsonl"
    return {
        row["material_id"]: row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


def _identity(row: dict[str, object]) -> MaterialIdentityRecord:
    return MaterialIdentityRecord(
        source_kind="materials_project",
        source_material_id=str(row["material_id"]),
        formula=str(row["formula_pretty"]),
        chemsys=str(row["chemsys"]),
        structure_fingerprint=(
            None if row.get("structure_hash") is None else str(row["structure_hash"])
        ),
        space_group_number=int(str(row["spacegroup_number"])),
    )


def test_resolver_matches_all_ra0_identity_boundaries() -> None:
    cases = json.loads(
        (BENCHMARK / "identity_cases" / "cases.json").read_text(encoding="utf-8")
    )
    records = _records()
    resolver = MaterialIdentityResolver()
    expected = {
        "same_source_material_id": IdentityRelation.EXACT_SOURCE_ID,
        "same_formula_different_phase": (
            IdentityRelation.SAME_FORMULA_DIFFERENT_PHASE
        ),
        "same_chemsys_only": IdentityRelation.SAME_CHEMSYS_ONLY,
    }

    for case in cases:
        assessment = resolver.resolve(
            _identity(records[case["left_material_id"]]),
            _identity(records[case["right_material_id"]]),
        )
        assert assessment.relation is expected[case["expected_relationship"]], case


def test_formula_only_never_becomes_exact_identity() -> None:
    assessment = MaterialIdentityResolver().resolve(
        MaterialIdentityRecord(source_kind="paper", formula="Fe2O3"),
        MaterialIdentityRecord(source_kind="materials_project", formula="Fe4O6"),
    )

    assert assessment.relation is IdentityRelation.SAME_FORMULA_INCOMPLETE_STRUCTURE
    assert assessment.identity_use is IdentityUse.BACKGROUND_ONLY
    assert not assessment.same_structure_candidate
    assert assessment.requires_human_review


def test_matching_structure_requires_formula_spacegroup_and_fingerprint() -> None:
    left = MaterialIdentityRecord(
        source_kind="database_a",
        formula="TiO2",
        chemsys="Ti-O",
        structure_fingerprint="abc123",
        space_group_number=136,
    )
    right = left.model_copy(
        update={"source_kind": "database_b", "source_material_id": "other-1"}
    )

    assessment = MaterialIdentityResolver().resolve(left, right)

    assert assessment.relation is IdentityRelation.STRUCTURE_MATCH
    assert assessment.identity_use is IdentityUse.CANDIDATE_PROPERTY
    assert assessment.same_structure_candidate


def test_same_name_only_is_background_not_identity() -> None:
    assessment = MaterialIdentityResolver().resolve(
        MaterialIdentityRecord(source_kind="paper", material_name="alpha phase"),
        MaterialIdentityRecord(source_kind="database", material_name=" Alpha  Phase "),
    )

    assert assessment.relation is IdentityRelation.NAME_ONLY
    assert assessment.identity_use is IdentityUse.BACKGROUND_ONLY


def test_phase_conditions_are_preserved_as_review_requirement() -> None:
    left = MaterialIdentityRecord(
        source_kind="database",
        source_material_id="mp-1",
        phase_conditions=("ambient",),
    )
    right = left.model_copy(update={"phase_conditions": ("30 GPa",)})

    assessment = MaterialIdentityResolver().resolve(left, right)

    assert assessment.relation is IdentityRelation.EXACT_SOURCE_ID
    assert assessment.requires_human_review
    assert any("可比性" in reason for reason in assessment.reasons)
