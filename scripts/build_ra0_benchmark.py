"""Build the frozen RA-0 crystalline-inorganic benchmark assets.

This is a benchmark-data builder, not production screening code. It deliberately
uses a small, independently reviewable reference evaluator and existing immutable
Materials Project snapshots. Running it twice must produce byte-identical assets.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "tests" / "fixtures" / "research" / "benchmark_v1"
SOURCE_QUERIES = (
    "query-0fcc4e292c4d4896ab51a8846951d347",
    "query-e7798308f9154323b28e4dc58693e3bf",
    "query-6e2ecc32641b4adda78906dd50f834f2",
)
NON_OXIDE_ANIONS = {
    "H",
    "C",
    "N",
    "F",
    "P",
    "S",
    "Cl",
    "Se",
    "Br",
    "I",
    "Te",
    "At",
    "He",
    "Ne",
    "Ar",
    "Kr",
    "Xe",
    "Rn",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_oxide_only(record: dict[str, Any]) -> bool:
    elements = set(record.get("elements") or ())
    return "O" in elements and not bool(elements & NON_OXIDE_ANIONS)


def _compact(
    record: dict[str, Any], query_id: str, database_version: str
) -> dict[str, Any]:
    symmetry = record.get("symmetry") or {}
    return {
        "source": "materials_project",
        "source_query_ids": [query_id],
        "database_version": database_version,
        "material_id": record["material_id"],
        "formula_pretty": record["formula_pretty"],
        "elements": record.get("elements") or [],
        "chemsys": record.get("chemsys"),
        "band_gap_ev": record.get("band_gap_ev"),
        "energy_above_hull_ev_atom": record.get("energy_above_hull_ev_atom"),
        "formation_energy_ev_atom": record.get("formation_energy_ev_atom"),
        "density_g_cm3": record.get("density_g_cm3"),
        "is_metal": record.get("is_metal"),
        "is_gap_direct": record.get("is_gap_direct"),
        "is_stable": record.get("is_stable"),
        "theoretical": record.get("theoretical"),
        "deprecated": record.get("deprecated"),
        "crystal_system": symmetry.get("crystal_system"),
        "spacegroup_symbol": symmetry.get("symbol"),
        "spacegroup_number": symmetry.get("number"),
        "structure_hash": record.get("structure_hash"),
    }


def build_snapshot() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    for query_id in SOURCE_QUERIES:
        folder = ROOT / "data" / "material_queries" / query_id
        metadata = _read_json(folder / "metadata.json")
        records = [
            json.loads(line)
            for line in (folder / "records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        pools[query_id] = [
            _compact(record, query_id, metadata["database_version"])
            for record in records
            if _is_oxide_only(record)
        ]
        sources.append(
            {
                "query_id": query_id,
                "database_version": metadata["database_version"],
                "created_at": metadata["created_at"],
                "source_record_count": len(records),
                "eligible_oxide_count": len(pools[query_id]),
                "metadata_sha256": _sha256(folder / "metadata.json"),
                "records_sha256": _sha256(folder / "records.jsonl"),
            }
        )

    selected: dict[str, dict[str, Any]] = {}

    def add(record: dict[str, Any]) -> None:
        existing = selected.get(record["material_id"])
        if existing is None:
            selected[record["material_id"]] = record
            return
        existing["source_query_ids"] = sorted(
            set(existing["source_query_ids"]) | set(record["source_query_ids"])
        )

    # Preserve all broad-query and Fe2O3 boundary records, then fill with stable
    # wide-gap oxides. This intentionally keeps both ordinary and adversarial rows.
    for query_id in (SOURCE_QUERIES[0], SOURCE_QUERIES[2]):
        for record in pools[query_id]:
            add(record)
    for record in pools[SOURCE_QUERIES[1]]:
        if len(selected) >= 300:
            break
        add(record)

    rows = sorted(selected.values(), key=lambda row: row["material_id"])
    if len(rows) != 300:
        raise RuntimeError(f"Expected 300 unique records, found {len(rows)}")
    return rows, sources


CASES: list[dict[str, Any]] = [
    {
        "case_id": "screen-001",
        "question": (
            "筛选不含 Pb、Cd、Hg，带隙 2–4 eV、能量高于凸包不超过 "
            "0.05 eV/atom 的稳定非金属氧化物。"
        ),
        "constraints": {
            "excluded_elements": ["Pb", "Cd", "Hg"],
            "band_gap_ev": [2.0, 4.0],
            "energy_above_hull_ev_atom": [None, 0.05],
            "is_stable": True,
            "is_metal": False,
            "deprecated": False,
        },
        "sort": {"field": "energy_above_hull_ev_atom", "direction": "asc"},
        "limit": 10,
    },
    {
        "case_id": "screen-002",
        "question": "筛选带隙至少 4 eV、密度不超过 5 g/cm³ 的稳定非金属氧化物。",
        "constraints": {
            "band_gap_ev": [4.0, None],
            "density_g_cm3": [None, 5.0],
            "is_stable": True,
            "is_metal": False,
            "deprecated": False,
        },
        "sort": {"field": "band_gap_ev", "direction": "desc"},
        "limit": 10,
    },
    {
        "case_id": "screen-003",
        "question": "筛选带隙 1–3 eV 的直接带隙非金属氧化物。",
        "constraints": {
            "band_gap_ev": [1.0, 3.0],
            "is_gap_direct": True,
            "is_metal": False,
            "deprecated": False,
        },
        "sort": {"field": "band_gap_ev", "direction": "asc"},
        "limit": 10,
    },
    {
        "case_id": "screen-004",
        "question": "筛选能量高于凸包不超过 0.10 eV/atom 的亚稳氧化物。",
        "constraints": {
            "energy_above_hull_ev_atom": [0.0000001, 0.10],
            "is_stable": False,
            "deprecated": False,
        },
        "sort": {"field": "energy_above_hull_ev_atom", "direction": "asc"},
        "limit": 10,
    },
    {
        "case_id": "screen-005",
        "question": "筛选已有非理论来源标记的稳定氧化物。",
        "constraints": {"is_stable": True, "theoretical": False, "deprecated": False},
        "sort": {"field": "material_id", "direction": "asc"},
        "limit": 10,
    },
    {
        "case_id": "screen-006",
        "question": "筛选密度 3–6 g/cm³ 的非金属氧化物。",
        "constraints": {
            "density_g_cm3": [3.0, 6.0],
            "is_metal": False,
            "deprecated": False,
        },
        "sort": {"field": "density_g_cm3", "direction": "asc"},
        "limit": 10,
    },
    {
        "case_id": "screen-007",
        "question": "筛选含 Fe 和 O、且不含 Li 的氧化物。",
        "constraints": {
            "required_elements": ["Fe", "O"],
            "excluded_elements": ["Li"],
            "deprecated": False,
        },
        "sort": {"field": "energy_above_hull_ev_atom", "direction": "asc"},
        "limit": 10,
    },
    {
        "case_id": "screen-008",
        "question": "筛选立方晶系氧化物。",
        "constraints": {"crystal_system": "Cubic", "deprecated": False},
        "sort": {"field": "density_g_cm3", "direction": "asc"},
        "limit": 10,
    },
    {
        "case_id": "screen-009",
        "question": "查找空间群编号为 167 的氧化物。",
        "constraints": {"spacegroup_number": 167, "deprecated": False},
        "sort": {"field": "material_id", "direction": "asc"},
        "limit": 10,
    },
    {
        "case_id": "screen-010",
        "question": "在稳定非金属氧化物中寻找带隙最接近 3.0 eV 的候选。",
        "constraints": {"is_stable": True, "is_metal": False, "deprecated": False},
        "sort": {"field": "band_gap_ev", "direction": "target", "target": 3.0},
        "limit": 10,
    },
    {
        "case_id": "screen-011",
        "question": "按密度从低到高列出稳定氧化物。",
        "constraints": {"is_stable": True, "deprecated": False},
        "sort": {"field": "density_g_cm3", "direction": "asc"},
        "limit": 10,
    },
    {
        "case_id": "screen-012",
        "question": "按能量高于凸包从低到高列出非理论来源的氧化物。",
        "constraints": {"theoretical": False, "deprecated": False},
        "sort": {"field": "energy_above_hull_ev_atom", "direction": "asc"},
        "limit": 10,
    },
    {
        "case_id": "screen-013",
        "question": "筛选密度小于 0 g/cm³ 的氧化物，预期零结果且不得放宽条件。",
        "constraints": {"density_g_cm3": [None, -0.000001]},
        "sort": {"field": "material_id", "direction": "asc"},
        "limit": 10,
        "expected_behavior": "zero_results_no_relaxation",
    },
    {
        "case_id": "screen-014",
        "question": "筛选未废弃、非理论来源、带隙大于 0 的氧化物。",
        "constraints": {
            "band_gap_ev": [0.0000001, None],
            "theoretical": False,
            "deprecated": False,
        },
        "sort": {"field": "band_gap_ev", "direction": "desc"},
        "limit": 10,
    },
    {
        "case_id": "screen-015",
        "question": "列出 Fe2O3 多晶型，并按能量高于凸包排序。",
        "constraints": {"formula_pretty": "Fe2O3", "deprecated": False},
        "sort": {"field": "energy_above_hull_ev_atom", "direction": "asc"},
        "limit": 26,
    },
]


def _matches(record: dict[str, Any], constraints: dict[str, Any]) -> bool:
    elements = set(record.get("elements") or ())
    if not set(constraints.get("required_elements", ())).issubset(elements):
        return False
    if elements & set(constraints.get("excluded_elements", ())):
        return False
    for key, expected in constraints.items():
        if key in {"required_elements", "excluded_elements"}:
            continue
        actual = record.get(key)
        if isinstance(expected, list) and len(expected) == 2:
            if actual is None:
                return False
            lower, upper = expected
            if lower is not None and actual < lower:
                return False
            if upper is not None and actual > upper:
                return False
        elif actual != expected:
            return False
    return True


def _rank_key(record: dict[str, Any], sort: dict[str, Any]) -> tuple[Any, ...]:
    field = sort["field"]
    value = record.get(field)
    if sort["direction"] == "target":
        distance = float("inf") if value is None else abs(value - sort["target"])
        return (distance, record["material_id"])
    missing = value is None
    if sort["direction"] == "desc" and isinstance(value, (int, float)):
        value = -value
    return (missing, value, record["material_id"])


def build_screening_cases(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected: list[dict[str, Any]] = []
    for case in CASES:
        matched = [row for row in records if _matches(row, case["constraints"])]
        ranked = sorted(matched, key=lambda row: _rank_key(row, case["sort"]))
        expected.append(
            {
                "case_id": case["case_id"],
                "matched_count": len(matched),
                "expected_all_material_ids": [row["material_id"] for row in ranked],
                "expected_material_ids": [
                    row["material_id"] for row in ranked[: case["limit"]]
                ],
                "expected_status": "zero_results" if not matched else "completed",
                "review_status": "reference_logic_verified_pending_second_person",
            }
        )
    return CASES, expected


def build_identity_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_formula: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_chemsys: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_formula[row["formula_pretty"]].append(row)
        if row.get("chemsys"):
            by_chemsys[row["chemsys"]].append(row)

    cases: list[dict[str, Any]] = []
    for row in records[:5]:
        cases.append(
            {
                "case_id": f"identity-{len(cases) + 1:03d}",
                "left_material_id": row["material_id"],
                "right_material_id": row["material_id"],
                "expected_relationship": "same_source_material_id",
                "review_status": "deterministic",
            }
        )

    duplicate_groups = [
        group
        for group in by_formula.values()
        if len({item.get("spacegroup_number") for item in group}) > 1
    ]
    duplicate_groups.sort(
        key=lambda group: (group[0]["formula_pretty"], group[0]["material_id"])
    )
    for group in duplicate_groups[:10]:
        ordered = sorted(
            group,
            key=lambda item: (item.get("spacegroup_number") or -1, item["material_id"]),
        )
        left = ordered[0]
        right = next(
            item
            for item in ordered[1:]
            if item.get("spacegroup_number") != left.get("spacegroup_number")
        )
        cases.append(
            {
                "case_id": f"identity-{len(cases) + 1:03d}",
                "left_material_id": left["material_id"],
                "right_material_id": right["material_id"],
                "formula": left["formula_pretty"],
                "left_spacegroup": left.get("spacegroup_number"),
                "right_spacegroup": right.get("spacegroup_number"),
                "expected_relationship": "same_formula_different_phase",
                "review_status": "pending_domain_review",
            }
        )

    chemsys_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for group in by_chemsys.values():
        formulas = defaultdict(list)
        for item in group:
            formulas[item["formula_pretty"]].append(item)
        if len(formulas) < 2:
            continue
        names = sorted(formulas)
        chemsys_pairs.append(
            (
                sorted(formulas[names[0]], key=lambda x: x["material_id"])[0],
                sorted(formulas[names[1]], key=lambda x: x["material_id"])[0],
            )
        )
    chemsys_pairs.sort(
        key=lambda pair: (
            pair[0]["chemsys"],
            pair[0]["material_id"],
            pair[1]["material_id"],
        )
    )
    for left, right in chemsys_pairs[:5]:
        cases.append(
            {
                "case_id": f"identity-{len(cases) + 1:03d}",
                "left_material_id": left["material_id"],
                "right_material_id": right["material_id"],
                "chemsys": left["chemsys"],
                "expected_relationship": "same_chemsys_only",
                "review_status": "deterministic",
            }
        )
    if len(cases) != 20:
        raise RuntimeError(f"Expected 20 identity cases, found {len(cases)}")
    return cases


def build_literature_index() -> list[dict[str, Any]]:
    ids = (
        "doc-3250ce3cd68ac48d95b866ae",
        "doc-6be19aada0b95fe176f931ee",
        "doc-8433c47dfe7828426efe0c59",
        "doc-cb4c833fc0948254786473b9",
        "doc-ed7f2da942310a8e026086fc",
    )
    rows: list[dict[str, Any]] = []
    for document_id in ids:
        dossier = ROOT / "data" / "literature_dossiers" / f"{document_id}.json"
        review_folder = ROOT / "data" / "literature_dossier_reviews" / document_id
        reviews = sorted(review_folder.glob("*.json"))
        approved = []
        for review in reviews:
            payload = _read_json(review)
            if payload.get("decision") == "approved":
                approved.append(review)
        if not dossier.exists() or not approved:
            raise RuntimeError(f"Missing approved dossier asset for {document_id}")
        dossier_payload = _read_json(dossier)
        rows.append(
            {
                "document_id": document_id,
                "title": dossier_payload.get("title"),
                "dossier_path": dossier.relative_to(ROOT).as_posix(),
                "dossier_sha256": _sha256(dossier),
                "review_path": approved[-1].relative_to(ROOT).as_posix(),
                "review_sha256": _sha256(approved[-1]),
                "review_decision": "approved",
                "benchmark_role": (
                    "seed_evidence_pending_task_specific_comparability_review"
                ),
            }
        )
    return rows


def build_literature_candidates() -> list[dict[str, Any]]:
    """Register task-specific open-full-text leads without promoting them to gold."""
    rows = [
        {
            "candidate_id": "lit-candidate-001",
            "title": (
                "Facile synthesis of epsilon iron oxides via spray drying for "
                "millimeter-wave absorption"
            ),
            "doi": "10.1039/D2CC03168J",
            "publisher": "Royal Society of Chemistry",
            "article_url": (
                "https://pubs.rsc.org/en/content/articlelanding/2022/cc/d2cc03168j"
            ),
            "open_full_text_url": (
                "https://pubs.rsc.org/en/content/articlepdf/2022/cc/d2cc03168j"
            ),
            "task_ids": [
                "e2e-b-polymorphs",
                "e2e-c-metastable-synthesized",
            ],
            "evidence_role": "phase_specific_experimental_synthesis",
            "material_scope": "epsilon-Fe2O3 embedded in silica-derived particles",
            "screening_note": (
                "出版社标记为开放获取；论文报告了制备高纯度 ε-Fe2O3 的喷雾干燥路线。"
                "材料身份、物相比例、退火条件和可比性仍需逐页建档并审核。"
            ),
            "local_pdf_path": None,
            "pdf_filename": "Facile synthesis of epsilon iron oxides via spray.pdf",
            "review_status": "pending_local_ingestion_and_domain_review",
        },
        {
            "candidate_id": "lit-candidate-002",
            "title": "Zeta-Fe2O3 - A new stable polymorph in iron(III) oxide family",
            "doi": "10.1038/srep15091",
            "publisher": "Scientific Reports",
            "article_url": ("https://pmc.ncbi.nlm.nih.gov/articles/PMC4606832/"),
            "open_full_text_url": "https://www.nature.com/articles/srep15091.pdf",
            "task_ids": [
                "e2e-b-polymorphs",
                "e2e-c-metastable-synthesized",
            ],
            "evidence_role": "phase_specific_high_pressure_synthesis",
            "material_scope": "zeta-Fe2O3 formed from beta-Fe2O3 under pressure",
            "screening_note": (
                "开放全文报告了一种在 30 GPa 以上形成、卸压后仍保留的单斜 Fe2O3 相。"
                "高压路线、纳米前驱体和物相指认必须在结论中明确保留。"
            ),
            "local_pdf_path": None,
            "pdf_filename": "Zeta-Fe2O3 – A new stable.pdf",
            "review_status": "pending_local_ingestion_and_domain_review",
        },
        {
            "candidate_id": "lit-candidate-003",
            "title": (
                "Physical descriptor for the Gibbs energy of inorganic crystalline "
                "solids and temperature-dependent materials chemistry"
            ),
            "doi": "10.1038/s41467-018-06682-4",
            "publisher": "Nature Communications",
            "article_url": ("https://www.nature.com/articles/s41467-018-06682-4"),
            "open_full_text_url": (
                "https://www.nature.com/articles/s41467-018-06682-4.pdf"
            ),
            "task_ids": ["e2e-c-metastable-synthesized"],
            "evidence_role": "cross_material_metastability_boundary",
            "material_scope": "ICSD-linked inorganic crystalline solids",
            "screening_note": (
                "这是跨材料边界证据：部分实验已实现结构在 0 K 计算中位于凸包之上；"
                "它不能单独证明某个具体数据库候选已经被合成。"
            ),
            "local_pdf_path": None,
            "pdf_filename": "s41467-018-06682-4.pdf",
            "review_status": "pending_local_ingestion_and_domain_review",
        },
    ]
    for row in rows:
        pdf_path = ROOT / "data" / "literature_pdfs" / "ra0" / row.pop("pdf_filename")
        if not pdf_path.exists():
            continue
        pdf_sha256 = _sha256(pdf_path)
        document_id = f"doc-{pdf_sha256[:24]}"
        dossier_path = (
            ROOT / "data" / "literature_dossier_candidates" / f"{document_id}.json"
        )
        row.update(
            {
                "document_id": document_id,
                "local_pdf_path": pdf_path.relative_to(ROOT).as_posix(),
                "pdf_sha256": pdf_sha256,
                "dossier_candidate_path": (
                    dossier_path.relative_to(ROOT).as_posix()
                    if dossier_path.exists()
                    else None
                ),
                "dossier_candidate_sha256": (
                    _sha256(dossier_path) if dossier_path.exists() else None
                ),
                "review_status": (
                    "pending_domain_review"
                    if dossier_path.exists()
                    else "pending_dossier_extraction"
                ),
            }
        )
    return rows


def build_end_to_end_tasks(
    expected_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_by_id = {row["case_id"]: row for row in expected_cases}
    return [
        {
            "task_id": "e2e-a-wide-gap-oxides",
            "title": "稳定宽禁带氧化物候选筛选",
            "screening_case_id": "screen-001",
            "database_expected": expected_by_id["screen-001"],
            "literature_seed_document_ids": ["doc-3250ce3cd68ac48d95b866ae"],
            "required_claim_checks": [
                "计算带隙与实验带隙必须分区",
                "不得从带隙直接断言绝缘强度或器件性能",
                "掺杂 TiO2 文献不得自动等同纯 TiO2 候选",
            ],
            "review_status": "draft_pending_domain_review",
        },
        {
            "task_id": "e2e-b-polymorphs",
            "title": "Fe2O3 同化学式多晶型比较",
            "screening_case_id": "screen-015",
            "database_expected": expected_by_id["screen-015"],
            "literature_seed_document_ids": [],
            "literature_candidate_ids": [
                "lit-candidate-001",
                "lit-candidate-002",
            ],
            "required_claim_checks": [
                "不同空间群不得按化学式合并",
                "计算基态与亚稳结构必须分开",
                "无实验物相证据时结论必须标记证据不足",
            ],
            "review_status": "draft_pending_literature_and_domain_review",
        },
        {
            "task_id": "e2e-c-metastable-synthesized",
            "title": "亚稳但具有实验合成证据的候选",
            "screening_case_id": "screen-004",
            "database_expected": expected_by_id["screen-004"],
            "literature_seed_document_ids": [],
            "literature_candidate_ids": [
                "lit-candidate-001",
                "lit-candidate-002",
                "lit-candidate-003",
            ],
            "required_claim_checks": [
                "is_stable=False 不得写成不能合成",
                "实验合成必须核验物相和条件",
                "未获得相应文献时不得伪造支持",
            ],
            "review_status": "draft_pending_literature_and_domain_review",
        },
    ]


def main() -> None:
    records, sources = build_snapshot()
    cases, expected = build_screening_cases(records)
    identity_cases = build_identity_cases(records)
    literature = build_literature_index()
    literature_candidates = build_literature_candidates()
    tasks = build_end_to_end_tasks(expected)

    snapshot_path = (
        BENCHMARK / "database_snapshots" / "materials_project_oxide_300.jsonl"
    )
    _write_jsonl(snapshot_path, records)
    _write_json(BENCHMARK / "database_snapshots" / "sources.json", sources)
    _write_json(BENCHMARK / "screening_cases" / "cases.json", cases)
    _write_json(BENCHMARK / "expected" / "screening_expected.json", expected)
    _write_json(BENCHMARK / "identity_cases" / "cases.json", identity_cases)
    _write_json(BENCHMARK / "literature_evidence" / "index.json", literature)
    _write_json(
        BENCHMARK / "literature_evidence" / "candidates.json",
        literature_candidates,
    )
    _write_json(BENCHMARK / "end_to_end_tasks" / "tasks.json", tasks)

    manifest = {
        "benchmark_id": "crystalline-inorganic-benchmark-v1",
        "phase": "RA-0A",
        "status": "conditional_silver_baseline",
        "gold_status": "pending_domain_review",
        "ra1_development_allowed": True,
        "generated_from_existing_snapshots": True,
        "record_count": len(records),
        "screening_case_count": len(cases),
        "identity_case_count": len(identity_cases),
        "approved_seed_dossier_count": len(literature),
        "pending_task_specific_literature_count": len(literature_candidates),
        "end_to_end_task_count": len(tasks),
        "minimum_approved_dossiers": 8,
        "domain_review_required": True,
        "assets": {
            "database_snapshot": {
                "path": snapshot_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(snapshot_path),
            },
            "screening_cases": "screening_cases/cases.json",
            "screening_expected": "expected/screening_expected.json",
            "identity_cases": "identity_cases/cases.json",
            "literature_index": "literature_evidence/index.json",
            "literature_candidates": "literature_evidence/candidates.json",
            "end_to_end_tasks": "end_to_end_tasks/tasks.json",
        },
        "known_gaps": [
            (
                "Five approved seed dossiers and three task-specific local PDF silver "
                "dossiers are registered; the three silver dossiers still require "
                "domain approval before the eight-dossier gold gate is met."
            ),
            (
                "Ten same-formula/different-space-group identity labels require "
                "domain review."
            ),
            "All three end-to-end task answer keys require domain approval.",
            "Tasks B and C still require task-specific approved literature evidence.",
            (
                "The frozen 300-record slice is oxide-heavy and does not establish "
                "full crystalline-inorganic generalization."
            ),
            (
                "Repository-wide Ruff and mypy gates have pre-existing failures "
                "outside the RA-0 files; the full offline pytest gate passes."
            ),
        ],
    }
    _write_json(BENCHMARK / "manifest.json", manifest)


if __name__ == "__main__":
    main()
