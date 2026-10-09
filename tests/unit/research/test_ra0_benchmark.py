from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "tests" / "fixtures" / "research" / "benchmark_v1"


def _json(path: Path):  # type: ignore[no-untyped-def]
    return json.loads(path.read_text(encoding="utf-8"))


def test_ra0_manifest_and_snapshot_are_consistent() -> None:
    manifest = _json(BENCHMARK / "manifest.json")
    snapshot = ROOT / manifest["assets"]["database_snapshot"]["path"]
    rows = [
        json.loads(line) for line in snapshot.read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["benchmark_id"] == "crystalline-inorganic-benchmark-v1"
    assert manifest["status"] == "conditional_silver_baseline"
    assert manifest["gold_status"] == "pending_domain_review"
    assert manifest["ra1_development_allowed"] is True
    assert len(rows) == manifest["record_count"] == 300
    assert len({row["material_id"] for row in rows}) == 300
    assert all("O" in row["elements"] for row in rows)
    assert (
        hashlib.sha256(snapshot.read_bytes()).hexdigest()
        == manifest["assets"]["database_snapshot"]["sha256"]
    )


def test_ra0_has_required_case_layers() -> None:
    manifest = _json(BENCHMARK / "manifest.json")
    screening = _json(BENCHMARK / "screening_cases" / "cases.json")
    expected = _json(BENCHMARK / "expected" / "screening_expected.json")
    identities = _json(BENCHMARK / "identity_cases" / "cases.json")
    literature = _json(BENCHMARK / "literature_evidence" / "index.json")
    candidates = _json(BENCHMARK / "literature_evidence" / "candidates.json")
    tasks = _json(BENCHMARK / "end_to_end_tasks" / "tasks.json")

    assert len(screening) == len(expected) == manifest["screening_case_count"] == 15
    assert {case["case_id"] for case in screening} == {
        case["case_id"] for case in expected
    }
    assert len(identities) == manifest["identity_case_count"] == 20
    assert len(literature) == manifest["approved_seed_dossier_count"] == 5
    assert len(candidates) == manifest["pending_task_specific_literature_count"] == 3
    assert len(tasks) == manifest["end_to_end_task_count"] == 3
    assert any(item["expected_status"] == "zero_results" for item in expected)


def test_ra0_identity_cases_include_required_boundaries() -> None:
    identities = _json(BENCHMARK / "identity_cases" / "cases.json")
    relationships = {case["expected_relationship"] for case in identities}

    assert relationships == {
        "same_source_material_id",
        "same_formula_different_phase",
        "same_chemsys_only",
    }
    assert (
        sum(case["review_status"] == "pending_domain_review" for case in identities)
        == 10
    )


def test_ra0_literature_assets_are_approved_and_immutable() -> None:
    literature = _json(BENCHMARK / "literature_evidence" / "index.json")
    for item in literature:
        dossier = ROOT / item["dossier_path"]
        review = ROOT / item["review_path"]
        assert item["review_decision"] == "approved"
        assert dossier.exists() and review.exists()
        assert (
            hashlib.sha256(dossier.read_bytes()).hexdigest() == item["dossier_sha256"]
        )
        assert hashlib.sha256(review.read_bytes()).hexdigest() == item["review_sha256"]


def test_ra0_task_specific_literature_leads_are_not_promoted_to_gold() -> None:
    candidates = _json(BENCHMARK / "literature_evidence" / "candidates.json")

    assert len(candidates) == 3
    assert all(candidate["doi"] for candidate in candidates)
    assert all(
        candidate["open_full_text_url"].startswith("https://")
        for candidate in candidates
    )
    assert all(
        candidate["review_status"] == "pending_domain_review"
        for candidate in candidates
    )
    for candidate in candidates:
        pdf = ROOT / candidate["local_pdf_path"]
        dossier = ROOT / candidate["dossier_candidate_path"]
        assert pdf.read_bytes().startswith(b"%PDF")
        assert hashlib.sha256(pdf.read_bytes()).hexdigest() == candidate["pdf_sha256"]
        assert dossier.exists()
        assert (
            hashlib.sha256(dossier.read_bytes()).hexdigest()
            == candidate["dossier_candidate_sha256"]
        )
        assert _json(dossier)["review_status"] == "pending"
    assert {task for candidate in candidates for task in candidate["task_ids"]} == {
        "e2e-b-polymorphs",
        "e2e-c-metastable-synthesized",
    }


def test_ra0_truthfully_records_unfinished_gold_review() -> None:
    manifest = _json(BENCHMARK / "manifest.json")
    tasks = _json(BENCHMARK / "end_to_end_tasks" / "tasks.json")

    assert manifest["domain_review_required"] is True
    assert (
        manifest["approved_seed_dossier_count"] < manifest["minimum_approved_dossiers"]
    )
    assert all(task["review_status"].startswith("draft_pending") for task in tasks)
    assert manifest["known_gaps"]


def test_ra0_current_baseline_is_present_and_bounded() -> None:
    baseline = _json(BENCHMARK / "expected" / "current_baseline.json")

    assert baseline["baseline_kind"] == "existing_implementation_before_ra1"
    assert (
        baseline["filter_cases_evaluated"] + baseline["unsupported_filter_cases"] == 15
    )
    assert baseline["filter_cases_exact"] <= baseline["filter_cases_evaluated"]
    assert baseline["task_specific_ranking_cases_supported"] == 0
    assert baseline["identity_relationship_cases_supported"] == 0
    assert baseline["end_to_end_tasks_completed"] == 0
    assert baseline["end_to_end_task_count"] == 3
