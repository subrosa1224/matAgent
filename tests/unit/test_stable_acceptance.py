from __future__ import annotations

import copy
from pathlib import Path

import pytest

from materials_screening.evaluation.stable_acceptance import (
    digest,
    frozen_expectation,
    ledger_template,
    matches,
    read_json,
    reference_records,
    score_ledger,
    validate_suite,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/research/stable_acceptance_v1"


@pytest.fixture
def inputs():
    return read_json(FIXTURE / "suite.json"), read_json(FIXTURE / "references.json")


def reviewed_ledger(root, suite, references):
    """Synthetic scorer test only; never presented as an actual agent run."""
    path = root / "audit.md"
    path.write_text("Synthetic fixture, not research evidence", encoding="utf-8")
    evidence = [{"path": "audit.md", "sha256": digest(path), "locator": "test fixture"}]
    ledger = ledger_template(suite, "synthetic-unit-test", "live")
    for n, entry in enumerate(ledger["observations"]):
        entry.update(
            execution_id=f"test-{n}",
            conversation_id=f"test-{n}",
            fresh_execution=True,
            cache_mode="cold",
            runtime_status="completed",
            origin="fault_adapter"
            if entry["tier"] == "fault_simulation"
            else "live_agent",
        )
        review = entry["review"]
        review.update(completed=True, reviewer="synthetic test fixture")
        for check in review["checks"].values():
            check.update(verdict="pass", evidence=copy.deepcopy(evidence))
        for check in review["critical_errors"].values():
            check.update(verdict="absent", evidence=copy.deepcopy(evidence))
        case = next(c for c in suite["cases"] if c["id"] == entry["case_id"])
        if case["kind"] == "normal":
            review["matched_reference_record_ids"] = [
                r["id"] for r in reference_records(references, case["reference_bundle"])
            ]
    return ledger


def test_fixed_suite_assets_and_split(inputs):
    suite, refs = inputs
    validate_suite(ROOT, suite, refs)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_missing_or_invalid_hard_numeric_field_rejected(value):
    assert not matches({"band_gap_ev": value}, {"band_gap_ev": [1, 4]})


def test_exact_elements_no_subset_and_zero_threshold():
    constraints = {
        "exact_elements": ["Li", "Fe", "O"],
        "energy_above_hull_ev_atom": [None, 0],
    }
    assert matches(
        {"elements": ["O", "Fe", "Li"], "energy_above_hull_ev_atom": 0}, constraints
    )
    assert not matches(
        {"elements": ["O", "Fe"], "energy_above_hull_ev_atom": 0}, constraints
    )
    assert not matches(
        {"elements": ["O", "Fe", "Li"], "energy_above_hull_ev_atom": 0.01}, constraints
    )


def test_nested_symmetry_supported():
    assert matches(
        {"symmetry": {"crystal_system": "Orthorhombic"}},
        {"crystal_system": "Orthorhombic"},
    )


def test_frozen_independent_expected_counts(inputs):
    suite, _ = inputs
    expected = {c["id"]: frozen_expectation(ROOT, suite, c) for c in suite["cases"]}
    assert expected["N05"]["count"] == 84
    assert expected["N06"]["count"] == 7
    assert expected["N08"]["count"] == 6
    assert expected["B01"]["count"] == 0
    assert len(expected["N05"]["groups"]["formula_pretty"]) == 42
    assert len(expected["N05"]["groups"]["crystal_system"]) == 7


def test_empty_and_runtime_only_never_pass(inputs):
    suite, refs = inputs
    ledger = ledger_template(suite, "v1", "live")
    result = score_ledger(ROOT, suite, refs, ledger)
    assert result["gate"] == "incomplete"
    assert result["normal"]["planned"] == 30
    assert result["normal"]["not_run"] == 30
    entry = ledger["observations"][0]
    entry.update(
        origin="live_agent",
        execution_id="one",
        conversation_id="one",
        runtime_status="completed",
    )
    result = score_ledger(ROOT, suite, refs, ledger)
    assert result["normal"]["passed"] == 0
    assert result["normal"]["unreviewed"] == 1


def test_synthetic_all_pass_and_fault_separate(tmp_path, inputs):
    suite, refs = inputs
    result = score_ledger(tmp_path, suite, refs, reviewed_ledger(tmp_path, suite, refs))
    assert result["gate"] == "pass_provisional"
    assert result["boundary_primary"]["planned"] == 12
    assert result["boundary_fault_simulation"]["planned"] == 3


@pytest.mark.parametrize(
    "mutation", ["critical", "zero_records", "cached", "duplicate_records"]
)
def test_hard_errors_or_unsupported_extraction_fail(tmp_path, inputs, mutation):
    suite, refs = inputs
    ledger = reviewed_ledger(tmp_path, suite, refs)
    entry = ledger["observations"][0]
    if mutation == "critical":
        entry["review"]["critical_errors"]["reference_as_own_work"]["verdict"] = (
            "present"
        )
    elif mutation == "cached":
        entry["cache_mode"] = "warm"
    elif mutation == "zero_records":
        entry["review"]["matched_reference_record_ids"] = []
    else:
        entry["review"]["matched_reference_record_ids"] *= 2
    result = score_ledger(tmp_path, suite, refs, ledger)
    assert result["results"][0]["status"] == "fail"
    if mutation == "critical":
        assert result["gate"] == "failed"


def test_missing_evidence_remains_unreviewed(tmp_path, inputs):
    suite, refs = inputs
    ledger = reviewed_ledger(tmp_path, suite, refs)
    ledger["observations"][0]["review"]["checks"]["report_grounding"]["evidence"] = []
    assert score_ledger(tmp_path, suite, refs, ledger)["gate"] == "incomplete"


def test_holdout_failure_cannot_hide_in_aggregate(tmp_path, inputs):
    suite, refs = inputs
    ledger = reviewed_ledger(tmp_path, suite, refs)
    next(e for e in ledger["observations"] if e["case_id"] == "N09")["review"][
        "checks"
    ]["experimental_analysis"]["verdict"] = "fail"
    result = score_ledger(tmp_path, suite, refs, ledger)
    assert result["normal"]["passed"] == 29
    assert result["gate"] == "failed"


@pytest.mark.parametrize(
    "mutation", ["duplicate_run", "mixed_version", "mixed_tier", "reused_context"]
)
def test_protocol_mixing_rejected(tmp_path, inputs, mutation):
    suite, refs = inputs
    ledger = reviewed_ledger(tmp_path, suite, refs)
    if mutation == "duplicate_run":
        ledger["observations"].append(copy.deepcopy(ledger["observations"][0]))
    elif mutation == "mixed_version":
        ledger["observations"][0]["version"] = "another-version"
    elif mutation == "mixed_tier":
        ledger["observations"][0]["tier"] = "offline_replay"
    else:
        ledger["observations"][1]["conversation_id"] = ledger["observations"][0][
            "conversation_id"
        ]
    with pytest.raises(ValueError):
        score_ledger(tmp_path, suite, refs, ledger)


def test_altered_evidence_not_pass(tmp_path, inputs):
    suite, refs = inputs
    ledger = reviewed_ledger(tmp_path, suite, refs)
    (tmp_path / "audit.md").write_text("changed", encoding="utf-8")
    assert score_ledger(tmp_path, suite, refs, ledger)["gate"] == "incomplete"
