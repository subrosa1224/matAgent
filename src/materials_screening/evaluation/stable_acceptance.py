"""Fixed, reviewer-led research acceptance; never imports production filters.

Runtime completion is deliberately not a scoring input. References and ledgers
are evaluator-only and must not be supplied to the research agents.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

PROPERTIES = ("energy_above_hull_ev_atom", "band_gap_ev", "density_g_cm3")
TIERS = {"live", "offline_replay", "fault_simulation"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def asset_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact outside workspace: {relative}")
    return path


def reference_records(references: dict, bundle_id: str) -> list[dict]:
    bundle = references["bundles"][bundle_id]
    if "extends" not in bundle:
        return bundle["records"]
    parent = reference_records(references, bundle["extends"])
    return [r for r in parent if r["id"] in bundle["record_ids"]]


def validate_suite(root: Path, suite: dict, references: dict) -> None:
    cases = suite["cases"]
    if len(cases) != 15 or len({c["id"] for c in cases}) != 15:
        raise ValueError("Expected 15 distinct cases")
    if Counter(c["kind"] for c in cases) != {"normal": 10, "boundary": 5}:
        raise ValueError("Expected 10 normal and 5 boundary cases")
    if Counter(c["split"] for c in cases) != {"development": 10, "holdout": 5}:
        raise ValueError("Expected 10 development and 5 holdout cases")
    if suite["repetitions"] != 3:
        raise ValueError("Expected three repetitions")
    for key, asset in suite["assets"].items():
        path = asset_path(root, asset["path"])
        if not path.is_file() or digest(path) != asset["sha256"]:
            raise ValueError(f"Missing or changed asset: {key}")
    for case in cases:
        if "reference_bundle" not in case:
            continue
        records = reference_records(references, case["reference_bundle"])
        ids = [r["id"] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate reference records: {case['id']}")
        if len(ids) < case["minimum_experimental_records"]:
            raise ValueError(f"Insufficient references: {case['id']}")


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def matches(record: dict, constraints: dict) -> bool:
    """Independent exact filtering. Missing hard fields fail closed."""
    elements = record.get("elements")
    if any(
        k in constraints
        for k in (
            "required_elements",
            "excluded_elements",
            "exact_elements",
            "num_elements",
        )
    ):
        if not isinstance(elements, list) or not elements:
            return False
        actual = set(elements)
        if not set(constraints.get("required_elements", [])).issubset(actual):
            return False
        if actual.intersection(constraints.get("excluded_elements", [])):
            return False
        if "exact_elements" in constraints and actual != set(
            constraints["exact_elements"]
        ):
            return False
        if "num_elements" in constraints and len(actual) != constraints["num_elements"]:
            return False
    if (
        "formulas" in constraints
        and record.get("formula_pretty") not in constraints["formulas"]
    ):
        return False
    if (
        "is_metal" in constraints
        and record.get("is_metal") is not constraints["is_metal"]
    ):
        return False
    if "crystal_system" in constraints:
        crystal = record.get("crystal_system") or (record.get("symmetry") or {}).get(
            "crystal_system"
        )
        if crystal != constraints["crystal_system"]:
            return False
    for key in PROPERTIES:
        if key not in constraints:
            continue
        value = record.get(key)
        low, high = constraints[key]
        if not _number(value):
            return False
        if (low is not None and value < low) or (high is not None and value > high):
            return False
    return True


def frozen_expectation(root: Path, suite: dict, case: dict) -> dict:
    asset = suite["assets"][case["snapshot"]]
    path = asset_path(root, asset["path"])
    selected = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = [r for r in selected if matches(r, case["constraints"])]
    ids = [r["material_id"] for r in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in frozen snapshot")
    groups = {}
    for field in dict.fromkeys([case["group_by"], case.get("additional_group_by")]):
        if field is None:
            continue
        grouped: dict[str, list[dict]] = {}
        for record in selected:
            value = record.get(field)
            if field == "crystal_system":
                value = value or (record.get("symmetry") or {}).get(field)
            grouped.setdefault(value or "unknown", []).append(record)
        table = {}
        for key, rows in sorted(grouped.items()):
            stats = {}
            for prop in PROPERTIES:
                values = [r[prop] for r in rows if _number(r.get(prop))]
                stats[prop] = {
                    "n": len(values),
                    "missing": len(rows) - len(values),
                    "mean": statistics.fmean(values) if values else None,
                    "median": statistics.median(values) if values else None,
                }
            table[key] = {"count": len(rows), "properties": stats}
        groups[field] = table
    return {
        "case_id": case["id"],
        "snapshot_sha256": asset["sha256"],
        "coverage": asset["coverage"],
        "count": len(selected),
        "material_ids": sorted(ids),
        "groups": groups,
        "warning": "仅限固定切片；live须另对实际查询存档独立计算，不能照搬这里的计数。",
    }


def ledger_template(suite: dict, version: str, primary_tier: str) -> dict:
    if primary_tier not in {"live", "offline_replay"}:
        raise ValueError("Primary tier must be live or offline_replay")
    observations = []
    for case in suite["cases"]:
        names = (
            suite["normal_checks"]
            if case["kind"] == "normal"
            else case["boundary_checks"]
        )
        for repeat in range(1, suite["repetitions"] + 1):
            observations.append(
                {
                    "case_id": case["id"],
                    "repeat": repeat,
                    "tier": "fault_simulation" if case["id"] == "B05" else primary_tier,
                    "version": version,
                    "execution_id": "",
                    "conversation_id": "",
                    "fresh_execution": False,
                    "cache_mode": "unknown",
                    "origin": "not_run",
                    "runtime_status": "not_run",
                    "report_path": "",
                    "review": {
                        "reviewer": "",
                        "completed": False,
                        "checks": {
                            name: {"verdict": "unknown", "evidence": []}
                            for name in names
                        },
                        "critical_errors": {
                            name: {"verdict": "unknown", "evidence": []}
                            for name in suite["critical_errors"]
                        },
                        "matched_reference_record_ids": [],
                    },
                }
            )
    return {
        "suite_id": suite["suite_id"],
        "version": version,
        "primary_tier": primary_tier,
        "observations": observations,
    }


def _has_evidence(root: Path, assessment: dict) -> bool:
    evidence = assessment.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        return False
    for item in evidence:
        if (
            not isinstance(item, dict)
            or not item.get("path")
            or not item.get("locator")
        ):
            return False
        path = asset_path(root, item["path"])
        if not path.is_file() or item.get("sha256") != digest(path):
            return False
    return True


def score_ledger(root: Path, suite: dict, references: dict, ledger: dict) -> dict:
    """Missing/unreviewed runs never shrink denominators or become successes."""
    if ledger.get("suite_id") != suite["suite_id"] or not ledger.get("version"):
        raise ValueError("Suite/version missing or mismatched")
    primary = ledger.get("primary_tier")
    if primary not in {"live", "offline_replay"}:
        raise ValueError("Invalid primary tier")
    cases = {c["id"]: c for c in suite["cases"]}
    observed = {}
    execution_ids = set()
    conversations = set()
    for entry in ledger.get("observations", []):
        case_id, repeat = entry["case_id"], entry["repeat"]
        if case_id not in cases or type(repeat) is not int or repeat not in range(1, 4):
            raise ValueError("Unknown case or invalid repetition")
        key = (case_id, repeat)
        if key in observed:
            raise ValueError("Duplicate case/repetition")
        expected_tier = "fault_simulation" if case_id == "B05" else primary
        if (
            entry.get("tier") != expected_tier
            or entry.get("version") != ledger["version"]
        ):
            raise ValueError("Mixed versions or tiers")
        if entry.get("origin") != "not_run":
            for field, seen in (
                ("execution_id", execution_ids),
                ("conversation_id", conversations),
            ):
                value = entry.get(field)
                if not value or value in seen:
                    raise ValueError(f"Missing or reused {field}")
                seen.add(value)
        observed[key] = entry
    results = []
    for case in suite["cases"]:
        for repeat in range(1, 4):
            entry = observed.get((case["id"], repeat))
            status, reasons, critical = "not_run", [], []
            if entry and entry.get("origin") != "not_run":
                review = entry.get("review") or {}
                checks = review.get("checks") or {}
                errors = review.get("critical_errors") or {}
                names = (
                    suite["normal_checks"]
                    if case["kind"] == "normal"
                    else case["boundary_checks"]
                )
                assessments = [checks.get(n, {}) for n in names]
                errors_list = [errors.get(n, {}) for n in suite["critical_errors"]]
                critical = [
                    n
                    for n in suite["critical_errors"]
                    if errors.get(n, {}).get("verdict") == "present"
                ]
                reviewed = bool(review.get("reviewer") and review.get("completed"))
                reviewed = reviewed and all(
                    a.get("verdict") in {"pass", "fail"} and _has_evidence(root, a)
                    for a in assessments
                )
                reviewed = reviewed and all(
                    a.get("verdict") in {"absent", "present"} and _has_evidence(root, a)
                    for a in errors_list
                )
                status = "unreviewed"
                if reviewed:
                    origin = {
                        "live": "live_agent",
                        "offline_replay": "offline_agent",
                        "fault_simulation": "fault_adapter",
                    }[entry["tier"]]
                    if entry.get("origin") != origin:
                        reasons.append("execution_origin_unverified")
                    if (
                        not entry.get("fresh_execution")
                        or entry.get("cache_mode") != "cold"
                    ):
                        reasons.append("cached_or_nonfresh_execution")
                    reasons.extend(n for n in names if checks[n]["verdict"] != "pass")
                    reasons.extend(critical)
                    if case["kind"] == "normal":
                        matched = review.get("matched_reference_record_ids") or []
                        allowed = {
                            r["id"]
                            for r in reference_records(
                                references, case["reference_bundle"]
                            )
                        }
                        if len(matched) != len(set(matched)) or not set(
                            matched
                        ).issubset(allowed):
                            reasons.append("invalid_or_duplicate_reference_match")
                        if (
                            len(set(matched).intersection(allowed))
                            < case["minimum_experimental_records"]
                        ):
                            reasons.append("insufficient_condition_bound_measurements")
                    status = "fail" if reasons else "pass"
                elif critical:
                    reasons.extend(critical)
            results.append(
                {
                    "case_id": case["id"],
                    "repeat": repeat,
                    "kind": case["kind"],
                    "split": case["split"],
                    "tier": "fault_simulation" if case["id"] == "B05" else primary,
                    "status": status,
                    "reasons": reasons,
                    "critical_errors": critical,
                }
            )
    normal = [r for r in results if r["kind"] == "normal"]
    boundary = [r for r in results if r["kind"] == "boundary" and r["tier"] == primary]
    faults = [r for r in results if r["tier"] == "fault_simulation"]

    def counts(rows: list[dict]) -> dict:
        counts_by_status = Counter(r["status"] for r in rows)
        return {
            "planned": len(rows),
            "passed": counts_by_status["pass"],
            "failed": counts_by_status["fail"],
            "unreviewed": counts_by_status["unreviewed"],
            "not_run": counts_by_status["not_run"],
            "confirmed_pass_fraction_of_planned": counts_by_status["pass"] / len(rows)
            if rows
            else None,
        }

    complete = all(r["status"] in {"pass", "fail"} for r in results)
    holdout_normal = [r for r in normal if r["split"] == "holdout"]
    threshold = suite["normal_completion_threshold"]
    passed = complete and counts(normal)["passed"] >= math.ceil(len(normal) * threshold)
    passed = passed and counts(holdout_normal)["passed"] >= math.ceil(
        len(holdout_normal) * threshold
    )
    passed = passed and all(r["status"] == "pass" for r in boundary + faults)
    passed = passed and not any(r["critical_errors"] for r in results)
    return {
        "suite_id": suite["suite_id"],
        "version": ledger["version"],
        "primary_tier": primary,
        "gate": "pass_provisional"
        if passed
        else ("failed" if complete else "incomplete"),
        "reference_status": references["status"],
        "normal": counts(normal),
        "normal_development": counts(
            [r for r in normal if r["split"] == "development"]
        ),
        "normal_holdout": counts(holdout_normal),
        "boundary_primary": counts(boundary),
        "boundary_fault_simulation": counts(faults),
        "results": results,
        "note": "实验/来源核对需独立审核；工具仅执行评分规则，不证明审核意见正确。"
        "通过只代表本套件限定范围工程验收，非专家金标准或全材料泛化。",
    }
