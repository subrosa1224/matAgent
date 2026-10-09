"""Data-format tests for the planner evaluation set (D2-M7)."""

import json
import re
from pathlib import Path

from materials_screening.evaluation.planner_evaluator import load_cases
from materials_screening.models import ScreeningRequest
from materials_screening.planner.models import PlannerStatus
from materials_screening.planner.rules import AmbiguityCode, ConflictCode

EVAL_FILE = Path(__file__).resolve().parent / "planner_eval.jsonl"
REPRESENTATIVE_FILE = (
    Path(__file__).resolve().parent / "planner_eval_representative_20.jsonl"
)

REQUIRED_KEYS = (
    "id",
    "query",
    "expected_status",
    "expected_request",
    "expected_ambiguity_codes",
    "expected_conflict_codes",
    "tags",
)

CATEGORY_MINIMUMS = {
    "cn_basic": 20,
    "en_basic": 10,
    "units": 10,
    "elements": 10,
    "ambiguous": 10,
    "conflict": 8,
    "unsupported": 6,
    "injection": 6,
}

HARD_INJECTION_MARKERS = (
    "忽略系统指令",
    "INTERN_API_KEY",
    "调用 Materials Project",
    "删除",
    "Markdown",
    "伪造",
    "shell 命令",
)


def _raw_lines() -> list[dict[str, object]]:
    lines = EVAL_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


class TestPlannerEvalDataFormat:
    def test_total_cases_at_least_80(self) -> None:
        cases = load_cases(EVAL_FILE)
        assert len(cases) >= 80

    def test_every_line_contains_all_required_keys(self) -> None:
        for line_number, payload in enumerate(_raw_lines(), start=1):
            missing = [key for key in REQUIRED_KEYS if key not in payload]
            assert not missing, f"line {line_number} missing keys: {missing}"

    def test_ids_are_unique(self) -> None:
        cases = load_cases(EVAL_FILE)
        ids = [case.id for case in cases]
        assert len(ids) == len(set(ids))

    def test_ids_and_queries_and_tags_are_non_empty(self) -> None:
        cases = load_cases(EVAL_FILE)
        for case in cases:
            assert case.id.strip()
            assert case.query.strip()
            assert case.tags

    def test_distribution_meets_minimums(self) -> None:
        cases = load_cases(EVAL_FILE)
        counts = {
            category: sum(1 for case in cases if category in case.tags)
            for category in CATEGORY_MINIMUMS
        }
        for category, minimum in CATEGORY_MINIMUMS.items():
            assert counts[category] >= minimum, (
                f"category {category!r}: {counts[category]} < {minimum}"
            )

    def test_expected_status_is_valid(self) -> None:
        cases = load_cases(EVAL_FILE)
        assert all(isinstance(case.expected_status, PlannerStatus) for case in cases)

    def test_expected_request_is_valid_request_subset(self) -> None:
        cases = load_cases(EVAL_FILE)
        for case in cases:
            unknown_keys = set(case.expected_request) - set(
                ScreeningRequest.model_fields
            )
            assert not unknown_keys, f"{case.id}: unknown request keys {unknown_keys}"
            ScreeningRequest.model_validate(case.expected_request)

    def test_ambiguity_codes_are_valid(self) -> None:
        cases = load_cases(EVAL_FILE)
        valid = {code.value for code in AmbiguityCode}
        for case in cases:
            assert set(case.expected_ambiguity_codes) <= valid, case.id

    def test_conflict_codes_are_valid(self) -> None:
        cases = load_cases(EVAL_FILE)
        valid = {code.value for code in ConflictCode}
        for case in cases:
            assert set(case.expected_conflict_codes) <= valid, case.id

    def test_non_ready_cases_have_empty_expected_request(self) -> None:
        cases = load_cases(EVAL_FILE)
        for case in cases:
            if case.expected_status is not PlannerStatus.READY:
                assert case.expected_request == {}, case.id

    def test_conflict_codes_only_on_conflict_cases(self) -> None:
        cases = load_cases(EVAL_FILE)
        for case in cases:
            if case.expected_conflict_codes:
                assert case.expected_status is PlannerStatus.INVALID, case.id

    def test_queries_do_not_reference_database_content(self) -> None:
        cases = load_cases(EVAL_FILE)
        material_id_pattern = re.compile(r"mp-\d+", re.IGNORECASE)
        for case in cases:
            assert not material_id_pattern.search(case.query), case.id
            assert "http://" not in case.query and "https://" not in case.query

    def test_hard_injection_cases_are_preserved(self) -> None:
        cases = load_cases(EVAL_FILE)
        injection_queries = " ".join(
            case.query for case in cases if "injection" in case.tags
        )
        for marker in HARD_INJECTION_MARKERS:
            assert marker in injection_queries, marker


class TestRepresentative20DataFormat:
    def test_exactly_twenty_unique_cases(self) -> None:
        cases = load_cases(REPRESENTATIVE_FILE)
        assert len(cases) == 20
        assert len({case.id for case in cases}) == 20

    def test_all_ids_exist_in_full_set(self) -> None:
        full_ids = {case.id for case in load_cases(EVAL_FILE)}
        representative = load_cases(REPRESENTATIVE_FILE)
        assert all(case.id in full_ids for case in representative)

    def test_covers_all_categories(self) -> None:
        cases = load_cases(REPRESENTATIVE_FILE)
        for category in CATEGORY_MINIMUMS:
            assert any(category in case.tags for case in cases), category

    def test_loads_with_required_keys(self) -> None:
        for payload in _raw_representative_lines():
            assert all(key in payload for key in REQUIRED_KEYS)


def _raw_representative_lines() -> list[dict[str, object]]:
    lines = REPRESENTATIVE_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
