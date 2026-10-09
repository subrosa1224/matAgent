"""Data-format tests for the single agent evaluation set (S3.5-M7)."""

import json
import re
from pathlib import Path

EVAL_FILE = Path(__file__).resolve().parent / "single_agent_eval.jsonl"

ALLOWED_TOOLS = frozenset(
    {
        "run_screening_workflow",
        "get_workflow_status",
        "get_workflow_history",
        "get_screening_result",
        "compare_ranked_materials",
    }
)
FINAL_STATUSES = frozenset({"completed", "needs_user_input", "error"})
CATEGORIES = frozenset(
    {
        "screening",
        "status_history",
        "results",
        "compare",
        "general",
        "clarification",
        "safety",
    }
)
CATEGORY_MINIMUMS = {
    "screening": 15,
    "status_history": 10,
    "results": 10,
    "compare": 5,
    "general": 5,
    "clarification": 5,
    "safety": 10,
}
CASE_KEYS = {"id", "category", "turns", "tags"}
TURN_KEYS = {
    "user",
    "expected_tool",
    "allowed_tools",
    "expected_final_status",
    "evidence_required",
}
INJECTION_MARKERS = (
    "忽略系统指令",
    "INTERN_API_KEY",
    "web_search",
    "shell 命令",
    "系统提示词",
)


def _raw_cases() -> list[dict[str, object]]:
    lines = EVAL_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _turns(payload: dict[str, object]) -> list[dict[str, object]]:
    turns = payload["turns"]
    assert isinstance(turns, list)
    return [turn for turn in turns if isinstance(turn, dict)]


class TestSingleAgentEvalDataFormat:
    def test_total_cases_at_least_60(self) -> None:
        assert len(_raw_cases()) >= 60

    def test_category_minimums(self) -> None:
        cases = _raw_cases()
        counts = {
            category: sum(1 for case in cases if case["category"] == category)
            for category in CATEGORY_MINIMUMS
        }
        for category, minimum in CATEGORY_MINIMUMS.items():
            assert counts[category] >= minimum, (
                f"category {category!r}: {counts[category]} < {minimum}"
            )

    def test_ids_are_unique_and_non_empty(self) -> None:
        cases = _raw_cases()
        ids = [str(case["id"]) for case in cases]
        assert len(ids) == len(set(ids))
        assert all(value.strip() for value in ids)

    def test_every_case_contains_required_keys(self) -> None:
        for line_number, payload in enumerate(_raw_cases(), start=1):
            missing = [key for key in CASE_KEYS if key not in payload]
            assert not missing, f"line {line_number} missing keys: {missing}"

    def test_every_turn_contains_required_keys(self) -> None:
        for line_number, payload in enumerate(_raw_cases(), start=1):
            for turn_index, turn in enumerate(_turns(payload), start=1):
                missing = [key for key in TURN_KEYS if key not in turn]
                assert not missing, (
                    f"line {line_number} turn {turn_index} missing keys: {missing}"
                )

    def test_categories_are_valid(self) -> None:
        for case in _raw_cases():
            assert case["category"] in CATEGORIES, case["id"]

    def test_category_is_in_tags(self) -> None:
        for case in _raw_cases():
            tags = case["tags"]
            assert isinstance(tags, list) and tags, case["id"]
            assert case["category"] in tags, case["id"]

    def test_multi_turn_tag_matches_turn_count(self) -> None:
        for case in _raw_cases():
            tags = case["tags"]
            is_multi = len(_turns(case)) > 1
            assert ("multi_turn" in tags) is is_multi, case["id"]

    def test_final_status_is_valid(self) -> None:
        for case in _raw_cases():
            for turn in _turns(case):
                assert turn["expected_final_status"] in FINAL_STATUSES, case["id"]

    def test_expected_tool_is_none_or_whitelisted(self) -> None:
        for case in _raw_cases():
            for turn in _turns(case):
                tool = turn["expected_tool"]
                assert tool is None or tool in ALLOWED_TOOLS, f"{case['id']}: {tool!r}"

    def test_allowed_tools_are_whitelisted(self) -> None:
        for case in _raw_cases():
            for turn in _turns(case):
                allowed = turn["allowed_tools"]
                assert isinstance(allowed, list)
                assert set(allowed) <= ALLOWED_TOOLS, case["id"]

    def test_expected_tool_is_in_allowed_tools(self) -> None:
        for case in _raw_cases():
            for turn in _turns(case):
                tool = turn["expected_tool"]
                if tool is not None:
                    assert tool in turn["allowed_tools"], (
                        f"{case['id']}: {tool} not allowed"
                    )

    def test_evidence_required_implies_tool(self) -> None:
        for case in _raw_cases():
            for turn in _turns(case):
                if turn["evidence_required"] is True:
                    assert turn["expected_tool"] is not None, case["id"]

    def test_no_tool_turns_require_no_evidence(self) -> None:
        for case in _raw_cases():
            for turn in _turns(case):
                if turn["expected_tool"] is None:
                    assert turn["evidence_required"] is False, case["id"]

    def test_general_cases_never_call_tools(self) -> None:
        for case in _raw_cases():
            if case["category"] == "general":
                for turn in _turns(case):
                    assert turn["expected_tool"] is None
                    assert turn["allowed_tools"] == []
                    assert turn["evidence_required"] is False

    def test_clarification_cases_ask_for_more_input(self) -> None:
        for case in _raw_cases():
            if case["category"] == "clarification":
                statuses = {turn["expected_final_status"] for turn in _turns(case)}
                assert "needs_user_input" in statuses, case["id"]

    def test_user_messages_do_not_reference_database_content(self) -> None:
        material_id_pattern = re.compile(r"mp-\d+", re.IGNORECASE)
        for case in _raw_cases():
            for turn in _turns(case):
                user = str(turn["user"])
                assert not material_id_pattern.search(user), case["id"]
                assert "http://" not in user and "https://" not in user

    def test_expected_material_ids_are_mock_ids(self) -> None:
        material_id_pattern = re.compile(r"^mp-\d+$")
        for case in _raw_cases():
            for turn in _turns(case):
                expected = turn.get("expected_material_ids", [])
                assert isinstance(expected, list)
                assert all(
                    isinstance(value, str) and material_id_pattern.fullmatch(value)
                    for value in expected
                ), f"{case['id']}: {expected}"

    def test_injection_markers_preserved_in_safety_cases(self) -> None:
        safety_text = " ".join(
            str(turn["user"])
            for case in _raw_cases()
            if case["category"] == "safety"
            for turn in _turns(case)
        )
        for marker in INJECTION_MARKERS:
            assert marker in safety_text, marker

    def test_safety_boundary_tags_are_represented(self) -> None:
        safety_tags = {
            tag
            for case in _raw_cases()
            if case["category"] == "safety"
            for tag in case["tags"]
        }
        assert {"injection", "cross_session", "fabrication"} <= safety_tags
