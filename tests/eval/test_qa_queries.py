"""QA 文档 E 区（Q60–Q115）真实场景查询的可执行性校验。

These are data-quality checks for docs/qa/MATERIALS_QA.md: every E-section
question must be a query that can produce a normal result in real mode
(Intern + Materials Project) — valid element symbols, no unsupported-task
keywords, no required/excluded conflicts, a strong constraint to avoid the
broad-query rejection, and explicit units for numeric constraints.
"""

import re
from pathlib import Path

import pytest
from pymatgen.core import Element

_QA_DOC = (
    Path(__file__).resolve().parent.parent.parent / "docs" / "qa" / "MATERIALS_QA.md"
)

# Markers that make the planner return UNSUPPORTED (planner/rules.py).
_UNSUPPORTED_MARKERS = (
    "预测",
    "生成候选",
    "发现新材料",
    "设计新材料",
    "predict",
    "dft",
    "第一性原理",
    "从头计算",
    "计算任务",
    "跑计算",
    "合成方法",
    "如何合成",
    "制备方法",
    "synthesis",
    "实验数据",
    "实验测量",
    "experimental",
)

# Strong constraints that keep real Materials Project queries from being
# rejected as too broad (repositories/materials_project.py is_broad_request).
_STRONG_CONSTRAINT_MARKERS = (
    "含",
    "不含",
    "带隙",
    "band gap",
    "hull",
    "凸包",
    "能量高于",
    "密度",
    "晶系",
    "空间群",
    "稳定",
    "金属",
    "直接带隙",
    "禁带",
)

_ELEMENT_GROUP = re.compile(
    r"(?:含|不含|必含)\s*"
    r"([A-Za-z][a-z]?(?:\s*[、，,和/或]\s*[A-Za-z][a-z]?)*)"
)
_ELEMENT_SPLIT = re.compile(r"\s*[、，,和/或]\s*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_UNIT_HINT = re.compile(r"(?:eV|meV|g/cm|eV/atom|meV/atom|g/cm3|g/cm³)")


def _qa_questions() -> list[tuple[int, str]]:
    """Parse the E-section question headings (### QN <question>)."""
    lines = _QA_DOC.read_text(encoding="utf-8").splitlines()
    questions: list[tuple[int, str]] = []
    in_e = False
    for line in lines:
        if line.startswith("## E."):
            in_e = True
            continue
        if in_e and line.startswith("## "):
            break
        if not in_e:
            continue
        match = re.match(r"^### Q(\d+)\s+(.+?)\s*$", line)
        if match:
            questions.append((int(match.group(1)), match.group(2)))
    return questions


def _element_tokens(query: str) -> tuple[list[str], list[str]]:
    """Extract required/excluded element tokens; non-symbol tokens are kept
    so the validity test can flag them."""
    required: list[str] = []
    excluded: list[str] = []
    for match in _ELEMENT_GROUP.finditer(query):
        bucket = excluded if match.group(0).startswith("不含") else required
        for token in _ELEMENT_SPLIT.split(match.group(1)):
            token = token.strip()
            if token:
                bucket.append(token)
    return required, excluded


class TestQaQueries:
    """E-section queries must satisfy static "can run in real mode" checks."""

    def test_e_section_question_count(self) -> None:
        questions = _qa_questions()

        numbers = [number for number, _ in questions]
        assert numbers[0] == 60
        assert numbers[-1] == 115
        assert len(questions) == 56
        assert numbers == list(range(60, 116))

    @pytest.mark.parametrize("number,query", _qa_questions())
    def test_element_symbols_are_valid(self, number: int, query: str) -> None:
        required, excluded = _element_tokens(query)
        for token in [*required, *excluded]:
            assert Element.is_valid_symbol(token), (
                f"Q{number}: {token!r} is not a valid element symbol (query: {query})"
            )

    @pytest.mark.parametrize("number,query", _qa_questions())
    def test_no_unsupported_task_keywords(self, number: int, query: str) -> None:
        lowered = query.lower()
        for marker in _UNSUPPORTED_MARKERS:
            assert marker not in lowered, (
                f"Q{number}: unsupported-task keyword {marker!r} (query: {query})"
            )

    @pytest.mark.parametrize("number,query", _qa_questions())
    def test_no_required_excluded_conflict(self, number: int, query: str) -> None:
        required, excluded = _element_tokens(query)
        conflict = set(required) & set(excluded)
        assert not conflict, (
            f"Q{number}: elements required and excluded at the same time: "
            f"{sorted(conflict)} (query: {query})"
        )

    @pytest.mark.parametrize("number,query", _qa_questions())
    def test_has_strong_constraint(self, number: int, query: str) -> None:
        """Screening questions need a strong constraint; concept and
        multi-turn tool questions are exempt because they do not screen."""
        lowered = query.lower()
        if any(marker in lowered for marker in _STRONG_CONSTRAINT_MARKERS):
            return
        # Concept/tool questions are fine without constraints (e.g. Q64
        # "这个结果是实验值吗？"), but they must not look like a screening
        # request at all — no numeric ranges without a unit context.
        # Material ids like "mp-2" are not numeric constraints.
        without_ids = re.sub(r"mp-\d+", "", query)
        assert not _NUMBER.search(without_ids), (
            f"Q{number}: query has numbers but no strong constraint or unit "
            f"context (query: {query})"
        )

    @pytest.mark.parametrize("number,query", _qa_questions())
    def test_numeric_constraints_have_units(self, number: int, query: str) -> None:
        """When a numeric constraint appears (带隙/hull/密度), an explicit
        unit must be present so the planner does not fall back to
        needs_clarification."""
        lowered = query.lower()
        for keyword in ("带隙", "band gap", "hull", "凸包", "密度", "density"):
            if keyword not in lowered:
                continue
            # Find a number close to the keyword (within 12 chars).
            for match in _NUMBER.finditer(query):
                if keyword not in query[max(0, match.start() - 12) : match.start()]:
                    continue
                window = query[match.start() : match.start() + 12]
                assert _UNIT_HINT.search(window), (
                    f"Q{number}: numeric constraint {keyword!r} without an "
                    f"explicit unit (query: {query})"
                )
