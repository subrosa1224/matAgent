from __future__ import annotations

import pytest

from materials_screening.sub_agents.literature.models import (
    LiteratureSearchInput,
    PaperRecord,
)
from materials_screening.sub_agents.literature.query_expansion import (
    DeterministicQueryExpander,
    plan_provider_queries,
)
from materials_screening.sub_agents.literature.service import _relevance


def _request(topic: str = "Li2FeO3 锂离子电池正极合成、容量与循环性能"):
    return LiteratureSearchInput(topic=topic, material_keywords=("Li2FeO3",))


@pytest.mark.parametrize(
    "suffix",
    ["，不复用前面的光催化报告。", "；不要沿用之前的光催化检索结果。"],
)
def test_negative_history_is_not_a_scientific_topic(suffix):
    request = _request(_request().topic + suffix)
    expanded = DeterministicQueryExpander().expand(request)
    assert expanded.original_topic == request.topic
    assert "photocatalysis" not in expanded.process_terms
    assert all("photocatalysis" not in q for q in expanded.search_queries)


def test_positive_photocatalysis_is_preserved():
    expanded = DeterministicQueryExpander().expand(
        LiteratureSearchInput(topic="TiO2 可见光光催化合成和降解，不复用之前的电池报告")
    )
    assert "photocatalysis" in expanded.process_terms
    assert all("photocatalysis" in q for q in expanded.search_queries)
    assert "battery" not in expanded.normalized_materials


@pytest.mark.parametrize(
    ("topic", "ion", "role"),
    [
        ("Li2FeO3 锂离子电池正极合成容量", "lithium-ion battery", "cathode"),
        ("NaFeO2 钠离子电池负极容量", "sodium-ion battery", "anode"),
        (
            "Li2FeO3 lithium ion battery positive electrode capacity",
            "lithium-ion battery",
            "cathode",
        ),
    ],
)
def test_battery_scope_survives_bounded_queries(topic, ion, role):
    expanded = DeterministicQueryExpander().expand(LiteratureSearchInput(topic=topic))
    assert ion in expanded.process_terms
    assert role in expanded.process_terms
    for provider in ("semantic_scholar", "openalex"):
        assert all(
            ion in q and role in q
            for q, _ in plan_provider_queries(expanded, provider).queries
        )


@pytest.mark.parametrize(
    ("title", "abstract", "expected"),
    [
        ("Li2FeO3 lithium-ion battery cathode synthesis capacity", None, "core"),
        ("Li2FeO3 lithium-ion battery anode synthesis capacity", None, None),
        ("Li2FeO3 sodium-ion battery cathode synthesis capacity", None, None),
        ("Li2FeO3 lithium-sulfur battery cathode synthesis capacity", None, None),
        ("Li2FeO3 synthesis and battery capacity", None, "extended"),
        (
            "Li2FeO3 lithium-ion battery anode and cathode synthesis capacity",
            None,
            "extended",
        ),
        (
            "Li2FeO3 sodium-ion battery anode synthesis capacity",
            "Compared with lithium-ion battery cathode synthesis capacity.",
            None,
        ),
        (
            "LiNi0.5Mn1.5O4 lithium-ion battery cathode synthesis capacity",
            None,
            "extended",
        ),
        (
            "Li2FeO3 synthesis capacity",
            "A positive electrode for lithium ion batteries.",
            "core",
        ),
    ],
)
def test_battery_relevance_does_not_promote_wrong_or_uncertain_evidence(
    title, abstract, expected
):
    request = _request()
    expanded = DeterministicQueryExpander().expand(request)
    paper = PaperRecord(
        paper_id="fixture", title=title, abstract=abstract, provenance=()
    )
    result = _relevance(paper, request, expanded)
    assert result.level == expected
    if expected == "extended":
        assert any(
            "电池" in c or "电极" in c or "候选" in c for c in result.missing_concepts
        )


def test_generic_battery_topic_does_not_impose_li_ion_or_cathode():
    request = LiteratureSearchInput(topic="battery synthesis capacity")
    expanded = DeterministicQueryExpander().expand(request)
    paper = PaperRecord(
        paper_id="fixture",
        title="Sodium-ion battery anode synthesis capacity",
        provenance=(),
    )
    assert _relevance(paper, request, expanded).level == "core"


def test_handoff_preserves_candidates_and_all_provider_application_terms():
    formulas = "Li2FeO3；LiFeO2；Li5FeO4；Li2FeO2；Li3FeO3；Li4FeO4；Li14Fe4O13"
    topic = (
        "检索相关文献。原始科研问题：Li-Fe-O 锂离子电池正极合成和容量，"
        "不得把计算稳定性当作已实验合成，不复用前面的光催化报告。"
        f"候选化学式：{formulas}。查询快照：query-fixture；使用通用检索。"
    )
    expanded = DeterministicQueryExpander().expand(LiteratureSearchInput(topic=topic))
    assert expanded.original_topic == topic
    assert "photocatalysis" not in expanded.process_terms
    assert "synthesis" in expanded.process_terms
    for provider in ("openalex", "semantic_scholar"):
        plan = plan_provider_queries(expanded, provider)
        assert plan.candidates == tuple(formulas.split("；"))
        assert all(
            "lithium-ion battery" in q and "cathode" in q and len(q) <= 300
            for q, _ in plan.queries
        )


def test_evidence_negation_does_not_erase_positive_scientific_topic():
    expanded = DeterministicQueryExpander().expand(
        _request("Li2FeO3 锂离子电池正极合成容量；不能把缺少容量数据写成性能最佳")
    )
    assert "capacity" in expanded.performance_terms
    assert "synthesis" in expanded.process_terms
