"""Regressions from the real non-UV chain, without external services."""

import json

import pytest

from materials_screening.agent.models import AgentFunctionOutputItem
from materials_screening.sub_agents.literature.models import LiteratureSearchInput
from materials_screening.sub_agents.literature.query_expansion import (
    DeterministicQueryExpander,
)
from materials_screening.sub_agents.materials_database.deterministic_model import (
    _final_from_tool,
    parse_standard_search,
)

QUESTION = (
    "我想为可见光驱动的光催化研究选择氧化物候选。请从 Materials Project "
    "筛选计算带隙为1.5～3.0 eV、凸包能不高于0.05 eV/atom且不含Pb、Cd、Hg"
    "的非金属氧化物，分析候选带隙、稳定性与密度分布，并结合实验文献评估"
    "前5名作为可见光光催化材料的研究可行性。"
)


def test_bounded_search_preserves_nonmetal_and_oxide_scope() -> None:
    search = parse_standard_search(QUESTION)
    assert search is not None
    assert {"Pb", "Cd", "Hg", "F", "P", "N", "S", "Cl"}.issubset(
        search.excluded_elements
    )
    assert search.required_elements == ("O",)
    assert ("is_metal", "eq", False) in [
        (item.field, item.operator.value, item.value) for item in search.filters
    ]
    assert "is_metal" in search.fields
    assert [
        (item.field, item.operator.value, item.value)
        for item in search.filters
        if item.field != "is_metal"
    ] == [
        ("band_gap_ev", "gte", 1.5),
        ("band_gap_ev", "lte", 3.0),
        ("energy_above_hull_ev_atom", "lte", 0.05),
    ]


def test_scope_approximation_is_visible_even_for_empty_search() -> None:
    search = parse_standard_search(QUESTION)
    assert search is not None
    response = _final_from_tool(
        AgentFunctionOutputItem(
            call_id="scope-test",
            output=json.dumps(
                {
                    "status": "ok",
                    "evidence_id": "test-evidence",
                    "output": {"query_id": "query-test", "materials": []},
                }
            ),
        ),
        search,
    )
    draft = json.loads(response.message_text)
    assert "is_metal = false" in draft["answer"]
    assert "近似" in draft["answer"]
    assert any("近似" in warning for warning in draft["warnings"])
    assert any(
        "排除" in warning and "氧化物" in warning for warning in draft["warnings"]
    )


@pytest.mark.parametrize("material_class", ["半导体氧化物", "非金属氧化物"])
def test_nonmetal_alias_is_not_ignored(material_class: str) -> None:
    search = parse_standard_search(
        f"筛选带隙为1.5～3.0 eV、凸包能不高于0.05 eV/atom的{material_class}"
    )
    assert search is not None
    assert any(
        item.field == "is_metal" and item.value is False for item in search.filters
    )


def test_an_explicit_mixed_anion_scope_is_not_silently_overridden() -> None:
    assert (
        parse_standard_search(
            "筛选带隙为1.5～3.0 eV、凸包能不高于0.05 eV/atom的氧化物和氧氟化物"
        )
        is None
    )


def test_workflow_screening_does_not_become_a_research_method() -> None:
    expanded = DeterministicQueryExpander().expand(
        LiteratureSearchInput(
            topic="筛选 Fe2O3、WO3 用于可见光光催化的候选并检索实验文献"
        )
    )
    assert "high-throughput screening" not in expanded.process_terms
    assert all("high-throughput" not in query for query in expanded.search_queries)
    assert all("visible light" in query for query in expanded.search_queries)
    assert all("photocatalysis" in query for query in expanded.search_queries)


def test_explicit_high_throughput_research_is_still_preserved() -> None:
    expanded = DeterministicQueryExpander().expand(
        LiteratureSearchInput(topic="Fe2O3 可见光光催化材料的高通量筛选研究")
    )
    assert "high-throughput screening" in expanded.process_terms


@pytest.mark.parametrize("keywords", [(), ("Fe2O3", "MP", "DOI", "PDF")])
def test_handoff_instructions_do_not_pollute_search_concepts(
    keywords: tuple[str, ...],
) -> None:
    task = (
        f"检索与以下原始科研问题及数据库候选材料相关的实验文献。\n原始科研问题：{QUESTION}"
        "\n候选化学式：Fe2O3；WO3。查询快照：query-test；"
        "使用 literature_search 通用检索。"
        "不预设其他应用的证据分级；不得把配方级论文对应为特定 MP 物相。"
        "列出 DOI，需要 PDF 正文；不要推断掺杂、降解或紫外探测器指标。"
    )
    expanded = DeterministicQueryExpander().expand(
        LiteratureSearchInput(topic=task, material_keywords=keywords)
    )
    assert expanded.original_topic == " ".join(task.split())
    assert expanded.normalized_materials == ("Fe2O3", "WO3")
    assert "doping" not in expanded.process_terms
    assert "degradation" not in expanded.performance_terms
    assert "UV photodetector" not in expanded.performance_terms
    assert all(
        "visible light" in query and "photocatalysis" in query
        for query in expanded.search_queries
    )


def test_relaxed_queries_keep_visible_light_and_photocatalysis() -> None:
    expanded = DeterministicQueryExpander().expand(
        LiteratureSearchInput(topic="掺杂 Fe2O3 的可见光光催化产氢实验")
    )
    assert all(
        "visible light" in query and "photocatalysis" in query
        for query in expanded.search_queries
    )
    assert 1 <= len(expanded.search_queries) <= 6
