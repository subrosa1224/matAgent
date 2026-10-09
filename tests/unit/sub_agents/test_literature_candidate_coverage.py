from __future__ import annotations

import hashlib
import json

import pytest
from tests.unit.sub_agents.test_literature_unified_search import _paper

from materials_screening.sub_agents.literature.models import LiteratureSearchInput
from materials_screening.sub_agents.literature.providers import LiteratureProviderError
from materials_screening.sub_agents.literature.query_expansion import (
    DeterministicQueryExpander,
    plan_provider_queries,
)
from materials_screening.sub_agents.literature.service import LiteratureSearchService
from materials_screening.sub_agents.literature.store import LiteratureQueryStore

FORMULAS = (
    "Ca3Cr2(GeO4)3",
    "Ba2TbNbO6",
    "Ba6Lu2(WO6)3",
    "Ba2NdNbO6",
    "KLa5V2O13",
    "Ba2ZrO4",
    "DyVO4",
    "HoVO4",
    "AlWO4",
    "BaZn2(AsO4)2",
    "Ba2PrNbO6",
    "Ca3Ga2(GeO4)3",
    "Cs2Zr(WO4)3",
    "Ho2Ti2O7",
    "GdVO4",
    "Cs7Mo8AsO30",
    "K3Ta3Si2O13",
    "BaCrO4",
    "ErVO4",
    "BaGd2Sc2O7",
)


def request(formulas=FORMULAS) -> LiteratureSearchInput:
    return LiteratureSearchInput(
        topic=(
            "原始科研问题：选择可见光光催化候选。候选化学式："
            + "；".join(formulas)
            + "。查询快照：query-test。"
        ),
        max_papers=20,
    )


class CaptureProvider:
    def __init__(self, name="openalex", fail_at=None, at_limit=False):
        self.name = name
        self.fail_at = fail_at
        self.at_limit = at_limit
        self.requests = []

    def search(self, value):
        self.requests.append(value)
        if self.fail_at == len(self.requests):
            raise LiteratureProviderError("PROVIDER_RATE_LIMIT", "limited")
        formula = next((item for item in FORMULAS if item in value.topic), "DyVO4")
        if self.at_limit:
            return tuple(
                _paper(
                    provider=self.name,
                    provider_id=f"A{index}",
                    title=f"{formula} visible light photocatalysis {index}",
                    doi=None,
                )
                for index in range(value.max_papers)
            )
        if formula not in value.topic:
            return ()
        return (
            _paper(
                provider=self.name,
                provider_id="A",
                title=f"{formula} visible light photocatalysis",
                doi=None,
            ),
        )


def search(tmp_path, *providers, value=None):
    return LiteratureSearchService(
        providers, LiteratureQueryStore(tmp_path)
    ).search_unified(value or request())


def test_openalex_groups_all_twenty_without_more_requests(tmp_path):
    provider = CaptureProvider()
    result = search(tmp_path, provider)
    assert len(provider.requests) == 4
    queries = [value.topic for value in provider.requests]
    for formula in FORMULAS:
        assert sum(f'"{formula}"' in query for query in queries) == 1
    assert all(" AND " in query and " OR " in query for query in queries)
    assert all(
        '"photocatalysis"' in query and '"visible light"' in query for query in queries
    )
    assert all(len(query) <= 300 for query in queries)
    status = result.provider_statuses[0]
    assert status.submitted_materials == FORMULAS
    assert not status.unqueried_materials
    assert not status.failed_materials
    assert status.search_queries == tuple(queries)
    assert result.returned_count > 0


def test_semantic_scholar_keeps_plain_queries_and_discloses_missing_candidates(
    tmp_path,
):
    openalex, semantic = CaptureProvider(), CaptureProvider("semantic_scholar")
    result = search(tmp_path, openalex, semantic)
    assert len(semantic.requests) == 6
    assert all(" OR " not in value.topic for value in semantic.requests)
    assert result.provider_statuses[1].submitted_materials == FORMULAS[:6]
    assert result.provider_statuses[1].unqueried_materials == FORMULAS[6:]
    assert any("CANDIDATE_COVERAGE_PARTIAL" in warning for warning in result.warnings)


def test_failed_batch_is_not_reported_as_successfully_queried(tmp_path):
    good = CaptureProvider("semantic_scholar")
    failed = CaptureProvider(fail_at=2)
    result = search(tmp_path, failed, good)
    status = result.provider_statuses[0]
    assert status.status == "degraded"
    assert status.queries_attempted == 2
    assert status.submitted_materials == FORMULAS[:10]
    assert status.failed_materials == FORMULAS[5:10]
    assert status.unqueried_materials == FORMULAS[10:]


def test_full_page_is_disclosed_without_claiming_exhaustive_retrieval(tmp_path):
    result = search(tmp_path, CaptureProvider(at_limit=True))
    assert result.provider_statuses[0].queries_at_record_limit == 4
    assert any(
        "PROVIDER_RECORD_LIMIT_REACHED" in warning for warning in result.warnings
    )


def test_ordinary_topic_uses_original_expansion(tmp_path):
    provider = CaptureProvider()
    result = search(
        tmp_path,
        provider,
        value=LiteratureSearchInput(topic="TiO2 photocatalysis hydrogen"),
    )
    assert all(" OR " not in value.topic for value in provider.requests)
    assert not result.provider_statuses[0].submitted_materials
    assert not any("CANDIDATE_COVERAGE" in warning for warning in result.warnings)


def test_parenthesized_formula_is_one_quoted_operand(tmp_path):
    provider = CaptureProvider()
    search(tmp_path, provider)
    assert '"Ca3Cr2(GeO4)3"' in provider.requests[0].topic
    assert '"Ba6Lu2(WO6)3"' in provider.requests[0].topic


@pytest.mark.parametrize("bad", ['TiO2" OR "ZnO', "", "scaffold"])
def test_invalid_explicit_list_does_not_create_boolean_query(tmp_path, bad):
    provider = CaptureProvider()
    search(tmp_path, provider, value=request((*FORMULAS[:6], bad)))
    assert all(" OR " not in value.topic for value in provider.requests)


def test_new_plan_does_not_reuse_old_zero_snapshot(tmp_path):
    provider = CaptureProvider()
    store = LiteratureQueryStore(tmp_path)
    service = LiteratureSearchService((provider,), store)
    value = request()
    expanded = service.expander.expand(value)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "request": value.model_dump(mode="json"),
                "expanded": expanded.model_dump(mode="json"),
                "providers": ["openalex"],
                "ranking": "unified-v12-uv-evidence-grade",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:24]
    from materials_screening.sub_agents.literature.models import LiteratureSearchOutput

    store.save(
        LiteratureSearchOutput(
            query_id=f"lit-{fingerprint}",
            provider="unified",
            papers=(),
            returned_count=0,
            expanded_query=expanded,
        )
    )
    result = service.search_unified(value)
    assert result.query_id != f"lit-{fingerprint}"
    assert len(provider.requests) == 4
    calls = len(provider.requests)
    cached = service.search_unified(value)
    assert cached.query_id == result.query_id and cached.papers == result.papers
    assert cached.created_at == result.created_at and cached.retrieval_mode == "cache"
    assert service.store.load(result.query_id) == result
    assert len(provider.requests) == calls


def test_irrelevant_papers_still_rejected(tmp_path):
    class WrongMaterial(CaptureProvider):
        def search(self, value):
            return (
                _paper(
                    provider="openalex",
                    provider_id="A",
                    doi=None,
                    title="SnO2 visible light photocatalysis",
                ),
            )

    assert search(tmp_path, WrongMaterial()).returned_count == 0


def test_long_terms_are_not_silently_cut_to_make_boolean_queries_fit(tmp_path):
    expanded = DeterministicQueryExpander().expand(request(FORMULAS[:7]))
    expanded = expanded.model_copy(
        update={"performance_terms": ("x" * 274,), "process_terms": ()}
    )
    plan = plan_provider_queries(expanded, "openalex")
    assert len(plan.queries) == 6
    assert all(len(query) <= 300 for query, _ in plan.queries)
    assert all('"' + "x" * 274 + '"' in query for query, _ in plan.queries)
    assert (
        tuple(value for _, values in plan.queries for value in values) == FORMULAS[:6]
    )


def test_duplicate_formulas_are_submitted_only_once(tmp_path):
    provider = CaptureProvider()
    result = search(tmp_path, provider, value=request((*FORMULAS, FORMULAS[-1])))
    assert len(provider.requests) == 4
    assert result.provider_statuses[0].submitted_materials == FORMULAS


def test_small_explicit_candidate_pool_keeps_plain_queries(tmp_path):
    provider = CaptureProvider()
    result = search(tmp_path, provider, value=request(FORMULAS[:3]))
    assert len(provider.requests) == 3
    assert all(" OR " not in value.topic for value in provider.requests)
    assert result.provider_statuses[0].submitted_materials == FORMULAS[:3]


def test_healthy_empty_search_does_not_use_stale_fallback_for_coverage_warning(
    tmp_path,
):
    provider = CaptureProvider("semantic_scholar")
    provider.search = lambda value: ()
    store = LiteratureQueryStore(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("a coverage warning must not trigger provider-failure fallback")

    store.latest_compatible_success = forbidden
    result = LiteratureSearchService((provider,), store).search_unified(request())
    assert result.returned_count == 0
    assert result.provider_statuses[0].unqueried_materials == FORMULAS[6:]


def test_report_uses_actual_provider_queries_and_discloses_coverage(tmp_path):
    from materials_screening.agent.intern_model import InternAgentModel
    from materials_screening.agent.model_base import MaterialAgentRequest
    from materials_screening.agent.models import (
        AgentFinalDraft,
        AgentFunctionOutputItem,
        AgentMessageItem,
    )

    result = search(tmp_path, CaptureProvider(), CaptureProvider("semantic_scholar"))
    agent_request = MaterialAgentRequest(
        instructions="literature",
        input_items=(
            AgentMessageItem(role="user", content=request().topic),
            AgentFunctionOutputItem(
                call_id="call-1",
                output=json.dumps(
                    {
                        "status": "ok",
                        "tool_name": "literature_search",
                        "evidence_id": "ev-1",
                        "output": result.model_dump(mode="json"),
                    }
                ),
            ),
        ),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        allow_tool_calls=False,
    )
    grounded = InternAgentModel._ground_final_draft(
        agent_request, AgentFinalDraft(status="completed", answer="summary")
    )
    assert "已提交 20 种" in grounded.answer
    assert "未查询 14 种" in grounded.answer
    assert "提交查询不表示已找到论文或完整召回" in grounded.answer
    assert all(
        query in grounded.answer
        for status in result.provider_statuses
        for query in status.search_queries
    )
    assert grounded.evidence_ids == ["ev-1"]


def test_battery_handoff_spends_plain_queries_on_candidates_not_generic_battery(
    tmp_path,
):
    formulas = (
        "Li2FeO3",
        "Li5FeO4",
        "LiFeO2",
        "Li2FeO2",
        "Li7Fe5O12",
        "Li3FeO4",
        "Li8Fe2O9",
    )
    value = LiteratureSearchInput(
        topic=(
            "原始科研问题：筛选锂离子电池正极，核验合成与容量。候选化学式："
            + "；".join(formulas)
            + "。查询快照：query-test。"
        )
    )
    expanded = DeterministicQueryExpander().expand(value)
    plan = plan_provider_queries(expanded, "semantic_scholar")
    assert len(plan.queries) == 6
    assert tuple(f for _, fs in plan.queries for f in fs) == formulas[:6]
    assert all(
        "lithium-ion battery" in q and "cathode" in q and fs for q, fs in plan.queries
    )
    assert all(not q.startswith("battery ") for q, _ in plan.queries)
    result = search(tmp_path, CaptureProvider("semantic_scholar"), value=value)
    assert result.provider_statuses[0].submitted_materials == formulas[:6]
    assert result.provider_statuses[0].unqueried_materials == formulas[6:]


def test_battery_formula_topic_does_not_add_generic_battery_query():
    expanded = DeterministicQueryExpander().expand(
        LiteratureSearchInput(topic="LiFeO2 锂离子电池正极合成容量")
    )
    assert all(q.startswith("LiFeO2 ") for q in expanded.search_queries)
