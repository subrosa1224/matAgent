"""Keep a verified candidate queue and its application through literature tools."""

import json

import pytest
from tests.unit.sub_agents.test_literature_research_scope import paper

from materials_screening.agent.intern_model import InternAgentModel
from materials_screening.agent.model_base import MaterialAgentRequest
from materials_screening.agent.models import (
    AgentFinalDraft,
    AgentFunctionOutputItem,
    AgentMessageItem,
)
from materials_screening.agent.tool_base import AgentToolDefinition, ToolSideEffect
from materials_screening.sub_agents.literature.models import LiteratureSearchInput
from materials_screening.sub_agents.literature.query_expansion import (
    DeterministicQueryExpander,
    plan_provider_queries,
)
from materials_screening.sub_agents.literature.service import (
    LiteratureSearchService,
    _relevance,
)
from materials_screening.sub_agents.literature.store import LiteratureQueryStore

QUESTION = (
    "重点考察 In₂O₃、SnO₂ 和 ZnO，先查询 Materials Project；再检索这些材料及其掺杂体系"
    "作为透明导电薄膜的实验研究。在我上传全文并确认分析后提取可见光透过率和电阻率。"
)
HANDOFF = (
    f"原始科研问题：{QUESTION}\n候选化学式：In2O3；SnO2；ZnO。查询快照：query-test。"
)


def tool_request(message=HANDOFF):
    return MaterialAgentRequest(
        instructions="literature",
        input_items=(AgentMessageItem(role="user", content=message),),
        tool_definitions=(
            AgentToolDefinition(
                name="literature_search",
                description="search",
                parameters=LiteratureSearchInput.model_json_schema(),
                side_effect=ToolSideEffect.READ_ONLY,
                version="1",
            ),
        ),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
    )


def normalized():
    args = InternAgentModel._normalize_tool_arguments(
        tool_request(),
        "literature_search",
        '{"topic":"In2O3 transparent electrode",'
        '"material_keywords":["In2O3","indium oxide"]}',
    )
    return LiteratureSearchInput.model_validate_json(args)


def test_tool_cannot_drop_two_handoff_candidates():
    value = normalized()
    assert value.research_question == QUESTION
    assert value.material_keywords == ("In2O3", "SnO2", "ZnO")
    assert "候选化学式：In2O3；SnO2；ZnO" in value.topic


@pytest.mark.parametrize("provider", ["openalex", "semantic_scholar"])
def test_three_target_queries_keep_transparent_conducting_application(provider):
    expanded = DeterministicQueryExpander().expand(normalized())
    plan = plan_provider_queries(expanded, provider)
    assert plan.candidates == ("In2O3", "SnO2", "ZnO")
    assert len(plan.queries) == 3
    assert tuple(f for _, fs in plan.queries for f in fs) == plan.candidates
    assert all("transparent conducting" in query for query, _ in plan.queries)


@pytest.mark.parametrize(
    "title,abstract",
    [
        ("In2O3 doping for gas sensing", "Nd-doped In2O3 porous particles detect NH3."),
        (
            "In2O3 co-doping for solid-state lithium metal batteries",
            "Doping improves ionic conductivity.",
        ),
        (
            "First-principles design of transparent conducting In2O3",
            "DFT calculations predict doped In2O3 band structure.",
        ),
    ],
)
def test_other_applications_or_theory_are_not_core_experimental_candidates(
    title, abstract
):
    value = normalized()
    relevance = _relevance(
        paper("background", title, abstract),
        value,
        DeterministicQueryExpander().expand(value),
    )
    assert relevance.level not in {"core", "high"}
    assert relevance.missing_concepts


def test_measured_transparent_conducting_film_is_a_related_fulltext_lead():
    value = normalized()
    relevance = _relevance(
        paper(
            "film",
            "Al-doped ZnO transparent conducting thin films",
            "Films were deposited by sputtering. Optical transmittance "
            "and electrical resistivity were measured.",
        ),
        value,
        DeterministicQueryExpander().expand(value),
    )
    assert relevance.level in {"core", "high"}


def test_report_distinguishes_submission_retained_leads_and_fulltext(tmp_path):
    class Provider:
        name = "openalex"

        def search(self, value):
            if value.topic.startswith("In2O3 "):
                return (
                    paper(
                        "film",
                        "Sn-doped In2O3 transparent conducting thin films",
                        "Sputtered films had measured transmittance and resistivity.",
                    ),
                )
            if value.topic.startswith("SnO2 "):
                return (
                    paper(
                        "sensor",
                        "SnO2 doping for gas sensing",
                        "Particles detect methane.",
                    ),
                )
            return ()

    result = LiteratureSearchService(
        (Provider(),), LiteratureQueryStore(tmp_path)
    ).search_unified(normalized())
    request = MaterialAgentRequest(
        instructions="literature",
        input_items=(
            AgentMessageItem(role="user", content=HANDOFF),
            AgentFunctionOutputItem(
                call_id="search",
                output=json.dumps(
                    {
                        "status": "ok",
                        "tool_name": "literature_search",
                        "evidence_id": "ev-search",
                        "output": result.model_dump(mode="json"),
                    }
                ),
            ),
        ),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
    )
    final = InternAgentModel._ground_final_draft(
        request, AgentFinalDraft(status="completed", answer="summary")
    )
    assert "各体系检索进度" in final.answer
    assert "| In2O3 | 已检索" in final.answer
    assert "| SnO2 | 已检索" in final.answer
    assert "| ZnO | 已检索" in final.answer
    assert "待上传全文核验" in final.answer
    assert "当前保留结果暂无目标应用线索" in final.answer
    assert "不代表没有相关论文" in final.answer


def test_standalone_topic_and_malformed_handoff_are_not_overwritten():
    raw = '{"topic":"ZnO transparent electrode","material_keywords":["ZnO"]}'
    for message in (
        "检索ZnO透明电极文献",
        HANDOFF.replace("SnO2", 'SnO2" OR scaffold'),
    ):
        args = json.loads(
            InternAgentModel._normalize_tool_arguments(
                tool_request(message), "literature_search", raw
            )
        )
        assert args["topic"] == "ZnO transparent electrode"
        assert args["material_keywords"] == ["ZnO"]


def test_failed_and_unqueried_targets_are_not_shown_as_healthy_empty_results(tmp_path):
    from materials_screening.sub_agents.literature.providers import (
        LiteratureProviderError,
    )
    from materials_screening.sub_agents.literature.search_coverage import (
        render_search_coverage,
    )

    class Provider:
        name = "openalex"

        def search(self, value):
            if value.topic.startswith("In2O3 "):
                return (paper("film", "In2O3 transparent conducting thin films"),)
            raise LiteratureProviderError("PROVIDER_RATE_LIMIT", "limited")

    service = LiteratureSearchService((Provider(),), LiteratureQueryStore(tmp_path))
    result = service.search_unified(normalized())
    text = render_search_coverage(result.model_dump(mode="json"))
    assert "| In2O3 | 已检索" in text
    assert "| SnO2 | 查询失败" in text
    assert "| ZnO | 未检索" in text
    assert "| SnO2 | 已检索" not in text
    assert "| ZnO | 已检索" not in text


def test_cached_coverage_labels_history_without_claiming_new_search(tmp_path):
    from materials_screening.sub_agents.literature.search_coverage import (
        render_search_coverage,
    )

    class Provider:
        name = "openalex"

        def search(self, value):
            return ()

    service = LiteratureSearchService((Provider(),), LiteratureQueryStore(tmp_path))
    first = service.search_unified(normalized())
    path = tmp_path / first.query_id / "result.json"
    before = path.read_bytes()
    cached = service.search_unified(normalized())
    text = render_search_coverage(cached.model_dump(mode="json"))
    assert "历史已检索（本次复用）" in text
    assert path.read_bytes() == before


def test_second_provider_success_can_recover_first_provider_failure(tmp_path):
    from materials_screening.sub_agents.literature.models import ProviderSearchStatus
    from materials_screening.sub_agents.literature.search_coverage import (
        render_search_coverage,
    )

    class Provider:
        name = "openalex"

        def search(self, value):
            return ()

    first = LiteratureSearchService(
        (Provider(),), LiteratureQueryStore(tmp_path)
    ).search_unified(normalized())
    statuses = (
        ProviderSearchStatus(
            provider="openalex",
            status="degraded",
            queries_attempted=1,
            records_returned=0,
            submitted_materials=("In2O3",),
            failed_materials=("In2O3",),
            unqueried_materials=("SnO2", "ZnO"),
        ),
        first.provider_statuses[0].model_copy(update={"provider": "semantic_scholar"}),
    )
    result = first.model_copy(update={"provider_statuses": statuses})
    text = render_search_coverage(result.model_dump(mode="json"))
    assert all(f"| {f} | 已检索" in text for f in ("In2O3", "SnO2", "ZnO"))
