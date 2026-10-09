"""Regression controls for target systems, phases, processes and cache receipts."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from materials_screening.agent.intern_model import InternAgentModel
from materials_screening.agent.model_base import MaterialAgentRequest
from materials_screening.agent.models import AgentFinalDraft, AgentMessageItem
from materials_screening.agent.tool_base import AgentToolDefinition, ToolSideEffect
from materials_screening.sub_agents.literature.models import (
    CandidateLiteratureScreenInput,
    LiteratureSearchInput,
    PaperProvenance,
    PaperRecord,
)
from materials_screening.sub_agents.literature.query_expansion import (
    DeterministicQueryExpander,
)
from materials_screening.sub_agents.literature.service import (
    LiteratureSearchService,
    _rank_key,
    _relevance,
)
from materials_screening.sub_agents.literature.store import LiteratureQueryStore


def paper(key, title, abstract=""):
    return PaperRecord(
        paper_id=key,
        title=title,
        abstract=abstract or None,
        doi=f"10.9999/{key}",
        year=2022,
        provenance=(
            PaperProvenance(
                provider="openalex",
                provider_id=key,
                retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
                raw_record_sha256=hashlib.sha256(key.encode()).hexdigest(),
            ),
        ),
    )


@pytest.mark.parametrize("oxide", ["GdVO4", "BiVO4", "ZnWO4"])
def test_compound_requires_both_components_before_core_promotion(oxide):
    request = LiteratureSearchInput(
        topic=f"{oxide}/g-C3N4 visible light tetracycline photocatalytic degradation",
        material_keywords=(oxide, "g-C3N4", f"{oxide}/g-C3N4"),
        sort_mode="relevance",
    )
    expanded = DeterministicQueryExpander().expand(request)
    unrelated = paper(
        "wrong",
        "Enhanced visible light photocatalytic degradation "
        "of tetracycline by MoS2/Ag/g-C3N4 Z-scheme composites",
    )
    target = paper(
        "target",
        "Photocatalytic composite for antibiotic removal",
        f"{oxide}/g-C3N4 was synthesized. Tetracycline degradation "
        "was tested under visible light.",
    )
    wrong = _relevance(unrelated, request, expanded)
    correct = _relevance(target, request, expanded)
    assert wrong.level == "extended"
    assert any(oxide in item for item in wrong.missing_concepts)
    assert correct.level == "core"
    assert _rank_key(target, request, expanded) < _rank_key(
        unrelated, request, expanded
    )


def test_alternative_materials_are_not_conjunctive_components():
    request = LiteratureSearchInput(topic="BiVO4 or GdVO4 visible light photocatalysis")
    expanded = DeterministicQueryExpander().expand(request)
    result = _relevance(
        paper("alternative", "BiVO4 visible light photocatalysis"), request, expanded
    )
    assert result.level == "core"
    assert not any("GdVO4" in item for item in result.missing_concepts)


@pytest.mark.parametrize(
    "description",
    [
        "BiVO4 and g-C3N4 composite visible light photocatalysis",
        "BiVO4与g-C3N4复合可见光光催化",
    ],
)
def test_composite_words_without_slash_still_require_both_components(description):
    request = LiteratureSearchInput(topic=description)
    expanded = DeterministicQueryExpander().expand(request)
    result = _relevance(
        paper("component", "g-C3N4 visible light photocatalysis"), request, expanded
    )
    assert result.level == "extended"
    assert any("BiVO4" in item for item in result.missing_concepts)


def test_formula_substring_is_not_a_target_component():
    request = LiteratureSearchInput(topic="TiO2/g-C3N4 visible light photocatalysis")
    expanded = DeterministicQueryExpander().expand(request)
    result = _relevance(
        paper("substring", "LiTiO2/g-C3N4 visible light photocatalysis"),
        request,
        expanded,
    )
    assert result.level != "core"
    assert any("TiO2" in item for item in result.missing_concepts)


def test_query_preserves_graphitic_component_label():
    request = LiteratureSearchInput(
        topic="BiVO4/g-C3N4 visible light photocatalysis",
        research_question="检索BiVO4/g-C3N4可见光光催化的实验论文。",
    )
    expanded = DeterministicQueryExpander().expand(request)
    assert any("BiVO4/g-C3N4" in q for q in expanded.search_queries)


def test_amorphous_component_is_not_graphitic_evidence():
    request = LiteratureSearchInput(topic="BiVO4/g-C3N4 visible light photocatalysis")
    expanded = DeterministicQueryExpander().expand(request)
    result = _relevance(
        paper("amorphous", "BiVO4/amorphous C3N4 visible light photocatalysis"),
        request,
        expanded,
    )
    assert result.level == "extended"
    assert any("g-C3N4" in item for item in result.missing_concepts)


def test_phase_in_wrong_title_is_not_rescued_by_comparison_in_abstract():
    request = LiteratureSearchInput(topic="ε-Fe2O3 spray drying annealing")
    expanded = DeterministicQueryExpander().expand(request)
    result = _relevance(
        paper(
            "comparison",
            "α-Fe2O3 synthesis by spray drying",
            "Annealing was performed. ε-Fe2O3 was discussed only as comparison.",
        ),
        request,
        expanded,
    )
    assert result.level == "extended"
    assert any("物相" in item for item in result.missing_concepts)


def test_original_literature_clause_excludes_negative_sample_caveat():
    question = "检索ZnO纳米结构的实验器件论文。不能把石墨烯/ZnO异质结指标归给纯ZnO。"
    value = LiteratureSearchInput.model_validate(
        {"topic": "ZnO device", "research_question": question}
    )
    expanded = DeterministicQueryExpander().expand(value)
    assert expanded.model_dump().get("research_question") == question
    assert not any("/" in q for q in expanded.search_queries)


@pytest.mark.parametrize("phase", ["ε", "epsilon"])
def test_explicit_phase_and_process_cannot_be_replaced_by_other_phase(phase):
    request = LiteratureSearchInput(topic=f"{phase}-Fe2O3 spray drying annealing")
    expanded = DeterministicQueryExpander().expand(request)
    target = paper(
        "right",
        "Spray drying synthesis of epsilon iron oxide",
        "ε-Fe2O3 was synthesized by spray drying followed by annealing.",
    )
    wrong = paper(
        "alpha",
        "α-Fe2O3/Fe3O4 photoanode via fast flame annealing",
        "α-Fe2O3 was tested for water oxidation.",
    )
    assert _relevance(target, request, expanded).level == "core"
    result = _relevance(wrong, request, expanded)
    assert result.level == "extended"
    assert any("物相" in item for item in result.missing_concepts)
    assert any("spray drying" in item for item in result.missing_concepts)


def test_correct_phase_wrong_process_is_background():
    request = LiteratureSearchInput(topic="ε-Fe2O3 spray drying annealing")
    expanded = DeterministicQueryExpander().expand(request)
    result = _relevance(
        paper("solgel", "ε-Fe2O3 nanoparticles by sol-gel annealing"), request, expanded
    )
    assert result.level == "extended"
    assert any("spray drying" in item for item in result.missing_concepts)


def test_compound_wrong_pollutant_is_background():
    request = LiteratureSearchInput(
        topic="BiVO4/g-C3N4 visible light tetracycline degradation"
    )
    expanded = DeterministicQueryExpander().expand(request)
    result = _relevance(
        paper("dye", "BiVO4/g-C3N4 visible light degradation of dyes"),
        request,
        expanded,
    )
    assert result.level == "extended"
    assert any("tetracycline" in item for item in result.missing_concepts)


def test_schema_normalizer_preserves_original_research_question():
    question = (
        "从 Materials Project 筛选Fe2O3、正交晶系、非金属的候选；"
        "检索喷雾干燥后退火制备ε-Fe2O3的实验文献，上传全文后分析物相比例。"
    )
    message = f"原始科研问题：{question}\n候选化学式：Fe2O3。查询快照：query-abc。"
    tool = AgentToolDefinition(
        name="literature_search",
        description="search",
        parameters=LiteratureSearchInput.model_json_schema(),
        side_effect=ToolSideEffect.READ_ONLY,
        version="1",
    )
    request = MaterialAgentRequest(
        instructions="literature",
        input_items=(AgentMessageItem(role="user", content=message),),
        tool_definitions=(tool,),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
    )
    normalized = json.loads(
        InternAgentModel._normalize_tool_arguments(
            request, "literature_search", '{"topic":"Fe2O3 nanospay drying annealing"}'
        )
    )
    assert normalized.get("research_question") == question
    expanded = DeterministicQueryExpander().expand(
        LiteratureSearchInput.model_validate(normalized)
    )
    assert any(
        "spray drying" in q and "annealing" in q for q in expanded.search_queries
    )
    assert not any("nanospay" in q for q in expanded.search_queries)
    assert any("ε" in q or "epsilon" in q for q in expanded.search_queries)
    assert not any("Orthorhombic" in q for q in expanded.search_queries)


class Provider:
    name = "openalex"

    def __init__(self):
        self.calls = 0

    def search(self, request):
        self.calls += 1
        return (
            paper(
                "uv",
                "ZnO ultraviolet photodetector",
                "A ZnO device was fabricated and responsivity was measured.",
            ),
        )


def test_candidate_screen_cache_receipt_has_no_new_network_attempt(tmp_path: Path):
    provider = Provider()
    service = LiteratureSearchService((provider,), LiteratureQueryStore(tmp_path))
    request = CandidateLiteratureScreenInput(materials=("ZnO",))
    fresh = service.screen_candidates(request)
    saved = tmp_path / fresh.screening_id / "result.json"
    before = saved.read_bytes()
    cached = service.screen_candidates(request)
    assert provider.calls == 1
    assert cached.model_dump().get("retrieval_mode") == "cache"
    assert cached.queries_attempted == 0
    assert cached.model_dump().get("cache_hits") == 1
    assert cached.screening_id == fresh.screening_id
    assert saved.read_bytes() == before
    answer = InternAgentModel._candidate_screen_answer(cached.model_dump(mode="json"))
    assert "缓存" in answer and "本次联网 0" in answer


def test_new_screen_using_provider_cache_marks_source_age(tmp_path: Path):
    provider = Provider()
    service = LiteratureSearchService((provider,), LiteratureQueryStore(tmp_path))
    service.screen_candidates(
        CandidateLiteratureScreenInput(materials=("ZnO",), final_limit=1)
    )
    second = service.screen_candidates(
        CandidateLiteratureScreenInput(materials=("ZnO",), final_limit=2)
    )
    assert provider.calls == 1
    assert second.model_dump().get("retrieval_mode") == "cache"
    assert second.queries_attempted == 0
    assert second.candidates[0].model_dump().get("source_created_at") is not None
    assert any("CACHED_PROVIDER_METADATA" in w for w in second.warnings)


def test_unified_cached_result_is_labeled_without_snapshot_overwrite(tmp_path: Path):
    provider = Provider()
    service = LiteratureSearchService((provider,), LiteratureQueryStore(tmp_path))
    request = LiteratureSearchInput(topic="ZnO ultraviolet photodetector")
    fresh = service.search_unified(request)
    path = tmp_path / fresh.query_id / "result.json"
    before, calls = path.read_bytes(), provider.calls
    cached = service.search_unified(request)
    assert cached.model_dump().get("retrieval_mode") == "cache"
    assert provider.calls == calls and path.read_bytes() == before
    assert any("CACHED_QUERY_METADATA" in w for w in cached.warnings)
