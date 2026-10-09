from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from materials_screening.sub_agents.literature.models import (
    CandidateLiteratureScreenInput,
    LiteratureSearchInput,
    PaperProvenance,
    PaperRecord,
)
from materials_screening.sub_agents.literature.providers import (
    LiteratureProviderError,
)
from materials_screening.sub_agents.literature.query_expansion import (
    DeterministicQueryExpander,
)
from materials_screening.sub_agents.literature.service import (
    LiteratureSearchService,
    _paper_mentions_formula,
)
from materials_screening.sub_agents.literature.store import LiteratureQueryStore


def _paper(
    *,
    provider: str,
    provider_id: str,
    title: str,
    doi: str | None,
    abstract: str | None = None,
    citations: int | None = None,
    open_access: bool | None = None,
    year: int = 2025,
) -> PaperRecord:
    return PaperRecord.model_validate(
        {
            "paper_id": f"paper-{provider_id}",
            "title": title,
            "doi": doi,
            "year": year,
            "abstract": abstract,
            "cited_by_count": citations,
            "open_access": open_access,
            "provenance": (
                PaperProvenance.model_validate(
                    {
                        "provider": provider,
                        "provider_id": provider_id,
                        "retrieved_at": datetime.now(UTC),
                        "raw_record_sha256": provider_id[0].lower() * 64,
                    }
                ),
            ),
        }
    )


class StubProvider:
    def __init__(
        self,
        name: str,
        papers: tuple[PaperRecord, ...] = (),
        error: LiteratureProviderError | None = None,
    ) -> None:
        self.name = name
        self.papers = papers
        self.error = error
        self.calls = 0

    def search(self, request: LiteratureSearchInput) -> tuple[PaperRecord, ...]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.papers


class CandidateStubProvider:
    name = "openalex"

    def __init__(self, papers_by_formula: dict[str, tuple[PaperRecord, ...]]) -> None:
        self.papers_by_formula = papers_by_formula
        self.calls = 0

    def search(self, request: LiteratureSearchInput) -> tuple[PaperRecord, ...]:
        self.calls += 1
        return next(
            (
                papers
                for formula, papers in self.papers_by_formula.items()
                if formula in request.topic
            ),
            (),
        )


def test_formula_match_uses_full_composition_not_substrings() -> None:
    paper = _paper(
        provider="openalex",
        provider_id="A",
        doi=None,
        title="LiTiO2 and β-Ga₂O₃ with Ca3Ga2(GeO4)3",
    )
    assert not _paper_mentions_formula(paper, "TiO2")
    assert _paper_mentions_formula(paper, "Ga2O3")
    assert _paper_mentions_formula(paper, "Ca3Ga2Ge3O12")
    assert not _paper_mentions_formula(paper, "ZnO")


def test_candidate_screen_failure_is_unknown_and_retried(tmp_path: Path) -> None:
    provider = StubProvider(
        "openalex", error=LiteratureProviderError("PROVIDER_RATE_LIMIT", "limited")
    )
    service = LiteratureSearchService((provider,), LiteratureQueryStore(tmp_path))
    request = CandidateLiteratureScreenInput(materials=("ZnO", "Ga2O3"))
    failed = service.screen_candidates(request)
    assert not failed.retrieval_complete
    assert failed.queries_attempted == 1
    assert [row.retrieval_status for row in failed.candidates] == [
        "error",
        "not_attempted",
    ]
    assert not failed.finalists
    provider.error = None
    recovered = service.screen_candidates(request)
    assert recovered.retrieval_complete
    assert recovered.screening_id != failed.screening_id
    assert provider.calls == 3
    cached = service.screen_candidates(request)
    assert cached.screening_id == recovered.screening_id
    assert cached.candidates == recovered.candidates
    assert cached.retrieval_mode == "cache" and cached.queries_attempted == 0
    assert service.store.load_candidate_screen(recovered.screening_id) == recovered
    assert provider.calls == 3


def test_reviews_and_simulations_are_not_experimental_device_evidence(
    tmp_path: Path,
) -> None:
    papers = tuple(
        _paper(
            provider="openalex",
            provider_id=identifier,
            doi=None,
            title=title,
            abstract=abstract,
        )
        for identifier, title, abstract in (
            (
                "A",
                "Review of ZnO ultraviolet photodetectors",
                "Fabricated devices with measured responsivity are reviewed.",
            ),
            (
                "B",
                "Simulation of ZnO ultraviolet photodetector responsivity",
                "Calculated detectivity and response time.",
            ),
        )
    )
    result = LiteratureSearchService(
        (StubProvider("openalex", papers),), LiteratureQueryStore(tmp_path)
    ).screen_candidates(CandidateLiteratureScreenInput(materials=("ZnO",)))
    assert result.candidates[0].evidence_grade == "C"
    assert not result.finalists
    assert not result.download_candidates


def test_query_expansion_is_bilingual_bounded_and_reproducible() -> None:
    request = LiteratureSearchInput(topic="二氧化钛光催化产氢", max_papers=10)
    first = DeterministicQueryExpander().expand(request)
    second = DeterministicQueryExpander().expand(request)
    assert first == second
    assert "TiO2" in first.normalized_materials
    assert "hydrogen evolution" in first.performance_terms
    assert "photocatalysis" in first.process_terms
    assert 1 <= len(first.search_queries) <= 6
    assert all(len(query) <= 300 for query in first.search_queries)

    porous = DeterministicQueryExpander().expand(
        LiteratureSearchInput(topic="磷酸钙多孔结构与成骨性能")
    )
    assert "porosity" in porous.process_terms
    assert any(
        query == "calcium phosphate osteogenic performance porosity"
        for query in porous.search_queries
    )
    assert all(
        "calcium phosphate bioceramic" not in query for query in porous.search_queries
    )

    glass = DeterministicQueryExpander().expand(
        LiteratureSearchInput(topic="3D打印生物活性玻璃支架孔结构与成骨")
    )
    assert "bioactive glass" in glass.normalized_materials
    assert "3D printing" in glass.process_terms
    assert "porosity" in glass.process_terms
    assert "osteogenic performance" in glass.performance_terms

    titanium = DeterministicQueryExpander().expand(
        LiteratureSearchInput(topic="增材制造钛合金孔隙缺陷与疲劳寿命关系的近年论文")
    )
    assert "titanium alloy" in titanium.normalized_materials
    assert "fatigue life" in titanium.performance_terms
    assert "additive manufacturing" in titanium.process_terms
    assert "porosity defects" in titanium.process_terms
    assert any(
        "titanium alloy" in query
        and "fatigue life" in query
        and "additive manufacturing" in query
        and "porosity defects" in query
        for query in titanium.search_queries
    )

    cfrp = DeterministicQueryExpander().expand(
        LiteratureSearchInput(topic="碳纤维增强聚合物界面改性与疲劳性能")
    )
    assert "carbon fiber reinforced polymer" in cfrp.normalized_materials
    assert "fatigue life" in cfrp.performance_terms
    assert "interface modification" in cfrp.process_terms
    assert any(
        query == "CFRP fatigue life interface modification"
        for query in cfrp.search_queries
    )
    assert all("碳纤维" not in query for query in cfrp.search_queries)

    uv = DeterministicQueryExpander().expand(
        LiteratureSearchInput(
            topic=("核验 Ba3Nb2O8、Ba11B26P2(H3O29)2 用于紫外光电探测的实验器件证据")
        )
    )
    assert "UV photodetector" in uv.performance_terms
    assert "Ba11B26P2(H3O29)2" in uv.normalized_materials
    assert "H3O29" not in uv.normalized_materials
    assert all("UV photodetector" in query for query in uv.search_queries)


def test_unified_search_merges_sources_ranks_and_reuses_snapshot(
    tmp_path: Path,
) -> None:
    openalex = StubProvider(
        "openalex",
        (
            _paper(
                provider="openalex",
                provider_id="A",
                title="TiO2 photocatalysis for hydrogen evolution",
                doi="10.1/shared",
                abstract="Short abstract.",
                citations=3,
            ),
        ),
    )
    semantic = StubProvider(
        "semantic_scholar",
        (
            _paper(
                provider="semantic_scholar",
                provider_id="B",
                title="TiO2 photocatalysis for hydrogen evolution",
                doi="10.1/shared",
                abstract="A longer abstract about TiO2 photocatalysis and hydrogen.",
                citations=8,
                open_access=True,
            ),
        ),
    )
    service = LiteratureSearchService(
        (openalex, semantic), LiteratureQueryStore(tmp_path)
    )
    request = LiteratureSearchInput(topic="TiO2 photocatalysis hydrogen")
    result = service.search_unified(request)
    calls = (openalex.calls, semantic.calls)
    assert result.provider == "unified"
    assert result.returned_count == 1
    assert len(result.papers[0].provenance) == 2
    assert result.papers[0].cited_by_count == 8
    assert result.papers[0].access_status == "open_access_reported"
    assert result.papers[0].selection_reason is not None
    cached = service.search_unified(request)
    assert cached.query_id == result.query_id and cached.papers == result.papers
    assert cached.created_at == result.created_at and cached.retrieval_mode == "cache"
    assert service.store.load(result.query_id) == result
    assert (openalex.calls, semantic.calls) == calls


def test_unified_search_returns_partial_result_when_one_provider_fails(
    tmp_path: Path,
) -> None:
    openalex = StubProvider(
        "openalex",
        (
            _paper(
                provider="openalex",
                provider_id="A",
                title="Calcium phosphate bioceramic scaffold",
                doi="10.2/example",
            ),
        ),
    )
    semantic = StubProvider(
        "semantic_scholar",
        error=LiteratureProviderError("PROVIDER_RATE_LIMIT", "limited"),
    )
    result = LiteratureSearchService(
        (openalex, semantic), LiteratureQueryStore(tmp_path)
    ).search_unified(LiteratureSearchInput(topic="calcium phosphate bioceramic"))
    assert result.returned_count == 1
    assert result.warnings == ("semantic_scholar: PROVIDER_RATE_LIMIT",)
    assert result.provider_statuses[1].status == "degraded"


def test_material_only_match_is_background_when_uv_performance_is_missing(
    tmp_path: Path,
) -> None:
    paper = _paper(
        provider="openalex",
        provider_id="C",
        title="Scintillation properties of CaHfO3 single crystals",
        doi="10.3/background",
        abstract="Photoluminescence and scintillation were measured.",
    )
    result = LiteratureSearchService(
        (StubProvider("openalex", (paper,)),), LiteratureQueryStore(tmp_path)
    ).search_unified(LiteratureSearchInput(topic="CaHfO3 紫外光电探测实验器件证据"))

    assert result.returned_count == 1
    assert result.papers[0].relevance_level == "extended"
    assert "性能" in result.papers[0].missing_concepts
    assert result.papers[0].application_evidence_grade == "B"


def test_uv_application_evidence_is_graded_and_ranked_a_before_b_before_c(
    tmp_path: Path,
) -> None:
    papers = (
        _paper(
            provider="openalex",
            provider_id="A",
            title="Ga2O3 ultraviolet photodetector with high responsivity",
            doi="10.3/a",
            abstract=(
                "A Ga2O3 device was fabricated and its detectivity and response "
                "time were measured under UV illumination."
            ),
        ),
        _paper(
            provider="openalex",
            provider_id="B",
            title="Experimental optical absorption of Ga2O3 single crystals",
            doi="10.3/b",
            abstract="Ga2O3 single crystals were grown and the band gap was measured.",
        ),
        _paper(
            provider="openalex",
            provider_id="C",
            title="First-principles Ga2O3 band structure for ultraviolet electronics",
            doi="10.3/c",
            abstract="Calculated electronic properties are reported.",
        ),
    )
    result = LiteratureSearchService(
        (StubProvider("openalex", papers),), LiteratureQueryStore(tmp_path)
    ).search_unified(LiteratureSearchInput(topic="Ga2O3 紫外光电探测实验器件证据"))

    assert [paper.application_evidence_grade for paper in result.papers] == [
        "A",
        "B",
        "C",
    ]
    assert "紫外探测器与器件性能指标" in (
        result.papers[0].application_evidence_reason or ""
    )


def test_candidate_pool_is_prescreened_before_finalists_are_selected(
    tmp_path: Path,
) -> None:
    provider = CandidateStubProvider(
        {
            "ZnO": (
                _paper(
                    provider="openalex",
                    provider_id="A",
                    title="ZnO ultraviolet photodetector with high responsivity",
                    doi="10.4/a",
                    abstract="A ZnO device was fabricated and detectivity measured.",
                ),
            ),
            "Ga2O3": (
                _paper(
                    provider="openalex",
                    provider_id="B",
                    title="Experimental optical absorption of Ga2O3 single crystals",
                    doi="10.4/b",
                    abstract="Ga2O3 crystals were grown and their band gap measured.",
                ),
            ),
            "CsAlSiO4": (
                _paper(
                    provider="openalex",
                    provider_id="C",
                    title="Preparation and crystal structure of CsAlSiO4",
                    doi="10.4/c",
                ),
            ),
        }
    )
    service = LiteratureSearchService((provider,), LiteratureQueryStore(tmp_path))
    request = CandidateLiteratureScreenInput(
        materials=("CsAlSiO4", "MgO", "Ga2O3", "ZnO"),
        final_limit=3,
    )

    result = service.screen_candidates(request)

    assert [row.formula for row in result.finalists] == [
        "ZnO",
        "Ga2O3",
    ]
    assert [row.evidence_grade for row in result.finalists] == ["A", "B"]
    assert len(result.candidates) == 4
    assert result.retrieval_complete
    assert result.qualifying_candidate_count == 2
    assert [paper.doi for paper in result.download_candidates] == [
        "10.4/a",
        "10.4/b",
    ]
    calls = provider.calls
    cached = service.screen_candidates(request)
    assert cached.screening_id == result.screening_id
    assert cached.candidates == result.candidates
    assert cached.retrieval_mode == "cache" and cached.queries_attempted == 0
    assert service.store.load_candidate_screen(result.screening_id) == result
    assert provider.calls == calls


def test_unified_search_reranks_compatible_success_snapshot_when_all_fail(
    tmp_path: Path,
) -> None:
    request = LiteratureSearchInput(
        topic="3D printed bioactive glass scaffold osteogenesis",
        material_keywords=("bioactive glass scaffold",),
    )
    paper = _paper(
        provider="semantic_scholar",
        provider_id="A",
        title="3D printed bioactive glass scaffold for osteogenesis",
        doi="10.6/snapshot",
        abstract="Bioactive glass pore architecture supports bone formation.",
    )
    source = LiteratureSearchService(
        (StubProvider("semantic_scholar", (paper,)),),
        LiteratureQueryStore(tmp_path / "source"),
    ).search_unified(request)
    store = LiteratureQueryStore(tmp_path / "active")
    store.save(source.model_copy(update={"query_id": "lit-historical123"}))
    failing = StubProvider(
        "semantic_scholar",
        error=LiteratureProviderError("PROVIDER_RATE_LIMIT", "limited"),
    )

    result = LiteratureSearchService((failing,), store).search_unified(request)

    assert [row.doi for row in result.papers] == ["10.6/snapshot"]
    assert any("STALE_SNAPSHOT_FALLBACK" in item for item in result.warnings)
    assert result.provider_statuses[0].status == "degraded"


def test_unified_search_retries_a_degraded_cached_snapshot(tmp_path: Path) -> None:
    openalex = StubProvider(
        "openalex",
        (
            _paper(
                provider="openalex",
                provider_id="A",
                title="Porous bioceramic scaffold for osteogenesis",
                doi="10.2/recovered",
            ),
        ),
        error=LiteratureProviderError("PROVIDER_UNAVAILABLE", "temporary"),
    )
    semantic = StubProvider(
        "semantic_scholar",
        (
            _paper(
                provider="semantic_scholar",
                provider_id="B",
                title="Porous bioceramic scaffold for osteogenesis",
                doi="10.2/available",
            ),
        ),
    )
    service = LiteratureSearchService(
        (openalex, semantic), LiteratureQueryStore(tmp_path)
    )
    request = LiteratureSearchInput(topic="porous bioceramic osteogenesis")
    degraded = service.search_unified(request)
    first_calls = openalex.calls
    assert degraded.provider_statuses[0].status == "degraded"

    openalex.error = None
    recovered = service.search_unified(request)

    assert openalex.calls > first_calls
    assert recovered.provider_statuses[0].status == "ok"


def test_concept_gate_filters_highly_cited_but_topic_irrelevant_paper(
    tmp_path: Path,
) -> None:
    provider = StubProvider(
        "openalex",
        (
            _paper(
                provider="openalex",
                provider_id="A",
                title=("Macropore calcium phosphate scaffold for osteoinductivity"),
                doi="10.3/relevant",
                abstract="Pore architecture affects bone formation.",
                citations=2,
            ),
            _paper(
                provider="openalex",
                provider_id="B",
                title="Highly cited semiconductor device review",
                doi="10.3/irrelevant",
                abstract="Electronic transport in silicon devices.",
                citations=100_000,
            ),
        ),
    )
    result = LiteratureSearchService(
        (provider,), LiteratureQueryStore(tmp_path)
    ).search_unified(LiteratureSearchInput(topic="磷酸钙多孔支架成骨性能"))

    assert [paper.doi for paper in result.papers] == ["10.3/relevant"]
    assert "calcium phosphate" in (result.papers[0].selection_reason or "")


def test_explicit_material_anchor_filters_generic_scaffold_match(
    tmp_path: Path,
) -> None:
    papers = (
        _paper(
            provider="openalex",
            provider_id="A",
            title="3D printed calcium phosphate scaffold for osteogenesis",
            doi="10.5/wrong-material",
            abstract="Pore architecture controls bone formation.",
        ),
        _paper(
            provider="openalex",
            provider_id="B",
            title="3D printed bioactive glass scaffold for osteogenesis",
            doi="10.5/right-material",
            abstract="Bioactive glass pore architecture controls bone formation.",
        ),
    )
    result = LiteratureSearchService(
        (StubProvider("openalex", papers),), LiteratureQueryStore(tmp_path)
    ).search_unified(
        LiteratureSearchInput(
            topic="3D printed scaffold pore architecture and osteogenesis",
            material_keywords=("bioactive glass scaffold", "pore architecture"),
        )
    )

    assert [paper.doi for paper in result.papers] == ["10.5/right-material"]


def test_chinese_material_anchor_matches_english_provider_records(
    tmp_path: Path,
) -> None:
    paper = _paper(
        provider="semantic_scholar",
        provider_id="BG",
        title="3D printed bioactive glass scaffold for bone engineering",
        doi="10.5/chinese-query",
        abstract="Bioactive glass supports osteogenesis in printed scaffolds.",
    )
    result = LiteratureSearchService(
        (StubProvider("semantic_scholar", (paper,)),),
        LiteratureQueryStore(tmp_path),
    ).search_unified(
        LiteratureSearchInput(
            topic="3D打印生物活性玻璃支架",
            material_keywords=("生物活性玻璃", "3D打印", "支架", "生物活性"),
        )
    )

    assert [item.doi for item in result.papers] == ["10.5/chinese-query"]
    assert "3D打印" not in result.expanded_query.normalized_materials
    assert "支架" not in result.expanded_query.normalized_materials


def test_titanium_fatigue_defect_topic_requires_defect_evidence(
    tmp_path: Path,
) -> None:
    papers = (
        _paper(
            provider="openalex",
            provider_id="A",
            title="Fatigue life of additively manufactured Ti-6Al-4V",
            doi="10.6/broad-fatigue",
            abstract="Heat treatment changes cyclic mechanical performance.",
        ),
        _paper(
            provider="openalex",
            provider_id="B",
            title="Porosity-controlled fatigue of additively manufactured Ti-6Al-4V",
            doi="10.6/defect-fatigue",
            abstract="Lack-of-fusion defects act as fatigue crack initiation sites.",
        ),
    )
    result = LiteratureSearchService(
        (StubProvider("openalex", papers),), LiteratureQueryStore(tmp_path)
    ).search_unified(
        LiteratureSearchInput(topic="增材制造钛合金孔隙缺陷与疲劳寿命关系")
    )

    assert [paper.doi for paper in result.papers] == ["10.6/defect-fatigue"]


def test_recent_sort_prefers_newer_paper_within_same_relevance_tier(
    tmp_path: Path,
) -> None:
    papers = (
        _paper(
            provider="openalex",
            provider_id="A",
            title="Porous calcium phosphate scaffold for osteogenesis",
            doi="10.4/old",
            abstract="Porosity affects osteogenic differentiation.",
            year=2021,
        ),
        _paper(
            provider="openalex",
            provider_id="B",
            title="Porous calcium phosphate scaffold for osteogenesis",
            doi="10.4/new",
            abstract="Porosity affects osteogenic differentiation.",
            year=2025,
        ),
    )
    result = LiteratureSearchService(
        (StubProvider("openalex", papers),), LiteratureQueryStore(tmp_path)
    ).search_unified(
        LiteratureSearchInput(topic="磷酸钙多孔结构成骨性能", sort_mode="recent")
    )

    assert [paper.year for paper in result.papers] == [2025, 2021]


def test_exploration_cannot_fill_strict_finalists(tmp_path: Path) -> None:
    paper = _paper(
        provider="openalex",
        provider_id="A",
        doi="10.7/test",
        title="ZnO ultraviolet photodetector",
        abstract="Devices fabricated and responsivity measured under UV illumination.",
    )
    service = LiteratureSearchService(
        (CandidateStubProvider({"ZnO": (paper,)}),), LiteratureQueryStore(tmp_path)
    )
    result = service.screen_candidates(
        CandidateLiteratureScreenInput(
            materials=("CsAlSiO4", "ZnO"),
            supplementary_materials=("ZnO",),
        )
    )
    assert not result.finalists
    assert [row.formula for row in result.supplementary_finalists] == ["ZnO"]
    assert result.supplementary_finalists[0].pool == "supplementary"
    assert result.download_candidates[0].doi == "10.7/test"


def test_candidate_screen_uses_secondary_source_on_primary_failure(
    tmp_path: Path,
) -> None:
    primary = StubProvider(
        "openalex", error=LiteratureProviderError("PROVIDER_RATE_LIMIT", "limited")
    )
    paper = _paper(
        provider="semantic_scholar",
        provider_id="A",
        doi=None,
        title="ZnO ultraviolet photodetector",
        abstract="A device was fabricated and its responsivity measured.",
    )
    secondary = StubProvider("semantic_scholar", (paper,))
    service = LiteratureSearchService(
        (primary, secondary), LiteratureQueryStore(tmp_path)
    )
    result = service.screen_candidates(
        CandidateLiteratureScreenInput(materials=("ZnO",))
    )
    assert result.retrieval_complete
    assert result.queries_attempted == 2
    assert result.finalists[0].evidence_grade == "A"
    assert result.download_candidates[0].provenance[0].provider == "semantic_scholar"
    assert "openalex: PROVIDER_RATE_LIMIT" in result.warnings


def test_partial_retry_reuses_successful_candidate_requests(tmp_path: Path) -> None:
    class FailsSecond(StubProvider):
        def search(self, request: LiteratureSearchInput) -> tuple[PaperRecord, ...]:
            self.calls += 1
            if self.error is not None and "Ga2O3" in request.topic:
                raise self.error
            return ()

    provider = FailsSecond(
        "openalex", error=LiteratureProviderError("PROVIDER_UNAVAILABLE", "down")
    )
    service = LiteratureSearchService((provider,), LiteratureQueryStore(tmp_path))
    request = CandidateLiteratureScreenInput(materials=("ZnO", "Ga2O3"))
    assert not service.screen_candidates(request).retrieval_complete
    assert provider.calls == 2
    provider.error = None
    assert service.screen_candidates(request).retrieval_complete
    assert provider.calls == 3
