"""Deterministic literature search orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal

from pymatgen.core import Composition

from .models import (
    CandidateEvidenceSummary,
    CandidateLiteratureScreenInput,
    CandidateLiteratureScreenOutput,
    ExpandedQuery,
    LiteratureSearchInput,
    LiteratureSearchOutput,
    PaperProvenance,
    PaperRecord,
    ProviderSearchStatus,
)
from .providers import LiteratureProvider, LiteratureProviderError
from .query_expansion import DeterministicQueryExpander, plan_provider_queries
from .retrieval_scope import phase_missing, retrieval_scope
from .store import LiteratureQueryStore
from .topic_scope import battery_evidence_scope, battery_query_terms, scientific_topic


class LiteratureSearchService:
    def __init__(
        self,
        providers: tuple[LiteratureProvider, ...],
        store: LiteratureQueryStore,
        expander: DeterministicQueryExpander | None = None,
    ) -> None:
        self._providers = {provider.name: provider for provider in providers}
        self.store = store
        self.expander = expander or DeterministicQueryExpander()

    def screen_candidates(
        self, request: CandidateLiteratureScreenInput
    ) -> CandidateLiteratureScreenOutput:
        """Pre-screen a broad candidate pool before choosing final materials."""

        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "request": request.model_dump(mode="json"),
                    "providers": sorted(
                        (name, type(provider).__name__)
                        for name, provider in self._providers.items()
                    ),
                    "version": "candidate-screen-v6-cache-receipt",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:24]
        screening_id = f"lit-screen-{fingerprint}"
        if self.store.candidate_screen_exists(screening_id):
            cached = self.store.load_candidate_screen(screening_id)
            return cached.model_copy(
                update={
                    "retrieval_mode": "cache",
                    "queries_attempted": 0,
                    "cache_hits": 1,
                    "warnings": tuple(
                        dict.fromkeys(
                            (
                                *cached.warnings,
                                f"CACHED_SCREEN_METADATA: {screening_id}; "
                                f"source_created_at={cached.created_at.isoformat()}",
                            )
                        )
                    ),
                }
            )

        provider = self._providers.get("openalex")
        if provider is None:
            provider = next(iter(self._providers.values()), None)
        if provider is None:
            raise LiteratureProviderError(
                "PROVIDER_UNAVAILABLE", "no literature providers configured"
            )
        summaries: list[CandidateEvidenceSummary] = []
        warnings: list[str] = []
        attempted = 0
        cache_hits = 0
        provider_failed = False
        for original_rank, formula in enumerate(request.materials, 1):
            papers: tuple[PaperRecord, ...] = ()
            source_result = None
            retrieval_status: Literal["ok", "error", "not_attempted"] = "not_attempted"
            if not provider_failed:
                search_request = LiteratureSearchInput(
                    # Application-first retrieval prevents highly cited general
                    # optical papers from crowding devices out of the top page.
                    topic=f"{formula} ultraviolet photodetector"
                    if request.application == "UV photodetector"
                    else f"{formula} {request.application}",
                    material_keywords=(formula,),
                    max_papers=50,
                    sort_mode="relevance",
                )
                candidates_providers = (
                    provider,
                    *(
                        source
                        for source in self._providers.values()
                        if source.name != provider.name
                    ),
                )
                for source in candidates_providers:
                    attempted += 1
                    try:
                        source_result, cache_hit = self._cached_candidate_search(
                            source, search_request
                        )
                        papers = source_result.papers
                        if cache_hit:
                            attempted -= 1
                            cache_hits += 1
                            warnings.append(
                                "CACHED_PROVIDER_METADATA: "
                                f"{source_result.query_id}; "
                                f"source_created_at={source_result.created_at.isoformat()}"
                            )
                        retrieval_status = "ok"
                        break
                    except LiteratureProviderError as exc:
                        warnings.append(f"{source.name}: {exc.code}")
                else:
                    provider_failed = True
                    retrieval_status = "error"
                expanded = self.expander.expand(
                    search_request.model_copy(
                        update={"topic": f"{formula} {request.application}"}
                    )
                )
                decorated = [
                    _decorate_paper(paper, search_request, expanded)
                    for paper in _deduplicate(papers)
                    if _paper_mentions_formula(paper, formula)
                ]
                papers = tuple(sorted(decorated, key=_evidence_paper_key))
            grade = _best_candidate_grade(papers)
            summaries.append(
                CandidateEvidenceSummary(
                    formula=formula,
                    original_rank=original_rank,
                    evidence_grade=grade,
                    pool="supplementary"
                    if formula in request.supplementary_materials
                    else "strict",
                    retrieval_status=retrieval_status,
                    matching_paper_count=len(papers),
                    source_query_id=source_result.query_id if source_result else None,
                    source_created_at=source_result.created_at
                    if source_result
                    else None,
                    papers=papers[: request.papers_per_candidate],
                    note=(
                        _candidate_grade_note(grade)
                        if retrieval_status == "ok"
                        else "检索失败，证据情况未知"
                        if retrieval_status == "error"
                        else "因文献源故障未检索，证据情况未知"
                    ),
                )
            )

        ordered = tuple(sorted(summaries, key=_candidate_summary_key))
        finalists = tuple(
            row
            for row in ordered
            if row.evidence_grade in {"A", "B"} and row.pool == "strict"
        )[: request.final_limit]
        supplementary_finalists = tuple(
            row
            for row in ordered
            if row.evidence_grade in {"A", "B"} and row.pool == "supplementary"
        )[: request.final_limit]
        qualifying = sum(row.evidence_grade in {"A", "B"} for row in ordered)
        if len(finalists) < request.final_limit:
            warnings.append(
                "EVIDENCE_SHORTFALL: fewer than the requested final candidates "
                "have A/B experimental evidence"
            )
        unique_downloads = {
            paper.paper_id: paper
            for row in (*finalists, *supplementary_finalists)
            if row.evidence_grade in {"A", "B"}
            for paper in tuple(
                paper
                for paper in row.papers
                if paper.application_evidence_grade in {"A", "B"}
            )[:3]
        }
        download_candidates = tuple(unique_downloads.values())
        if provider_failed:
            # Preserve failed attempts without poisoning the reusable successful
            # cache key. A new invocation retries the same request.
            screening_id = (
                "lit-screen-"
                + hashlib.sha256(
                    f"{fingerprint}:{datetime.now(UTC).isoformat()}".encode()
                ).hexdigest()[:24]
            )
        result = CandidateLiteratureScreenOutput(
            screening_id=screening_id,
            application=request.application,
            candidate_count=len(request.materials),
            qualifying_candidate_count=qualifying,
            candidates=ordered,
            finalists=finalists,
            supplementary_finalists=supplementary_finalists,
            download_candidates=download_candidates,
            provider=provider.name,
            queries_attempted=attempted,
            retrieval_complete=not provider_failed,
            final_limit=request.final_limit,
            warnings=tuple(dict.fromkeys(warnings)),
            retrieval_mode=(
                "mixed"
                if cache_hits and attempted
                else "cache"
                if cache_hits
                else "fresh"
            ),
            cache_hits=cache_hits,
        )
        self.store.save_candidate_screen(result)
        return result

    def _cached_candidate_search(
        self, provider: LiteratureProvider, request: LiteratureSearchInput
    ) -> tuple[LiteratureSearchOutput, bool]:
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "provider": provider.name,
                    "provider_type": type(provider).__name__,
                    # Preserve compatibility with existing device retrieval
                    # keys. The candidate path has no original-question field.
                    "request": request.model_dump(
                        mode="json", exclude={"research_question"}
                    ),
                    "kind": "candidate-device-retrieval-v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:24]
        query_id = f"lit-{fingerprint}"
        if self.store.exists(query_id):
            return self.store.load(query_id), True
        papers = provider.search(request)
        result = LiteratureSearchOutput(
            query_id=query_id,
            provider=provider.name,  # type: ignore[arg-type]
            papers=papers,
            returned_count=len(papers),
            retrieval_mode="fresh",
        )
        self.store.save(result)
        return result, False

    def search_unified(self, request: LiteratureSearchInput) -> LiteratureSearchOutput:
        expanded = self.expander.expand(request)
        plans = {
            name: plan_provider_queries(expanded, name) for name in self._providers
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "request": request.model_dump(mode="json"),
                    "expanded": expanded.model_dump(mode="json"),
                    "providers": sorted(self._providers),
                    "ranking": "unified-v13-explicit-research-scope",
                    "query_planning": "bounded-candidates-v1",
                    "provider_plans": {
                        name: asdict(plan) for name, plan in plans.items()
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:24]
        query_id = f"lit-{fingerprint}"
        cached_degraded: LiteratureSearchOutput | None = None
        if self.store.exists(query_id):
            cached = self.store.load(query_id)
            provider_degraded = any(
                status.status == "degraded" for status in cached.provider_statuses
            )
            if not provider_degraded:
                return cached.model_copy(
                    update={
                        "retrieval_mode": "cache",
                        "warnings": tuple(
                            dict.fromkeys(
                                (
                                    *cached.warnings,
                                    f"CACHED_QUERY_METADATA: {query_id}; "
                                    f"source_created_at={cached.created_at.isoformat()}",
                                )
                            )
                        ),
                    }
                )
            cached_degraded = cached

        collected: list[PaperRecord] = []
        statuses: list[ProviderSearchStatus] = []
        warnings: list[str] = []
        for provider_name, provider in self._providers.items():
            provider_rows: list[PaperRecord] = []
            warning: str | None = None
            attempted = 0
            submitted: set[str] = set()
            failed: set[str] = set()
            actual_queries: list[str] = []
            at_limit = 0
            plan = plans[provider_name]
            for query, materials in plan.queries:
                attempted += 1
                actual_queries.append(query)
                submitted.update(materials)
                provider_request = request.model_copy(
                    update={
                        "topic": query,
                        "material_keywords": (),
                        "max_papers": min(max(request.max_papers * 3, 30), 100),
                    }
                )
                try:
                    rows = provider.search(provider_request)
                    provider_rows.extend(rows)
                    at_limit += len(rows) >= provider_request.max_papers
                except LiteratureProviderError as exc:
                    failed.update(materials)
                    warning = f"{provider_name}: {exc.code}"
                    break
            if warning:
                warnings.append(warning)
            unqueried = tuple(
                value for value in plan.candidates if value not in submitted
            )
            if unqueried or failed:
                warnings.append(
                    f"CANDIDATE_COVERAGE_PARTIAL: {provider_name}; "
                    f"submitted={len(submitted)}/{len(plan.candidates)}; "
                    f"failed={len(failed)}; unqueried={len(unqueried)}"
                )
            if at_limit:
                warnings.append(
                    f"PROVIDER_RECORD_LIMIT_REACHED: {provider_name}; "
                    f"queries={at_limit}; results are bounded, not exhaustive"
                )
            collected.extend(provider_rows)
            statuses.append(
                ProviderSearchStatus(
                    provider=provider_name,  # type: ignore[arg-type]
                    status="degraded" if warning else "ok",
                    queries_attempted=attempted,
                    records_returned=len(provider_rows),
                    warning=warning,
                    search_queries=tuple(actual_queries),
                    submitted_materials=tuple(
                        value for value in plan.candidates if value in submitted
                    ),
                    failed_materials=tuple(
                        value for value in plan.candidates if value in failed
                    ),
                    unqueried_materials=unqueried,
                    queries_at_record_limit=at_limit,
                )
            )
        if not collected and any(status.status == "degraded" for status in statuses):
            fallback = self.store.latest_compatible_success(
                expanded, exclude_query_id=query_id
            )
            if fallback is None:
                raise LiteratureProviderError(
                    "PROVIDER_UNAVAILABLE", "all literature providers failed"
                )
            collected.extend(fallback.papers)
            warnings.append(
                "STALE_SNAPSHOT_FALLBACK: all current providers failed; "
                f"reranked {fallback.query_id}"
            )
        merged = _merge_papers(collected)
        relevant = [
            paper
            for paper in merged
            if _relevance(paper, request, expanded).level is not None
        ]
        if not relevant and any(status.status == "degraded" for status in statuses):
            fallback = self.store.latest_compatible_success(
                expanded, exclude_query_id=query_id
            )
            if fallback is not None:
                fallback_papers = _merge_papers([*merged, *fallback.papers])
                relevant = [
                    paper
                    for paper in fallback_papers
                    if _relevance(paper, request, expanded).level is not None
                ]
                if relevant:
                    warnings.append(
                        "STALE_SNAPSHOT_FALLBACK: current provider response was "
                        f"insufficient; reranked {fallback.query_id}"
                    )
        ranked = sorted(
            relevant,
            key=lambda paper: _rank_key(paper, request, expanded),
        )
        selected = tuple(
            _decorate_paper(paper, request, expanded)
            for paper in ranked[: request.max_papers]
        )
        if merged and not relevant:
            warnings.append("RELEVANCE_FILTER_EMPTY: no candidate met the topic gate")
        result = LiteratureSearchOutput(
            query_id=query_id,
            provider="unified",
            papers=selected,
            returned_count=len(selected),
            expanded_query=expanded,
            provider_statuses=tuple(statuses),
            warnings=tuple(warnings),
            created_at=datetime.now(UTC),
            retrieval_mode=(
                "mixed"
                if any("STALE_SNAPSHOT_FALLBACK" in w for w in warnings)
                else "fresh"
            ),
        )
        if cached_degraded is not None and not any(
            status.status == "degraded" for status in result.provider_statuses
        ):
            self.store.replace_degraded(result)
        elif cached_degraded is None:
            self.store.save(result)
        return result

    def search(
        self, provider_name: str, request: LiteratureSearchInput
    ) -> LiteratureSearchOutput:
        provider = self._providers[provider_name]
        papers = provider.search(request)
        ranked = tuple(
            sorted(_deduplicate(papers), key=lambda paper: _rank_key(paper, request))
        )
        fingerprint_payload = {
            "provider": provider_name,
            "request": request.model_dump(mode="json"),
            "paper_ids": [paper.paper_id for paper in ranked],
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()[:24]
        query_id = f"lit-{fingerprint}"
        if self.store.exists(query_id):
            return self.store.load(query_id)
        result = LiteratureSearchOutput(
            query_id=query_id,
            provider=provider_name,  # type: ignore[arg-type]
            papers=ranked[: request.max_papers],
            returned_count=min(len(ranked), request.max_papers),
            created_at=datetime.now(UTC),
        )
        self.store.save(result)
        return result

    def degraded_result(
        self,
        provider_name: str,
        request: LiteratureSearchInput,
        warning: str,
    ) -> LiteratureSearchOutput:
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "provider": provider_name,
                    "request": request.model_dump(mode="json"),
                    "degraded": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:24]
        query_id = f"lit-{fingerprint}"
        if self.store.exists(query_id):
            return self.store.load(query_id)
        result = LiteratureSearchOutput(
            query_id=query_id,
            provider=provider_name,  # type: ignore[arg-type]
            papers=(),
            returned_count=0,
            warnings=(warning,),
            created_at=datetime.now(UTC),
        )
        self.store.save(result)
        return result


def _deduplicate(papers: tuple[PaperRecord, ...]) -> list[PaperRecord]:
    seen: set[str] = set()
    output: list[PaperRecord] = []
    for paper in papers:
        key = (
            f"doi:{paper.doi}"
            if paper.doi
            else f"title:{_normalized_title(paper.title)}:{paper.year}"
        )
        if key not in seen:
            seen.add(key)
            output.append(paper)
    return output


def _normalized_title(title: str) -> str:
    return "".join(character for character in title.casefold() if character.isalnum())


def _merge_papers(papers: list[PaperRecord]) -> list[PaperRecord]:
    merged: dict[str, PaperRecord] = {}
    for paper in papers:
        key = (
            f"doi:{paper.doi}"
            if paper.doi
            else f"title:{_normalized_title(paper.title)}:{paper.year}"
        )
        existing = merged.get(key)
        if existing is None:
            merged[key] = paper
            continue
        provenance = _merge_provenance(existing.provenance, paper.provenance)
        abstracts = [value for value in (existing.abstract, paper.abstract) if value]
        authors = tuple(dict.fromkeys((*existing.authors, *paper.authors)))
        citations = [
            value
            for value in (existing.cited_by_count, paper.cited_by_count)
            if value is not None
        ]
        merged[key] = existing.model_copy(
            update={
                "abstract": max(abstracts, key=len) if abstracts else None,
                "authors": authors,
                "venue": existing.venue or paper.venue,
                "cited_by_count": max(citations) if citations else None,
                "open_access": bool(existing.open_access or paper.open_access),
                "landing_page_url": (
                    existing.landing_page_url or paper.landing_page_url
                ),
                "provenance": provenance,
            }
        )
    return list(merged.values())


def _merge_provenance(
    left: tuple[PaperProvenance, ...], right: tuple[PaperProvenance, ...]
) -> tuple[PaperProvenance, ...]:
    values = {
        (item.provider, item.provider_id, item.raw_record_sha256): item
        for item in (*left, *right)
    }
    return tuple(values[key] for key in sorted(values))


def _decorate_paper(
    paper: PaperRecord,
    request: LiteratureSearchInput,
    expanded: ExpandedQuery | None = None,
) -> PaperRecord:
    providers = tuple(dict.fromkeys(item.provider for item in paper.provenance))
    access = (
        "open_access_reported"
        if paper.open_access
        else "abstract_available"
        if paper.abstract
        else "metadata_only"
    )
    reason_parts = [f"来源：{' + '.join(providers)}"]
    if paper.abstract:
        reason_parts.append("有摘要")
    if paper.open_access:
        reason_parts.append("来源报告开放获取")
    relevance = _relevance(paper, request, expanded)
    evidence_grade, evidence_reason = _application_evidence_grade(paper, expanded)
    if relevance.matched_terms:
        reason_parts.append("匹配：" + "、".join(relevance.matched_terms[:8]))
    if relevance.missing_concepts:
        reason_parts.append("摘要未明确：" + "、".join(relevance.missing_concepts))
    reason_parts.append(f"相关性：{relevance.score:.2f}")
    if evidence_grade is not None and evidence_reason is not None:
        reason_parts.append(f"紫外应用证据 {evidence_grade} 级：{evidence_reason}")
    return paper.model_copy(
        update={
            "selection_reason": "；".join(reason_parts),
            "access_status": access,
            "relevance_level": relevance.level,
            "relevance_score": relevance.score,
            "matched_concepts": relevance.matched_terms,
            "missing_concepts": relevance.missing_concepts,
            "application_evidence_grade": evidence_grade,
            "application_evidence_reason": evidence_reason,
        }
    )


def _rank_key(
    paper: PaperRecord,
    request: LiteratureSearchInput,
    expanded: ExpandedQuery | None = None,
) -> tuple[int, int, float, str]:
    relevance = _relevance(paper, request, expanded)
    evidence_grade, _ = _application_evidence_grade(paper, expanded)
    evidence_rank = {"A": 0, "B": 1, "C": 2, None: 3}[evidence_grade]
    citation_bonus = min(math.log1p(paper.cited_by_count or 0) / 250.0, 0.025)
    year = paper.year or 0
    level_rank = {"core": 0, "high": 1, "extended": 2, None: 3}[relevance.level]
    if request.sort_mode == "recent":
        return (evidence_rank, level_rank, -float(year), paper.paper_id)
    recency_bonus = 0.0
    age = max(datetime.now(UTC).year - year, 0) if year else 100
    if request.sort_mode == "balanced":
        recency_bonus = 0.15 if age <= 3 else 0.08 if age <= 5 else 0.0
    score = relevance.score + citation_bonus + recency_bonus
    return (evidence_rank, level_rank, -score, paper.paper_id)


def _application_evidence_grade(
    paper: PaperRecord,
    expanded: ExpandedQuery | None,
) -> tuple[Literal["A", "B", "C"] | None, str | None]:
    """Grade UV-device evidence conservatively from title and abstract metadata."""

    if expanded is None or "UV photodetector" not in expanded.performance_terms:
        return None, None
    text = f"{paper.title} {paper.abstract or ''}".casefold()
    device_hit = _contains_any(
        text,
        (
            "photodetector",
            "photo detector",
            "ultraviolet detector",
            "uv detector",
            "photoconductive detector",
        ),
    )
    metric_hit = _contains_any(
        text,
        (
            "responsivity",
            "detectivity",
            "response time",
            "rise time",
            "decay time",
            "on/off",
            "on-off",
            "external quantum efficiency",
            "eqe",
        ),
    )
    uv_hit = (
        _contains_any(
            text,
            (
                "ultraviolet",
                "deep-ultraviolet",
                "deep ultraviolet",
                "solar-blind",
                "solar blind",
            ),
        )
        or re.search(r"\buv(?:a|b|c)?\b", text) is not None
    )
    experimental_hit = _contains_any(
        text,
        (
            "synthesized",
            "prepared",
            "fabricated",
            "grown",
            "measured",
            "experimental",
        ),
    )
    optical_hit = _contains_any(
        text,
        (
            "band gap",
            "absorption",
            "photoluminescence",
            "photoresponse",
            "scintillation",
            "luminescence",
        ),
    )
    review_hit = (
        re.search(r"\b(?:review|overview|perspective)\b", paper.title.casefold())
        is not None
    )
    review_hit = (
        review_hit
        or re.search(
            r"\b(?:this review|we review|is reviewed|are reviewed|"
            r"review article|an overview|we summarize)\b",
            text,
        )
        is not None
    )
    if review_hit:
        return "C", "综述或概览只能提供背景，不能作为该材料的原始器件实验"
    title = paper.title.casefold()
    theory_hit = _contains_any(
        title,
        (
            "first-principles",
            "first principles",
            "calculation",
            "simulation",
            "density functional",
            "theoretical",
            "computational",
        ),
    )
    if theory_hit and "experimental" not in title:
        return "C", "题名指向计算或模拟研究，不能当作器件实验；待全文确认"
    if device_hit and metric_hit and uv_hit and experimental_hit:
        return "A", "摘要/题名同时出现紫外探测器与器件性能指标及实验描述；待全文核验"
    if experimental_hit and (optical_hit or device_hit):
        return "B", "有实验光学或器件描述，但缺少完整紫外器件指标证据"
    return "C", "仅有配方、计算或一般背景，未形成紫外器件实验闭环"


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _paper_mentions_formula(paper: PaperRecord, formula: str) -> bool:
    # Compare entire chemical tokens by composition, not case-insensitive
    # substrings: TiO2 must not match LiTiO2, and polymorphs still need review.
    try:
        expected = Composition(formula).fractional_composition
    except ValueError:
        return False
    text = f"{paper.title} {paper.abstract or ''}".translate(
        str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    )
    atom = r"[A-Z][a-z]?(?:\d+(?:\.\d+)?)?"
    token = rf"(?<![A-Za-z0-9])(?:{atom}|\((?:{atom})+\)\d*)+(?![A-Za-z0-9])"
    for match in re.finditer(token, text):
        try:
            actual = Composition(match.group()).fractional_composition
            if len(actual.elements) >= 2 and actual.almost_equals(expected):
                return True
        except ValueError:
            continue
    return False


def _best_candidate_grade(
    papers: tuple[PaperRecord, ...],
) -> Literal["A", "B", "C", "NONE"]:
    grades = {paper.application_evidence_grade for paper in papers}
    for grade in ("A", "B", "C"):
        if grade in grades:
            return grade
    return "NONE"


def _candidate_grade_note(grade: Literal["A", "B", "C", "NONE"]) -> str:
    return {
        "A": "有紫外探测器及器件性能指标，进入全文验证",
        "B": "有实验光学或器件描述，但器件指标不完整",
        "C": "仅有计算、合成或一般材料背景",
        "NONE": "本轮未找到化学式匹配的相关记录",
    }[grade]


def _candidate_summary_key(
    row: CandidateEvidenceSummary,
) -> tuple[int, int, str]:
    grade_rank = {"A": 0, "B": 1, "C": 2, "NONE": 3}[row.evidence_grade]
    return (grade_rank, row.original_rank, row.formula)


def _evidence_paper_key(paper: PaperRecord) -> tuple[int, int, float, str]:
    grade_rank = {"A": 0, "B": 1, "C": 2, None: 3}[paper.application_evidence_grade]
    relevance_rank = {"core": 0, "high": 1, "extended": 2, None: 3}[
        paper.relevance_level
    ]
    return (
        grade_rank,
        relevance_rank,
        -float(paper.cited_by_count or 0),
        paper.paper_id,
    )


@dataclass(frozen=True)
class _Relevance:
    score: float
    level: Literal["core", "high", "extended"] | None
    matched_terms: tuple[str, ...]
    missing_concepts: tuple[str, ...]


def _relevance(
    paper: PaperRecord,
    request: LiteratureSearchInput,
    expanded: ExpandedQuery | None,
) -> _Relevance:
    title = paper.title.casefold()
    abstract = (paper.abstract or "").casefold()
    material_terms = _search_terms(
        (
            *(expanded.normalized_materials if expanded else ()),
            *(expanded.synonyms if expanded else ()),
            *(request.material_keywords if expanded is None else ()),
        ),
        expand_tokens=False,
    )
    performance_terms = _search_terms(expanded.performance_terms if expanded else ())
    process_terms = _search_terms(expanded.process_terms if expanded else ())
    target_terms = tuple(dict.fromkeys((*performance_terms, *process_terms)))
    original_terms = _search_terms(
        tuple(re.findall(r"[A-Za-z0-9+.-]{2,}", scientific_topic(request.topic)))
    )
    all_terms = tuple(dict.fromkeys((*material_terms, *target_terms, *original_terms)))
    title_hits = tuple(term for term in all_terms if _term_in(term, title))
    abstract_hits = tuple(
        term
        for term in all_terms
        if term not in title_hits and _term_in(term, abstract)
    )
    material_hit = not material_terms or any(
        _term_in(term, title) or _term_in(term, abstract) for term in material_terms
    )
    material_anchors = _material_anchors(request)
    material_anchor_hit = not material_anchors or any(
        _term_in(term, title) or _term_in(term, abstract) for term in material_anchors
    )
    performance_hit = not performance_terms or any(
        _term_in(term, title) or _term_in(term, abstract) for term in performance_terms
    )
    process_hit = not process_terms or any(
        _term_in(term, title) or _term_in(term, abstract) for term in process_terms
    )
    required_topic_anchors = _required_topic_anchors(expanded)
    required_topic_anchor_hit = not required_topic_anchors or any(
        _term_in(term, title) or _term_in(term, abstract)
        for term in required_topic_anchors
    )
    denominator = max(len(all_terms), 1)
    score = (2.0 * len(title_hits) + 0.7 * len(abstract_hits)) / denominator
    if paper.abstract:
        score += 0.03
    asks_for_review = bool(re.search(r"\breview\b|综述", request.topic.casefold()))
    if not asks_for_review and re.search(r"\breview\b", title):
        score -= 0.08
    categories = (
        ("材料", bool(material_terms), material_hit),
        ("性能", bool(performance_terms), performance_hit),
        ("工艺/结构", bool(process_terms), process_hit),
    )
    active = tuple(item for item in categories if item[1])
    matched_categories = sum(item[2] for item in active)
    missing = tuple(item[0] for item in active if not item[2])
    material_required = bool(material_terms)
    level: Literal["core", "high", "extended"] | None
    if not material_anchor_hit:
        level = None
        missing = tuple(dict.fromkeys((*missing, "显式材料体系")))
    elif not required_topic_anchor_hit:
        level = None
        missing = tuple(dict.fromkeys((*missing, "孔隙/缺陷")))
    elif active and matched_categories == len(active) and score >= 0.12:
        level = "core"
    elif performance_terms and not performance_hit and material_hit and score >= 0.06:
        # A material-name match without the requested performance outcome is
        # useful background, but must never look like application validation.
        level = "extended"
    elif (
        matched_categories >= max(len(active) - 1, 1)
        and (material_hit or not material_required)
        and score >= 0.10
    ):
        level = "high"
    elif material_hit and score >= 0.06:
        level = "extended"
    else:
        level = None
    battery_mismatch, battery_missing = battery_evidence_scope(
        request.topic, title, abstract
    )
    if battery_query_terms(request.topic) and expanded:
        formulas = tuple(
            value
            for value in expanded.normalized_materials
            if re.fullmatch(r"(?:[A-Z][a-z]?\d*|\((?:[A-Z][a-z]?\d*)+\)\d*){2,}", value)
        )
        if formulas and not any(_paper_mentions_formula(paper, f) for f in formulas):
            battery_missing = (*battery_missing, "候选化学式待核验")
    if battery_mismatch:
        level = None
    elif battery_missing and level is not None:
        level = "extended"
    missing = tuple(dict.fromkeys((*missing, *battery_missing)))
    scope = retrieval_scope(request)
    scope_missing = []
    if scope.material_alternatives and not any(
        all(_paper_mentions_formula(paper, formula) for formula in group)
        for group in scope.material_alternatives
    ):
        nearest = min(
            scope.material_alternatives,
            key=lambda group: sum(not _paper_mentions_formula(paper, f) for f in group),
        )
        scope_missing.extend(
            f"目标组分 {f} 待核验"
            for f in nearest
            if not _paper_mentions_formula(paper, f)
        )
    for phase, formula in scope.phases:
        if phase_missing(paper.title, paper.abstract or "", phase, formula):
            scope_missing.append(f"目标物相 {phase}-{formula} 待核验")
    for name, pattern in scope.conditions:
        if not re.search(pattern, f"{paper.title} {paper.abstract or ''}", re.I):
            scope_missing.append(f"目标条件 {name} 待核验")
    if any(name == "transparent conducting" for name, _ in scope.conditions):
        subject = scientific_topic(request.research_question or request.topic)
        if re.search(r"实验|\bexperimental\b", subject, re.I):
            evidence_text = f"{paper.title} {paper.abstract or ''}"
            theoretical = re.search(
                r"first[- ]principles|\bDFT\b|ab\s+initio|theoretic\w*"
                r"|density.functional|计算研究",
                evidence_text,
                re.I,
            )
            measured = re.search(
                r"measur\w*|experiment\w*|fabricat\w*|deposit\w*|sputter\w*|synthesi[sz]\w*|实验|测量|制备",
                evidence_text,
                re.I,
            )
            if theoretical and not measured:
                scope_missing.append("实验研究待核验（当前仅见理论/计算线索）")
    if scope_missing and level is not None:
        level = "extended"
    missing = tuple(dict.fromkeys((*missing, *scope_missing)))
    return _Relevance(
        score=max(score, 0.0),
        level=level,
        matched_terms=(*title_hits, *abstract_hits),
        missing_concepts=missing,
    )


def _required_topic_anchors(expanded: ExpandedQuery | None) -> tuple[str, ...]:
    """Keep explicit narrow defect requests from degrading into broad AM fatigue."""
    if expanded is None or "porosity defects" not in expanded.process_terms:
        return ()
    return (
        "porosity",
        "pore",
        "defect",
        "lack of fusion",
        "gas pore",
    )


def _material_anchors(request: LiteratureSearchInput) -> tuple[str, ...]:
    """Extract chemistry-bearing user constraints, excluding generic structure terms."""
    material_markers = {
        "alloy",
        "bioceramic",
        "calcium phosphate",
        "ceramic",
        "glass",
        "hydroxyapatite",
        "metal",
        "polymer",
        "tantalum",
        "tcp",
        "tio2",
        "titanium",
        "玻璃",
        "陶瓷",
        "磷酸钙",
        "聚合物",
        "钽",
        "钛",
    }
    anchors: list[str] = []
    canonical_markers = (
        ("生物活性玻璃", "bioactive glass"),
        ("磷酸钙", "calcium phosphate"),
        ("生物陶瓷", "bioceramic"),
        ("聚合物", "polymer"),
        ("二氧化钛", "titanium dioxide"),
        ("钽", "tantalum"),
        ("钛", "titanium"),
    )
    for value in request.material_keywords:
        normalized = " ".join(value.casefold().split())
        if not any(marker in normalized for marker in material_markers):
            continue
        canonical = next(
            (
                replacement
                for marker, replacement in canonical_markers
                if marker in normalized
            ),
            None,
        )
        if canonical is not None:
            anchors.append(canonical)
            continue
        anchor = re.sub(
            r"\b(?:porous|scaffold|structure|architecture)\b", "", normalized
        )
        anchor = re.sub(r"(?:多孔|支架|结构)", "", anchor)
        anchor = " ".join(anchor.split())
        if len(anchor) >= 2:
            anchors.append(anchor)
    return tuple(dict.fromkeys(anchors))


def _search_terms(
    values: tuple[str, ...], *, expand_tokens: bool = True
) -> tuple[str, ...]:
    stopwords = {
        "and",
        "for",
        "material",
        "materials",
        "performance",
        "scaffold",
        "structure",
    }
    terms: list[str] = []
    for value in values:
        normalized = " ".join(value.casefold().split())
        if len(normalized) >= 2:
            terms.append(normalized)
        if expand_tokens:
            terms.extend(
                token
                for token in re.findall(r"[a-z0-9+.-]{3,}", normalized)
                if token not in stopwords
            )
    return tuple(dict.fromkeys(terms))


def _term_in(term: str, text: str) -> bool:
    if term == "porosity":
        return re.search(r"\bpor(?:e|es|ous|osity)\b", text) is not None
    if term == "osteogenic":
        return re.search(r"\bosteo[a-z-]+\b", text) is not None
    if re.fullmatch(r"[a-z0-9+.-]+", term):
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        return re.search(pattern, text) is not None
    return term in text
