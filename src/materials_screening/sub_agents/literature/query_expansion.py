"""Deterministic, bounded query expansion for materials literature discovery."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .models import ExpandedQuery, LiteratureSearchInput
from .retrieval_scope import retrieval_scope, search_subject
from .topic_scope import battery_query_terms, explicit_candidate_formulas

_CONCEPTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("tio2", "titanium dioxide", "二氧化钛"), ("TiO2", "titanium dioxide")),
    (("perovskite", "钙钛矿"), ("perovskite",)),
    (("calcium phosphate", "cap", "磷酸钙"), ("calcium phosphate", "CaP")),
    (
        ("bioactive glass", "bioglass", "生物活性玻璃"),
        ("bioactive glass", "bioglass", "bioactive glass scaffold"),
    ),
    (("bioceramic", "生物陶瓷"), ("bioceramic", "ceramic scaffold")),
    (("battery", "电池"), ("battery",)),
    (
        ("titanium alloy", "ti alloy", "钛合金"),
        ("titanium alloy", "Ti-6Al-4V", "Ti alloy"),
    ),
    (
        (
            "carbon fiber reinforced polymer",
            "carbon fibre reinforced polymer",
            "cfrp",
            "碳纤维增强聚合物",
            "碳纤维增强树脂",
            "碳纤维复合材料",
        ),
        (
            "carbon fiber reinforced polymer",
            "CFRP",
            "carbon fiber polymer composite",
            "carbon fibre reinforced plastic",
        ),
    ),
)

_PERFORMANCE: tuple[tuple[tuple[str, ...], str], ...] = (
    (("efficiency", "pce", "效率"), "efficiency"),
    (("degradation", "降解"), "degradation"),
    (("hydrogen", "产氢"), "hydrogen evolution"),
    (("bone", "osteo", "成骨", "骨诱导"), "osteogenic performance"),
    (("capacity", "容量"), "capacity"),
    (("solar cell", "photovoltaic", "光伏", "太阳能电池"), "photovoltaic performance"),
    (
        (
            "uv photodetector",
            "ultraviolet photodetector",
            "紫外光电探测",
            "紫外探测器",
            "紫外探测",
        ),
        "UV photodetector",
    ),
    (("fatigue life", "fatigue", "疲劳寿命", "疲劳"), "fatigue life"),
    (
        ("crack initiation", "crack propagation", "裂纹萌生", "裂纹扩展"),
        "fatigue crack initiation",
    ),
)

_PROCESS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("dop", "掺杂"), "doping"),
    (("3d print", "打印"), "3D printing"),
    (("synthesis", "制备", "合成"), "synthesis"),
    (
        ("high-throughput screening", "high throughput screening", "高通量筛选"),
        "high-throughput screening",
    ),
    (("photocatalysis", "photocatalytic", "光催化"), "photocatalysis"),
    (("visible light", "visible-light", "可见光"), "visible light"),
    (
        ("porosity", "porous", "pore architecture", "多孔", "孔隙率", "孔型", "孔结构"),
        "porosity",
    ),
    (
        ("additive manufacturing", "additively manufactured", "增材制造"),
        "additive manufacturing",
    ),
    (
        (
            "lack of fusion",
            "gas pore",
            "pore defect",
            "porosity defect",
            "未熔合",
            "气孔",
            "孔隙缺陷",
            "孔洞缺陷",
        ),
        "porosity defects",
    ),
    (
        (
            "interface modification",
            "interfacial modification",
            "interface engineering",
            "surface oxidation",
            "plasma treatment",
            "nanocoating",
            "界面改性",
            "表面氧化",
            "等离子处理",
            "纳米涂层",
        ),
        "interface modification",
    ),
)

_GENERIC_MATERIAL_KEYWORDS = frozenset(
    {
        "3d print",
        "3d printing",
        "3d打印",
        "bioactive",
        "scaffold",
        "scaffolds",
        "structure",
        "支架",
        "生物活性",
        "结构",
    }
)


class DeterministicQueryExpander:
    """Expand common bilingual materials concepts without a remote model call."""

    version = "deterministic-v11-candidate-application-scope"

    def expand(self, request: LiteratureSearchInput) -> ExpandedQuery:
        topic = search_subject(request)
        scope = retrieval_scope(request)
        source = " ".join((topic, *request.material_keywords))
        folded = source.casefold()
        concepts: list[str] = []
        synonyms: list[str] = []
        for triggers, values in _CONCEPTS:
            if any(trigger in folded for trigger in triggers):
                concepts.append(values[0])
                synonyms.extend(values)
        formulas = re.findall(
            r"\b(?:[A-Z][a-z]?\d*|\((?:[A-Z][a-z]?\d*)+\)\d*){2,}\b",
            source,
        )
        concepts.extend(
            value for value in formulas if value not in {"MP", "PDF", "DOI"}
        )
        concepts.extend(
            value for group in scope.material_alternatives for value in group
        )
        concepts.extend(
            value
            for value in request.material_keywords
            if value.casefold().strip() not in _GENERIC_MATERIAL_KEYWORDS
            and value.casefold().strip() not in {"mp", "pdf", "doi"}
        )
        performance = [
            value
            for triggers, value in _PERFORMANCE
            if any(trigger in folded for trigger in triggers)
        ]
        processes = [
            value
            for triggers, value in _PROCESS
            if any(trigger in folded for trigger in triggers)
        ]
        battery_terms = battery_query_terms(source)
        processes.extend(battery_terms)
        processes.extend(name for name, _ in scope.conditions)
        materials = _unique(concepts)
        synonym_values = _unique(synonyms)
        performance_values = _unique(performance)
        process_values = _unique(processes)
        # Relax a treatment term, never the explicitly requested illumination
        # or photocatalytic application. These terms also remain in relevance
        # metadata; no provider result or relevance gate is fabricated/relaxed.
        application_terms = tuple(
            value
            for value in process_values
            if value in {"visible light", "photocatalysis", "transparent conducting"}
            or value in battery_terms
        )
        # English scholarly indexes respond poorly to long Chinese prose.  Once
        # bilingual concepts are available, spend the bounded provider calls on
        # compact English concept queries instead of the original paragraph.
        has_chinese = bool(re.search(r"[\u3400-\u9fff]", source))
        queries = (
            []
            if materials and (has_chinese or topic != request.topic or battery_terms)
            else [source]
        )
        material_variants = _unique((*materials, *synonym_values))
        if scope.query_materials:
            material_variants = scope.query_materials
        if battery_terms and formulas:
            material_variants = _unique(
                [value for value in formulas if value not in {"MP", "PDF", "DOI"}]
            )
        for material in material_variants:
            terms = _unique((material, *performance_values, *process_values))
            queries.append(" ".join(terms))
        if material_variants and performance_values and process_values:
            primary = material_variants[0]
            # A relaxed pair preserves the requested material and outcome while
            # recovering papers that describe the treatment with different words.
            queries.append(
                " ".join(
                    _unique(
                        (
                            primary,
                            *performance_values,
                            *application_terms,
                        )
                    )
                )
            )
            for process in process_values:
                queries.append(
                    " ".join(_unique((primary, process, *application_terms)))
                )
        bounded = tuple(value[:300] for value in _unique(queries)[:6] if value.strip())
        return ExpandedQuery(
            original_topic=request.topic,
            research_question=request.research_question,
            normalized_materials=materials,
            synonyms=synonym_values,
            performance_terms=performance_values,
            process_terms=process_values,
            search_queries=bounded or (request.topic,),
            expansion_version=self.version,
        )


def _unique(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            output.append(normalized)
    return tuple(output)


@dataclass(frozen=True)
class ProviderQueryPlan:
    """Query text and associated candidates; submission is not recall proof."""

    candidates: tuple[str, ...]
    queries: tuple[tuple[str, tuple[str, ...]], ...]


def plan_provider_queries(expanded: ExpandedQuery, provider: str) -> ProviderQueryPlan:
    candidates = explicit_candidate_formulas(expanded.original_topic)
    if (
        candidates
        and battery_query_terms(expanded.original_topic)
        and (provider != "openalex" or len(candidates) <= 6)
    ):
        terms = _unique((*expanded.performance_terms, *expanded.process_terms))
        queries = tuple(
            (" ".join((candidate, *terms)), (candidate,))
            for candidate in candidates
            if len(" ".join((candidate, *terms))) <= 300
        )[:6]
        return ProviderQueryPlan(candidates=candidates, queries=queries)
    if provider != "openalex" or len(candidates) <= 6:
        return ProviderQueryPlan(
            candidates=candidates,
            queries=tuple(
                (
                    query,
                    tuple(
                        candidate
                        for candidate in candidates
                        if re.search(
                            rf"(?<![A-Za-z0-9]){re.escape(candidate)}"
                            r"(?![A-Za-z0-9])",
                            query,
                        )
                    ),
                )
                for query in expanded.search_queries
            ),
        )

    # Quote full formula operands, including parentheses, as literals. Never
    # truncate Boolean syntax. Terms remain exactly those of the existing
    # expander; the only changed dimension is candidate coverage.
    terms = _unique((*expanded.performance_terms, *expanded.process_terms))
    suffix = "".join(" AND " + json.dumps(term, ensure_ascii=False) for term in terms)

    def query_for(group: tuple[str, ...]) -> str:
        return "(" + " OR ".join(json.dumps(value) for value in group) + ")" + suffix

    groups: list[tuple[str, ...]] = []
    current: tuple[str, ...] = ()
    for candidate in candidates:
        proposed = (*current, candidate)
        if current and (len(proposed) > 5 or len(query_for(proposed)) > 300):
            groups.append(current)
            current = ()
        if len(query_for((candidate,))) <= 300:
            current = (*current, candidate)
    if current:
        groups.append(current)
    queries = tuple((query_for(group), group) for group in groups[:6])
    return ProviderQueryPlan(candidates=candidates, queries=queries)
