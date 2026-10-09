"""LLM-assisted, evidence-gated automation for LiteratureAgent dossiers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from openai import OpenAIError

from materials_screening.llm.base import StructuredLLM
from materials_screening.llm.errors import LLMError

from .dossier import (
    DossierExtractionBatch,
    DossierItem,
    DossierItemCandidate,
    DossierTranslationBatch,
    PaperDossier,
)
from .evidence_scope import is_prior_work, is_reference_evidence
from .rag import ChunkRecord

_SYSTEM_PROMPT = """You extract a structured scientific-paper dossier.
Use only the supplied evidence chunks. Every item must include one verbatim
source_quote copied from its named chunk_id. Do not infer missing values.
Keep summaries concise and write them in Chinese, while preserving chemical
formulae, symbols, units, and standard scientific abbreviations. Extract only
meaningful paper content; ignore references, author metadata, and boilerplate.
It is valid to return an empty items list when the batch has no useful evidence.
"""

_TRANSLATION_PROMPT = """Translate scientific dossier summaries into concise
Chinese. Use only the supplied source_quote. Preserve chemical formulae,
symbols, abbreviations, numbers, ranges, and units exactly. Do not add a number
or claim absent from that item's source_quote. Return one result for every
item_key. The chinese_summary must contain Chinese; source quotes stay English.
"""

_CATEGORY_TASKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "research_problem": (
        "研究问题、现有瓶颈和研究动机",
        ("challenge", "problem", "limit", "bottleneck", "however", "aim"),
    ),
    "innovation": (
        "本文的新方法、设计或相对已有工作的创新",
        ("novel", "new", "we report", "we propose", "develop", "innovation"),
    ),
    "materials": (
        "材料体系、组成、配方、掺杂或实验组",
        (
            "material",
            "composition",
            "doped",
            "ratio",
            "sample",
            "scaffold",
            "electrode",
        ),
    ),
    "preparation": (
        "材料合成和制备步骤及条件",
        (
            "synthesi",
            "prepared",
            "fabricat",
            "anneal",
            "sinter",
            "hydrothermal",
            "rpm",
            "method",
        ),
    ),
    "device_fabrication": (
        "器件结构、组装、膜层和电极制备；无器件则留空",
        ("device", "electrode", "substrate", "film", "cell", "FTO", "ITO"),
    ),
    "characterization": (
        "表征、测试方法和测试条件",
        ("characteri", "measured", "spectra", "microscop", "XRD", "SEM", "test"),
    ),
    "structural_result": (
        "晶相、形貌、孔结构、微结构等结果",
        ("structure", "phase", "morpholog", "pore", "grain", "crystal", "surface"),
    ),
    "optical_result": (
        "光学、能带、发光、电荷动力学结果；不适用则留空",
        ("optical", "bandgap", "absorption", "photolum", "lifetime", "spectrum"),
    ),
    "performance_result": (
        "主要性能、定量结果和最佳条件",
        ("performance", "efficien", "highest", "increase", "decrease", "result", "%"),
    ),
    "mechanism": (
        "作者提出且有正文依据的作用机理",
        (
            "mechanism",
            "attribut",
            "due to",
            "because",
            "indicat",
            "suggest",
            "responsible",
        ),
    ),
    "limitations": (
        "论文明确说明的限制、边界、失败条件或证据不足",
        ("limit", "however", "remain", "deterior", "only", "future", "challenge"),
    ),
    "reproducibility": (
        "复现实验所需的关键配方、时间、温度、设备和测试参数",
        (
            "experimental",
            "method",
            "°c",
            "min",
            "hour",
            "rpm",
            "concentration",
            "supplement",
        ),
    ),
}

_CORE_CATEGORIES = (
    "research_problem",
    "innovation",
    "materials",
    "preparation",
    "characterization",
    "structural_result",
    "performance_result",
    "mechanism",
    "limitations",
    "reproducibility",
)

_NARRATIVE_LABELS = {
    "research_problem": "研究问题",
    "innovation": "核心创新",
    "materials": "材料体系",
    "preparation": "制备方法",
    "device_fabrication": "样品或器件制备",
    "characterization": "表征方法",
    "structural_result": "结构结果",
    "optical_result": "光学结果",
    "performance_result": "性能结果",
    "mechanism": "作用机理",
    "limitations": "局限与边界",
    "reproducibility": "复现参数",
}


@dataclass(frozen=True)
class DossierTask:
    category: str
    instruction: str
    keywords: tuple[str, ...]
    chunks: tuple[ChunkRecord, ...]


class AutomatedDossierExtractor:
    def __init__(
        self,
        llm: StructuredLLM,
        *,
        batch_chars: int = 6800,
        chunks_per_task: int = 5,
    ) -> None:
        self.llm = llm
        self.batch_chars = batch_chars
        self.chunks_per_task = chunks_per_task

    def extract(
        self,
        *,
        document_id: str,
        title: str,
        chunks: Sequence[ChunkRecord],
        max_output_tokens: int = 4096,
    ) -> tuple[PaperDossier, tuple[str, ...]]:
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        candidates: list[tuple[str, DossierItemCandidate]] = []
        warnings: list[str] = []
        tasks = plan_dossier_tasks(chunks, chunks_per_task=self.chunks_per_task)
        for batch_index, task in enumerate(tasks, 1):
            response = None
            for attempt in (1, 2):
                try:
                    response = self.llm.generate_structured(
                        system_prompt=_SYSTEM_PROMPT,
                        user_text=_task_text(task, self.batch_chars),
                        output_model=DossierExtractionBatch,
                        schema_name="literature_dossier_batch_v1",
                        max_output_tokens=max_output_tokens,
                    )
                    break
                except OpenAIError as exc:
                    if attempt == 2:
                        warnings.append(f"batch {batch_index} rejected: {exc}")
                except LLMError as exc:
                    warnings.append(f"batch {batch_index} rejected: {exc}")
                    break
            if response is None:
                continue
            candidates.extend((task.category, item) for item in response.parsed.items)
        accepted: list[DossierItem] = []
        seen: set[tuple[str, str]] = set()
        for expected_category, candidate in candidates:
            chunk = chunks_by_id.get(candidate.chunk_id)
            if candidate.category != expected_category:
                warnings.append(
                    f"{candidate.category} rejected: expected {expected_category}"
                )
                continue
            grounded_quote, quote_quality = (
                _ground_quote_with_quality(candidate.source_quote, chunk.text)
                if chunk is not None
                else (None, None)
            )
            reason = _rejection_reason(grounded_quote, candidate.summary, chunk)
            summary = candidate.summary
            forced_high_risk = False
            if reason == "summary contains numbers absent from source_quote":
                assert grounded_quote is not None
                summary = grounded_quote
                forced_high_risk = True
                warnings.append(
                    f"{candidate.category}: ungrounded summary replaced "
                    "with source_quote"
                )
            elif reason is not None:
                warnings.append(f"{candidate.category} rejected: {reason}")
                continue
            assert chunk is not None
            if is_reference_evidence(grounded_quote, chunk, chunks):
                warnings.append(f"{candidate.category} rejected: reference section")
                continue
            key = (candidate.category, _normalize(summary))
            if key in seen:
                continue
            seen.add(key)
            assert grounded_quote is not None
            confidence, risk = _risk(
                grounded_quote, summary, quote_quality=quote_quality
            )
            if is_prior_work(grounded_quote, chunk.text):
                forced_high_risk = True
                warnings.append(f"{candidate.category}: prior work quarantined")
            if forced_high_risk:
                confidence, risk = 0.6, "high"
            accepted.append(
                DossierItem(
                    category=candidate.category,
                    summary=summary,
                    source_quote=grounded_quote,
                    chunk_id=chunk.chunk_id,
                    page=chunk.page_from,
                    document_id=document_id,
                    page_to=chunk.page_to,
                    source_text_sha256=chunk.text_sha256,
                    confidence=confidence,
                    risk_level=risk,
                    llm_extracted=True,
                )
            )
        if not accepted:
            detail = "; ".join(warnings[:8]) or "model returned no candidates"
            raise ValueError(
                "no evidence-grounded dossier items were extracted: " + detail
            )
        accepted = list(_prune_dossier_items(accepted, warnings=warnings))
        accepted = list(
            self._translate_to_chinese(
                accepted,
                warnings=warnings,
                max_output_tokens=max_output_tokens,
            )
        )
        usable_items = [
            item
            for item in accepted
            if not item.summary.startswith("该候选摘要未能安全翻译")
            and item.risk_level != "high"
        ]
        covered = tuple(
            category
            for category in _CATEGORY_TASKS
            if any(item.category == category for item in usable_items)
        )
        missing_core = tuple(
            category for category in _CORE_CATEGORIES if category not in covered
        )
        completeness = 1 - len(missing_core) / len(_CORE_CATEGORIES)
        if missing_core:
            warnings.append(
                "missing core dossier categories: " + ", ".join(missing_core)
            )
        return (
            PaperDossier(
                document_id=document_id,
                title=title,
                extraction_method="llm",
                review_status="pending",
                items=tuple(accepted),
                warnings=tuple(warnings),
                extraction_version="task-retrieval-v3-risk-calibrated",
                narrative_summary=_build_narrative(usable_items),
                covered_categories=covered,
                missing_core_categories=missing_core,
                completeness_score=completeness,
            ),
            tuple(warnings),
        )

    def _translate_to_chinese(
        self,
        items: Sequence[DossierItem],
        *,
        warnings: list[str],
        max_output_tokens: int,
    ) -> tuple[DossierItem, ...]:
        targets = list(items)
        if not targets:
            return tuple(items)
        translated: dict[str, str] = {}
        indexed_targets = tuple(enumerate(targets))
        for group in _translation_groups(indexed_targets):
            payload = "\n\n".join(
                f"[item_key=item-{index}; category={item.category}]\n"
                f"draft_summary: {item.summary}\n"
                f"source_quote: {item.source_quote}"
                for index, item in group
            )
            try:
                response = self.llm.generate_structured(
                    system_prompt=_TRANSLATION_PROMPT,
                    user_text=payload,
                    output_model=DossierTranslationBatch,
                    schema_name="literature_dossier_translation_v1",
                    max_output_tokens=max_output_tokens,
                )
                translated.update(
                    {
                        item.item_key: item.chinese_summary
                        for item in response.parsed.items
                    }
                )
            except (OpenAIError, LLMError) as exc:
                warnings.append(f"summary translation rejected: {exc}")

        replacements: dict[int, DossierItem] = {}
        for index, item in enumerate(targets):
            candidate = translated.get(f"item-{index}", "").strip()
            valid = _valid_translation(candidate, item.source_quote)
            if not valid:
                candidate = self._retry_single_translation(
                    index=index,
                    item=item,
                    max_output_tokens=max_output_tokens,
                    warnings=warnings,
                )
                valid = _valid_translation(candidate, item.source_quote)
            if not valid:
                candidate = "该候选摘要未能安全翻译，请根据下方原文人工核对。"
                warnings.append(
                    f"{item.category}: unsafe translation replaced with review notice"
                )
            translated_risk = "high" if not valid else item.risk_level
            replacements[id(item)] = item.model_copy(
                update={
                    "summary": candidate,
                    "confidence": min(item.confidence, 0.9 if valid else 0.5),
                    "risk_level": translated_risk,
                }
            )
        return tuple(replacements.get(id(item), item) for item in items)

    def _retry_single_translation(
        self,
        *,
        index: int,
        item: DossierItem,
        max_output_tokens: int,
        warnings: list[str],
    ) -> str:
        item_key = f"item-{index}"
        payload = (
            f"[item_key={item_key}; category={item.category}]\n"
            f"draft_summary: {item.summary}\n"
            f"source_quote: {item.source_quote}"
        )
        try:
            response = self.llm.generate_structured(
                system_prompt=(
                    _TRANSLATION_PROMPT
                    + "\nThis is a single-item retry. Return exactly one item."
                ),
                user_text=payload,
                output_model=DossierTranslationBatch,
                schema_name="literature_dossier_translation_retry_v1",
                max_output_tokens=max_output_tokens,
            )
        except (OpenAIError, LLMError) as exc:
            warnings.append(f"{item.category}: translation retry rejected: {exc}")
            return ""
        return next(
            (
                result.chinese_summary.strip()
                for result in response.parsed.items
                if result.item_key == item_key
            ),
            "",
        )


def _translation_groups(
    targets: Sequence[tuple[int, DossierItem]], *, max_chars: int = 6000
) -> tuple[tuple[tuple[int, DossierItem], ...], ...]:
    groups: list[tuple[tuple[int, DossierItem], ...]] = []
    current: list[tuple[int, DossierItem]] = []
    size = 0
    for indexed_item in targets:
        _, item = indexed_item
        item_size = len(item.summary) + len(item.source_quote) + 120
        if current and size + item_size > max_chars:
            groups.append(tuple(current))
            current = []
            size = 0
        current.append(indexed_item)
        size += item_size
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _build_narrative(items: Sequence[DossierItem]) -> str:
    """Build a complete Chinese paper story without generating new facts."""
    sections: list[str] = []
    for category in _CATEGORY_TASKS:
        category_items = [item for item in items if item.category == category]
        if not category_items:
            continue
        summaries = "；".join(
            item.summary.rstrip("。；") for item in category_items[:2]
        )
        pages = "、".join(str(item.page) for item in category_items[:2])
        sections.append(
            f"{_NARRATIVE_LABELS[category]}：{summaries}。（证据页：{pages}）"
        )
    return "\n".join(sections)


def _prune_dossier_items(
    items: Sequence[DossierItem],
    *,
    warnings: list[str],
    max_per_category: int = 3,
) -> tuple[DossierItem, ...]:
    kept: list[DossierItem] = []
    category_counts: dict[str, int] = {}
    seen_quotes: set[tuple[str, str]] = set()
    for item in items:
        if not _category_evidence_consistent(item.category, item.source_quote):
            warnings.append(
                f"{item.category}: low-value or mismatched evidence removed"
            )
            continue
        quote_key = (item.category, _normalize(item.source_quote))
        if quote_key in seen_quotes:
            warnings.append(f"{item.category}: duplicate evidence removed")
            continue
        count = category_counts.get(item.category, 0)
        if count >= max_per_category:
            warnings.append(f"{item.category}: excess candidate removed")
            continue
        seen_quotes.add(quote_key)
        category_counts[item.category] = count + 1
        kept.append(item)
    if not kept:
        raise ValueError("all dossier candidates were removed by quality filters")
    return tuple(kept)


def _category_evidence_consistent(category: str, quote: str) -> bool:
    text = _normalize(quote)
    if len(text) < 35:
        return False
    if re.search(r"\b(performed|measurements|analyses) by\b", text):
        return False
    if text.startswith(("fig. ", "figure ", "table ")):
        return False
    if category == "device_fabrication":
        if any(term in text for term in ("sem image", "xrd", "spectrum", "spectra")):
            return False
        return any(
            term in text
            for term in (
                "fabricat",
                "prepared",
                "deposited",
                "spin coat",
                "screen-print",
                "sinter",
                "anneal",
                "electrode",
            )
        )
    if category == "innovation":
        return any(
            term in text
            for term in (
                "we report",
                "first time",
                "novel",
                "new ",
                "develop",
                "establish",
                "propos",
                "methodology",
                "fabricated",
            )
        )
    if category == "limitations":
        return any(
            term in text
            for term in (
                "limit",
                "deterior",
                "however",
                "remain",
                "not included",
                "future",
            )
        )
    return True


def _batches(
    chunks: Sequence[ChunkRecord], limit: int
) -> tuple[tuple[ChunkRecord, ...], ...]:
    batches: list[tuple[ChunkRecord, ...]] = []
    current: list[ChunkRecord] = []
    size = 0
    for chunk in chunks:
        if current and size + len(chunk.text) > limit:
            batches.append(tuple(current))
            current = []
            size = 0
        current.append(chunk)
        size += len(chunk.text)
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _batch_text(chunks: Sequence[ChunkRecord]) -> str:
    return "\n\n".join(
        f"[chunk_id={chunk.chunk_id}; page={chunk.page_from}]\n{chunk.text}"
        for chunk in chunks
    )


def plan_dossier_tasks(
    chunks: Sequence[ChunkRecord], *, chunks_per_task: int = 6
) -> tuple[DossierTask, ...]:
    """Select evidence independently for every dossier category."""
    if chunks_per_task < 1:
        raise ValueError("chunks_per_task must be positive")
    tasks: list[DossierTask] = []
    for category, (instruction, keywords) in _CATEGORY_TASKS.items():
        ranked = sorted(
            chunks,
            key=lambda chunk: (
                -_task_score(chunk, keywords, category),
                chunk.page_from,
                chunk.chunk_id,
            ),
        )
        tasks.append(
            DossierTask(
                category=category,
                instruction=instruction,
                keywords=keywords,
                chunks=tuple(ranked[:chunks_per_task]),
            )
        )
    return tuple(tasks)


def _task_score(chunk: ChunkRecord, keywords: Sequence[str], category: str) -> float:
    text = chunk.text.casefold()
    score = sum(text.count(keyword.casefold()) for keyword in keywords)
    if category in {"research_problem", "innovation"} and chunk.page_from <= 2:
        score += 4
    if category in {
        "preparation",
        "characterization",
        "reproducibility",
    } and any(
        term in text for term in ("experimental", "methods", "supporting information")
    ):
        score += 4
    if category == "limitations" and any(
        term in text for term in ("conclusion", "discussion", "future")
    ):
        score += 3
    if category in {
        "performance_result",
        "structural_result",
        "optical_result",
        "mechanism",
    }:
        score += min(3, len(re.findall(r"\d+(?:\.\d+)?", text)) // 3)
    return float(score)


def _task_text(task: DossierTask, limit: int) -> str:
    header = (
        f"Target category: {task.category}\n"
        f"Task: {task.instruction}\n"
        "Return zero or more items only for the target category. "
        "Do not fill a category when evidence is absent. Every number and unit "
        "in a summary must occur in that item's single source_quote; create "
        "separate items instead of combining facts from different chunks.\n\n"
    )
    body = "\n\n".join(
        (
            f"[chunk_id={chunk.chunk_id}; page={chunk.page_from}]\n"
            f"{_evidence_excerpt(chunk.text, task.keywords)}"
        )
        for chunk in task.chunks
    )
    return header + body[: max(0, limit - len(header))]


def _evidence_excerpt(
    text: str, keywords: Sequence[str], *, max_chars: int = 1200
) -> str:
    """Keep a verbatim window around the densest task-keyword region."""
    if len(text) <= max_chars:
        return text
    lowered = text.casefold()
    positions: list[int] = []
    for keyword in keywords:
        start = 0
        needle = keyword.casefold()
        while (position := lowered.find(needle, start)) >= 0:
            positions.append(position)
            start = position + max(1, len(needle))
    center = max(
        positions,
        key=lambda position: sum(abs(other - position) <= 600 for other in positions),
        default=0,
    )
    start = max(0, center - max_chars // 4)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    return text[start:end]


def _rejection_reason(
    quote: str | None, summary: str, chunk: ChunkRecord | None
) -> str | None:
    if chunk is None:
        return "unknown chunk_id"
    if quote is None:
        return "source_quote is not verbatim evidence"
    quote_numbers = set(_numbers(quote))
    summary_numbers = set(_numbers(summary))
    if not summary_numbers.issubset(quote_numbers):
        return "summary contains numbers absent from source_quote"
    return None


def _ground_quote(quote: str, chunk_text: str) -> str | None:
    """Return exact chunk text, tolerating only minor PDF extraction noise."""
    grounded, _ = _ground_quote_with_quality(quote, chunk_text)
    return grounded


def _ground_quote_with_quality(
    quote: str, chunk_text: str
) -> tuple[str | None, Literal["exact", "repaired"] | None]:
    normalized_quote = _normalize(quote)
    if not normalized_quote:
        return None, None
    if normalized_quote in _normalize(chunk_text):
        return quote, "exact"
    sentences = tuple(
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n{2,}", chunk_text)
        if part.strip()
    )
    best_text: str | None = None
    best_score = 0.0
    for width in (1, 2, 3):
        for index in range(0, len(sentences) - width + 1):
            candidate = " ".join(sentences[index : index + width])
            score = SequenceMatcher(
                None, normalized_quote, _normalize(candidate)
            ).ratio()
            if score > best_score:
                best_score = score
                best_text = candidate
    return (best_text, "repaired") if best_score >= 0.88 else (None, None)


def _risk(
    quote: str,
    summary: str,
    *,
    quote_quality: Literal["exact", "repaired"] | None,
) -> tuple[float, Literal["low", "medium", "high"]]:
    if quote_quality != "exact":
        return 0.65, "high"
    summary_numbers = set(_numbers(summary))
    if summary_numbers:
        return 0.85, "medium"
    if len(quote) >= 50:
        return 0.95, "low"
    return 0.8, "medium"


def _numbers(value: str) -> tuple[str, ...]:
    normalized = value.replace("−", "-").replace("×", "x")
    return tuple(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", normalized))


def _is_chinese_summary(value: str) -> bool:
    han_count = len(re.findall(r"[\u3400-\u9fff]", value))
    return han_count >= 4


def _valid_translation(summary: str, source_quote: str) -> bool:
    return _is_chinese_summary(summary) and set(_numbers(summary)).issubset(
        set(_numbers(source_quote))
    )


def _normalize(value: str) -> str:
    value = value.replace("−", "-").replace("–", "-").replace("×", "x")
    value = value.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"\s+", " ", value).strip().casefold()
