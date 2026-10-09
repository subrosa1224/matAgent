"""One-call, evidence-bound previews used before expensive deep analysis."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.llm.base import StructuredLLM
from materials_screening.llm.errors import LLMError

from .rag import ChunkRecord

_PREVIEW_VERSION = "preview-v4-boundary-gated"

_PROMPT = """Create a concise screening preview for one scientific paper.
Use only the supplied evidence. Copy evidence_quote verbatim from its chunk_id.
Separate interpretation from citation:
- Summaries may paraphrase: research_question, methods, key_findings, and reason
  should concisely explain the evidence in Chinese, not mechanically translate it.
- evidence_quote is NOT a summary. Select one continuous passage from ONE supplied
  chunk, preferably a short complete sentence or consecutive sentences, and copy
  it exactly in its original language. Set chunk_id to that same chunk's ID.
  Do not stitch together distant sentences or passages from different chunks.
  Do not translate, polish, correct typographical errors, insert ellipses, or
  rewrite punctuation, formulas, exponents, minus signs, or units. Preserve the
  source characters even when they look unusual; JSON escaping is allowed.
  If a long quote is difficult to copy, choose a shorter faithful passage instead
  of reconstructing it from memory. Before returning, check that the exact quote
  appears continuously in the named supplied chunk and supports your summaries.
Do not invent numbers, methods, or conclusions. The preview is for deciding
whether the paper deserves expensive deep analysis, not for creating approved
scientific facts. Keep the screening summaries qualitative and omit numerical
details; the deep-analysis stage handles exact numbers. research_question,
methods, key_findings, and reason MUST be written in Chinese. Preserve formulas,
abbreviations, and assay names such as CCK-8. The user topic is only a relevance
question: NEVER copy a material, fabrication method, 3D-printing claim, mechanism,
or outcome from the user topic into the paper summary. Report the paper's actual
method even when it conflicts with the topic, and lower relevance accordingly.
evidence_quote must directly support the stated methods and main finding. Return
exactly one JSON object and no Markdown or explanatory text.

Treat explicit material and synthesis-route restrictions in the user topic as hard
screening boundaries. Put a boundary_findings item only when the supplied paper
evidence explicitly demonstrates a violation or a causal-attribution risk. Use:
- material_scope + exclude for doped, substituted, loaded, coated, composite, or
  otherwise modified material when the topic explicitly requires a pure or
  undoped material;
- route_scope + background_only when the actual preparation route is outside an
  explicitly allowed route list;
- multifactor_confounding + caution when two or more preparation variables change
  together, so a single-factor causal claim is unsafe;
- particle_size_ambiguity + caution when crystallite, primary-particle, aggregate,
  or hydrodynamic sizes are conflated.
Every finding must quote the paper verbatim and name its chunk_id. Do not create a
finding from the user topic alone. An empty list is correct when no explicit,
evidence-backed issue is present."""


class BoundaryFinding(BaseModel):
    """One evidence-backed reason to cap or qualify a paper preview."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    finding_type: Literal[
        "material_scope",
        "route_scope",
        "multifactor_confounding",
        "particle_size_ambiguity",
    ]
    severity: Literal["exclude", "background_only", "caution"]
    reason: str = Field(min_length=1, max_length=500)
    evidence_quote: str = Field(min_length=1, max_length=2000)
    chunk_id: str = Field(min_length=1)


class PreviewBoundaryPolicy(BaseModel):
    """Deterministic task-specific caps applied after the LLM preview."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_route_terms: tuple[str, ...] = Field(default=(), max_length=30)
    excluded_material_term_groups: tuple[tuple[str, ...], ...] = Field(
        default=(), max_length=20
    )


def infer_preview_boundary_policy(topic: str) -> PreviewBoundaryPolicy | None:
    """Infer only explicit, high-confidence screening boundaries from a topic."""
    normalized = _normalize(topic)
    route_aliases: list[str] = []
    for triggers, aliases in (
        (("沉淀", "precipitation"), ("沉淀", "湿化学沉淀", "precipitation")),
        (("水热", "hydrothermal"), ("水热", "水热法", "hydrothermal")),
        (
            ("溶胶-凝胶", "溶胶凝胶", "sol-gel", "sol–gel"),
            ("溶胶-凝胶", "溶胶凝胶", "sol-gel", "sol–gel"),
        ),
    ):
        if any(_normalize(trigger) in normalized for trigger in triggers):
            route_aliases.extend(aliases)

    material_restricted = any(
        term in normalized
        for term in (
            "纯",
            "未掺杂",
            "无掺杂",
            "未负载",
            "无负载",
            "undoped",
            "unloaded",
            "unmodified",
        )
    )
    material_groups: tuple[tuple[str, ...], ...] = ()
    if material_restricted:
        material_groups = (
            ("微量元素", "微量矿物", "trace mineral", "trace element"),
            ("掺杂", "doped", "doping", "substituted hydroxyapatite"),
            ("负载", "loaded hydroxyapatite", "hydroxyapatite loaded with"),
            ("复合材料", "hydroxyapatite composite", "composite hydroxyapatite"),
            ("羟基磷灰石涂层", "hydroxyapatite coating", "coated hydroxyapatite"),
        )
    if not route_aliases and not material_groups:
        return None
    return PreviewBoundaryPolicy(
        allowed_route_terms=tuple(dict.fromkeys(route_aliases)),
        excluded_material_term_groups=material_groups,
    )


class PreviewCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    article_type: Literal["research", "review", "other"]
    topic_relevance: Literal["core", "high", "extended", "low"]
    research_question: str = Field(min_length=1, max_length=800)
    methods: str = Field(min_length=1, max_length=800)
    key_findings: str = Field(min_length=1, max_length=1200)
    recommendation: Literal["deep_analyze", "background_only", "exclude"]
    reason: str = Field(min_length=1, max_length=600)
    evidence_quote: str = Field(min_length=1, max_length=3000)
    chunk_id: str = Field(min_length=1)
    boundary_findings: tuple[BoundaryFinding, ...] = Field(default=(), max_length=8)


class PreviewSelectionCandidate(BaseModel):
    """The provider chooses a supplied passage, never rewrites its text."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    article_type: Literal["research", "review", "other"]
    topic_relevance: Literal["core", "high", "extended", "low"]
    research_question: str = Field(min_length=1, max_length=800)
    methods: str = Field(min_length=1, max_length=800)
    key_findings: str = Field(min_length=1, max_length=1200)
    recommendation: Literal["deep_analyze", "background_only", "exclude"]
    reason: str = Field(min_length=1, max_length=600)
    passage_id: str = Field(pattern=r"^passage-[0-9]{4}$")
    boundary_findings: tuple[BoundaryFinding, ...] = Field(default=(), max_length=8)


def _numbered_passages(chunks, max_chars):
    """Only complete, supplied spans enter the request-local identifier map."""
    passages, blocks = {}, []
    used = 0
    for chunk in chunks:
        for start in range(0, len(chunk.text), 700):
            text = chunk.text[start : start + 700].strip()
            if not text:
                continue
            key = f"passage-{len(passages) + 1:04d}"
            block = (
                f"[passage_id={key}; chunk_id={chunk.chunk_id}; "
                f"page={chunk.page_from}]\n{text}"
            )
            if used + len(block) + 2 > max_chars:
                return passages, "\n\n".join(blocks)
            passages[key] = (chunk, text)
            blocks.append(block)
            used += len(block) + 2
    return passages, "\n\n".join(blocks)


class PaperPreview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    topic: str
    title: str
    article_type: Literal["research", "review", "other"]
    topic_relevance: Literal["core", "high", "extended", "low"]
    research_question: str
    methods: str
    key_findings: str
    recommendation: Literal["deep_analyze", "background_only", "exclude"]
    reason: str
    evidence_quote: str
    chunk_id: str
    boundary_findings: tuple[BoundaryFinding, ...] = ()
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_quality: Literal["verbatim", "fallback_chunk"]
    evidence_origin: Literal["model_quote", "selected_passage"] = "model_quote"
    selected_passage_id: str | None = Field(default=None, pattern=r"^passage-[0-9]{4}$")
    proposed_evidence_quote: str | None = Field(default=None, max_length=3000)
    generation_status: Literal["model", "safe_fallback", "legacy_unknown"] = (
        "legacy_unknown"
    )
    review_status: Literal["preview_only"] = "preview_only"
    preview_version: str = _PREVIEW_VERSION
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PaperPreviewExtractor:
    def __init__(
        self,
        llm: StructuredLLM,
        *,
        max_chars: int = 14000,
        boundary_policy: PreviewBoundaryPolicy | None = None,
    ) -> None:
        self.llm = llm
        self.max_chars = max_chars
        self.boundary_policy = boundary_policy

    def extract(
        self,
        *,
        document_id: str,
        title: str,
        topic: str,
        chunks: Sequence[ChunkRecord],
        max_output_tokens: int = 2048,
        screening_topic: str | None = None,
    ) -> PaperPreview:
        selected = _preview_chunks(chunks)
        passages, evidence = _numbered_passages(selected, self.max_chars)
        if not passages:
            raise ValueError("No complete source passage fits the preview budget")
        # The stored topic remains the original task identity. A combined
        # workflow may supply its paper-only scope for relevance judgments.
        user_text = (
            f"User topic: {screening_topic if screening_topic is not None else topic}"
            f"\nPaper title: {title}\n\n{evidence}"
        )
        candidate: PreviewCandidate | None = None
        last_error: Exception | None = None
        selected_passage_id = None
        selection_prompt = _PROMPT + (
            "\nOUTPUT CONTRACT: The numbered-passage selection schema replaces "
            "the main evidence_quote/chunk_id fields described above. For the main "
            "citation output ONLY passage_id from the supplied passage list. "
            "Do not output a main evidence_quote or chunk_id, or reproduce its text. "
            "Choose a passage supporting your Chinese summaries; the server retrieves "
            "its exact source text and chunk/page identity. Boundary findings still "
            "require their own verbatim quotes under the existing boundary rules."
        )
        for attempt in (1, 2):
            try:
                response = self.llm.generate_structured(
                    system_prompt=(
                        selection_prompt
                        if attempt == 1
                        else selection_prompt
                        + "\nRetry: output valid JSON and Chinese summaries only."
                    ),
                    user_text=user_text,
                    output_model=PreviewSelectionCandidate,
                    schema_name="literature_paper_preview_selection_v1",
                    max_output_tokens=max_output_tokens,
                )
                candidate = response.parsed
                if isinstance(candidate, PreviewSelectionCandidate):
                    choice = passages.get(candidate.passage_id)
                    if choice is None:
                        raise ValueError("Unknown or unsupplied preview passage")
                    source, quote = choice
                    selected_passage_id = candidate.passage_id
                    candidate = PreviewCandidate(
                        **candidate.model_dump(exclude={"passage_id"}),
                        evidence_quote=quote,
                        chunk_id=source.chunk_id,
                    )
                if _has_chinese_preview(candidate):
                    try:
                        _validate_method_grounding(candidate.methods, evidence)
                    except ValueError as exc:
                        last_error = exc
                        candidate = None
                        continue
                    break
                last_error = ValueError("preview summaries are not Chinese")
                candidate = None
            except LLMError as exc:
                last_error = exc
        used_fallback = candidate is None
        if used_fallback:
            candidate = _safe_chinese_fallback(
                title=title,
                selected=selected,
                reason=type(last_error).__name__ if last_error else "invalid_output",
            )
        chunk = next(
            (item for item in selected if item.chunk_id == candidate.chunk_id), None
        )
        chunk = chunk or selected[0]
        boundary_findings = _resolve_boundary_findings(
            candidate,
            selected,
            self.boundary_policy,
            title=title,
        )
        candidate = _enforce_boundary_findings(candidate, boundary_findings)
        is_verbatim = _normalize(candidate.evidence_quote) in _normalize(chunk.text)
        evidence_quote = (
            candidate.evidence_quote if is_verbatim else chunk.text[:1200].strip()
        )
        supported_numbers = _numbers(evidence_quote)
        candidate_data = candidate.model_dump()
        candidate_data.update(
            {
                field: _redact_unsupported_numbers(
                    str(candidate_data[field]), supported_numbers
                )
                for field in (
                    "research_question",
                    "methods",
                    "key_findings",
                    "reason",
                )
            }
        )
        candidate_data["evidence_quote"] = evidence_quote
        candidate_data["boundary_findings"] = boundary_findings
        _validate_method_grounding(str(candidate_data["methods"]), evidence)
        return PaperPreview(
            document_id=document_id,
            topic=topic,
            title=title,
            **candidate_data,
            page_from=chunk.page_from,
            page_to=chunk.page_to,
            source_text_sha256=chunk.text_sha256,
            evidence_quality="verbatim" if is_verbatim else "fallback_chunk",
            evidence_origin="selected_passage"
            if selected_passage_id and not used_fallback
            else "model_quote",
            selected_passage_id=selected_passage_id if not used_fallback else None,
            proposed_evidence_quote=candidate.evidence_quote
            if not is_verbatim and not used_fallback
            else None,
            generation_status="safe_fallback" if used_fallback else "model",
        )


class PaperPreviewStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def load(self, document_id: str, topic: str) -> PaperPreview | None:
        path = (self.root / _preview_name(document_id, topic)).resolve()
        if path.parent != self.root or not path.exists():
            return None
        return PaperPreview.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def save(self, preview: PaperPreview) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = (
            self.root / _preview_name(preview.document_id, preview.topic)
        ).resolve()
        if target.parent != self.root:
            raise ValueError("invalid preview document id")
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(preview.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, target)
        return target


def _preview_chunks(chunks: Sequence[ChunkRecord]) -> tuple[ChunkRecord, ...]:
    if not chunks:
        raise ValueError("paper preview requires indexed chunks")
    ranked = sorted(
        chunks,
        key=lambda chunk: (
            -sum(
                term in chunk.text.casefold()
                for term in (
                    "abstract",
                    "conclusion",
                    "results",
                    "method",
                    "discussion",
                )
            ),
            chunk.page_from,
        ),
    )
    return tuple(ranked[:6])


def _numbers(value: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", value))


def _normalize(value: str) -> str:
    value = value.replace("−", "-").replace("–", "-")
    return re.sub(r"\s+", " ", value).strip().casefold()


def _redact_unsupported_numbers(value: str, supported: set[str]) -> str:
    return re.sub(
        r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?",
        lambda match: _number_or_placeholder(match, value, supported),
        value,
    )


def _number_or_placeholder(
    match: re.Match[str], value: str, supported: set[str]
) -> str:
    prefix = value[max(0, match.start() - 12) : match.start()]
    suffix = value[match.end() : match.end() + 2]
    if (
        match.group(0) in supported
        or re.search(r"[A-Za-z]+[- ]$", prefix)
        or re.match(r"[A-Za-z]", suffix)
    ):
        return match.group(0)
    return "[数值待深度分析]"


def _has_chinese_preview(candidate: PreviewCandidate) -> bool:
    text = "".join(
        (
            candidate.research_question,
            candidate.methods,
            candidate.key_findings,
            candidate.reason,
        )
    )
    return len(re.findall(r"[\u3400-\u9fff]", text)) >= 12


def _validate_method_grounding(methods: str, evidence_quote: str) -> None:
    method_anchors = {
        "3d打印": (
            "3d print",
            "3d-print",
            "three-dimensional print",
            "additive manufactur",
        ),
        "定向冷冻": ("directional freez", "radial freez"),
        "冷冻干燥": ("freeze-dry", "freeze dry", "lyophiliz"),
        "数字光处理": ("digital light processing", "dlp"),
    }
    normalized_methods = methods.casefold().replace(" ", "")
    evidence = evidence_quote.casefold()
    for label, source_terms in method_anchors.items():
        if label in normalized_methods and not any(
            term in evidence for term in source_terms
        ):
            raise ValueError(f"preview method is not grounded by evidence: {label}")


def _validated_boundary_findings(
    findings: Sequence[BoundaryFinding],
    chunks: Sequence[ChunkRecord],
) -> tuple[BoundaryFinding, ...]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    accepted: list[BoundaryFinding] = []
    for finding in findings:
        chunk = chunks_by_id.get(finding.chunk_id)
        if chunk is None:
            continue
        if _normalize(finding.evidence_quote) not in _normalize(chunk.text):
            continue
        accepted.append(finding)
    return tuple(accepted)


def _resolve_boundary_findings(
    candidate: PreviewCandidate,
    chunks: Sequence[ChunkRecord],
    policy: PreviewBoundaryPolicy | None,
    *,
    title: str,
) -> tuple[BoundaryFinding, ...]:
    validated = _validated_boundary_findings(candidate.boundary_findings, chunks)
    if policy is None:
        return validated

    # Model-proposed hard decisions are advisory. A deterministic task policy
    # must corroborate material and route exclusions before they can change the
    # recommendation. Evidence-backed caution findings remain useful as flags.
    accepted = [item for item in validated if item.severity == "caution"]
    summary = _normalize(
        " ".join(
            (
                candidate.research_question,
                candidate.methods,
                candidate.key_findings,
            )
        )
    )
    evidence = _normalize(" ".join(chunk.text for chunk in chunks))

    for aliases in policy.excluded_material_term_groups:
        clean_aliases = tuple(alias for alias in aliases if alias.strip())
        if not clean_aliases:
            continue
        summary_match = next(
            (alias for alias in clean_aliases if _normalize(alias) in summary), None
        )
        evidence_match = next(
            (alias for alias in clean_aliases if _normalize(alias) in evidence), None
        )
        if summary_match is None or evidence_match is None:
            continue
        chunk = _first_chunk_containing(chunks, evidence_match)
        if chunk is None:
            continue
        accepted.append(
            BoundaryFinding(
                finding_type="material_scope",
                severity="exclude",
                reason=(
                    f"论文明确涉及“{summary_match}”，不满足本任务的纯、未掺杂材料边界"
                ),
                evidence_quote=chunk.text[:2000].strip(),
                chunk_id=chunk.chunk_id,
            )
        )
        break

    route_terms = tuple(term for term in policy.allowed_route_terms if term.strip())
    # Route caps must not trust the generated method summary: it can copy an
    # allowed route from the topic. Use only source-controlled text here.
    normalized_route_description = _normalize(f"{title} {candidate.evidence_quote}")
    if route_terms and not any(
        _normalize(term) in normalized_route_description for term in route_terms
    ):
        named_chunk = next(
            (chunk for chunk in chunks if chunk.chunk_id == candidate.chunk_id),
            chunks[0],
        )
        accepted.append(
            BoundaryFinding(
                finding_type="route_scope",
                severity="background_only",
                reason="论文实际制备方法不在本任务限定的合成路线内",
                evidence_quote=named_chunk.text[:2000].strip(),
                chunk_id=named_chunk.chunk_id,
            )
        )
    return tuple(accepted)


def _first_chunk_containing(
    chunks: Sequence[ChunkRecord], term: str
) -> ChunkRecord | None:
    normalized_term = _normalize(term)
    return next(
        (chunk for chunk in chunks if normalized_term in _normalize(chunk.text)),
        None,
    )


def _enforce_boundary_findings(
    candidate: PreviewCandidate,
    findings: Sequence[BoundaryFinding],
) -> PreviewCandidate:
    if not findings:
        return candidate
    reasons = "；".join(dict.fromkeys(item.reason.rstrip("。") for item in findings))
    reason = f"{candidate.reason.rstrip('。')}。硬边界复核：{reasons}。"
    severities = {item.severity for item in findings}
    if "exclude" in severities:
        return candidate.model_copy(
            update={
                "topic_relevance": "low",
                "recommendation": "exclude",
                "reason": reason,
            }
        )
    if "background_only" in severities:
        relevance = (
            "extended"
            if candidate.topic_relevance in {"core", "high"}
            else candidate.topic_relevance
        )
        return candidate.model_copy(
            update={
                "topic_relevance": relevance,
                "recommendation": "background_only",
                "reason": reason,
            }
        )
    return candidate.model_copy(update={"reason": reason})


def _safe_chinese_fallback(
    *, title: str, selected: Sequence[ChunkRecord], reason: str
) -> PreviewCandidate:
    chunk = selected[0]
    evidence = " ".join(item.text for item in selected).casefold()
    method = "论文制备方法需结合下方代表原文核对。"
    for terms, label in (
        (
            ("directional freeze-casting", "radial freezing"),
            "原文出现定向冷冻/冻铸方法，是否为本文核心制备方法需核对。",
        ),
        (
            ("3d printing", "3d-printed", "three-dimensional printing"),
            "原文出现3D打印方法，是否为本文核心制备方法需核对。",
        ),
        (
            ("freeze-drying", "freeze drying", "lyophiliz"),
            "原文出现冷冻干燥方法，是否为本文核心制备方法需核对。",
        ),
        (
            ("digital light processing", "dlp"),
            "原文出现数字光处理打印方法，是否为本文核心制备方法需核对。",
        ),
    ):
        if any(term in evidence for term in terms):
            method = label
            break
    article_type: Literal["research", "review", "other"] = (
        "review" if "review" in title.casefold() else "other"
    )
    return PreviewCandidate(
        article_type=article_type,
        topic_relevance="extended",
        research_question="模型未能稳定生成中文摘要，请结合题名与代表原文判断研究问题。",
        methods=method,
        key_findings="为避免误述，自动回退模式不生成未经核验的论文结论。",
        recommendation="background_only",
        reason=f"已启用安全中文回退（{reason}）；建议先阅读代表原文再决定是否深度分析。",
        evidence_quote=chunk.text[:1200].strip(),
        chunk_id=chunk.chunk_id,
    )


def preview_batch_id(topic: str, document_ids: Sequence[str]) -> str:
    identity = "|".join((topic.strip(), *sorted(document_ids)))
    return f"preview-batch-{hashlib.sha256(identity.encode()).hexdigest()[:24]}"


def _preview_name(document_id: str, topic: str) -> str:
    identity = f"{_PREVIEW_VERSION}|{topic.strip()}"
    topic_hash = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return f"{document_id}-{topic_hash}.json"
