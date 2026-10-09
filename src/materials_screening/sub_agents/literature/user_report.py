"""Ordinary-user reports assembled without knowledge-base approval."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.llm.base import StructuredLLM

from .dossier import PaperDossier, PaperDossierStore
from .evidence_scope import is_prior_work, is_reference_evidence
from .integration import sanitize_dossier_for_report
from .models import LiteratureDocumentMetadata

_REPORT_VERSION = "user-report-v9-reference-scope"


class MetadataStore(Protocol):
    def get_document_metadata(
        self, document_id: str
    ) -> LiteratureDocumentMetadata | None: ...


class UserReportEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    category: str
    summary: str
    page: int = Field(ge=1)
    source_quote: str
    risk_level: Literal["low", "medium"]
    display_status: Literal["reliable", "check_recommended"]


class UserPaperReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    title: str
    year: int | None = None
    doi: str | None = None
    dossier_status: Literal["pending", "approved"]
    evidence: tuple[UserReportEvidence, ...]
    low_risk_count: int = Field(ge=0)
    medium_risk_count: int = Field(ge=0)
    excluded_high_risk_count: int = Field(ge=0)
    excluded_unsafe_count: int = Field(ge=0)


class UserReportNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    markdown: str = Field(min_length=20, max_length=12000)


class LiteratureUserReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: str
    topic: str
    report_version: str = _REPORT_VERSION
    papers: tuple[UserPaperReport, ...]
    narrative: UserReportNarrative | None = None
    warnings: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LiteratureUserReportService:
    def __init__(
        self,
        *,
        metadata_store: MetadataStore,
        pending_store: PaperDossierStore,
        approved_store: PaperDossierStore,
    ) -> None:
        self.metadata_store = metadata_store
        self.pending_store = pending_store
        self.approved_store = approved_store

    def build(self, *, topic: str, document_ids: Sequence[str]) -> LiteratureUserReport:
        normalized_topic = " ".join(topic.split())
        if not normalized_topic:
            raise ValueError("user report topic must not be blank")
        selected_ids = tuple(dict.fromkeys(document_ids))
        if not selected_ids:
            raise ValueError("user report requires at least one document")
        papers: list[UserPaperReport] = []
        warnings: list[str] = []
        for document_id in selected_ids:
            dossier = self.approved_store.load(document_id)
            if dossier is None:
                dossier = self.pending_store.load(document_id)
            if dossier is None or dossier.review_status == "rejected":
                warnings.append(f"{document_id}: 没有可用于普通报告的论文档案。")
                continue
            papers.append(self._paper_report(dossier))
        if not papers:
            raise ValueError("no usable paper dossiers were found")
        identity = "|".join(
            (
                _REPORT_VERSION,
                normalized_topic,
                *(item.evidence_id for paper in papers for item in paper.evidence),
            )
        )
        return LiteratureUserReport(
            report_id=f"user-lit-{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
            topic=normalized_topic,
            papers=tuple(papers),
            warnings=tuple(warnings),
        )

    def _paper_report(self, dossier: PaperDossier) -> UserPaperReport:
        sanitized = sanitize_dossier_for_report(dossier)
        loader = getattr(self.metadata_store, "get_document_chunks", None)
        records = tuple(loader(dossier.document_id)) if loader else ()
        chunks = {chunk.chunk_id: chunk for chunk in records}
        # Check old pending dossiers too; do not silently approve/overwrite them.
        sanitized = sanitized.model_copy(
            update={
                "items": tuple(
                    item
                    for item in sanitized.items
                    if not is_prior_work(
                        item.source_quote,
                        chunks[item.chunk_id].text
                        if item.chunk_id in chunks
                        else item.source_quote,
                    )
                    and not (
                        item.chunk_id in chunks
                        and is_reference_evidence(
                            item.source_quote, chunks[item.chunk_id], records
                        )
                    )
                )
            }
        )
        unsafe_count = len(dossier.items) - len(sanitized.items)
        high_count = sum(item.risk_level == "high" for item in sanitized.items)
        included = [
            item for item in sanitized.items if item.risk_level in {"low", "medium"}
        ]
        seen: set[tuple[str, str, int]] = set()
        evidence: list[UserReportEvidence] = []
        for item in included:
            identity = (item.category, " ".join(item.summary.split()), item.page)
            if identity in seen:
                continue
            seen.add(identity)
            evidence.append(
                UserReportEvidence(
                    evidence_id=_evidence_id(dossier.document_id, *identity),
                    category=item.category,
                    summary=item.summary,
                    page=item.page,
                    source_quote=item.source_quote,
                    risk_level=cast(Literal["low", "medium"], item.risk_level),
                    display_status=(
                        "reliable" if item.risk_level == "low" else "check_recommended"
                    ),
                )
            )
        metadata = self.metadata_store.get_document_metadata(dossier.document_id)
        return UserPaperReport(
            document_id=dossier.document_id,
            title=(metadata.title if metadata and metadata.title else dossier.title),
            year=metadata.year if metadata else None,
            doi=metadata.doi if metadata else None,
            dossier_status=cast(Literal["pending", "approved"], dossier.review_status),
            evidence=tuple(evidence),
            low_risk_count=sum(item.risk_level == "low" for item in evidence),
            medium_risk_count=sum(item.risk_level == "medium" for item in evidence),
            excluded_high_risk_count=high_count,
            excluded_unsafe_count=unsafe_count,
        )


class LiteratureUserReportSynthesizer:
    _PROMPT = """Write a concise Chinese literature report for an ordinary user.
Use only the supplied risk-filtered evidence. Reorganize evidence into a coherent
topic overview, one paragraph per paper, cross-paper consensus, differences,
design implications, and limitations. Every statement must cite one or more
evidence_id values at the end in the form [E-xxxxxxxxxx]. Use these exact headings:
主题概述、逐篇解读、共同结论、关键差异、设计启示、适用边界. Keep the complete
report within 1200 Chinese characters. Do not add numbers, mechanisms, comparisons,
or conclusions absent from cited evidence. Do not mention internal risk machinery,
approval, chunks, or JSON. Directional freeze-casting is NOT 3D printing. Never
claim that all papers share a method or conclusion unless every paper has direct
evidence for it. Use bullets rather than numbered-list digits. Return exactly
The evidence is a filtered subset, not the entire paper. If a metric is absent
from this subset, say 当前已提取证据尚未覆盖该指标，需要继续核对全文. Never
claim the paper did not report/provide it. Do not infer standardized procedures
or reproducibility from a single method description.
{"markdown": "..."}."""

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def synthesize(
        self, report: LiteratureUserReport, *, max_output_tokens: int = 3072
    ) -> LiteratureUserReport:
        evidence = {
            item.evidence_id: item for paper in report.papers for item in paper.evidence
        }
        if not evidence:
            return report.model_copy(
                update={
                    "warnings": (
                        *report.warnings,
                        "没有可用于叙事综合的低风险或中风险证据。",
                    )
                }
            )
        payload = [f"Topic: {report.topic}"]
        for paper in report.papers:
            payload.append(f"\n[document_id={paper.document_id}; title={paper.title}]")
            for item in paper.evidence:
                payload.append(
                    f"[{item.evidence_id}; page={item.page}; "
                    f"category={item.category}]\n"
                    f"summary: {item.summary}\n"
                    f"source_quote: {item.source_quote[:1000]}"
                )
        response = self.llm.generate_structured(
            system_prompt=self._PROMPT,
            user_text="\n\n".join(payload),
            output_model=UserReportNarrative,
            schema_name="literature_user_report_narrative_v2_flat",
            max_output_tokens=max_output_tokens,
        )
        narrative = _drop_cross_paper_inconsistencies(response.parsed, report, evidence)
        narrative = _drop_unsupported_numeric_sentences(narrative, evidence)
        narrative = _complete_numeric_citations(narrative, evidence)
        _validate_narrative(narrative, report, evidence)
        narrative = _remove_empty_sections(narrative)
        return report.model_copy(update={"narrative": narrative})


class LiteratureUserReportStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def save(self, report: LiteratureUserReport) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = (self.root / f"{report.report_id}.json").resolve()
        if target.parent != self.root:
            raise ValueError("invalid user report id")
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, target)
        return target

    def load(self, report_id: str) -> LiteratureUserReport | None:
        target = (self.root / f"{report_id}.json").resolve()
        if target.parent != self.root or not target.is_file():
            return None
        return LiteratureUserReport.model_validate_json(
            target.read_text(encoding="utf-8")
        )

    def find_latest(self, topic: str) -> LiteratureUserReport | None:
        """Return the newest safely stored report for the same normalized topic."""

        normalized = " ".join(topic.split()).casefold()
        if not normalized or not self.root.is_dir():
            return None
        matches: list[LiteratureUserReport] = []
        for path in self.root.glob("user-lit-*.json"):
            try:
                report = self.load(path.stem)
            except (OSError, ValueError):
                # Topic discovery must tolerate reports written by older schemas.
                continue
            if report is None:
                continue
            if " ".join(report.topic.split()).casefold() == normalized:
                matches.append(report)
        return max(matches, key=lambda item: item.created_at) if matches else None


def _evidence_id(document_id: str, category: str, summary: str, page: int) -> str:
    identity = f"{document_id}|{category}|{summary}|{page}"
    return f"E-{hashlib.sha256(identity.encode()).hexdigest()[:10]}"


def _validate_narrative(
    narrative: UserReportNarrative,
    report: LiteratureUserReport,
    evidence: dict[str, UserReportEvidence],
) -> None:
    if len(re.findall(r"[\u3400-\u9fff]", narrative.markdown)) < 20:
        raise ValueError("narrative report is not Chinese")
    headings = {
        "主题概述",
        "逐篇解读",
        "共同结论",
        "关键差异",
        "设计启示",
        "适用边界",
    }
    for heading in headings:
        if heading not in narrative.markdown:
            raise ValueError(f"narrative report is missing heading: {heading}")
    cited_documents: set[str] = set()
    evidence_documents = {
        item.evidence_id: paper.document_id
        for paper in report.papers
        for item in paper.evidence
    }
    globally_supported_numbers = {
        number
        for item in evidence.values()
        for number in _numbers(f"{item.summary} {item.source_quote}")
    }
    factual_lines = [
        line.strip()
        for line in narrative.markdown.splitlines()
        if _is_factual_line(line, headings)
    ]
    for line in factual_lines:
        for sentence in re.split(r"[。！？;；]", line):
            for absence in re.finditer(
                r"(?:未|没有)(?:提供|报告|给出|披露|提及)|未讨论", sentence
            ):
                prefix = sentence[: absence.start()]
                if not re.search(r"(?:当前|本轮).*(?:证据|提取|抽取|系统)", prefix):
                    raise ValueError(
                        "narrative contains an unscoped paper absence claim"
                    )
        citation_ids = tuple(dict.fromkeys(re.findall(r"E-[a-f0-9]{10}", line)))
        if not citation_ids:
            raise ValueError("narrative contains a factual line without citation")
        if any(evidence_id not in evidence for evidence_id in citation_ids):
            raise ValueError("narrative contains an unknown evidence citation")
        cited_documents.update(evidence_documents[item] for item in citation_ids)
        supported_text = " ".join(
            f"{evidence[evidence_id].summary} {evidence[evidence_id].source_quote}"
            for evidence_id in citation_ids
        )
        line_without_citations = re.sub(r"\[E-[^\]]+\]", "", line)
        statement_numbers = set(_numbers(line_without_citations))
        cited_numbers = set(_numbers(supported_text))
        cited_number_match = statement_numbers.issubset(cited_numbers)
        global_number_match = statement_numbers.issubset(globally_supported_numbers)
        if not cited_number_match and not global_number_match:
            raise ValueError("narrative contains a number absent from cited evidence")
    expected_documents = {paper.document_id for paper in report.papers}
    if cited_documents != expected_documents:
        raise ValueError("narrative does not cite evidence from every selected paper")


def _complete_numeric_citations(
    narrative: UserReportNarrative,
    evidence: dict[str, UserReportEvidence],
) -> UserReportNarrative:
    """Attach evidence that contains a number omitted from a model citation."""
    evidence_numbers = {
        evidence_id: set(_numbers(f"{item.summary} {item.source_quote}"))
        for evidence_id, item in evidence.items()
    }
    repaired_lines: list[str] = []
    for line in narrative.markdown.splitlines():
        citation_ids = list(dict.fromkeys(re.findall(r"E-[a-f0-9]{10}", line)))
        text = re.sub(r"\[E-[^\]]+\]", "", line)
        if not citation_ids:
            headings = {
                "主题概述",
                "逐篇解读",
                "共同结论",
                "关键差异",
                "设计启示",
                "适用边界",
            }
            if not _is_factual_line(line, headings):
                repaired_lines.append(line)
                continue
            matching_id = _best_matching_evidence_id(text, evidence)
            if matching_id is None:
                repaired_lines.append(line)
                continue
            citation_ids.append(matching_id)
        cited_numbers = {
            number
            for evidence_id in citation_ids
            for number in evidence_numbers.get(evidence_id, set())
        }
        for number in set(_numbers(text)) - cited_numbers:
            matching_id = next(
                (
                    evidence_id
                    for evidence_id, numbers in evidence_numbers.items()
                    if number in numbers
                ),
                None,
            )
            if matching_id is not None and matching_id not in citation_ids:
                citation_ids.append(matching_id)
        if citation_ids:
            text = text.rstrip() + f" [{', '.join(citation_ids)}]"
        repaired_lines.append(text)
    return narrative.model_copy(update={"markdown": "\n".join(repaired_lines)})


def _drop_unsupported_numeric_sentences(
    narrative: UserReportNarrative,
    evidence: dict[str, UserReportEvidence],
) -> UserReportNarrative:
    """Remove only sentences containing a number absent from all safe evidence."""
    supported = {
        number
        for item in evidence.values()
        for number in _numbers(f"{item.summary} {item.source_quote}")
    }
    cleaned_lines: list[str] = []
    for line in narrative.markdown.splitlines():
        citation_ids = tuple(dict.fromkeys(re.findall(r"E-[a-f0-9]{10}", line)))
        text = re.sub(r"\[E-[^\]]+\]", "", line).rstrip()
        if not set(_numbers(text)) - supported:
            cleaned_lines.append(line)
            continue
        sentences = re.split(r"(?<=[。！？])", text)
        kept = [
            sentence
            for sentence in sentences
            if not (set(_numbers(sentence)) - supported)
        ]
        cleaned = "".join(kept).strip()
        if cleaned:
            if citation_ids:
                cleaned += f" [{', '.join(citation_ids)}]"
            cleaned_lines.append(cleaned)
    return narrative.model_copy(update={"markdown": "\n".join(cleaned_lines)})


def _drop_cross_paper_inconsistencies(
    narrative: UserReportNarrative,
    report: LiteratureUserReport,
    evidence: dict[str, UserReportEvidence],
) -> UserReportNarrative:
    """Remove unsupported universal claims and fabrication-method conflation."""
    evidence_documents = {
        item.evidence_id: paper.document_id
        for paper in report.papers
        for item in paper.evidence
    }
    expected_documents = {paper.document_id for paper in report.papers}
    cleaned_lines: list[str] = []
    for line in narrative.markdown.splitlines():
        citation_ids = tuple(dict.fromkeys(re.findall(r"E-[a-f0-9]{10}", line)))
        cited_documents = {
            evidence_documents[item]
            for item in citation_ids
            if item in evidence_documents
        }
        text = re.sub(r"\[E-[^\]]+\]", "", line).rstrip()
        sentences = re.split(r"(?<=[。！？])", text)
        kept: list[str] = []
        for sentence in sentences:
            universal = bool(
                re.search(
                    r"(?:三篇|所有(?:研究|论文)|各(?:篇|项)?研究|上述结论|均)",
                    sentence,
                )
            )
            method_conflation = "3D打印" in sentence and "定向冷冻" in sentence
            if method_conflation:
                continue
            if universal and cited_documents != expected_documents:
                continue
            if (
                universal
                and "3D打印" in sentence
                and not _all_papers_use_3d_printing(report)
            ):
                continue
            kept.append(sentence)
        cleaned = "".join(kept).strip()
        if cleaned:
            if citation_ids and cleaned != text:
                cleaned += f" [{', '.join(citation_ids)}]"
            elif citation_ids:
                cleaned = line
            cleaned_lines.append(cleaned)
    return narrative.model_copy(update={"markdown": "\n".join(cleaned_lines)})


def _all_papers_use_3d_printing(report: LiteratureUserReport) -> bool:
    preparation_categories = {"preparation", "device_fabrication"}
    source_terms = ("3d print", "3d-print", "three-dimensional print")
    return all(
        any(
            item.category in preparation_categories
            and any(
                term in f"{item.summary} {item.source_quote}".casefold()
                for term in source_terms
            )
            for item in paper.evidence
        )
        for paper in report.papers
    )


def _remove_empty_sections(
    narrative: UserReportNarrative,
) -> UserReportNarrative:
    lines = narrative.markdown.splitlines()
    cleaned: list[str] = []
    for index, line in enumerate(lines):
        if line.startswith("## "):
            following = next(
                (item.strip() for item in lines[index + 1 :] if item.strip()),
                "",
            )
            if not following or following.startswith("## "):
                continue
        cleaned.append(line)
    return narrative.model_copy(update={"markdown": "\n".join(cleaned)})


def _is_factual_line(line: str, headings: set[str]) -> bool:
    stripped = line.strip()
    return (
        len(re.findall(r"[\u3400-\u9fff]", stripped)) >= 4
        and stripped.lstrip("#").strip() not in headings
        and (
            bool(re.search(r"E-[a-f0-9]{10}", stripped))
            or stripped.rstrip("*_ ").endswith(("。", "！", "？", ".", "!", "?"))
        )
    )


def _best_matching_evidence_id(
    text: str, evidence: dict[str, UserReportEvidence]
) -> str | None:
    terms = _matching_terms(text)
    if not terms:
        return None
    scored = [
        (
            len(terms & _matching_terms(f"{item.summary} {item.source_quote}")),
            evidence_id,
        )
        for evidence_id, item in evidence.items()
    ]
    score, evidence_id = max(scored, default=(0, ""))
    return evidence_id if score > 0 else None


def _matching_terms(value: str) -> set[str]:
    normalized = value.lower()
    words = set(re.findall(r"[a-zα-ωβγ]+\d*(?:[-/][a-z0-9α-ωβγ]+)*", normalized))
    chinese = "".join(re.findall(r"[\u3400-\u9fff]", normalized))
    bigrams = {chinese[index : index + 2] for index in range(len(chinese) - 1)}
    return words | bigrams


def _numbers(value: str) -> tuple[str, ...]:
    normalized = value.replace("−", "-").replace("×", "x")
    return tuple(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", normalized))
