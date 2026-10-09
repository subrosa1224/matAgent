"""Evaluate blind P2 dossier candidates against approved category coverage."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from materials_screening.sub_agents.literature.dossier import PaperDossier
from materials_screening.sub_agents.literature.rag import (
    PyMuPdfParser,
    parse_pdf_chunks,
)


def _normalize(value: str) -> str:
    value = value.replace("−", "-").replace("–", "-").replace("×", "x")
    value = value.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"\s+", " ", value).strip().casefold()


def _numbers(value: str) -> set[str]:
    normalized = value.replace("−", "-").replace("×", "x")
    return set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", normalized))


def _contains_chinese(value: str) -> bool:
    return len(re.findall(r"[\u3400-\u9fff]", value)) >= 4


def main() -> None:
    approved_root = Path("data/literature_dossiers")
    candidate_root = Path("data/literature_dossier_candidates")
    pdf_root = Path("data/literature_pdfs")
    pdf_by_document: dict[str, Path] = {}
    for pdf_path in pdf_root.glob("*.pdf"):
        digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        pdf_by_document[f"doc-{digest[:24]}"] = pdf_path

    documents: list[dict[str, object]] = []
    category_recalls: list[float] = []
    for candidate_path in sorted(candidate_root.glob("doc-*.json")):
        document_id = candidate_path.stem
        approved_path = approved_root / candidate_path.name
        resolved_pdf = pdf_by_document.get(document_id)
        if not approved_path.exists() or resolved_pdf is None:
            continue
        candidate = PaperDossier.model_validate_json(
            candidate_path.read_text(encoding="utf-8")
        )
        approved = PaperDossier.model_validate_json(
            approved_path.read_text(encoding="utf-8")
        )
        chunks = {
            chunk.chunk_id: chunk
            for chunk in parse_pdf_chunks(
                resolved_pdf,
                parser=PyMuPdfParser(),
                document_id=document_id,
            )
        }
        reference_categories = {item.category for item in approved.items}
        candidate_categories = {
            item.category
            for item in candidate.items
            if not item.summary.startswith("该候选摘要未能安全翻译")
        }
        resolvable = 0
        quote_resolvable = 0
        unsupported_numeric_items = 0
        chinese_summary_items = 0
        translation_review_notices = 0
        for item in candidate.items:
            chunk = chunks.get(item.chunk_id)
            if (
                chunk is not None
                and item.document_id == document_id
                and item.page == chunk.page_from
                and item.page_to == chunk.page_to
                and item.source_text_sha256 == chunk.text_sha256
            ):
                resolvable += 1
            if chunk is not None and _normalize(item.source_quote) in _normalize(
                chunk.text
            ):
                quote_resolvable += 1
            if not _numbers(item.summary).issubset(_numbers(item.source_quote)):
                unsupported_numeric_items += 1
            if _contains_chinese(item.summary):
                chinese_summary_items += 1
            if item.summary.startswith("该候选摘要未能安全翻译"):
                translation_review_notices += 1
        count = len(candidate.items)
        category_recall = round(
            len(reference_categories & candidate_categories)
            / len(reference_categories),
            4,
        )
        category_recalls.append(category_recall)
        documents.append(
            {
                "document_id": document_id,
                "candidate_items": count,
                "reference_categories": sorted(reference_categories),
                "candidate_categories": sorted(candidate_categories),
                "category_recall": category_recall,
                "evidence_resolvability": round(resolvable / count, 4),
                "quote_resolvability": round(quote_resolvable / count, 4),
                "unsupported_numeric_items": unsupported_numeric_items,
                "chinese_summary_rate": round(chinese_summary_items / count, 4),
                "translation_review_notices": translation_review_notices,
            }
        )
    macro_recall = (
        round(
            sum(category_recalls) / len(category_recalls),
            4,
        )
        if category_recalls
        else 0.0
    )
    print(
        json.dumps(
            {"documents": documents, "macro_category_recall": macro_recall},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
