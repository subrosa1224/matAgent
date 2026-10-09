"""Create evidence-located, pending RA-0 dossiers without a database or LLM."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from materials_screening.sub_agents.literature.dossier import (  # noqa: E402
    DossierItem,
    PaperDossier,
    PaperDossierStore,
)


@dataclass(frozen=True)
class EvidenceSpec:
    category: str
    page: int
    anchor: str
    summary: str
    length: int = 420


@dataclass(frozen=True)
class PaperSpec:
    filename: str
    title: str
    summary: str
    evidence: tuple[EvidenceSpec, ...]


PAPERS = (
    PaperSpec(
        filename="Facile synthesis of epsilon iron oxides via spray.pdf",
        title=(
            "Facile synthesis of epsilon iron oxides via spray drying for "
            "millimeter-wave absorption"
        ),
        summary=(
            "论文提供 ε-Fe2O3 的喷雾干燥-退火实验路线、物相比例和磁性表征。"
            "它能支持特定条件下的实验制备证据，不能直接证明数据库中的任意 Fe2O3 "
            "结构记录与样品相同。"
        ),
        evidence=(
            EvidenceSpec(
                "research_problem",
                1,
                "A simple, scalable spray drying method was developed",
                "研究目标是建立可扩展的高产率 ε-Fe2O3 合成方法。",
            ),
            EvidenceSpec(
                "preparation",
                2,
                "For the sample annealed in air at 930",
                "论文比较了不同空气退火温度下的 Fe2O3 物相演变。",
            ),
            EvidenceSpec(
                "structural_result",
                2,
                "A pure e-Fe2O3 phase is observed at 1180",
                "XRD 显示 1180 °C 条件得到 ε 相；更高温度出现 α 相共存。",
            ),
            EvidenceSpec(
                "characterization",
                3,
                "The relative areas of e-Fe2O3 and a-Fe2O3",
                "穆斯堡尔拟合给出 ε-Fe2O3 与 α-Fe2O3 的相对比例。",
            ),
            EvidenceSpec(
                "limitations",
                4,
                "In this study, a simple and scalable method",
                "结论适用于喷雾干燥、二氧化硅限域和规定退火条件，不能脱离工艺外推。",
            ),
        ),
    ),
    PaperSpec(
        filename="Zeta-Fe2O3 – A new stable.pdf",
        title="Zeta-Fe2O3 - A new stable polymorph in iron(III) oxide family",
        summary=(
            "论文报告 β-Fe2O3 纳米颗粒在高压下转变为单斜 ζ-Fe2O3，并在卸压后保留。"
            "该证据依赖高压、纳米前驱体和物相表征，不能泛化为普通条件可合成。"
        ),
        evidence=(
            EvidenceSpec(
                "research_problem",
                8,
                "The pressure-induced transformation of the rare",
                "研究考察稀有 β-Fe2O3 的压力诱导转变并识别新多晶型。",
            ),
            EvidenceSpec(
                "preparation",
                9,
                "Synthesis of b-Fe2O3 nanoparticles",
                "β-Fe2O3 前驱体由 NaCl 与 Fe2(SO4)3 的空气固相反应制得。",
            ),
            EvidenceSpec(
                "characterization",
                9,
                "In-situ high-pressure X-ray diffraction experiments",
                "采用金刚石对顶砧同步辐射原位高压 XRD，并记录压力范围。",
            ),
            EvidenceSpec(
                "structural_result",
                6,
                "After releasing the pressure",
                "卸压后 ζ-Fe2O3 保留，并由 XRD 与选区电子衍射进一步支持。",
                620,
            ),
            EvidenceSpec(
                "limitations",
                8,
                "Its stability is thus strongly linked",
                "论文明确指出相稳定性与纳米颗粒尺寸紧密相关。",
            ),
        ),
    ),
    PaperSpec(
        filename="s41467-018-06682-4.pdf",
        title=(
            "Physical descriptor for the Gibbs energy of inorganic crystalline "
            "solids and temperature-dependent materials chemistry"
        ),
        summary=(
            "论文以 ICSD 关联的晶态无机材料为总体样本，说明 0 K 凸包距离不能单独"
            "作为实验可合成性的否决条件。它是跨材料边界证据，不是任何具体候选的"
            "合成证明。"
        ),
        evidence=(
            EvidenceSpec(
                "research_problem",
                1,
                "The Gibbs energy, G, determines",
                "研究目标是建立温度相关无机固体吉布斯能描述符并分析稳定性。",
            ),
            EvidenceSpec(
                "materials",
                1,
                "We then apply this descriptor to",
                "分析对象约为 3 万种由 ICSD 整理的已知材料。",
            ),
            EvidenceSpec(
                "structural_result",
                6,
                "At 0 K, 54% of metastable",
                "研究量化了实验已实现但计算为亚稳的材料在不同凸包距离上的比例。",
            ),
            EvidenceSpec(
                "limitations",
                6,
                "An important distinction between structures and compositions",
                "同一组成的不同结构必须与组成层面的统计结论区分。",
            ),
            EvidenceSpec(
                "mechanism",
                6,
                "A number of routes exist for accessing metastable structures",
                "非平衡合成和合金化是获得亚稳结构的可能路径，但不保证具体候选成功。",
            ),
        ),
    ),
)


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _excerpt(text: str, anchor: str, length: int) -> str:
    position = text.casefold().find(anchor.casefold())
    if position < 0:
        raise ValueError(f"anchor not found: {anchor}")
    return text[position : position + length].strip()


def build_dossier(spec: PaperSpec) -> PaperDossier:
    pdf_path = ROOT / "data" / "literature_pdfs" / "ra0" / spec.filename
    pdf_bytes = pdf_path.read_bytes()
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError(f"not a PDF: {pdf_path}")
    document_id = f"doc-{hashlib.sha256(pdf_bytes).hexdigest()[:24]}"
    reader = PdfReader(pdf_path)
    pages = [_normalise(page.extract_text() or "") for page in reader.pages]
    items: list[DossierItem] = []
    for evidence in spec.evidence:
        page_text = pages[evidence.page - 1]
        items.append(
            DossierItem(
                category=evidence.category,  # type: ignore[arg-type]
                summary=evidence.summary,
                source_quote=_excerpt(page_text, evidence.anchor, evidence.length),
                chunk_id=f"{document_id}-page-{evidence.page:03d}",
                page=evidence.page,
                document_id=document_id,
                source_text_sha256=hashlib.sha256(page_text.encode()).hexdigest(),
                confidence=0.85,
                risk_level="medium",
                llm_extracted=False,
            )
        )
    covered = tuple(dict.fromkeys(item.category for item in items))
    return PaperDossier(
        document_id=document_id,
        title=spec.title,
        extraction_method="manual_pilot",
        review_status="pending",
        items=tuple(items),
        warnings=(
            "RA-0 silver-label draft; requires source-page and domain review",
            (
                "paper evidence must not be mapped to a database structure by "
                "formula alone"
            ),
        ),
        extraction_version="ra0-manual-pilot-v1",
        narrative_summary=spec.summary,
        covered_categories=covered,
        missing_core_categories=(),
        completeness_score=len(covered) / 12,
    )


def main() -> None:
    store = PaperDossierStore(ROOT / "data" / "literature_dossier_candidates")
    for spec in PAPERS:
        dossier = build_dossier(spec)
        output = store.save_pending(dossier)
        print(f"{dossier.document_id}\t{len(dossier.items)}\t{output}")


if __name__ == "__main__":
    main()
