"""Local Gradio UI for the ordinary-user LiteratureAgent workflow."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr

PDF_ROOT = Path("data/literature_pdfs").resolve()

_MATERIAL_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("生物活性玻璃", "bioactive glass"), "bioactive glass"),
    (("磷酸钙", "calcium phosphate"), "calcium phosphate"),
    (("生物陶瓷", "bioceramic"), "bioceramic"),
    (("二氧化钛", "tio2", "titanium dioxide"), "TiO2"),
)

CSS = """
footer { display: none !important; }
body, .gradio-container { background: #f4f7fb !important; }
.gradio-container { max-width: 1560px !important; margin: 0 auto !important;
  padding: 12px !important;
  font-family: "Segoe UI", "Microsoft YaHei", sans-serif !important; }
#lit-hero { background: linear-gradient(120deg, #12233f, #1c3d68); color: #fff;
  border-radius: 14px; padding: 16px 20px; margin-bottom: 12px;
  box-shadow: 0 12px 30px rgba(18, 35, 63, .18); }
#lit-hero h1 { margin: 0 0 4px; font-size: 21px; }
#lit-hero p { margin: 0; color: #c8d8ee; font-size: 13px; }
.lit-main { align-items: stretch !important; gap: 12px !important; }
.agent-panel { background: #fff !important; border: 1px solid #dce4ef !important;
  border-radius: 14px !important; padding: 12px !important;
  box-shadow: 0 5px 18px rgba(20, 38, 66, .06) !important; }
#literature-chat { min-height: 500px !important; }
.bubble-wrap .message-row.bubble.user-row { justify-content: flex-end; }
.bubble-wrap .message-row.bubble.bot-row { justify-content: flex-start; }
.bubble-wrap .user.message { background: #2347b8 !important; border: none !important;
  border-radius: 12px 12px 4px 12px !important; }
.bubble-wrap .user.message, .bubble-wrap .user.message * { color: #fff !important; }
.bubble-wrap .bot.message { background: #eef3fa !important; color: #17243a !important;
  border: 1px solid #dce4ef !important; border-radius: 12px 12px 12px 4px !important;
  width: 100% !important; max-width: 96% !important; }
.quick-chip { border-radius: 999px !important; border: 1px solid #cfdaea !important;
  background: #f7f9fc !important; color: #34445e !important;
  font-size: 12px !important; }
.quick-chip:hover { border-color: #3157d5 !important; color: #3157d5 !important; }
button.primary { background: #3157d5 !important; border-color: #3157d5 !important; }
@media (max-width: 900px) { .lit-main { flex-wrap: wrap !important; }
  .agent-panel { min-width: 100% !important; } }
"""

QUICK_PROMPTS = (
    ("主题检索", "帮我检索3D打印生物活性玻璃支架孔结构与成骨的近年论文"),
    ("快速阅读", "预览我上传的这些论文"),
    ("深度分析", "深度分析第1篇论文"),
    ("多文献综合", "综合这些论文并生成主题报告"),
)

DOSSIER_LABELS = {
    "research_problem": "研究问题",
    "innovation": "核心创新",
    "materials": "材料体系",
    "preparation": "制备与实验方法",
    "device_fabrication": "样品制备",
    "characterization": "表征方法",
    "structural_result": "结构结果",
    "optical_result": "光学结果",
    "performance_result": "性能结果",
    "mechanism": "作用机理",
    "limitations": "局限性",
    "reproducibility": "复现要点",
}

_PDF_ANALYSIS_PATTERN = re.compile(
    r"深度分析|详细分析|分析第|分析这|分析这些|分析(?:一下)?(?:论文|文献)|"
    r"解读|概括|总结这",
    re.I,
)
_GENERIC_PDF_PROMPT = re.compile(
    r"^(?:请|帮我|麻烦)?(?:预览|快速阅读|分析|解读|概括|总结)?"
    r"(?:一下)?(?:我上传的这些|我上传的|上传的这些|上传的|这些|这篇|这批)?"
    r"(?:论文|文献|pdf)?[。！!？?]*$",
    re.I,
)
_INTERNAL_PLACEHOLDERS = (
    "未能安全翻译",
    "模型未能",
    "自动回退",
    "请根据下方原文",
    "请人工核对",
    "候选摘要",
)


def _user_facing_summary(value: str) -> str | None:
    summary = " ".join(value.split()).strip()
    if not summary or any(marker in summary for marker in _INTERNAL_PLACEHOLDERS):
        return None
    return summary


def _run_literature(args: list[str], *, timeout: int = 3600) -> str:
    command = [
        sys.executable,
        "-m",
        "materials_screening.cli",
        "literature",
        *args,
    ]
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        env=os.environ.copy(),
    )
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if completed.returncode:
        return f"操作未完成（退出码 {completed.returncode}）\n\n{output}"
    return output or "操作已完成。"


def _normalize_search_prompt(message: str) -> str:
    topic = " ".join(message.strip().split())
    topic = re.sub(
        r"^(?:请|麻烦)?(?:帮我|为我)?(?:检索|搜索|查找|找)(?:一下)?",
        "",
        topic,
    ).strip(" ：:，,")
    topic = re.sub(r"(?:的)?(?:近年|近期|最新)(?:相关)?论文$", "", topic).strip()
    topic = re.sub(r"(?:相关)?(?:文献|论文)$", "", topic).strip()
    return topic or message.strip()


def _inferred_materials(topic: str) -> tuple[str, ...]:
    folded = topic.casefold()
    return tuple(
        value
        for triggers, value in _MATERIAL_HINTS
        if any(trigger in folded for trigger in triggers)
    )


def _search(topic: str, material: str, year_from: float, limit: float) -> str:
    normalized_topic = _normalize_search_prompt(topic)
    if not normalized_topic:
        return "请输入检索主题。"
    args = [
        "search",
        normalized_topic,
        "--year-from",
        str(int(year_from)),
        "--max-papers",
        str(int(limit)),
        "--sort",
        "balanced",
        "--show-abstract",
    ]
    explicit_materials = tuple(
        item.strip() for item in re.split(r"[,，]", material) if item.strip()
    )
    materials = (*explicit_materials, *_inferred_materials(normalized_topic))
    for value in dict.fromkeys(materials):
        if value:
            args.extend(("--material", value))
    return _run_literature(args, timeout=180)


def _safe_uploaded_pdf(source: str | Path) -> Path:
    path = Path(source).resolve()
    if path.suffix.casefold() != ".pdf" or not path.is_file():
        raise ValueError("只支持PDF文件。")
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("单个PDF不能超过100 MB。")
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError("文件不是有效PDF。")
    PDF_ROOT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._") or "paper"
    target = (PDF_ROOT / f"{stem[:80]}-{digest}.pdf").resolve()
    if target.parent != PDF_ROOT:
        raise ValueError("PDF文件名无效。")
    if not target.exists():
        shutil.copy2(path, target)
    return target


def _prepare_files(files: list[Any] | None) -> tuple[list[Path], str | None]:
    if not files:
        return [], "请至少上传一篇PDF。"
    paths: list[Path] = []
    try:
        for item in files:
            source = item if isinstance(item, (str, Path)) else item.name
            paths.append(_safe_uploaded_pdf(source))
    except (OSError, ValueError) as exc:
        return [], str(exc)
    return paths, None


def _preview(topic: str, files: list[Any] | None) -> tuple[str, str, list[str]]:
    effective_topic = topic.strip() or "自动识别论文的研究问题、方法和主要结论"
    paths, error = _prepare_files(files)
    if error:
        return error, "", []
    args = ["batch-preview", "--topic", effective_topic]
    for path in paths:
        args.extend(("--pdf", str(path)))
    raw_output = _run_literature(args)
    if raw_output.startswith("操作未完成"):
        return raw_output, "", [str(path) for path in paths]
    document_ids = _document_ids_for_paths([str(path) for path in paths])
    rendered = _render_previews(
        document_ids,
        effective_topic,
        show_relevance=bool(topic.strip()),
    )
    return (
        rendered or raw_output,
        "\n".join(document_ids),
        [str(path) for path in paths],
    )


def _render_previews(
    document_ids: list[str], topic: str, *, show_relevance: bool
) -> str:
    from materials_screening.sub_agents.literature.dossier import PaperDossierStore
    from materials_screening.sub_agents.literature.preview import PaperPreviewStore

    preview_store = PaperPreviewStore(Path("data/literature_previews"))
    approved = PaperDossierStore(Path("data/literature_dossiers"))
    pending = PaperDossierStore(Path("data/literature_dossier_candidates"))
    relevance_labels = {
        "core": "核心相关",
        "high": "高度相关",
        "extended": "扩展阅读",
        "low": "低相关",
    }
    sections: list[str] = []
    for index, document_id in enumerate(document_ids, 1):
        preview = preview_store.load(document_id, topic)
        if preview is None:
            continue
        lines = [f"### {index}. {preview.title}"]
        if show_relevance:
            relevance = relevance_labels[preview.topic_relevance]
            lines.append(f"**与主题的关系：** {relevance}")
        preview_text = " ".join(
            (preview.research_question, preview.methods, preview.key_findings)
        )
        fallback = preview.evidence_quality == "fallback_chunk" or any(
            marker in preview_text
            for marker in ("模型未能", "自动回退", "未能安全翻译")
        )
        if fallback:
            dossier = approved.load(document_id) or pending.load(document_id)
            if dossier is None:
                lines.append(
                    "本篇快速摘要未通过证据校验，暂不自动概括结论；"
                    "可以继续进行深度分析。"
                )
            else:
                grouped: dict[str, list[Any]] = {}
                for item in dossier.items:
                    grouped.setdefault(item.category, []).append(item)
                mapping = (
                    ("research_problem", "研究内容"),
                    ("preparation", "主要方法"),
                    ("performance_result", "主要发现"),
                )
                for category, label in mapping:
                    items = grouped.get(category, ())
                    summary = next(
                        (
                            clean
                            for item in items
                            if (clean := _user_facing_summary(item.summary)) is not None
                        ),
                        None,
                    )
                    if summary:
                        lines.append(f"**{label}：** {summary}")
                    elif category == "preparation":
                        lines.append(
                            "**主要方法：** 快速预览阶段尚未获得足够可靠的"
                            "方法信息，可在深度分析时补充。"
                        )
                lines.append(
                    "**阅读建议：** 本条概览由已有证据档案生成；若需要引用"
                    "具体数值或实验条件，建议继续深度分析并核对原文。"
                )
        else:
            research, _ = _clean_preview_placeholders(preview.research_question)
            methods, _ = _clean_preview_placeholders(preview.methods)
            findings, findings_trimmed = _clean_preview_placeholders(
                preview.key_findings
            )
            if findings_trimmed:
                findings = f"{findings} 具体定量结果需在深度分析中核对。".strip()
            lines.extend(
                (
                    f"**研究内容：** {research or '快速预览阶段尚未稳定提取。'}",
                    f"**主要方法：** {methods or '快速预览阶段尚未稳定提取。'}",
                    f"**初步发现：** {findings or '具体结论需在深度分析中核对。'}",
                    "**阅读建议：** "
                    + (
                        "建议优先深读，进一步提取关键实验条件和定量结果。"
                        if preview.recommendation == "deep_analyze"
                        else "可作为背景材料，是否深读取决于当前研究目标。"
                        if preview.recommendation == "background_only"
                        else "目前不建议优先深读。"
                    ),
                )
            )
        sections.append("\n\n".join(lines))
    if not sections:
        return ""
    intro = f"已快速阅读 {len(sections)} 篇论文。"
    if not show_relevance:
        intro += "本次未设置统一主题，因此仅逐篇概括，不进行相关性排名。"
    return intro + "\n\n" + "\n\n---\n\n".join(sections)


def _clean_preview_placeholders(text: str) -> tuple[str, bool]:
    """Remove sentences containing internal unresolved-number placeholders."""

    marker = "[数值待深度分析]"
    if marker not in text:
        return text.strip(), False
    sentences = re.split(r"(?<=[。！？.!?])\s*", text.strip())
    clean = "".join(sentence for sentence in sentences if marker not in sentence)
    return clean.strip(), True


def _analyze(paths: list[str] | None) -> str:
    if not paths:
        return "请先在“上传与预览”中上传PDF。"
    args = ["batch-analyze"]
    for path in paths:
        args.extend(("--pdf", path))
    raw_output = _run_literature(args)
    if raw_output.startswith("操作未完成"):
        return raw_output
    document_ids = list(dict.fromkeys(re.findall(r"doc-[0-9a-f]{24}", raw_output)))
    rendered = _render_analyzed_dossiers(document_ids)
    return rendered or "深度分析已完成，但暂未生成可展示的论文档案。"


def _document_ids_for_paths(paths: list[str]) -> list[str]:
    from materials_screening.sub_agents.literature.batch import document_id_for_pdf

    return [document_id_for_pdf(Path(path)) for path in paths]


def _render_analyzed_dossiers(document_ids: list[str]) -> str:
    from materials_screening.sub_agents.literature.dossier import PaperDossierStore

    approved = PaperDossierStore(Path("data/literature_dossiers"))
    pending = PaperDossierStore(Path("data/literature_dossier_candidates"))
    sections: list[str] = []
    for index, document_id in enumerate(document_ids, 1):
        dossier = approved.load(document_id) or pending.load(document_id)
        if dossier is None:
            continue
        lines = [f"## {index}. {dossier.title}"]
        grouped: dict[str, list[Any]] = {}
        for item in dossier.items:
            grouped.setdefault(item.category, []).append(item)
        for category, label in DOSSIER_LABELS.items():
            items = grouped.get(category, ())
            if not items:
                continue
            summaries: list[str] = []
            seen: set[str] = set()
            for item in items:
                summary = _user_facing_summary(item.summary)
                if summary is None or summary in seen:
                    continue
                seen.add(summary)
                summaries.append(f"{summary}（原文第 {item.page} 页）")
                if len(summaries) == 2:
                    break
            if summaries:
                lines.extend(("", f"### {label}"))
                lines.extend(f"- {summary}" for summary in summaries)
        coverage = (
            f"{dossier.completeness_score:.0%}"
            if dossier.completeness_score is not None
            else "未计算"
        )
        lines.extend(
            (
                "",
                "### 阅读与引用说明",
                f"本次分析的内容覆盖度为 {coverage}。以上要点均标注了原文页码；"
                "如果准备在报告或研究中引用具体数值、实验条件或因果结论，"
                "建议打开对应页码复核原文。",
            )
        )
        sections.append("\n".join(lines))
    if not sections:
        return ""
    return "已完成所选论文的深度阅读。\n\n" + "\n\n---\n\n".join(sections)


def _report(topic: str, document_ids: str) -> str:
    selected = list(dict.fromkeys(re.findall(r"doc-[0-9a-f]{24}", document_ids)))
    if not topic.strip() or not selected:
        return "请输入主题，并从预览结果带入至少一个文档ID。"
    args = ["user-report", "--topic", topic.strip()]
    for document_id in selected:
        args.extend(("--document-id", document_id))
    raw_output = _run_literature(args)
    if raw_output.startswith("操作未完成"):
        return raw_output
    return _clean_user_report(raw_output, paper_count=len(selected))


def _clean_user_report(raw_output: str, *, paper_count: int) -> str:
    payload = _load_user_report_payload(raw_output)
    if payload is not None:
        return _render_structured_user_report(payload)
    start = raw_output.find("## 主题概述")
    if start < 0:
        return raw_output
    body = raw_output[start:]
    end_markers = ("\n证据覆盖", "\n详细证据与报告JSON已保存")
    end_positions = [body.find(marker) for marker in end_markers]
    valid_ends = [position for position in end_positions if position >= 0]
    if valid_ends:
        body = body[: min(valid_ends)]
    body = re.sub(
        r"\n## 设计启示\n.*?(?=\n## |\Z)",
        "",
        body,
        flags=re.S,
    )
    body = re.sub(
        r"\n## 适用边界\n.*?(?=\n## |\Z)",
        "",
        body,
        flags=re.S,
    )
    body = body.replace("## 共同结论", "## 跨论文共同观察（综合判断）")
    body = body.replace("## 关键差异", "## 跨论文差异（综合判断）")
    body = body.replace("三篇研究均证实", "综合现有证据可以观察到")
    body = body.replace("所有研究均强调", "这些研究共同关注")
    evidence_map = _report_evidence_pages(raw_output)

    def replace_evidence(match: re.Match[str]) -> str:
        references = []
        for evidence_id in re.findall(r"E-[0-9a-f]+", match.group(0)):
            reference = evidence_map.get(evidence_id)
            if reference and reference not in references:
                references.append(reference)
        return f"（证据：{'；'.join(references)}）" if references else ""

    body = re.sub(r"\s*\[(?:E-[0-9a-f]+(?:,\s*)?)+\]", replace_evidence, body)
    return (
        f"已根据 {paper_count} 篇论文生成综合报告。\n\n"
        "下列“共同观察”和“差异”是基于已提取证据的跨论文归纳，"
        "不等同于论文作者的直接结论。\n\n" + body.strip()
    )


def _report_evidence_pages(raw_output: str) -> dict[str, str]:
    payload = _load_user_report_payload(raw_output)
    if payload is None:
        return {}
    mapping: dict[str, str] = {}
    for paper_index, paper in enumerate(payload.get("papers", ()), 1):
        for evidence in paper.get("evidence", ()):
            evidence_id = evidence.get("evidence_id")
            page = evidence.get("page")
            if isinstance(evidence_id, str) and isinstance(page, int):
                mapping[evidence_id] = f"论文{paper_index}第{page}页"
    return mapping


def _load_user_report_payload(raw_output: str) -> dict[str, Any] | None:
    match = re.search(r"报告ID：(user-lit-[0-9a-f]{24})", raw_output)
    if match is None:
        return None
    path = Path("data/literature_user_reports") / f"{match.group(1)}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _render_structured_user_report(payload: dict[str, Any]) -> str:
    papers = payload.get("papers", ())
    if not isinstance(papers, list) or not papers:
        return "当前没有足够的论文证据用于综合。"
    lines = [
        f"已根据 {len(papers)} 篇论文生成结构化综合报告。",
        "",
        "报告先分别概括每篇论文，再进行横向比较。跨论文部分只总结"
        "已提取证据共同覆盖的维度，不扩展临床适用性。",
        "",
        "## 纳入论文",
    ]
    for index, paper in enumerate(papers, 1):
        lines.append(f"- **论文{index}：** {paper.get('title', '题名未知')}")
    lines.extend(("", "## 逐篇证据归纳"))
    section_categories = (
        ("研究目标", ("research_problem",)),
        ("材料与结构设计", ("materials", "structural_result")),
        ("制备与评价方法", ("preparation", "device_fabrication", "characterization")),
        ("主要结果", ("performance_result", "optical_result")),
        ("作用机理", ("mechanism",)),
        ("论文陈述的局限或边界", ("limitations",)),
    )
    for index, paper in enumerate(papers, 1):
        lines.extend(("", f"### 论文{index}：{paper.get('title', '题名未知')}"))
        evidence = paper.get("evidence", ())
        for label, categories in section_categories:
            selected = _select_report_evidence(evidence, categories, limit=2)
            if not selected:
                continue
            lines.extend(("", f"**{label}**"))
            for item in selected:
                risk = "可靠" if item.get("risk_level") == "low" else "建议核对"
                lines.append(
                    f"- {item.get('summary', '')}"
                    f"（论文{index}第{item.get('page', '?')}页，{risk}）"
                )
    if len(papers) > 1:
        lines.extend(("", "## 横向对比"))
        lines.append("| 论文 | 材料/结构 | 制备路线 | 主要结果 | 已提取边界 |")
        lines.append("|---|---|---|---|---|")
        for index, paper in enumerate(papers, 1):
            evidence = paper.get("evidence", ())
            cells = (
                _compact_report_cell(evidence, ("materials", "structural_result")),
                _compact_report_cell(evidence, ("preparation", "device_fabrication")),
                _compact_report_cell(evidence, ("performance_result", "mechanism")),
                _compact_report_cell(evidence, ("limitations",)),
            )
            lines.append(f"| 论文{index} | " + " | ".join(cells) + " |")
        lines.extend(
            (
                "",
                "## 跨论文观察（综合判断）",
                f"- 本报告实际纳入 {len(papers)} 篇论文；共同点和差异只依据"
                "逐篇证据和上表归纳。",
                "- 若材料体系、实验条件或评价指标不同，不进行未经统一口径的"
                "数值优劣排序。",
            )
        )
    else:
        lines.extend(
            (
                "",
                "## 比较范围说明",
                "- 当前只有1篇论文具备可用证据，因此不生成跨论文共同结论或横向差异。",
            )
        )
    lines.extend(
        (
            "",
            "## 引用说明",
            "本报告基于论文原文证据生成。引用具体数值、实验条件或因果结论时，"
            "请根据标注页码核对原文。",
        )
    )
    return "\n".join(lines)


def _select_report_evidence(
    evidence: Any, categories: tuple[str, ...], *, limit: int
) -> list[dict[str, Any]]:
    if not isinstance(evidence, list):
        return []
    selected: list[dict[str, Any]] = []
    for category in categories:
        for item in evidence:
            if item.get("category") != category:
                continue
            summary = _user_facing_summary(str(item.get("summary", "")))
            if summary is None:
                continue
            selected.append({**item, "summary": summary})
            if len(selected) >= limit:
                return selected
    return selected


def _compact_report_cell(evidence: Any, categories: tuple[str, ...]) -> str:
    selected = _select_report_evidence(evidence, categories, limit=1)
    if not selected:
        return "当前证据未覆盖"
    summary = str(selected[0]["summary"]).replace("|", "／")
    return summary if len(summary) <= 90 else summary[:87].rstrip() + "…"


def _initial_state() -> dict[str, Any]:
    return {"topic": "", "paths": [], "document_ids": [], "last_action": "等待提问"}


def _trace(state: dict[str, Any]) -> str:
    topic = state.get("topic") or "尚未设置"
    paths = state.get("paths") or []
    document_ids = state.get("document_ids") or []
    return (
        "### 当前对话上下文\n\n"
        f"- **主题**：{topic}\n"
        f"- **已上传PDF**：{len(paths)} 篇\n"
        f"- **已识别文档**：{len(document_ids)} 篇\n"
        f"- **最近动作**：{state.get('last_action', '等待提问')}\n\n"
        "文档ID：\n" + ("\n".join(f"- `{item}`" for item in document_ids) or "- 暂无")
    )


def _selection(total: int, message: str) -> list[int]:
    requested = {
        int(value) - 1
        for value in re.findall(r"(?:第\s*)?(\d+)\s*篇?", message)
        if 1 <= int(value) <= total
    }
    return sorted(requested) if requested else list(range(total))


def _chat(
    message: str,
    history: list[dict[str, Any]] | None,
    files: list[Any] | None,
    state: dict[str, Any] | None,
    material: str,
    year_from: float,
    limit: float,
) -> Iterator[tuple[list[dict[str, Any]], str, dict[str, Any], str, None]]:
    current: dict[str, Any] = dict(state or _initial_state())
    conversation = list(history or [])
    prompt = message.strip()
    if not prompt and not files:
        yield conversation, _trace(current), current, "", None
        return
    display = prompt or "请预览我上传的论文。"
    if files:
        display += f"\n\n（上传 {len(files)} 篇PDF）"
    conversation.append({"role": "user", "content": display})
    pending = [
        *conversation,
        {"role": "assistant", "content": "正在处理，请稍候……"},
    ]
    yield pending, _trace(current), current, "", None

    if files:
        paths, error = _prepare_files(files)
        if error:
            answer = error
        else:
            current["paths"] = [str(path) for path in paths]
            is_generic_action = bool(_GENERIC_PDF_PROMPT.fullmatch(prompt))
            if prompt and not is_generic_action:
                current["topic"] = prompt
            if _PDF_ANALYSIS_PATTERN.search(prompt):
                saved_paths = [str(path) for path in paths]
                answer = _analyze(saved_paths)
                current.update(
                    paths=saved_paths,
                    document_ids=_document_ids_for_paths(saved_paths),
                    last_action=f"深度分析 {len(saved_paths)} 篇论文",
                )
            else:
                topic = "" if is_generic_action else current.get("topic") or ""
                output, ids, saved_paths = _preview(topic, files)
                current.update(
                    paths=saved_paths,
                    document_ids=ids.splitlines() if ids else [],
                    last_action="批量快速预览",
                )
                answer = output
    elif re.search(r"帮助|怎么用|能做什么", prompt):
        answer = (
            "你可以直接说：\n\n"
            "- 帮我检索‘多孔生物陶瓷孔结构与成骨性能’的近年论文；\n"
            "- 上传PDF后说‘预览这些论文’；\n"
            "- 说‘深度分析第1、3篇’；\n"
            "- 说‘综合这些论文并生成主题报告’。"
        )
        current["last_action"] = "使用帮助"
    elif re.search(r"综合|报告|对比|总结", prompt):
        report_ids: list[str] = list(current.get("document_ids") or [])
        selected = _selection(len(report_ids), prompt)
        chosen_ids = [report_ids[index] for index in selected]
        if not chosen_ids:
            answer = "当前对话还没有可综合的文档。请先上传并预览PDF。"
        else:
            available_paths = list(current.get("paths") or [])
            chosen_paths = [
                available_paths[index]
                for index in selected
                if index < len(available_paths)
            ]
            if chosen_paths:
                _analyze(chosen_paths)
            topic = current.get("topic") or ("上传论文的研究问题、方法、主要发现及差异")
            answer = _report(topic, "\n".join(chosen_ids))
            current["last_action"] = f"综合 {len(chosen_ids)} 篇论文"
    elif _PDF_ANALYSIS_PATTERN.search(prompt):
        analysis_paths: list[str] = list(current.get("paths") or [])
        selected = _selection(len(analysis_paths), prompt)
        chosen_paths = [analysis_paths[index] for index in selected]
        if not chosen_paths:
            answer = "当前对话还没有已上传的PDF。"
        else:
            answer = _analyze(chosen_paths)
            current["last_action"] = f"深度分析 {len(chosen_paths)} 篇论文"
    elif re.search(r"预览|快速阅读", prompt):
        preview_paths: list[str] = list(current.get("paths") or [])
        if not preview_paths:
            answer = "请先通过左侧上传PDF。"
        else:
            topic = current.get("topic") or prompt
            args = ["batch-preview", "--topic", topic]
            for path in preview_paths:
                args.extend(("--pdf", path))
            answer = _run_literature(args)
            preview_ids = list(dict.fromkeys(re.findall(r"doc-[0-9a-f]{24}", answer)))
            current.update(document_ids=preview_ids, last_action="重新快速预览")
    else:
        current["topic"] = prompt
        answer = _search(prompt, material, year_from, limit)
        current["last_action"] = "主题文献检索"

    conversation.append({"role": "assistant", "content": answer})
    yield conversation, _trace(current), current, "", None


def _reset_chat() -> tuple[list[Any], str, dict[str, Any], str, None]:
    state = _initial_state()
    return [], _trace(state), state, "", None


def _available_port(preferred: int, *, attempts: int = 20) -> int:
    """Return the preferred local port or the next available one."""
    for port in range(preferred, min(preferred + attempts, 65536)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError(f"端口 {preferred}-{min(preferred + attempts - 1, 65535)} 均被占用。")


def _create_ui() -> gr.Blocks:
    demo = gr.Blocks(title="LiteratureAgent 文献工作台")
    with demo:
        gr.HTML(
            "<section id='lit-hero'><h1>LiteratureAgent 文献工作台</h1>"
            "<p>像与研究助理对话一样完成检索、选文、分析和证据综合</p></section>"
        )
        session_state = gr.State(value=_initial_state())
        with gr.Row(elem_classes=["lit-main"]):
            with gr.Column(scale=2, min_width=250, elem_classes=["agent-panel"]):
                gr.Markdown("### 运行配置")
                gr.Markdown("✅ **LiteratureAgent 已就绪**")
                material = gr.Textbox(
                    label="材料硬约束（可选）",
                    placeholder="bioactive glass scaffold",
                )
                with gr.Row():
                    year_from = gr.Number(value=2020, label="起始年份")
                    limit = gr.Slider(5, 30, value=15, step=1, label="候选数")
                pdf_files = gr.File(
                    label="PDF附件",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath",
                )
                reset_button = gr.Button("新建对话")
                gr.Markdown(
                    "**能力范围**  文献检索 · PDF快速阅读 · 深度分析 · 多文献综合"
                )
            with gr.Column(scale=6, min_width=600, elem_classes=["agent-panel"]):
                chatbot = gr.Chatbot(
                    label="文献知识对话",
                    height=500,
                    layout="bubble",
                    elem_id="literature-chat",
                )
                with gr.Row():
                    message = gr.Textbox(
                        label="",
                        placeholder="例如：帮我检索多孔生物陶瓷孔结构与成骨性能的近年论文",
                        lines=2,
                        scale=6,
                    )
                    send_button = gr.Button("发送", variant="primary", scale=1)
                gr.Markdown("**功能演示命令**")
                quick_buttons: list[tuple[gr.Button, str]] = []
                with gr.Row():
                    for label, prompt in QUICK_PROMPTS:
                        quick_buttons.append(
                            (
                                gr.Button(
                                    label, size="sm", elem_classes=["quick-chip"]
                                ),
                                prompt,
                            )
                        )
            with gr.Column(scale=2, min_width=260, elem_classes=["agent-panel"]):
                gr.Markdown("### 子 Agent 执行轨迹")
                context = gr.Markdown(_trace(_initial_state()))
                gr.Markdown(
                    "### 工作方式\n\n"
                    "- 检索结果来自OpenAlex与Semantic Scholar；\n"
                    "- 上传后先快速预览，再按你的指令深度分析；\n"
                    "- 普通模式跳过实验矩阵；\n"
                    "- 综合报告保留证据编号并隔离高风险内容。"
                )

        inputs = [
            message,
            chatbot,
            pdf_files,
            session_state,
            material,
            year_from,
            limit,
        ]
        outputs = [chatbot, context, session_state, message, pdf_files]
        send_button.click(_chat, inputs, outputs)
        message.submit(_chat, inputs, outputs)
        reset_button.click(
            _reset_chat,
            None,
            [chatbot, context, session_state, message, pdf_files],
        )
        for button, prompt in quick_buttons:
            button.click(lambda value=prompt: value, None, message)
    return demo


def main() -> None:
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "8503"))
    port = _available_port(preferred_port)
    if port != preferred_port:
        print(f"端口 {preferred_port} 已被占用，自动改用 http://127.0.0.1:{port}")
    _create_ui().launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        css=CSS,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )


if __name__ == "__main__":
    main()
