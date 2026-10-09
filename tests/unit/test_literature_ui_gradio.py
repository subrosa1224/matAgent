from __future__ import annotations

import socket
from pathlib import Path

import pytest

from materials_screening import literature_ui_gradio as ui


def test_safe_uploaded_pdf_copies_into_authorized_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "论文 sample.pdf"
    source.write_bytes(b"%PDF-1.7\ncontent")
    target_root = tmp_path / "ingest"
    monkeypatch.setattr(ui, "PDF_ROOT", target_root.resolve())

    target = ui._safe_uploaded_pdf(source)

    assert target.parent == target_root.resolve()
    assert target.read_bytes() == source.read_bytes()


def test_safe_uploaded_pdf_rejects_fake_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fake.pdf"
    source.write_text("not a pdf", encoding="utf-8")
    monkeypatch.setattr(ui, "PDF_ROOT", (tmp_path / "ingest").resolve())

    with pytest.raises(ValueError, match="有效PDF"):
        ui._safe_uploaded_pdf(source)


def test_report_accepts_only_stable_document_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_run(args: list[str], *, timeout: int = 3600) -> str:
        captured.extend(args)
        return "ok"

    monkeypatch.setattr(ui, "_run_literature", fake_run)

    result = ui._report(
        "主题",
        "doc-1234567890abcdef12345678\ninvalid\ndoc-1234567890abcdef12345678",
    )

    assert result == "ok"
    assert captured.count("--document-id") == 1


def test_selection_understands_numbered_papers() -> None:
    assert ui._selection(4, "请深度分析第1、3篇") == [0, 2]
    assert ui._selection(2, "综合这些论文") == [0, 1]


def test_search_prompt_is_normalized_and_material_is_inferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_run(args: list[str], *, timeout: int = 3600) -> str:
        captured.extend(args)
        return "ok"

    monkeypatch.setattr(ui, "_run_literature", fake_run)
    result = ui._search(
        "帮我检索3D打印生物活性玻璃支架孔结构与成骨的近年论文",
        "",
        2020,
        15,
    )

    assert result == "ok"
    assert captured[1] == "3D打印生物活性玻璃支架孔结构与成骨"
    assert captured[-2:] == ["--material", "bioactive glass"]


def test_chat_routes_plain_question_to_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    def fake_search(topic: str, material: str, year: float, limit: float) -> str:
        captured.extend((topic, material, year, limit))
        return "候选论文"

    monkeypatch.setattr(ui, "_search", fake_search)
    updates = list(
        ui._chat(
            "多孔生物陶瓷孔结构与成骨性能",
            [],
            None,
            ui._initial_state(),
            "β-TCP",
            2021,
            12,
        )
    )
    pending_history = updates[0][0]
    history, _trace, state, cleared, uploads = updates[-1]

    assert captured == ["多孔生物陶瓷孔结构与成骨性能", "β-TCP", 2021, 12]
    assert pending_history[0]["content"] == "多孔生物陶瓷孔结构与成骨性能"
    assert pending_history[-1]["content"] == "正在处理，请稍候……"
    assert history[-1]["content"] == "候选论文"
    assert state["last_action"] == "主题文献检索"
    assert cleared == ""
    assert uploads is None


def test_available_port_falls_forward_when_preferred_is_busy() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        preferred = occupied.getsockname()[1]
        selected = ui._available_port(preferred, attempts=2)

    assert selected == preferred + 1


def test_preview_does_not_require_user_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ui, "_prepare_files", lambda files: ([Path("paper.pdf")], None))
    monkeypatch.setattr(ui, "_document_ids_for_paths", lambda paths: [])
    captured: list[str] = []

    def fake_run(args: list[str], *, timeout: int = 3600) -> str:
        captured.extend(args)
        return "ok"

    monkeypatch.setattr(ui, "_run_literature", fake_run)
    output, _ids, _paths = ui._preview("", ["paper.pdf"])

    assert output == "ok"
    assert captured[:3] == [
        "batch-preview",
        "--topic",
        "自动识别论文的研究问题、方法和主要结论",
    ]


def test_uploaded_these_papers_prompt_is_topic_free() -> None:
    assert ui._GENERIC_PDF_PROMPT.fullmatch("预览我上传的这些论文") is not None


def test_internal_preview_placeholders_are_not_user_facing() -> None:
    assert ui._user_facing_summary("正常的实验方法摘要") == "正常的实验方法摘要"
    assert ui._user_facing_summary("该候选摘要未能安全翻译，请人工核对") is None


def test_numeric_preview_placeholders_are_removed_by_sentence() -> None:
    cleaned, trimmed = ui._clean_preview_placeholders(
        "材料具有良好结晶性。薄膜厚度约为[数值待深度分析]毫米。电化学性能稳定。"
    )

    assert trimmed is True
    assert cleaned == "材料具有良好结晶性。电化学性能稳定。"
    assert "数值待深度分析" not in cleaned


def test_user_report_hides_internal_metadata_and_unsafe_inference_sections() -> None:
    raw = (
        "普通用户文献报告\n报告ID：user-lit-1234567890abcdef12345678\n"
        "主题：综合这些论文并生成主题报告\n"
        "## 主题概述\n共同问题。\n"
        "## 共同结论\n三篇研究均证实某现象。\n"
        "## 设计启示\n未经充分验证的设计建议。\n"
        "## 适用边界\n未经验证的临床适用性。\n"
        "证据覆盖\n内部计数"
    )

    result = ui._clean_user_report(raw, paper_count=3)

    assert "报告ID" not in result
    assert "设计启示" not in result
    assert "适用边界" not in result
    assert "证据覆盖" not in result
    assert "跨论文共同观察（综合判断）" in result


def test_structured_report_hides_evidence_risk_counts() -> None:
    payload = {
        "papers": [
            {
                "title": "Paper A",
                "evidence": [],
                "low_risk_count": 2,
                "medium_risk_count": 8,
            }
        ]
    }

    result = ui._render_structured_user_report(payload)

    assert "可靠证据 2 条" not in result
    assert "建议核对 8 条" not in result
    assert "## 证据边界" not in result
    assert "## 引用说明" in result
    assert "横向对比" not in result
    assert "三篇论文" not in result
    assert "生物活性玻璃" not in result
    assert "当前只有1篇论文" in result


def test_report_request_analyzes_all_selected_uploaded_papers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzed: list[list[str]] = []
    reported: list[str] = []
    monkeypatch.setattr(
        ui,
        "_analyze",
        lambda paths: analyzed.append(list(paths)) or "analyzed",
    )
    monkeypatch.setattr(
        ui,
        "_report",
        lambda topic, ids: reported.append(ids) or "report",
    )
    state = ui._initial_state()
    state.update(
        paths=["paper-1.pdf", "paper-2.pdf"],
        document_ids=[
            "doc-111111111111111111111111",
            "doc-222222222222222222222222",
        ],
    )

    updates = list(
        ui._chat(
            "综合这些论文并生成主题报告",
            [],
            None,
            state,
            "",
            2020,
            15,
        )
    )

    assert analyzed == [["paper-1.pdf", "paper-2.pdf"]]
    assert "doc-111111111111111111111111" in reported[0]
    assert "doc-222222222222222222222222" in reported[0]
    assert updates[-1][0][-1]["content"] == "report"
