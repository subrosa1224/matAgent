"""Internal execution diagnostics are folded, not scientific limitations."""

from materials_screening.master.user_answer import render_user_answer


def test_partial_report_has_clear_user_message_and_folded_diagnostics():
    raw = (
        "这是部分执行报告，不是完整链路成功报告。\n\n调用计数为 1/3；交接条件不足。\n\n"
        "## 数据库初筛：已返回结果\n\n| mp-1 | ZnO |\n\n"
        "## 数据分析：未执行\n\n没有本轮返回结果，不等于零条命中。\n\n"
        "## 文献检索：未执行\n\n没有本轮返回结果。\n\n"
        "## 结论边界与下一步\n\n查询快照仍保留。"
    )
    rendered = render_user_answer(raw)
    main = rendered.split("<details>")[0]
    assert "尚未完成" in main and "ZnO" in main
    assert "调用计数" not in main and "交接条件" not in main
    assert "实验" in main and "文献检索" in main
    assert "<summary>技术详情</summary>" in rendered
    assert "1/3" in rendered and "<details open" not in rendered


def test_other_answers_are_not_rewritten():
    assert render_user_answer("文献预览，等待确认。") == "文献预览，等待确认。"


def test_combined_report_keeps_numbers_and_papers_not_internal_ids():
    raw = (
        "## 1. Materials Project 初筛\n\n查询快照：query-a\n| mp-1 | ZnO | 0.7 |\n\n"
        "## 3. 候选数据分析\n\n数据集：`dataset-x`；分析：`analysis-x`。\n"
        "band_gap_ev 均值 0.7\n\n## 4. 目标应用文献检索与证据缺口\n\n"
        "论文题名，DOI 10.1234/example，尚未核验全文。\n\n## 5. 结论边界与下一步\n"
        "快照最多保存1000条。"
    )
    answer = render_user_answer(raw)
    main = answer.split("<details>")[0]
    assert "0.7" in main and "10.1234/example" in main
    assert "query-a" not in main and "dataset-x" not in main
    assert "尚未核验全文" in main and "确认后再详细分析" in main
