"""Preview presentation never repairs evidence or invents completed stages."""

import html

from materials_screening.master.user_answer import render_user_answer
from tests.unit.master.test_fulltext_preview import run, setup


def report(
    *,
    progress="已预览 10/10 篇，等待你确认分析范围。",
    ending="已按要求只预览，未启动详细提取。",
):
    return (
        "全文预览（仅用于选文，不是实验统计或专业验证）：\n\n"
        + progress
        + "\n\n"
        + "第1篇：SnO2 thin films\n研究问题：透明导电薄膜\n"
        + "方法：spray pyrolysis\n主要发现（预览）：[数值待深度分析] Ω cm。\n"
        + "原文依据（第9页）：figure of merit 3.16 \x02 10 2 (Ω) 1, 250 \x0eC.\n"
        + "阶段决定：hold；用户要求只预览，未启动详细提取。\n\n"
        + "复用并重新核对了 6 篇当前任务的已保存预览；这些预览未重新请求模型。\n\n"
        + ending
    )


def test_preview_main_shows_total_not_last_batch_cache_count():
    answer = render_user_answer(report())
    main = answer.split("<details>")[0]
    assert "已预览 10/10 篇" in main
    assert "等待你确认" in main
    assert "复用" not in main and "6 篇" not in main
    assert "hold" not in main and "阶段决定" not in main
    assert "仅预览" in main or "等待确认" in main
    assert "不是实验统计或专业验证" in main
    assert "原始预览与处理详情" in answer
    assert "6 篇" in answer.split("<details>")[1]


def test_controls_are_marked_not_guessed_and_audit_is_reversible():
    raw = report()
    answer = render_user_answer(raw)
    main, audit = answer.split("<details>", 1)
    assert "[符号识别待核对]" in main
    assert "\x02" not in answer and "\x0e" not in answer
    assert "250 [符号识别待核对]C" in main
    assert "250 °C" not in main
    assert "3.16 ×" not in main
    assert "[数值待深度分析]" in main and "尚未核验" in main
    assert r"\u0002" in audit and r"\u000e" in audit
    assert raw == report()


def test_partial_failure_does_not_become_complete_confirmation():
    raw = report(
        progress="已预览 6/10 篇。", ending="预览服务执行失败；已完成部分保留。"
    )
    main = render_user_answer(raw).split("<details>")[0]
    assert "6/10" in main and "执行失败" in main
    assert "10/10" not in main and "等待你确认" not in main


def test_old_report_without_denominator_does_not_invent_completion():
    raw = report(progress="")
    main = render_user_answer(raw).split("<details>")[0]
    assert "10/10" not in main and "6/6" not in main
    assert "复用" not in main


def test_audit_escapes_html_and_renderer_is_idempotent():
    raw = report().replace("SnO2 thin films", "SnO2 <script>not executable</script>")
    answer = render_user_answer(raw)
    audit = answer.split("<details>")[1]
    assert html.escape("<script>not executable</script>") in audit
    assert "<script>" not in answer
    assert render_user_answer(answer) == answer


def test_actual_batch_progress_uses_validated_selected_documents(tmp_path):
    env = setup(tmp_path, 10)
    env.task = env.task.model_copy(update={"user_instructions": ("只预览",)})
    result = run(env, budget=2)
    assert "已预览 2/10 篇" in result.response_text
    assert "10/10" not in render_user_answer(result.response_text)
    assert env.llm.calls == result.model_call_count == 2


def test_selected_subset_has_its_own_total_without_changing_choice(tmp_path):
    env = setup(tmp_path, 10)
    task = env.task.model_copy(update={"user_instructions": ("只预览第2篇",)})
    result = run(env, task)
    assert "已预览 1/1 篇" in result.response_text
    assert "本次选定 1 篇" in result.response_text
    assert env.saved[-1].preview_decisions[task.document_ids[1]].action == "hold"
    assert env.llm.calls == 1


def test_rendering_saved_result_does_not_mutate_evidence_or_call_model(tmp_path):
    env = setup(tmp_path)
    result = run(env, env.task.model_copy(update={"user_instructions": ("只预览",)}))
    task = env.saved[-1]
    original = task.model_dump_json()
    raw = result.response_text
    answer = render_user_answer(raw)
    assert answer != raw
    assert task.model_dump_json() == original
    assert result.response_text == raw
    assert env.llm.calls == 1
