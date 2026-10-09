"""Optional pre-analysis gate and escaped, text-only report appendix."""

import html
import re

from materials_screening.agent.models import AgentResult

from .fulltext_tasks import FulltextTask


class FigureReviewGate:
    def __init__(self, service):
        self.service = service

    def before_analysis(self, task, *, save_task, user_turn_id, model_calls):
        if task.figure_review_policy == "disabled":
            return task, None
        task = FulltextTask.model_validate(
            {
                **task.model_dump(),
                "stage": "awaiting_figure_review",
                "resume_stage": None,
            }
        )
        ref = self.service.start(task, conversation_id=task.conversation_id)
        task = FulltextTask.model_validate(
            {**task.model_dump(), "figure_evidence_ref": ref}
        )
        save_task(task)
        if self.service.store.load(ref).status == "closed":
            return task, None
        return task, AgentResult(
            conversation_id=task.conversation_id,
            user_turn_id=user_turn_id,
            status="completed",
            final_status="needs_user_input",
            model_call_count=model_calls,
            selected_tools=("literature",),
            response_text=(
                "全文提取结果已保存，尚未启动本任务统计。请打开图证据核对入口，"
                "选择论文、页码和区域，核对文字，或明确跳过后继续。\n"
                "文字核对不等于实验条件确认；普通“继续”不会批准图证据。\n任务："
                + task.task_id
            ),
        )


def _plain(text):
    text = html.escape(str(text), quote=True).replace("\n", " ").replace("\r", " ")
    return re.sub(r"([\\`*_{}\[\]()#!|>~])", r"\\\1", text)


def render_figure_appendix(batch):
    lines = [
        "",
        "## 图中文字核对",
        "",
        "图中文字仅作为人工核对的证据附录。实验条件尚未绑定；未改变测量、统计或审核状态。",
    ]
    if not batch.candidates:
        lines.append("本批次未选择图证据，已明确结束核对；不是论文没有图数据。")
    for candidate in batch.candidates:
        lines.extend(
            [
                "",
                f"- {_plain(candidate.figure_label)}：第 {candidate.page} 页；"
                f"文字状态 {_plain(candidate.text_review_status)}；"
                f"OCR {_plain(candidate.ocr.status)}。",
                f"  文档：{candidate.document_id}；"
                f"PDF SHA256：{candidate.pdf_sha256}。",
                f"  人工选区（PDF 页面点坐标）：{candidate.region.model_dump()}。",
                f"  局部图 SHA256：{candidate.crop_image.sha256}；"
                f"快照：{candidate.snapshot_ref.snapshot_id}。",
                f"  OCR 引擎：{_plain(candidate.ocr.engine)}；"
                f"语言：{_plain(candidate.ocr.language or '不可用')}；"
                f"失败代码：{_plain(candidate.ocr.error_code or '无')}。",
                "  手工转录："
                + ("是" if candidate.manual_transcription else "否")
                + "；人工选区，不声称已自动核实图号或样品身份。",
                "  原始识别："
                + _plain(
                    candidate.ocr.raw_text or "无可用识别文字，不代表图中没有文字。"
                ),
                "  用户核对文字："
                + _plain(candidate.reviewed_text or "未确认或已跳过。"),
            ]
        )
        if candidate.superseded_by is not None:
            lines.append("  已重新选择区域，旧候选仅保留历史，不作为当前已核对文字。")
    return "\n".join(lines)
