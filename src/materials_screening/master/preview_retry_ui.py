"""Explicit failed-preview regeneration, scoped to server checkpoint tasks."""

from __future__ import annotations

import hashlib

from .fulltext_preview import FulltextPreviewProcessor


class PreviewRetryController:
    def __init__(self, get_runner):
        self.get_runner = get_runner

    def tasks(self, session):
        conversation = (session or {}).get("master_conversation_id")
        runner = self.get_runner()
        if not conversation or runner is None:
            return ()
        return tuple(
            t
            for t in runner.get_fulltext_tasks(conversation)
            if any(FulltextPreviewProcessor.retry_allowed(t, d) for d in t.document_ids)
        )

    def open(self, session, task_id, document_id):
        task = next((t for t in self.tasks(session) if t.task_id == task_id), None)
        if task is None or document_id not in task.document_ids:
            raise ValueError("请刷新并选择当前任务的失败预览。")
        expected = FulltextPreviewProcessor.retry_token(task, document_id)
        scope = dict(
            conversation_id=task.conversation_id,
            task_id=task.task_id,
            document_id=document_id,
            expected=expected,
        )
        # Stable operation for one displayed revision; double-click/replay is safe.
        scope["operation_id"] = hashlib.sha256(
            (task.task_id + document_id + expected).encode()
        ).hexdigest()
        return scope, self.describe(task, document_id)

    @staticmethod
    def describe(task, document_id):
        preview = next((p for p in task.previews if p.document_id == document_id), None)
        records = [
            a for a in task.preview_retry_history if a.document_id == document_id
        ]
        lines = [
            f"原科研问题：{task.original_question}",
            f"已使用手动重试：{len(records)}/2",
        ]
        if preview:
            lines.extend(
                [
                    f"当前预览：{preview.title}",
                    f"研究问题：{preview.research_question}",
                    f"方法：{preview.methods}",
                    f"主要发现：{preview.key_findings}",
                    f"原文校验：{preview.evidence_quality}；生成状态：{preview.generation_status}",
                    f"来源第{preview.page_from}页：{preview.evidence_quote}",
                ]
            )
        for index, attempt in enumerate(records, 1):
            lines.append(f"重试{index}：{attempt.status}；{attempt.reason}")
            if attempt.previous_preview:
                lines.append("保留的旧预览：" + attempt.previous_preview.key_findings)
            if (
                attempt.proposed_preview
                and attempt.proposed_preview.proposed_evidence_quote
            ):
                lines.append(
                    "模型候选引文（未通过校验，不作为原文证据）："
                    + attempt.proposed_preview.proposed_evidence_quote
                )
        lines.append(
            "重试不会启动详细分析。成功后，请在对话中明确选择论文并要求详细分析。"
        )
        return "\n\n".join(lines)

    def retry(self, session, scope):
        if not scope or (session or {}).get("master_conversation_id") != scope.get(
            "conversation_id"
        ):
            raise ValueError("会话已改变，请刷新待重试预览。")
        task = self.get_runner().regenerate_preview(**scope)
        return self.describe(task, scope["document_id"])


def mount_preview_retry(session_state, get_runner):
    import gradio as gr

    controller = PreviewRetryController(get_runner)
    with gr.Accordion("失败文献预览重试", open=False):
        gr.Markdown("每篇最多手动重试2次。保留旧记录，不重新检索，不自动启动详细分析。")
        scope = gr.State({})
        refresh_button = gr.Button("刷新失败预览", size="sm")
        task = gr.Dropdown(choices=[], label="当前任务")
        document = gr.Dropdown(choices=[], label="失败预览论文")
        detail = gr.Textbox(label="预览与重试记录", lines=10, interactive=False)
        retry_button = gr.Button("重新生成预览", interactive=False)

    def guarded(fn):
        def wrapper(*args):
            try:
                return fn(*args)
            except Exception as exc:
                text = str(exc)
                if not text.startswith(
                    ("请", "会话已", "预览状态已", "每篇论文", "当前会话", "仅可重试")
                ):
                    text = "操作未完成，请刷新预览记录；旧结果未被清除。"
                raise gr.Error(text) from None

        return wrapper

    def refresh(session):
        tasks = controller.tasks(session)
        return (
            gr.update(
                choices=[(t.original_question[:70], t.task_id) for t in tasks],
                value=None,
            ),
            gr.update(choices=[], value=None),
            {},
            "请选择失败预览。",
            gr.update(interactive=False),
        )

    def choose_task(session, task_id):
        selected = next(
            (t for t in controller.tasks(session) if t.task_id == task_id), None
        )
        titles = {p.document_id: p.title for p in selected.previews} if selected else {}
        choices = (
            [
                (titles.get(d, d), d)
                for d in selected.document_ids
                if FulltextPreviewProcessor.retry_allowed(selected, d)
            ]
            if selected
            else []
        )
        return (
            gr.update(choices=choices, value=None),
            {},
            "请选择论文。",
            gr.update(interactive=False),
        )

    def choose_document(session, task_id, doc):
        if not doc:
            return {}, "请选择论文。", gr.update(interactive=False)
        current, text = controller.open(session, task_id, doc)
        selected = next(t for t in controller.tasks(session) if t.task_id == task_id)
        used = sum(a.document_id == doc for a in selected.preview_retry_history)
        return current, text, gr.update(interactive=used < 2)

    def retry(session, current):
        text = controller.retry(session, current)
        # Retain operation scope on failure/replay; refresh explicitly for attempt 2.
        return text, gr.update(interactive=False)

    outputs = [task, document, scope, detail, retry_button]
    refresh_button.click(guarded(refresh), session_state, outputs)
    task.change(
        guarded(choose_task),
        [session_state, task],
        [document, scope, detail, retry_button],
    )
    document.change(
        guarded(choose_document),
        [session_state, task, document],
        [scope, detail, retry_button],
    )
    retry_button.click(guarded(retry), [session_state, scope], [detail, retry_button])
    return guarded(refresh), outputs
