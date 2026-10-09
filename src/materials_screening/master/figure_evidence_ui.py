"""Explicit local figure-text actions; browser state never grants source authority."""

import hashlib
import json
import math
import threading

from .figure_evidence_contracts import FigureBatchReference


class FigureReviewController:
    def __init__(self, get_runner):
        self.get_runner = get_runner
        self._cancellations = {}
        self._cancel_lock = threading.Lock()

    def runner(self):
        runner = self.get_runner()
        if runner is None:
            raise ValueError("请先初始化主控。")
        return runner

    def tasks(self, session):
        conversation = (session or {}).get("master_conversation_id")
        if not conversation or self.get_runner() is None:
            return ()
        return tuple(
            t
            for t in self.runner().get_fulltext_tasks(conversation)
            if t.conversation_id == conversation and t.stage == "awaiting_figure_review"
        )

    def open(self, session, task_id):
        if task_id not in {t.task_id for t in self.tasks(session)}:
            raise ValueError("请选择当前会话的待核对任务。")
        conversation = session["master_conversation_id"]
        batch = self.runner().get_figure_review(
            conversation_id=conversation, task_id=task_id
        )
        return dict(
            conversation_id=conversation,
            task_id=task_id,
            head=self.reference(batch),
            corners=[],
        )

    @staticmethod
    def reference(batch):
        # Reference digest is server-owned, not reconstructed from UI fields.
        from .figure_evidence_store import batch_reference

        return batch_reference(batch).model_dump(mode="json")

    def arguments(self, session, scope):
        if not scope or (session or {}).get("master_conversation_id") != scope.get(
            "conversation_id"
        ):
            raise ValueError("会话已改变，请重新选择核对任务。")
        return dict(conversation_id=scope["conversation_id"], task_id=scope["task_id"])

    def batch(self, session, scope):
        batch = self.runner().get_figure_review(**self.arguments(session, scope))
        if self.reference(batch) != scope.get("head"):
            raise ValueError("核对状态已改变，请刷新后重试。")
        return batch

    def action(self, session, scope, action, **parameters):
        batch = self.runner().figure_review_action(
            **self.arguments(session, scope),
            expected=FigureBatchReference.model_validate(scope["head"]),
            operation_id="ui-"
            + hashlib.sha256(
                json.dumps(
                    dict(
                        scope=self.arguments(session, scope),
                        head=scope["head"],
                        action=action,
                        parameters={
                            k: v for k, v in parameters.items() if k != "cancel_event"
                        },
                    ),
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ).hexdigest()[:32],
            action=action,
            **parameters,
        )
        return {**scope, "head": self.reference(batch)}, batch

    def page(self, session, scope, document_id, page):
        if (
            isinstance(page, bool)
            or not isinstance(page, (int, float))
            or not math.isfinite(page)
            or int(page) != page
        ):
            raise ValueError("页码必须是正整数。")
        page = int(page)
        scope, batch = self.action(
            session, scope, "page", document_id=document_id, page=page
        )
        view = next(
            v
            for v in batch.page_views
            if v.document_id == document_id and v.page == page
        )
        path = self.runner().get_figure_image(
            **self.arguments(session, scope), document_id=document_id, page=page
        )
        return {
            **scope,
            "document_id": document_id,
            "page": page,
            "image_sha256": view.image.sha256,
            "corners": [],
        }, path

    @staticmethod
    def corner(scope, index):
        if not scope or not scope.get("image_sha256"):
            raise ValueError("请先查看来源页。")
        if (
            not isinstance(index, (list, tuple))
            or len(index) != 2
            or any(
                isinstance(n, bool)
                or not isinstance(n, (int, float))
                or not math.isfinite(n)
                or n < 0
                for n in index
            )
        ):
            raise ValueError("无效区域坐标。")
        corners = list(scope.get("corners", []))
        if len(corners) >= 2:
            corners = []
        return {**scope, "corners": [*corners, list(index)]}

    def add(self, session, scope, figure_label, kind, *, replace=False):
        corners = (scope or {}).get("corners", [])
        if len(corners) != 2:
            raise ValueError("请依次点击区域的两个对角。")
        (x0, y0), (x1, y1) = corners
        key = tuple(self.arguments(session, scope).values())
        event = threading.Event()
        with self._cancel_lock:
            if key in self._cancellations:
                raise ValueError("该任务已有本地识别在运行。")
            self._cancellations[key] = event
        try:
            parameters = dict(
                document_id=scope["document_id"],
                page=scope["page"],
                image_sha256=scope["image_sha256"],
                region=dict(
                    x0=min(x0, x1), y0=min(y0, y1), x1=max(x0, x1), y1=max(y0, y1)
                ),
                figure_label=figure_label,
                kind=kind,
                cancel_event=event,
            )
            if replace:
                parameters["replace_candidate_id"] = scope.get("candidate_id")
                if not parameters["replace_candidate_id"]:
                    raise ValueError("请先选择要替代的候选。")
            scope, batch = self.action(session, scope, "add_screen", **parameters)
            candidate = batch.candidates[-1]
            path = self.runner().get_figure_image(
                **self.arguments(session, scope), candidate_id=candidate.candidate_id
            )
            return {**scope, "candidate_id": candidate.candidate_id}, candidate, path
        finally:
            with self._cancel_lock:
                self._cancellations.pop(key, None)

    def cancel(self, session, scope):
        key = tuple(self.arguments(session, scope).values())
        with self._cancel_lock:
            event = self._cancellations.get(key)
            if event:
                event.set()
        return "已请求取消本地识别；未确认的文字不会获批。"

    def candidate(self, session, scope, candidate_id):
        batch = self.batch(session, scope)
        candidate = next(
            (c for c in batch.candidates if c.candidate_id == candidate_id), None
        )
        if candidate is None or candidate.superseded_by:
            raise ValueError("请选择当前候选。")
        path = self.runner().get_figure_image(
            **self.arguments(session, scope), candidate_id=candidate_id
        )
        return {**scope, "candidate_id": candidate_id}, candidate, path

    def decide(self, session, scope, candidate_id, decision, text=None):
        scope, _ = self.action(
            session,
            scope,
            "decide",
            candidate_id=candidate_id,
            decision=decision,
            text=text if decision == "text_verified" else None,
        )
        return scope

    def finish(self, session, scope, *, skip_remaining=False, abandon_all=False):
        batch = self.batch(session, scope)
        pending = [c for c in batch.candidates if c.text_review_status == "pending"]
        if pending and not skip_remaining and not abandon_all:
            raise ValueError(
                f"还有 {len(pending)} 个图未作决定：请确认文字、跳过此图，"
                "或明确跳过所有尚未决定的图。"
            )
        if batch.status != "closed":
            scope, _ = self.action(
                session,
                scope,
                "close",
                skip_remaining=skip_remaining,
                abandon_all=abandon_all,
            )
        # Structured close is committed before the ordinary continuation. If that
        # continuation fails the closed batch remains resumable without approving again.
        result = self.runner().ask(message="继续", **self.arguments(session, scope))
        return scope, result.response_text


def mount_figure_review(session_state, chatbot, get_runner):
    """Mount a standard Gradio panel; return a refresh binding for chat completion."""
    import gradio as gr

    controller = FigureReviewController(get_runner)
    with gr.Accordion("图中文字核对（可选）", open=False):
        gr.Markdown(
            "仅核对可见文字。**实验条件尚未绑定**；不会写入测量时间或统计输入。"
        )
        scope = gr.State({})
        refresh_button = gr.Button("刷新待核对任务", size="sm")
        task = gr.Dropdown(choices=[], label="当前待核对任务")
        question = gr.Textbox(label="原科研问题", interactive=False)
        with gr.Row():
            document = gr.Dropdown(choices=[], label="授权论文", scale=4)
            page = gr.Number(value=1, precision=0, label="页码（从 1 开始）", scale=1)
            view_button = gr.Button("查看来源页", scale=1)
        source = gr.Image(
            label="来源页：依次点击区域两个对角", type="filepath", interactive=False
        )
        selection = gr.Textbox(label="所选区域（原图像素）", interactive=False)
        with gr.Row():
            figure_label = gr.Textbox(label="人工图号 / 面板说明", max_lines=2)
            kind = gr.Dropdown(
                choices=[
                    ("时间轴文字", "axis_label"),
                    ("时间轴刻度", "axis_ticks"),
                    ("图例文字", "legend_text"),
                    ("其他图中文字", "other_text"),
                ],
                value="axis_label",
                label="文字类型",
            )
        with gr.Row():
            add_button = gr.Button("本地识别所选区域")
            replace_button = gr.Button("重新选区域（替代当前候选）")
            cancel_button = gr.Button("取消本地识别")
        candidate = gr.Dropdown(choices=[], label="图文字候选")
        crop = gr.Image(label="局部原图", type="filepath", interactive=False)
        raw = gr.Textbox(label="原始识别文字（不覆盖）", interactive=False, lines=3)
        revised = gr.Textbox(label="修订文字（仅核对文字）", lines=3)
        status = gr.Textbox(label="核对状态 / 提示", interactive=False, lines=3)
        with gr.Row():
            confirm_button = gr.Button("确认文字")
            skip_button = gr.Button("跳过此图")
        skip_remaining = gr.Checkbox(label="明确跳过所有尚未决定的图", value=False)
        with gr.Row():
            finish_button = gr.Button("完成核对并继续", variant="primary")
            abandon_button = gr.Button("跳过图证据并继续")

    def guarded(fn):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                # Do not display private paths, OCR stderr or evidence as executable UI.
                detail = (
                    str(exc)
                    if str(exc).startswith(
                        (
                            "请选择",
                            "请先",
                            "请依次",
                            "会话已",
                            "核对状态已",
                            "页码必须",
                            "无效区域",
                            "还有",
                        )
                    )
                    else "操作未完成：请刷新核对状态、检查页码和区域，或明确跳过。"
                )
                raise gr.Error(detail) from None

        return wrapper

    def refresh(session):
        tasks = controller.tasks(session)
        choices = [(t.original_question[:80], t.task_id) for t in tasks]
        return (
            gr.update(
                choices=choices, value=tasks[0].task_id if len(tasks) == 1 else None
            ),
            {},
            "有多个任务时请明确选择。"
            if len(tasks) > 1
            else "请选择待核对任务；没有图时可直接跳过。",
        )

    def open_task(session, task_id):
        if not task_id:
            return (
                {},
                "",
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                None,
                None,
                "",
                "",
                "",
                "",
            )
        current = controller.open(session, task_id)
        batch = controller.batch(session, current)
        actual = next(t for t in controller.tasks(session) if t.task_id == task_id)
        titles = {p.document_id: p.title for p in actual.previews}
        docs = [(titles.get(d, d), d) for d in batch.selected_snapshots]
        candidates = [
            (f"{c.figure_label} · {c.text_review_status}", c.candidate_id)
            for c in batch.candidates
            if not c.superseded_by
        ]
        return (
            current,
            actual.original_question,
            gr.update(choices=docs, value=docs[0][1] if len(docs) == 1 else None),
            gr.update(choices=candidates, value=None),
            None,
            None,
            "",
            "",
            "",
            "已恢复当前批次。实验条件尚未绑定。",
        )

    def show_page(session, current, doc, number):
        current, path = controller.page(session, current, doc, number)
        return current, str(path), "", "请点击区域的两个对角；第三次点击重新开始。"

    def click_corner(current, evt: gr.SelectData):
        current = controller.corner(current, evt.index)
        return current, str(current["corners"])

    def candidate_outputs(current, c, path):
        return (
            current,
            str(path),
            c.ocr.raw_text,
            c.reviewed_text if c.reviewed_text is not None else c.ocr.raw_text,
            (
                f"文字：{c.text_review_status}；OCR：{c.ocr.status}"
                f"（{c.ocr.error_code or c.ocr.engine}）；实验条件尚未绑定。"
            ),
        )

    def add_candidate(session, current, label, category, replace=False):
        current, c, path = controller.add(
            session, current, label, category, replace=replace
        )
        batch = controller.batch(session, current)
        choices = [
            (f"{x.figure_label} · {x.text_review_status}", x.candidate_id)
            for x in batch.candidates
            if not x.superseded_by
        ]
        return (
            *candidate_outputs(current, c, path),
            gr.update(choices=choices, value=c.candidate_id),
        )

    def select_candidate(session, current, candidate_id):
        if not candidate_id:
            return current, None, "", "", "请选择候选。"
        return candidate_outputs(*controller.candidate(session, current, candidate_id))

    def decide(session, current, candidate_id, text, decision):
        current = controller.decide(session, current, candidate_id, decision, text)
        batch = controller.batch(session, current)
        choices = [
            (f"{c.figure_label} · {c.text_review_status}", c.candidate_id)
            for c in batch.candidates
            if not c.superseded_by
        ]
        return (
            *select_candidate(session, current, candidate_id),
            gr.update(choices=choices, value=candidate_id),
        )

    def finish(session, current, history, remaining=False, abandon=False):
        current, text = controller.finish(
            session, current, skip_remaining=remaining, abandon_all=abandon
        )
        return (
            current,
            [*(history or []), {"role": "assistant", "content": text}],
            "图核对批次已关闭；接续结果见对话。实验条件仍未绑定。",
        )

    refresh_outputs = [task, scope, status]
    refresh_button.click(guarded(refresh), session_state, refresh_outputs)
    task.change(
        guarded(open_task),
        [session_state, task],
        [
            scope,
            question,
            document,
            candidate,
            source,
            crop,
            selection,
            raw,
            revised,
            status,
        ],
    )
    view_button.click(
        guarded(show_page),
        [session_state, scope, document, page],
        [scope, source, selection, status],
    )

    # Gradio injects SelectData only if the annotation survives the wrapper.
    def on_select(current, evt: gr.SelectData):
        return guarded(click_corner)(current, evt)

    source.select(on_select, scope, [scope, selection])
    add_button.click(
        guarded(add_candidate),
        [session_state, scope, figure_label, kind],
        [scope, crop, raw, revised, status, candidate],
    )
    replace_button.click(
        guarded(lambda *args: add_candidate(*args, replace=True)),
        [session_state, scope, figure_label, kind],
        [scope, crop, raw, revised, status, candidate],
    )
    cancel_button.click(
        guarded(controller.cancel), [session_state, scope], status, queue=False
    )
    candidate.change(
        guarded(select_candidate),
        [session_state, scope, candidate],
        [scope, crop, raw, revised, status],
    )
    confirm_button.click(
        guarded(lambda *args: decide(*args, decision="text_verified")),
        [session_state, scope, candidate, revised],
        [scope, crop, raw, revised, status, candidate],
    )
    skip_button.click(
        guarded(lambda *args: decide(*args, decision="skipped")),
        [session_state, scope, candidate, revised],
        [scope, crop, raw, revised, status, candidate],
    )
    finish_button.click(
        guarded(finish),
        [session_state, scope, chatbot, skip_remaining],
        [scope, chatbot, status],
    )
    abandon_button.click(
        guarded(lambda *args: finish(*args, abandon=True)),
        [session_state, scope, chatbot],
        [scope, chatbot, status],
    )
    return guarded(refresh), refresh_outputs
