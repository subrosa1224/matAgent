"""Bounded automatic continuation of successful saved fulltext work only."""

import json
from queue import Queue
from threading import Thread

_BUDGET_CODES = {
    "FULLTEXT_PREVIEW_BUDGET",
    "FULLTEXT_EXTRACTION_BUDGET",
    "FULLTEXT_ANALYSIS_BUDGET",
}


def run_saved_batches(
    processor,
    task,
    *,
    save_task,
    model_budget,
    cancel_event,
    user_turn_id,
    progress=None,
):
    current = task
    calls = tools = 0
    seen = set()
    batch_budget = processor.batch_model_budget
    if batch_budget is None:
        batch_budget = model_budget

    def saved(value):
        nonlocal current
        save_task(value)  # Progress is emitted only after durable save succeeds.
        current = value
        if progress is not None:
            try:
                selected, _ = processor.selection(value)
            except ValueError:
                progress("literature：任务已保存，请确认要处理的论文范围。")
                return
            ids = {value.document_ids[i] for i in selected}
            completed = sum(p.document_id in ids for p in value.previews)
            extracted = len(
                set(value.extraction_snapshots) | set(value.staged_extractions)
            )
            progress(
                f"literature：全文预览已完成 {completed}/{len(ids)} 篇；"
                f"已保存提取进度 {extracted} 篇。"
            )

    result = None
    for index in range(processor.max_batches):
        if progress is not None:
            progress(f"literature：开始第{index + 1}批全文处理，已有成功结果将复用。")
        result = processor._run_single_batch(
            current,
            save_task=saved,
            model_budget=batch_budget,
            cancel_event=cancel_event,
            user_turn_id=user_turn_id,
        )
        calls += result.model_call_count
        tools += result.tool_call_count
        result = result.model_copy(
            update={"model_call_count": calls, "tool_call_count": tools}
        )
        if result.status == "cancelled" or (
            cancel_event is not None and cancel_event.is_set()
        ):
            return result
        code = result.error.get("code") if result.error else None
        if code not in _BUDGET_CODES:
            return result  # Human confirmation, provider/source failure, or completion.
        identity = json.dumps(
            {
                "previews": [p.model_dump(mode="json") for p in current.previews],
                "snapshots": {
                    k: v.model_dump(mode="json")
                    for k, v in current.extraction_snapshots.items()
                },
                "analysis": current.fulltext_analysis_ref.model_dump(mode="json")
                if current.fulltext_analysis_ref
                else None,
                "staged": processor.extraction_processor.progress_signature(current)
                if current.staged_extractions
                and hasattr(processor.extraction_processor, "progress_signature")
                else {},
                "stage": current.resume_stage,
            },
            sort_keys=True,
        )
        if identity in seen:
            return result.model_copy(
                update={
                    "response_text": result.response_text
                    + "\n连续批次没有产生新进度，已停止自动接续；已有结果保留。",
                }
            )
        seen.add(identity)
        if progress is not None:
            progress("literature：本批预算已到，成功部分已保存，自动接续未完成部分。")
    return result.model_copy(
        update={
            "response_text": result.response_text
            + "\n已达到自动批次上限；成功结果保留，可继续未完成部分。",
        }
    )


def stream_saved_operation(operation, cancel_event):
    """Yield saved progress while retaining ownership until worker exit.

    Closing the stream requests a safe-point stop, never abandons an active
    checkpoint writer or releases its conversation lock ahead of it.
    """
    queue = Queue()

    def work():
        try:
            result = operation(lambda message: queue.put(("progress", message)))
        except BaseException as exc:
            queue.put(("error", exc))
        else:
            queue.put(("result", result))

    worker = Thread(target=work, name="fulltext-saved-batches")
    worker.start()
    finished = False
    try:
        while True:
            kind, value = queue.get()
            if kind == "progress":
                yield value
            else:
                finished = True
                if kind == "error":
                    raise value
                return value
    finally:
        if not finished:
            cancel_event.set()
        # Keep checkpoint ownership until the in-flight request reaches a safe
        # point; cancellation must not permit a concurrent writer to this task.
        while worker.is_alive():
            worker.join(timeout=0.2)
