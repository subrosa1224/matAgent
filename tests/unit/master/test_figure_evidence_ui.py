"""Local UI callbacks using real scoped sources; no remote model calls."""

import threading
from types import SimpleNamespace

import pytest

from materials_screening.master.figure_evidence_ui import (
    FigureReviewController,
    mount_figure_review,
)
from materials_screening.master.master_runner import MasterAgentRunner
from tests.unit.master.test_figure_evidence_review import setup as figure_setup
from tests.unit.master.test_figure_evidence_review import update

setup = figure_setup


@pytest.fixture
def controller(setup):
    s = setup
    runner = object.__new__(MasterAgentRunner)
    runner._figure_review_service = s.service
    runner._store = SimpleNamespace(conversation_lock=lambda c: lock)
    lock = threading.Lock()
    runner.get_fulltext_tasks = lambda c: (
        (s.task,) if c == s.task.conversation_id else ()
    )
    runner._save_fulltext_task = lambda t: update(s, t.figure_evidence_ref)
    calls = []
    runner.ask = lambda **kw: (
        calls.append(kw) or SimpleNamespace(response_text="offline report")
    )
    return s, FigureReviewController(lambda: runner), calls


def test_ui_page_click_crop_confirm_and_continue(controller):
    s, ui, calls = controller
    session = {"master_conversation_id": s.task.conversation_id}
    scope = ui.open(session, s.task.task_id)
    scope, path = ui.page(session, scope, s.doc, 1)
    assert path.is_file()
    scope = ui.corner(scope, (10, 10))
    scope = ui.corner(scope, (210, 210))
    scope, candidate, crop = ui.add(session, scope, "Fig 1", "axis_label")
    assert crop.is_file() and candidate.text_review_status == "pending"
    raw = candidate.ocr.raw_text
    scope = ui.decide(
        session,
        scope,
        candidate.candidate_id,
        "text_verified",
        "<script>999 min</script>",
    )
    batch = ui.batch(session, scope)
    assert batch.candidates[0].ocr.raw_text == raw
    assert batch.candidates[0].condition_binding_status == "unbound"
    scope, report = ui.finish(session, scope)
    assert report == "offline report" and len(calls) == 1
    assert calls[0]["task_id"] == s.task.task_id
    assert ui.batch(session, scope).status == "closed"


def test_ui_scope_change_and_stale_head_rejected(controller):
    s, ui, _ = controller
    session = {"master_conversation_id": s.task.conversation_id}
    scope = ui.open(session, s.task.task_id)
    stale = dict(scope)
    scope, _ = ui.page(session, scope, s.doc, 1)
    replay, _ = ui.page(session, stale, s.doc, 1)
    assert replay["head"] == scope["head"]
    with pytest.raises(ValueError):
        ui.page(session, stale, s.doc, 2)
    with pytest.raises(ValueError):
        ui.page({"master_conversation_id": "other"}, scope, s.doc, 1)
    with pytest.raises(ValueError):
        ui.corner(scope, (float("nan"), 5))


def test_ui_requires_explicit_pending_skip(controller):
    s, ui, calls = controller
    session = {"master_conversation_id": s.task.conversation_id}
    scope = ui.open(session, s.task.task_id)
    scope, _ = ui.page(session, scope, s.doc, 1)
    scope = ui.corner(ui.corner(scope, (10, 10)), (210, 210))
    scope, candidate, _ = ui.add(session, scope, "Fig 1", "legend_text")
    with pytest.raises(ValueError):
        ui.finish(session, scope)
    assert not calls
    scope, _ = ui.finish(session, scope, skip_remaining=True)
    assert ui.batch(session, scope).candidates[0].text_review_status == "skipped"


def test_ui_cancel_reaches_local_work_without_approving_candidate(controller):
    s, ui, _ = controller
    session = {"master_conversation_id": s.task.conversation_id}
    scope = ui.open(session, s.task.task_id)
    scope, _ = ui.page(session, scope, s.doc, 1)
    scope = ui.corner(ui.corner(scope, (10, 10)), (210, 210))
    started = threading.Event()
    outcomes = []

    class SlowOcr:
        def recognize(self, path, *, cancel_event):
            started.set()
            assert cancel_event.wait(3)
            raise ValueError("local cancelled")

    s.service.ocr = SlowOcr()

    def worker():
        try:
            ui.add(session, scope, "Fig 1", "axis_label")
        except Exception as exc:
            outcomes.append(type(exc).__name__)

    thread = threading.Thread(target=worker)
    thread.start()
    assert started.wait(3)
    ui.cancel(session, scope)
    thread.join(3)
    assert not thread.is_alive() and outcomes
    assert not ui.batch(session, scope).candidates


def test_ui_retry_preserves_operation_id_after_head_commit_failure(controller):
    s, ui, _ = controller
    session = {"master_conversation_id": s.task.conversation_id}
    scope = ui.open(session, s.task.task_id)
    scope, _ = ui.page(session, scope, s.doc, 1)
    scope = ui.corner(ui.corner(scope, (10, 10)), (210, 210))
    runner = ui.runner()
    original = runner._save_fulltext_task

    def interrupted(task):
        raise OSError("simulated checkpoint write interruption")

    runner._save_fulltext_task = interrupted
    with pytest.raises(OSError):
        ui.add(session, scope, "Fig 1", "axis_label")
    runner._save_fulltext_task = original
    scope, candidate, _ = ui.add(session, scope, "Fig 1", "axis_label")
    assert len(s.calls) == 1 and len(ui.batch(session, scope).candidates) == 1


def test_ui_mount_exposes_plain_readonly_ocr_and_explicit_actions():
    import gradio as gr

    with gr.Blocks() as demo:
        session = gr.State({})
        chatbot = gr.Chatbot()
        mount_figure_review(session, chatbot, lambda: None)
    config = demo.get_config_file()
    labels = [c["props"].get("label") for c in config["components"]]
    assert "原始识别文字（不覆盖）" in labels
    assert "修订文字（仅核对文字）" in labels
    buttons = [
        c["props"].get("value") for c in config["components"] if c["type"] == "button"
    ]
    assert "确认文字" in buttons and "跳过图证据并继续" in buttons
