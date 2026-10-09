"""Entry routing tests only; fake runner outputs are not scientific validation."""

from contextlib import contextmanager

import pytest
from typer.testing import CliRunner

from materials_screening import cli
from materials_screening import unified_ui_gradio as ui
from materials_screening.agent.models import AgentResult
from materials_screening.master.artifact_registry import ArtifactRegistry
from materials_screening.master.fulltext_tasks import new_fulltext_task
from materials_screening.master.master_runner import AgentStreamEvent

QUESTION = (
    "查询GdVO4材料并分析稳定性；检索复合光催化实验论文，"
    "上传全文后提取降解率并统计变化。"
)


@pytest.fixture(autouse=True)
def no_unplanned_processing(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Route test must not invoke a real literature processor")

    monkeypatch.setattr(ui, "_dispatch_literature", unexpected)


class RecordingRunner:
    def __init__(self, registry, tasks=()):
        self.registry = registry
        self.tasks = tasks
        self.calls = []

    def start_conversation(self):
        return "conv-a"

    def get_fulltext_tasks(self, conversation_id):
        assert conversation_id == "conv-a"
        return self.tasks

    def register_pdf_attachment(self, source, *, conversation_id):
        return self.registry.register_pdf(source, conversation_id=conversation_id)

    def ask_stream(self, **arguments):
        self.calls.append(arguments)
        for ref in arguments.get("artifact_refs", ()):
            self.registry.resolve_pdf(ref, conversation_id=arguments["conversation_id"])
        yield AgentStreamEvent(
            node="fulltext_task",
            message="测试接续入口",
            is_final=True,
            result=AgentResult(
                conversation_id="conv-a",
                user_turn_id="turn-a",
                status="error",
                final_status="error",
                response_text="未执行全文分析",
                error={"code": "FULLTEXT_PROCESSOR_UNAVAILABLE"},
            ),
        )

    def ask(self, **arguments):
        return list(self.ask_stream(**arguments))[-1].result


def pdf(tmp_path):
    source = tmp_path / "uploaded.pdf"
    source.write_bytes(b"%PDF-1.7\nroute fixture")
    return source


def dispatch(message, files, state):
    return list(ui._dispatch(ui.MODE_AUTO, message, [], files, state, "", 2020, 10))


@pytest.mark.parametrize("message", ["", "继续", "只预览，暂不详细分析"])
def test_active_task_upload_routes_to_same_master(tmp_path, monkeypatch, message):
    registry = ArtifactRegistry(tmp_path / "artifacts")
    runner = RecordingRunner(registry, (new_fulltext_task("conv-a", QUESTION),))
    monkeypatch.setattr(ui, "_master_runner", runner)
    monkeypatch.setattr(ui, "ARTIFACTS", registry)
    state = ui._initial_state()
    state["master_conversation_id"] = "conv-a"
    result = dispatch(message, [str(pdf(tmp_path))], state)[-1]
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["conversation_id"] == "conv-a"
    assert call["message"] == (message or "继续")
    assert len(call["artifact_refs"]) == 1
    assert result[2]["artifact_ids"] == list(call["artifact_refs"])
    assert str(tmp_path) not in str(result)
    assert "未执行全文分析" in result[0][-1]["content"]


def test_first_question_with_pdf_does_not_skip_master(tmp_path, monkeypatch):
    registry = ArtifactRegistry(tmp_path / "artifacts")
    runner = RecordingRunner(registry)
    monkeypatch.setattr(ui, "_master_runner", runner)
    monkeypatch.setattr(ui, "ARTIFACTS", registry)
    result = dispatch(QUESTION, [str(pdf(tmp_path))], ui._initial_state())[-1]
    assert runner.calls[0]["message"] == QUESTION
    assert runner.calls[0]["conversation_id"] == "conv-a"
    assert runner.calls[0]["artifact_refs"]
    assert result[2]["master_conversation_id"] == "conv-a"


def test_bare_upload_without_task_stays_independent_preview(tmp_path, monkeypatch):
    runner = RecordingRunner(ArtifactRegistry(tmp_path / "artifacts"))
    monkeypatch.setattr(ui, "_master_runner", runner)
    preview_calls = []

    def preview(*args, **kwargs):
        preview_calls.append(args)
        yield [], "preview", args[3], "", None

    monkeypatch.setattr(ui, "_dispatch_literature", preview)
    dispatch("", [str(pdf(tmp_path))], ui._initial_state())
    assert preview_calls[0][0] == "请预览我上传的论文。"
    assert runner.calls == []


@pytest.mark.parametrize(
    "message",
    [
        "继续",
        "只分析第2篇",
        "只预览这些论文",
        "不分析第二篇",
        "只分析第一、第三篇",
    ],
)
def test_followup_precedes_browser_artifact_owner_routing(
    tmp_path, monkeypatch, message
):
    registry = ArtifactRegistry(tmp_path / "artifacts")
    runner = RecordingRunner(registry, (new_fulltext_task("conv-a", QUESTION),))
    monkeypatch.setattr(ui, "_master_runner", runner)
    monkeypatch.setattr(ui, "ARTIFACTS", registry)
    state = ui._initial_state()
    state["master_conversation_id"] = "conv-a"
    # Browser references are not task authority and must not redirect continuation.
    state["artifact_ids"] = ["forged-browser-artifact"]
    dispatch(message, None, state)
    assert runner.calls[0]["message"] == message
    assert "artifact_refs" not in runner.calls[0]


def test_corrupt_server_task_never_falls_back_to_independent_preview(
    tmp_path, monkeypatch
):
    runner = RecordingRunner(ArtifactRegistry(tmp_path / "artifacts"))

    def invalid(_conversation):
        raise ValueError("bad checkpoint")

    runner.get_fulltext_tasks = invalid
    monkeypatch.setattr(ui, "_master_runner", runner)
    state = ui._initial_state()
    state["master_conversation_id"] = "conv-a"
    result = dispatch("继续", [str(pdf(tmp_path))], state)[-1]
    assert "请求未完成" in result[0][-1]["content"]
    assert runner.calls == []


@pytest.mark.parametrize("progress", [False, True])
def test_cli_attachment_and_task_flags_share_master_entry(
    tmp_path, monkeypatch, progress
):
    registry = ArtifactRegistry(tmp_path / "artifacts")
    runner = RecordingRunner(registry)

    @contextmanager
    def open_runner(**kwargs):
        yield runner

    monkeypatch.setattr(cli, "_open_master_runner", open_runner)
    task_id = "task-fulltext-" + "a" * 32
    arguments = [
        "master",
        "ask",
        "--message",
        "继续",
        "--conversation-id",
        "conv-a",
        "--pdf",
        str(pdf(tmp_path)),
        "--task-id",
        task_id,
    ]
    if progress:
        arguments.append("--progress")
    result = CliRunner().invoke(cli.app, arguments)
    assert result.exit_code == 0, result.output
    assert runner.calls[0]["task_id"] == task_id
    assert runner.calls[0]["artifact_refs"]
    assert "未执行全文分析" in result.output


def test_cancelled_continuation_not_shown_as_completed():
    from materials_screening.master.ui_controller import stream_master_request

    class Cancelled:
        def ask_stream(self, **kwargs):
            yield AgentStreamEvent(
                node="fulltext_task",
                message="已停止",
                is_final=True,
                result=AgentResult(
                    conversation_id="conv-a",
                    user_turn_id="turn-a",
                    status="cancelled",
                    response_text="已停止",
                ),
            )

    result = list(
        stream_master_request(Cancelled(), message="继续", conversation_id="conv-a")
    )[-1]
    assert result.status == "已停止"
