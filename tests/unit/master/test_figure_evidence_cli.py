"""CLI routing is local, explicit and uses the same task-bound service."""

from contextlib import contextmanager

import typer
from typer.testing import CliRunner

from materials_screening.master.figure_evidence_cli import (
    register_figure_review_command,
)
from tests.unit.master.test_figure_evidence_review import setup as figure_setup
from tests.unit.master.test_figure_evidence_ui import controller as figure_controller

controller = figure_controller
setup = figure_setup


def test_cli_show_page_add_confirm_close_share_service(controller):
    s, ui, calls = controller
    opened = []

    @contextmanager
    def factory(**kw):
        opened.append(kw)
        yield ui.runner()

    app = typer.Typer()
    register_figure_review_command(app, factory)
    runner = CliRunner()
    base = ["-c", s.task.conversation_id, "--task-id", s.task.task_id]
    result = runner.invoke(app, ["show", *base])
    assert result.exit_code == 0, result.output
    assert "unbound" in result.output

    def invoke(action, *parameters):
        head = s.task.figure_evidence_ref
        result = runner.invoke(
            app,
            [
                action,
                *base,
                "--batch-id",
                head.record_id,
                "--batch-sha256",
                head.content_sha256,
                "--operation-id",
                action,
                *parameters,
            ],
        )
        assert result.exit_code == 0, result.output

    invoke("page", "--document-id", s.doc)
    invoke(
        "add",
        "--document-id",
        s.doc,
        "--figure-label",
        "Fig 1",
        "--region",
        "20",
        "--region",
        "20",
        "--region",
        "150",
        "--region",
        "80",
    )
    candidate = s.store.load(s.task.figure_evidence_ref).candidates[0]
    invoke("confirm", "--candidate-id", candidate.candidate_id, "--text", "999 min")
    invoke("close")
    assert s.store.load(s.task.figure_evidence_ref).status == "closed"
    assert not calls and all(kw["llm_provider"] == "mock" for kw in opened)


def test_cli_write_requires_explicit_revision_before_opening_runner():
    app = typer.Typer()
    register_figure_review_command(
        app, lambda **kw: (_ for _ in ()).throw(AssertionError("must not open"))
    )
    result = CliRunner().invoke(
        app, ["skip-all", "-c", "conv", "--task-id", "task-fulltext-" + "a" * 32]
    )
    assert result.exit_code == 2 and "写操作必须" in result.output
