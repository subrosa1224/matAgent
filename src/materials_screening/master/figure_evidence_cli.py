"""Register explicit, revision-bound figure review commands without model approval."""

import json
from pathlib import Path
from typing import Literal

import typer


def register_figure_review_command(app, open_runner):
    @app.command("figure-review")
    def figure_review(
        action: Literal[
            "show", "page", "add", "confirm", "skip", "close", "skip-all"
        ] = typer.Argument(...),
        conversation_id: str = typer.Option(..., "--conversation-id", "-c"),
        task_id: str = typer.Option(..., "--task-id"),
        batch_id: str | None = typer.Option(None, "--batch-id"),
        batch_sha256: str | None = typer.Option(None, "--batch-sha256"),
        operation_id: str | None = typer.Option(None, "--operation-id"),
        document_id: str | None = typer.Option(None, "--document-id"),
        page: int = typer.Option(1, "--page", min=1),
        region: list[float] | None = typer.Option(
            None, "--region", help="PDF page points: repeat four times, x0 y0 x1 y1."
        ),
        figure_label: str = typer.Option("", "--figure-label"),
        kind: Literal[
            "axis_label", "axis_ticks", "legend_text", "other_text"
        ] = typer.Option("axis_label", "--kind"),
        candidate_id: str | None = typer.Option(None, "--candidate-id"),
        replace_candidate_id: str | None = typer.Option(None, "--replace-candidate-id"),
        text: str | None = typer.Option(
            None,
            "--text",
            help="Explicit human transcription; never an experimental condition.",
        ),
        skip_remaining: bool = typer.Option(False, "--skip-remaining"),
        run_root: Path = typer.Option(Path("data/workflow_runs"), "--output", "-o"),
    ):
        """Review local figure text. Use master ask to resume after closing."""
        from .figure_evidence_contracts import FigureBatchReference
        from .figure_evidence_store import batch_reference

        try:
            if action != "show" and not all((batch_id, batch_sha256, operation_id)):
                raise ValueError(
                    "写操作必须提供当前 batch-id、batch-sha256 "
                    "和 operation-id；先运行 show。"
                )
            scope = dict(conversation_id=conversation_id, task_id=task_id)
            with open_runner(
                run_root=run_root,
                llm_provider="mock",
                planner_fixture=None,
                materials_repository="mock",
                materials_fixture=None,
            ) as runner:
                if action == "show":
                    batch = runner.get_figure_review(**scope)
                else:
                    parameters = {}
                    service_action = action
                    if action in {"page", "add"}:
                        if document_id is None:
                            raise ValueError("必须选择当前任务的 document-id。")
                        parameters = dict(document_id=document_id, page=page)
                    if action == "add":
                        if region is None or len(region) != 4:
                            raise ValueError(
                                "必须提供四个 --region 值（PDF 页面点坐标）。"
                            )
                        parameters.update(
                            region=dict(
                                zip(("x0", "y0", "x1", "y1"), region, strict=True)
                            ),
                            figure_label=figure_label,
                            kind=kind,
                        )
                        if replace_candidate_id:
                            parameters["replace_candidate_id"] = replace_candidate_id
                    if action in {"confirm", "skip"}:
                        if candidate_id is None:
                            raise ValueError("必须选择 candidate-id。")
                        service_action = "decide"
                        parameters = dict(
                            candidate_id=candidate_id,
                            decision="text_verified"
                            if action == "confirm"
                            else "skipped",
                            text=text,
                        )
                    if action in {"close", "skip-all"}:
                        service_action = "close"
                        parameters = dict(
                            skip_remaining=skip_remaining,
                            abandon_all=action == "skip-all",
                        )
                    batch = runner.figure_review_action(
                        **scope,
                        expected=FigureBatchReference(
                            record_id=batch_id, content_sha256=batch_sha256
                        ),
                        operation_id=operation_id,
                        action=service_action,
                        **parameters,
                    )
                result = {
                    "reference": batch_reference(batch).model_dump(mode="json"),
                    "batch": batch.model_dump(mode="json"),
                    "condition_binding_status": "unbound",
                }
                if action == "page":
                    result["source_page"] = str(
                        runner.get_figure_image(
                            **scope, document_id=document_id, page=page
                        )
                    )
                typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
                if batch.status == "closed":
                    typer.echo(
                        "图核对已关闭，实验条件尚未绑定。使用 master ask "
                        "--message 继续 --conversation-id <当前会话> "
                        "--task-id <当前任务> 接续。"
                    )
        except Exception as exc:
            # Strict validation errors can contain supplied evidence; avoid dumping
            # internals or OCR stderr into a machine-readable response.
            detail = (
                str(exc)
                if str(exc).startswith(("写操作必须", "必须选择", "必须提供"))
                else "核对操作失败：检查当前引用、来源、页码和区域，或显式跳过。"
            )
            typer.echo(detail, err=True)
            raise typer.Exit(2) from None
