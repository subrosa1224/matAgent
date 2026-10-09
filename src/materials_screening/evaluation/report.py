"""Evaluation report builders: JSON and Markdown, redacted failures (D2-M8)."""

import json
from pathlib import Path

from materials_screening.evaluation.planner_evaluator import (
    PlannerCaseResult,
    PlannerEvalMetrics,
)


def _redacted_failure(result: PlannerCaseResult) -> dict[str, object]:
    """Redacted failure record: no query, no raw output, no error body."""
    return {
        "id": result.id,
        "tags": list(result.tags),
        "expected_status": result.expected_status.value,
        "schema_success": result.schema_success,
        "actual_status": (
            result.actual_status.value if result.actual_status is not None else None
        ),
        "error_type": result.error_type,
    }


def _failures(
    outcomes: tuple[PlannerCaseResult, ...],
) -> list[PlannerCaseResult]:
    return [
        outcome
        for outcome in outcomes
        if not outcome.schema_success
        or outcome.actual_status != outcome.expected_status
    ]


def metrics_to_json(metrics: PlannerEvalMetrics) -> str:
    """Serialize metrics as indented JSON."""
    return metrics.model_dump_json(indent=2)


def failures_to_json(outcomes: tuple[PlannerCaseResult, ...]) -> str:
    """Serialize redacted failure records as indented JSON."""
    payload = [_redacted_failure(result) for result in _failures(outcomes)]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def metrics_to_markdown(metrics: PlannerEvalMetrics) -> str:
    """Render metrics as a Markdown report."""
    lines = [
        "# Planner Evaluation Metrics",
        "",
        f"- Total cases: {metrics.total}",
        (
            f"- Schema success rate: {metrics.schema_success_rate:.3f} "
            f"({metrics.schema_success}/{metrics.total})"
        ),
        (
            f"- Status accuracy: {metrics.status_accuracy:.3f} "
            f"({metrics.status_accurate}/{metrics.total})"
        ),
        f"- Ready precision: {metrics.ready_precision:.3f}",
        f"- Ready recall: {metrics.ready_recall:.3f}",
        (
            f"- Request field exact match: {metrics.request_exact_match_rate:.3f} "
            f"({metrics.request_exact_match}/{metrics.request_exact_instances})"
        ),
        (
            f"- Numeric constraint accuracy: "
            f"{metrics.numeric_constraint_accuracy:.3f} "
            f"({metrics.numeric_matches}/{metrics.numeric_instances})"
        ),
        (
            f"- Element constraint accuracy: "
            f"{metrics.element_constraint_accuracy:.3f} "
            f"({metrics.element_matches}/{metrics.element_instances})"
        ),
        f"- Ambiguity recall: {metrics.ambiguity_recall:.3f}",
        f"- Conflict recall: {metrics.conflict_recall:.3f}",
        f"- Unsupported accuracy: {metrics.unsupported_accuracy:.3f}",
        f"- Injection resilience: {metrics.injection_resilience:.3f}",
        f"- Avg latency ms: {metrics.average_latency_ms:.3f}",
        f"- Avg input tokens: {metrics.average_input_tokens:.3f}",
        f"- Avg output tokens: {metrics.average_output_tokens:.3f}",
        f"- Avg reasoning tokens: {metrics.average_reasoning_tokens:.3f}",
        "",
    ]
    return "\n".join(lines)


def failures_to_markdown(outcomes: tuple[PlannerCaseResult, ...]) -> str:
    """Render redacted failures as a Markdown table."""
    failures = _failures(outcomes)
    lines = [
        "# Redacted Failures",
        "",
        "| id | tags | expected | actual | schema | error |",
        "|---|---|---|---|---|---|",
    ]
    for result in failures:
        actual = result.actual_status.value if result.actual_status is not None else "-"
        lines.append(
            f"| {result.id} | {', '.join(result.tags)} | "
            f"{result.expected_status.value} | {actual} | "
            f"{result.schema_success} | {result.error_type or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_eval_report(
    run_dir: Path,
    metrics: PlannerEvalMetrics,
    outcomes: tuple[PlannerCaseResult, ...],
) -> None:
    """Write metrics and redacted failures as JSON and Markdown."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(metrics_to_json(metrics), encoding="utf-8")
    (run_dir / "metrics.md").write_text(metrics_to_markdown(metrics), encoding="utf-8")
    (run_dir / "failures.json").write_text(failures_to_json(outcomes), encoding="utf-8")
    (run_dir / "failures.md").write_text(
        failures_to_markdown(outcomes), encoding="utf-8"
    )
