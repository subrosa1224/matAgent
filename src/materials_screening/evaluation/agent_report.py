"""Single-agent evaluation report builders (S3.5-M7).

Failures are redacted: only ids, expectations and safe observed flags are
reported, never user text, tool arguments, raw model responses or reasoning.
"""

import json
from pathlib import Path

from materials_screening.evaluation.agent_evaluator import (
    AgentCaseResult,
    AgentEvalMetrics,
    AgentTurnObservation,
    _turn_failed,
)


def _case_failed(result: AgentCaseResult) -> bool:
    if result.error_type is not None:
        return True
    if result.multi_turn and result.multi_turn_success is False:
        return True
    return any(_turn_failed(turn) for turn in result.turns)


def _redacted_turn(turn: AgentTurnObservation) -> dict[str, object]:
    """Redacted per-turn failure record; no user text or raw output."""
    return {
        "turn_index": turn.turn_index,
        "expected_tool": turn.expected_tool,
        "emitted_tools": list(turn.emitted_tools),
        "expected_final_status": turn.expected_final_status,
        "final_status": turn.final_status,
        "error_code": turn.error_code,
        "flags": [
            flag
            for flag, value in (
                ("tool_selection", not turn.tool_selection_ok),
                ("argument_schema", not turn.tool_argument_schema_ok),
                ("unauthorized_tool", turn.unauthorized_tool_used),
                ("workflow_bypass", turn.workflow_bypassed),
                ("duplicate_side_effect", turn.duplicate_side_effect),
                ("non_ready_access", turn.non_ready_access),
                ("final_schema", not turn.final_schema_ok),
                ("evidence_grounding", not turn.evidence_grounded),
                ("active_thread", not turn.active_thread_ok),
                ("loop_limit", turn.loop_limit_violation),
                ("secret_leak", turn.secret_leaked),
            )
            if value
        ],
    }


def _redacted_failure(result: AgentCaseResult) -> dict[str, object]:
    return {
        "id": result.id,
        "category": result.category,
        "tags": list(result.tags),
        "error_type": result.error_type,
        "turns": [_redacted_turn(turn) for turn in result.turns],
    }


def _failures(results: tuple[AgentCaseResult, ...]) -> list[AgentCaseResult]:
    return [result for result in results if _case_failed(result)]


def metrics_to_json(metrics: AgentEvalMetrics) -> str:
    """Serialize metrics as indented JSON."""
    return metrics.model_dump_json(indent=2)


def failures_to_json(results: tuple[AgentCaseResult, ...]) -> str:
    """Serialize redacted failure records as indented JSON."""
    payload = [_redacted_failure(result) for result in _failures(results)]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def metrics_to_markdown(metrics: AgentEvalMetrics) -> str:
    """Render metrics as a Markdown report."""
    return "\n".join(
        [
            "# Single Agent Evaluation Metrics",
            "",
            f"- Total cases: {metrics.total_cases}",
            f"- Total turns: {metrics.total_turns}",
            (
                f"- Tool selection accuracy: {metrics.tool_selection_accuracy:.3f} "
                f"({metrics.tool_selection_accurate}/{metrics.total_turns})"
            ),
            (
                f"- Tool argument schema success: "
                f"{metrics.tool_argument_schema_success_rate:.3f} "
                f"({metrics.tool_argument_schema_ok_turns} turns)"
            ),
            (
                f"- Unauthorized tool rate: {metrics.unauthorized_tool_rate:.3f} "
                f"({metrics.unauthorized_tool_turns})"
            ),
            (
                f"- Workflow bypass rate: {metrics.workflow_bypass_rate:.3f} "
                f"({metrics.workflow_bypasses})"
            ),
            (
                f"- Duplicate side effect rate: "
                f"{metrics.duplicate_side_effect_rate:.3f} "
                f"({metrics.duplicate_side_effects})"
            ),
            (
                f"- Non-ready access rate: {metrics.non_ready_access_rate:.3f} "
                f"({metrics.non_ready_accesses})"
            ),
            (
                f"- Final schema success rate: {metrics.final_schema_success_rate:.3f} "
                f"({metrics.final_schema_success}/{metrics.total_turns})"
            ),
            (
                f"- Evidence grounding rate: {metrics.evidence_grounding_rate:.3f} "
                f"({metrics.evidence_grounded}/{metrics.evidence_required_turns})"
            ),
            (
                f"- Active thread accuracy: {metrics.active_thread_accuracy:.3f} "
                f"({metrics.active_thread_accurate}/{metrics.total_turns})"
            ),
            (
                f"- Multi-turn success rate: {metrics.multi_turn_success_rate:.3f} "
                f"({metrics.multi_turn_success}/{metrics.multi_turn_cases})"
            ),
            (
                f"- Loop limit violation rate: "
                f"{metrics.loop_limit_violation_rate:.3f} "
                f"({metrics.loop_limit_violations})"
            ),
            (
                f"- Injection resilience rate: "
                f"{metrics.injection_resilience_rate:.3f} "
                f"({metrics.injection_resilient}/{metrics.injection_cases})"
            ),
            f"- Avg model calls: {metrics.average_model_calls:.3f}",
            f"- Avg tool calls: {metrics.average_tool_calls:.3f}",
            f"- Avg turn latency ms: {metrics.average_turn_latency_ms:.3f}",
            f"- Avg input tokens: {metrics.average_input_tokens:.3f}",
            f"- Avg output tokens: {metrics.average_output_tokens:.3f}",
            f"- Avg reasoning tokens: {metrics.average_reasoning_tokens:.3f}",
            "",
        ]
    )


def failures_to_markdown(results: tuple[AgentCaseResult, ...]) -> str:
    """Render redacted failures as a Markdown table."""
    failures = _failures(results)
    lines = [
        "# Redacted Agent Failures",
        "",
        "| id | category | turns | error |",
        "|---|---|---|---|",
    ]
    for result in failures:
        lines.append(
            f"| {result.id} | {result.category} | {result.turn_count} | "
            f"{result.error_type or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_agent_eval_report(
    run_dir: Path,
    metrics: AgentEvalMetrics,
    results: tuple[AgentCaseResult, ...],
) -> None:
    """Write metrics and redacted failures as JSON and Markdown."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "agent_metrics.json").write_text(
        metrics_to_json(metrics), encoding="utf-8"
    )
    (run_dir / "agent_metrics.md").write_text(
        metrics_to_markdown(metrics), encoding="utf-8"
    )
    (run_dir / "agent_failures.json").write_text(
        failures_to_json(results), encoding="utf-8"
    )
    (run_dir / "agent_failures.md").write_text(
        failures_to_markdown(results), encoding="utf-8"
    )
