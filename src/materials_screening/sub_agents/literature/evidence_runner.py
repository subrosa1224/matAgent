"""Evidence-aware wrapper for the LiteratureAgent runner."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from materials_screening.agent.models import AgentResult
from materials_screening.master.literature_evidence_trial import (
    LiteratureEvidenceTrialService,
    LiteratureReportNotFoundError,
)
from materials_screening.sub_agents.literature.metric_coverage import (
    TaskMetricRequirements,
    load_task_metric_requirements,
    render_required_metric_coverage,
)

LITERATURE_DATASET_HANDOFF_MARKER = "LITERATURE_DATASET_HANDOFF"


class LiteratureRunner(Protocol):
    def ask(
        self, *, message: str, conversation_id: str | None = None
    ) -> AgentResult: ...


class EvidenceAwareLiteratureRunner:
    """Reuse exact saved evidence; otherwise run a fresh literature search."""

    def __init__(
        self,
        *,
        base_runner: LiteratureRunner,
        evidence_service: LiteratureEvidenceTrialService,
    ) -> None:
        self._base = base_runner
        self._evidence = evidence_service

    def ask(self, *, message: str, conversation_id: str | None = None) -> AgentResult:
        if "候选池文献预检" in message:
            # Pool pre-screening is a fresh metadata workflow, not an exact
            # full-text topic. Preserve its larger bounded formula contract.
            return self._base.ask(message=message, conversation_id=conversation_id)
        policy = _exact_metric_policy(message)
        try:
            if policy is None:
                handoff = self._evidence.prepare(topic=message)
            else:
                handoff = self._evidence.prepare(
                    topic=message, metric_requirements=policy
                )
        except LiteratureReportNotFoundError:
            return self._base.ask(message=message, conversation_id=conversation_id)

        digest = hashlib.sha256(f"{handoff.report_id}|{message}".encode()).hexdigest()[
            :24
        ]
        resolved_conversation = conversation_id or f"lit_evidence_{digest}"
        lines = [
            "已复用与当前问题完全匹配的本地文献证据报告。",
            "",
            f"- 文献报告 ID：`{handoff.report_id}`",
            f"- 纳入论文：{len(handoff.document_ids)} 篇",
            f"- 可数值化记录：{handoff.record_count} 条",
        ]
        if handoff.dataset_id is not None:
            lines.extend(
                (
                    f"- 文献测量数据集：`{handoff.dataset_id}`",
                    f"- {LITERATURE_DATASET_HANDOFF_MARKER}: "
                    f"dataset_id={handoff.dataset_id}",
                )
            )
        else:
            lines.append("- 未发现可安全交给数据分析 Agent 的数值记录。")
        report_path = Path("data/literature_user_reports") / f"{handoff.report_id}.json"
        if report_path.is_file():
            lines.append(f"- [完整证据报告]({report_path.resolve().as_posix()})")
        lines.extend(
            (
                "",
                render_required_metric_coverage(
                    handoff.required_metric_coverage,
                    scope_label="实际交给数据分析的记录",
                ),
                "",
                handoff.literature_markdown,
            )
        )
        return AgentResult(
            conversation_id=resolved_conversation,
            user_turn_id=f"lit_turn_{digest}",
            status="completed",
            final_status="completed",
            response_text="\n".join(lines),
            selected_tools=("reuse_exact_literature_report",),
            evidence_ids=handoff.evidence_ids,
            warnings=handoff.warnings,
        )


def _exact_metric_policy(message: str) -> TaskMetricRequirements | None:
    """Use only explicit local requirements for this exact normalized topic.

    These files hold metric names/units, never measurements or reference answers.
    Malformed or duplicate configurations fail visibly instead of hiding coverage.
    """
    root = Path("data/literature_metric_requirements")
    if not root.exists():
        return None
    paths = sorted(root.glob("*.json"))
    if len(paths) > 50:
        raise ValueError("必需指标配置文件超过50份，未自动选择清单")
    matches = []
    for path in paths:
        policy = load_task_metric_requirements(path)
        if policy.topic == " ".join(message.split()):
            matches.append(policy)
    if len(matches) > 1:
        raise ValueError("当前问题有多份必需指标清单，未自动选择或合并")
    return matches[0] if matches else None
