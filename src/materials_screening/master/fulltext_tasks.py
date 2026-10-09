"""Server-owned fulltext task contracts, independent of UI and model output."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from materials_screening.sub_agents.literature.preview import PaperPreview

from .figure_evidence_contracts import FigureBatchReference
from .fulltext_analysis_contracts import FulltextAnalysisReference
from .fulltext_snapshots import SnapshotReference
from .staged_fulltext_handoff import StagedHandoff
from .staged_fulltext_store import StagedReference

TaskStage = Literal[
    "waiting_fulltext",
    "previewing",
    "awaiting_clarification",
    "awaiting_figure_review",
    "extracting",
    "analyzing",
    "reporting",
    "ready_to_resume",
    "failed",
    "finished",
]
_CONVERSATION = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,253}$"


class PreviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["extract", "background", "exclude", "hold", "clarify"]
    reason: str = Field(min_length=1, max_length=5000)
    policy_version: Literal["master-preview-v1"] = "master-preview-v1"


class MeasurementHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    snapshot_id: str = Field(pattern=r"^snapshot-[a-f0-9]{32}$")
    dataset_id: str | None = Field(default=None, pattern=r"^dataset-[a-f0-9]{24}$")
    record_count: int = Field(ge=0)
    analysis_mode: Literal["trial"] = "trial"
    isolated_measurement_ids: tuple[str, ...] = ()
    isolation_reasons: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class PreviewRetryAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    document_id: str = Field(pattern=r"^doc-[a-f0-9]{24}$")
    operation_id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,100}$")
    expected: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: Literal["started", "succeeded", "failed"] = "started"
    previous_preview: PaperPreview | None = None
    proposed_preview: PaperPreview | None = None
    reason: str = Field(default="重试已登记；尚未完成。", max_length=1000)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FulltextTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(
        default_factory=lambda: "task-fulltext-" + uuid4().hex,
        pattern=r"^task-fulltext-[a-f0-9]{32}$",
    )
    conversation_id: str = Field(pattern=_CONVERSATION)
    version: Literal["fulltext-task-v1"] = "fulltext-task-v1"
    original_question: str = Field(min_length=1, max_length=122880)
    user_instructions: tuple[str, ...] = Field(default=(), max_length=100)
    stage: TaskStage = "waiting_fulltext"
    resume_stage: TaskStage | None = None
    analysis_mode: Literal["trial"] = "trial"
    artifact_refs: tuple[str, ...] = Field(default=(), max_length=100)
    document_ids: tuple[str, ...] = Field(default=(), max_length=100)
    database_query_ids: tuple[str, ...] = Field(default=(), max_length=100)
    database_dataset_ids: tuple[str, ...] = Field(default=(), max_length=100)
    analysis_ids: tuple[str, ...] = Field(default=(), max_length=100)
    previews: tuple[PaperPreview, ...] = Field(default=(), max_length=100)
    preview_decisions: dict[str, PreviewDecision] = Field(default_factory=dict)
    preview_retry_history: tuple[PreviewRetryAttempt, ...] = Field(
        default=(), max_length=200
    )
    # Instruction offsets survive restarts. Old detailed instructions cannot
    # approve a newly generated preview; only a subsequent explicit choice can.
    preview_choice_offsets: dict[str, int] = Field(default_factory=dict)
    extraction_snapshots: dict[str, SnapshotReference] = Field(default_factory=dict)
    snapshot_history: tuple[SnapshotReference, ...] = Field(default=(), max_length=1000)
    measurement_handoffs: dict[str, MeasurementHandoff] = Field(default_factory=dict)
    # Independent versioned references; legacy snapshots remain readable.
    staged_extractions: dict[str, StagedReference] = Field(default_factory=dict)
    staged_extraction_history: tuple[StagedReference, ...] = Field(
        default=(), max_length=1000
    )
    staged_handoffs: dict[str, StagedHandoff] = Field(default_factory=dict)
    staged_handoff_history: tuple[StagedHandoff, ...] = Field(
        default=(), max_length=1000
    )
    fulltext_analysis_ref: FulltextAnalysisReference | None = None
    figure_review_policy: Literal["disabled", "figure-evidence-review-v1"] = "disabled"
    figure_evidence_ref: FigureBatchReference | None = None
    warnings: tuple[str, ...] = Field(default=(), max_length=100)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def safe_references(self) -> FulltextTask:
        if (
            self.figure_evidence_ref is not None
            or self.stage == "awaiting_figure_review"
        ) and self.figure_review_policy == "disabled":
            raise ValueError("图证据核对没有当前任务的启用策略")
        patterns = {
            "artifact_refs": r"artifact-pdf-[a-f0-9]{24}",
            "document_ids": r"doc-[a-f0-9]{24}",
            "database_query_ids": r"query-[a-f0-9]{32}",
            "database_dataset_ids": r"dataset-[a-f0-9]{24}",
            "analysis_ids": r"analysis-[a-f0-9]{32}",
        }
        for field, pattern in patterns.items():
            values = getattr(self, field)
            if len(set(values)) != len(values) or any(
                re.fullmatch(pattern, value) is None for value in values
            ):
                raise ValueError("任务引用包含重复或无效标识")
        if self.stage == "ready_to_resume" and self.resume_stage in {
            None,
            "ready_to_resume",
            "finished",
        }:
            raise ValueError("暂停任务缺少可恢复阶段")
        preview_ids = [preview.document_id for preview in self.previews]
        if self.document_ids != tuple(
            "doc-" + ref.removeprefix("artifact-pdf-") for ref in self.artifact_refs
        ):
            raise ValueError("论文身份与附件哈希引用不一致")
        if (
            len(set(preview_ids)) != len(preview_ids)
            or any(
                preview.document_id not in self.document_ids
                or preview.topic != self.original_question
                for preview in self.previews
            )
            or not set(self.preview_decisions).issubset(preview_ids)
        ):
            raise ValueError("预览结果或决定不属于当前任务附件与原问题")
        if any(
            key != ref.document_id or key not in self.document_ids
            for key, ref in self.extraction_snapshots.items()
        ):
            raise ValueError("提取快照不属于当前任务附件")
        if len({a.operation_id for a in self.preview_retry_history}) != len(
            self.preview_retry_history
        ):
            raise ValueError("预览重试操作重复")
        for doc in self.document_ids:
            if sum(a.document_id == doc for a in self.preview_retry_history) > 2:
                raise ValueError("每篇论文最多手动重试2次")
        for attempt in self.preview_retry_history:
            if attempt.document_id not in self.document_ids or any(
                p is not None
                and (
                    p.document_id != attempt.document_id
                    or p.topic != self.original_question
                )
                for p in (attempt.previous_preview, attempt.proposed_preview)
            ):
                raise ValueError("重试记录不属于当前任务")
        if any(
            doc not in self.document_ids
            or not isinstance(offset, int)
            or not 0 <= offset <= len(self.user_instructions)
            for doc, offset in self.preview_choice_offsets.items()
        ):
            raise ValueError("新预览选择状态无效")
        if len({ref.snapshot_id for ref in self.snapshot_history}) != len(
            self.snapshot_history
        ):
            raise ValueError("重复的提取快照历史")
        history = {ref.snapshot_id: ref for ref in self.snapshot_history}
        if any(
            ref.document_id not in self.document_ids for ref in self.snapshot_history
        ) or any(
            history.get(ref.snapshot_id) != ref
            for ref in self.extraction_snapshots.values()
        ):
            raise ValueError("当前快照未关联任务历史")
        if any(
            key not in self.extraction_snapshots
            or handoff.snapshot_id != self.extraction_snapshots[key].snapshot_id
            for key, handoff in self.measurement_handoffs.items()
        ):
            raise ValueError("测量交接不属于当前提取快照")
        stage_history = {ref.record_id: ref for ref in self.staged_extraction_history}
        if (
            len(stage_history) != len(self.staged_extraction_history)
            or any(
                ref.document_id not in self.document_ids
                for ref in self.staged_extraction_history
            )
            or any(
                key != ref.document_id
                or key not in self.document_ids
                or stage_history.get(ref.record_id) != ref
                for key, ref in self.staged_extractions.items()
            )
        ):
            raise ValueError("分阶段提取引用不属于任务附件或历史")
        if any(
            key not in self.staged_extractions
            or handoff.reference != self.staged_extractions[key]
            for key, handoff in self.staged_handoffs.items()
        ) or any(
            row.reference.record_id not in stage_history
            or stage_history[row.reference.record_id] != row.reference
            for row in self.staged_handoff_history
        ):
            raise ValueError("分阶段交接不属于已保存来源记录")
        return self


class FulltextTaskSelectionError(ValueError):
    pass


def select_fulltext_task(
    tasks: tuple[FulltextTask, ...], *, conversation_id: str, task_id: str | None = None
) -> FulltextTask | None:
    if any(task.conversation_id != conversation_id for task in tasks):
        raise FulltextTaskSelectionError("任务不属于当前会话。")
    if task_id is not None:
        selected = next((task for task in tasks if task.task_id == task_id), None)
        if selected is None:
            raise FulltextTaskSelectionError("当前会话没有指定任务。")
        if selected.stage == "finished":
            raise FulltextTaskSelectionError("指定任务已经结束，请明确发起新的任务。")
        return selected
    active = tuple(task for task in tasks if task.stage != "finished")
    if len(active) > 1:
        raise FulltextTaskSelectionError(
            "当前有多个未完成任务，请明确选择要继续的任务。"
        )
    return active[0] if active else None


def new_fulltext_task(
    conversation_id: str,
    question: str,
    *,
    state: dict[str, Any] | None = None,
    task_id: str | None = None,
) -> FulltextTask:
    state = state or {}
    calls = set(state.get("executed_call_ids", []))
    current = [
        row
        for row in state.get("sub_agent_results", [])
        if isinstance(row, dict) and row.get("call_id") in calls
    ]
    receipt = "\n".join(str(row.get("response_text", "")) for row in current)

    def references(pattern: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(re.findall(pattern, receipt)))[:100]

    arguments = {"task_id": task_id} if task_id is not None else {}
    return FulltextTask(
        **arguments,
        conversation_id=conversation_id,
        original_question=question,
        database_query_ids=references(r"\bquery-[a-f0-9]{32}\b"),
        database_dataset_ids=references(r"\bdataset-[a-f0-9]{24}\b"),
        analysis_ids=references(r"\banalysis-[a-f0-9]{32}\b"),
    )


def requests_fulltext_task(message: str) -> bool:
    """Conservative bootstrap from the existing affirmative chain recognizer.

    Structured planning will extend coverage; an isolated '分析' or a PDF upload
    is not authority to initiate a full research chain.
    """
    from .master_nodes import _requires_material_screening_chain

    affirmative = re.sub(
        r"(?:不需要|不要|无需|不必|不用|不做)[^。；;，,\n]*", "", message.casefold()
    )
    return (
        _requires_material_screening_chain(affirmative)
        and re.search(
            r"全文|\bpdf\b|full.?text|提取[^。；;\n]{0,120}"
            r"(?:实验|指标|测量|降解|响应|物相)",
            affirmative,
        )
        is not None
    )


def requests_fulltext_resume(message: str) -> bool:
    """Recognize continuation references, not standalone topic questions.

    This never chooses a task or changes its analysis policy. The server task
    selector and later preview decision must still validate the request.
    """
    normalized = message.strip().casefold()
    return (
        normalized in {"", "继续", "continue", "重试"}
        # Users confirming the displayed previews may say "全部内容" rather
        # than repeat "论文". Recognize only an explicit opening command and
        # a complete scope phrase, not questions about that phrase. This is
        # routing only: server task selection and fresh-preview consent remain
        # authoritative, including ambiguity and conversation ownership checks.
        or re.match(
            r"(?:请\s*)?(?:确认|开始)\s*(?:详细分析|深度分析|分析)\s*"
            r"(?:全部|所有)(?:内容|论文|文献|pdf)(?=$|[。；;，,！!\s])",
            normalized,
        )
        is not None
        or re.search(
            r"第\s*[\d零〇一二三四五六七八九十两]+(?:\s*[、,，和及]\s*(?:第\s*)?[\d零〇一二三四五六七八九十两]+)*\s*篇|"
            r"这些(?:论文|文献|pdf)|上传的(?:论文|文献|pdf)|"
            r"(?:只|仅|暂不|不要|不用).{0,30}(?:预览|详细分析|深度分析)|"
            r"继续(?:详细分析|深度分析|提取|全文)|"
            r"(?:预览|深度分析|详细分析|综合|总结|对比).*(?:论文|文献|pdf)|"
            r"task-fulltext-[a-f0-9]{32}",
            normalized,
        )
        is not None
    )


def referenced_fulltext_task(message: str) -> str | None:
    references = tuple(
        dict.fromkeys(re.findall(r"task-fulltext-[a-f0-9]{32}", message))
    )
    if len(references) > 1:
        raise FulltextTaskSelectionError("一条接续消息只能指定一个全文任务。")
    return references[0] if references else None
