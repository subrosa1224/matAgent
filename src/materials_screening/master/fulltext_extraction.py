"""Bounded generic extraction with immutable per-response checkpoints."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

from materials_screening.agent.models import AgentResult
from materials_screening.sub_agents.literature.matrix import validate_matrix_evidence
from materials_screening.sub_agents.literature.matrix_automation import (
    AutomatedMatrixExtractor,
    _chunk_batches,
    _select_evidence,
)

from .artifact_registry import ArtifactRegistry
from .fulltext_handoff import snapshot_trial_handoff
from .fulltext_preview import _Pause
from .fulltext_snapshots import (
    CheckedChunkStore,
    ExtractionSnapshot,
    ExtractionSnapshotStore,
    MatrixPayload,
    source_identities,
)
from .fulltext_tasks import FulltextTask

_TASK_PROMPT_VERSION = "master-task-extraction-v2"


class _TaskCalls:
    def __init__(self, calls, task, *, chunks=(), source_selection=False):
        self.calls, self.task = calls, task
        self.chunks, self.source_selection = chunks, source_selection
        self._selection_attempts = {}

    def generate_structured(self, **kwargs):
        sources = None
        if self.source_selection:
            from .fulltext_matrix_selection import (
                SELECTION_PROMPT,
                SelectedMatrixBatch,
                supplied_sources,
            )

            sources, evidence = supplied_sources(kwargs["user_text"], self.chunks)
            key = hashlib.sha256(kwargs["user_text"].encode()).hexdigest()
            attempt = self._selection_attempts.get(key, 0) + 1
            self._selection_attempts[key] = attempt
            kwargs.update(
                user_text=evidence,
                system_prompt=SELECTION_PROMPT
                + (
                    "\nRetry: return valid compact JSON only; use only supplied "
                    "source IDs and exact sample labels. Omit missing or ambiguous "
                    "measurements rather than inventing fields."
                    if attempt > 1
                    else ""
                ),
                output_model=SelectedMatrixBatch,
                schema_name="master_matrix_source_selection_v1",
            )
        kwargs["system_prompt"] += (
            "\nThe user request below defines extraction scope, not paper facts. "
            "Keep database-computed properties separate. Never fill missing evidence "
            "from the task. Extract actual experimental metrics and preparation/test "
            "conditions needed by this task, preserving controls and sample identity."
            + (
                "\nReturn compact JSON. For source_quote copy the shortest contiguous "
                "verbatim sentence or table row that supports the sample/value, not "
                "the whole chunk or paragraph repeated for every row. Preserve the "
                "units and sample labels; do not shorten a quote if doing so makes "
                "the binding ambiguous. Use one local group_key per exact sample "
                "and include it in every measurement. "
                if sources is None
                else "\nUse only the supplied source_id and exact sample_label; "
                "source quotes and internal group keys are resolved by the server. "
            )
            + "Detailed extraction has been explicitly authorized for this selected "
            "paper; follow-up requirements are chronological, with newer "
            "instructions superseding "
            "older preview-only instructions. Never infer missing experimental facts."
        )
        kwargs["user_text"] = (
            "User request (requirements only):\n"
            + self.task.original_question
            + "\nFollow-up requirements:\n"
            + "\n".join(self.task.user_instructions)
            + "\n\nPaper evidence (the only source of facts):\n"
            + kwargs["user_text"]
        )
        response = self.calls.generate_structured(**kwargs)
        if sources is not None:
            from materials_screening.llm.base import StructuredProviderResponse

            from .fulltext_matrix_selection import resolve_selected_matrix

            return StructuredProviderResponse(
                **response.model_dump(exclude={"parsed"}),
                parsed=resolve_selected_matrix(response.parsed, sources),
            )
        return response


class FulltextExtractionProcessor:
    def __init__(
        self,
        *,
        artifacts: ArtifactRegistry,
        snapshots: ExtractionSnapshotStore,
        dataset_factory,
        model_profile: str,
        max_output_bytes=65536,
        max_output_tokens=8192,
        max_evidence_chars=24000,
        batch_chars=8000,
        analysis_processor=None,
        figure_review_service=None,
        source_selection=False,
    ):
        self.artifacts, self.snapshots = artifacts, snapshots
        self.dataset_factory, self.model_profile = dataset_factory, model_profile
        self.max_output_bytes = max_output_bytes
        self.source_selection = source_selection
        self.analysis_processor = analysis_processor
        self.figure_review_service = figure_review_service
        self.max_output_tokens = min(8192, max_output_tokens)
        self.max_evidence_chars = min(24000, max_evidence_chars)
        # A dense table plus repeated evidence quotes can exceed the configured
        # output budget even when the evidence fits the generic 8k window. Use
        # smaller deterministic batches, not a larger token/request allowance.
        self.batch_chars = min(8000, batch_chars, self.max_output_tokens)

    def _policy(self, task):
        return hashlib.sha256(
            json.dumps(
                {
                    "version": "master-matrix-v1",
                    "task_prompt_version": _TASK_PROMPT_VERSION,
                    "task_id": task.task_id,
                    "question": task.original_question,
                    "instructions": task.user_instructions,
                    "model_profile": self.model_profile,
                    "max_output_tokens": self.max_output_tokens,
                    "max_evidence_chars": self.max_evidence_chars,
                    "batch_chars": self.batch_chars,
                    **(
                        {"source_selection_version": "source-selection-v2"}
                        if self.source_selection
                        else {}
                    ),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()

    def run(
        self,
        task,
        *,
        chunks_by_document,
        refresh_sources,
        calls,
        save_task,
        user_turn_id,
    ):
        summaries, reused = [], 0
        latest = None

        def save(**updates):
            nonlocal task
            task = FulltextTask.model_validate(
                {
                    **task.model_dump(mode="json"),
                    **updates,
                    "updated_at": datetime.now(UTC),
                }
            )
            save_task(task)

        def checkpoint(snapshot):
            nonlocal latest
            ref = self.snapshots.save(snapshot)
            refs = {**task.extraction_snapshots, snapshot.document_id: ref}
            handoffs = {
                key: value
                for key, value in task.measurement_handoffs.items()
                if key != snapshot.document_id
            }
            save(
                extraction_snapshots=refs,
                snapshot_history=(*task.snapshot_history, ref),
                measurement_handoffs=handoffs,
            )
            latest = snapshot

        def on_batch(batch_index, parsed):
            updated = {**latest.batches, batch_index: parsed}
            checkpoint(
                latest.model_copy(
                    update={
                        "snapshot_id": "snapshot-" + uuid4().hex,
                        "parent_snapshot_id": latest.snapshot_id,
                        "batches": updated,
                        "created_at": datetime.now(UTC),
                    }
                )
            )

        def result(message, code, cancelled=False):
            body = (
                "详细提取阶段（新快照、待审核试运行，不是完整统计报告）：\n"
                + "\n".join(summaries)
            )
            if reused:
                body += (
                    f"\n复核并复用 {reused} 篇当前任务的成功提取快照，"
                    "未重新调用提取模型。"
                )
            text = body + "\n\n" + message
            warnings = ()
            if len(text.encode()) > self.max_output_bytes:
                warnings = ("显示已截断，完整提取记录保存在任务快照；统计尚未执行。",)
                prefix = message + "\n" + warnings[0] + "\n"
                text = prefix + body.encode()[
                    : max(0, self.max_output_bytes - len(prefix.encode()))
                ].decode("utf-8", "ignore")
                text = text.encode()[: self.max_output_bytes].decode("utf-8", "ignore")
            return AgentResult(
                conversation_id=task.conversation_id,
                user_turn_id=user_turn_id,
                status="cancelled" if cancelled else "error",
                final_status=None if cancelled else "error",
                response_text=text,
                model_call_count=calls.calls,
                selected_tools=("literature",),
                warnings=warnings,
                error=None
                if cancelled
                else {
                    "code": code,
                    "message": message,
                    "retryable": code
                    not in {
                        "FULLTEXT_ANALYSIS_UNAVAILABLE",
                        "FULLTEXT_REPORTING_UNAVAILABLE",
                    },
                },
            )

        try:
            save(stage="extracting", resume_stage=None)
            policy = self._policy(task)
            for doc_id, decision in task.preview_decisions.items():
                if decision.action != "extract":
                    continue
                calls.check()
                latest = None
                chunks = tuple(chunks_by_document[doc_id])
                identities = source_identities(chunks)
                selected = _select_evidence(
                    chunks, max_evidence_chars=self.max_evidence_chars
                )
                plan = tuple(_chunk_batches(selected, batch_chars=self.batch_chars))
                if not plan:
                    raise ValueError("No checked source chunks for extraction")
                old_ref = task.extraction_snapshots.get(doc_id)
                old = self.snapshots.load(old_ref) if old_ref else None
                if old is not None and (
                    old.task_id != task.task_id
                    or old.conversation_id != task.conversation_id
                    or old.document_id != doc_id
                    or old.source_identities != identities
                ):
                    raise ValueError("Snapshot belongs to a different task or source")
                matching = (
                    old is not None
                    and old.policy_sha256 == policy
                    and old.version == "master-matrix-v1"
                )
                if matching and old.status == "complete":
                    latest = old
                    validate_matrix_evidence(
                        CheckedChunkStore(chunks),
                        document_id=doc_id,
                        groups=old.matrix.groups,
                        measurements=old.matrix.measurements,
                    )
                    reused += 1
                else:
                    index = task.document_ids.index(doc_id)
                    artifact = self.artifacts.get(task.artifact_refs[index])
                    preview = next(
                        row for row in task.previews if row.document_id == doc_id
                    )
                    latest = ExtractionSnapshot(
                        attempt_id="attempt-" + uuid4().hex,
                        parent_snapshot_id=old.snapshot_id if old else None,
                        task_id=task.task_id,
                        conversation_id=task.conversation_id,
                        document_id=doc_id,
                        pdf_sha256=artifact.metadata["sha256"],
                        title=preview.title,
                        policy_sha256=policy,
                        original_question=task.original_question,
                        user_instructions=task.user_instructions,
                        model_profile=self.model_profile,
                        source_identities=identities,
                        planned_batches=len(plan),
                        batches=dict(old.batches) if matching else {},
                        status="collecting",
                    )
                    checkpoint(latest)

                    extracted = AutomatedMatrixExtractor(
                        _TaskCalls(
                            calls,
                            task,
                            chunks=chunks,
                            source_selection=self.source_selection,
                        ),
                        CheckedChunkStore(chunks),
                    ).extract(
                        document_id=doc_id,
                        chunks=chunks,
                        max_output_tokens=self.max_output_tokens,
                        max_evidence_chars=self.max_evidence_chars,
                        batch_chars=self.batch_chars,
                        cached_batches=latest.batches,
                        on_batch=on_batch,
                    )
                    calls.check()
                    if source_identities(refresh_sources(doc_id)) != identities:
                        raise ValueError("Source changed during extraction")
                    status = (
                        "complete" if len(latest.batches) == len(plan) else "partial"
                    )
                    payload = MatrixPayload.from_result(extracted)
                    checkpoint(
                        latest.model_copy(
                            update={
                                "snapshot_id": "snapshot-" + uuid4().hex,
                                "parent_snapshot_id": latest.snapshot_id,
                                "status": status,
                                "matrix": payload,
                                "warnings": (
                                    "提取采用有界证据窗口；必需指标覆盖尚未独立验收，不能据空结果断言论文没有实验数据。",
                                ),
                                "created_at": datetime.now(UTC),
                            }
                        )
                    )
                    if status != "complete":
                        save(stage="failed", resume_stage="extracting")
                        return result(
                            "部分证据批次提取失败；成功批次已保存，继续将复核来源并重试未成功批次。",
                            "FULLTEXT_EXTRACTION_FAILED",
                        )
                calls.check()
                prior_handoff = task.measurement_handoffs.get(doc_id)
                handoff = snapshot_trial_handoff(
                    latest,
                    chunks=chunks,
                    dataset_factory=self.dataset_factory,
                    existing=prior_handoff,
                )
                if prior_handoff is not None and handoff != prior_handoff:
                    raise ValueError("Successful snapshot handoff changed unexpectedly")
                save(
                    measurement_handoffs={**task.measurement_handoffs, doc_id: handoff}
                )
                summaries.append(
                    f"{latest.title}\n快照：{latest.snapshot_id}；"
                    f"已核对测量 {len(latest.matrix.measurements)} 条；"
                    f"试运行标量 {handoff.record_count} 条；"
                    f"隔离 {len(handoff.isolated_measurement_ids)} 条。"
                    + (
                        f" 数据集：{handoff.dataset_id}"
                        if handoff.dataset_id
                        else " 没有可安全交接的数值。"
                    )
                )
                summaries.append(_readable_details(latest, handoff))
                calls.check()
            has_data = any(
                row.record_count
                for doc_id, row in task.measurement_handoffs.items()
                if doc_id in task.preview_decisions
                and task.preview_decisions[doc_id].action == "extract"
            )
            save(
                stage="ready_to_resume",
                resume_stage="analyzing" if has_data else "reporting",
            )
            if task.figure_review_policy != "disabled":
                if self.figure_review_service is None:
                    raise ValueError("Figure review service unavailable")
                from .figure_evidence_gate import FigureReviewGate

                gate = FigureReviewGate(self.figure_review_service)
                task, paused = gate.before_analysis(
                    task,
                    save_task=save_task,
                    user_turn_id=user_turn_id,
                    model_calls=calls.calls,
                )
                if paused is not None:
                    return paused
            if self.analysis_processor is not None:
                return self.analysis_processor.run(
                    task,
                    calls=calls,
                    save_task=save_task,
                    chunks_by_document=chunks_by_document,
                    refresh_sources=refresh_sources,
                    user_turn_id=user_turn_id,
                )
            return result(
                "本任务的新实验快照与待审核试运行数据已准备；数据分析接续处理器尚未接入，未执行统计或最终报告。"
                if has_data
                else (
                    "已完成有界全文提取，但没有可安全交接的标量；"
                    "报告处理器尚未接入，不生成空表统计。"
                ),
                "FULLTEXT_ANALYSIS_UNAVAILABLE"
                if has_data
                else "FULLTEXT_REPORTING_UNAVAILABLE",
            )
        except _Pause as exc:
            if latest is not None and latest.status == "collecting":
                checkpoint(
                    latest.model_copy(
                        update={
                            "snapshot_id": "snapshot-" + uuid4().hex,
                            "parent_snapshot_id": latest.snapshot_id,
                            "status": "cancelled" if exc.cancelled else "partial",
                            "created_at": datetime.now(UTC),
                        }
                    )
                )
            save(stage="ready_to_resume", resume_stage="extracting")
            return result(
                "已停止；成功批次与快照保留，未启动下一请求或统计。"
                if exc.cancelled
                else "本轮调用或时间预算已到；成功批次已保存，继续只处理未完成提取。",
                "FULLTEXT_EXTRACTION_BUDGET",
                cancelled=exc.cancelled,
            )
        except Exception:
            save(stage="failed", resume_stage="extracting")
            return result(
                "提取、快照或测量交接执行失败；成功部分保留，旧矩阵未覆盖，未执行统计。可有界重试，无须改写原科研问题。",
                "FULLTEXT_EXTRACTION_FAILED",
            )


def _readable_details(snapshot, handoff):
    """Bounded caller display; the immutable snapshot retains complete evidence."""
    groups = {row.group_id: row for row in snapshot.matrix.groups}
    isolated = set(handoff.isolated_measurement_ids)
    lines = [
        f"提取时间：{snapshot.created_at.isoformat()}；审核：pending（待审核试运行）。",
        "以下是有界提取记录，不是实验统计、趋势排名或专业审核结论：",
    ]
    for row in snapshot.matrix.measurements:
        group = groups.get(row.group_id)
        label = group.label if group else row.group_id
        conditions = (
            json.dumps(group.conditions, ensure_ascii=False)
            if group and group.conditions
            else "未知（原文未绑定条件）"
        )
        variables = json.dumps(group.variables, ensure_ascii=False) if group else "未知"
        state = (
            "隔离："
            + "、".join(
                handoff.isolation_reasons.get(row.measurement_id, ("未通过交接检查",))
            )
            if row.measurement_id in isolated
            else "已交接 trial；任务指标与可比条件待核对"
        )
        quote = " ".join(row.source_quote.split())
        if len(quote) > 260:
            quote = quote[:260] + "…（显示截断，完整引文见快照）"
        page = (
            f"第 {row.page_from} 页"
            if row.page_from == row.page_to
            else f"第 {row.page_from}–{row.page_to} 页"
        )
        lines.append(
            f"\n样品：{label}；角色：{group.role if group else 'unknown'}；"
            f"指标：{row.metric}；原始值：{row.value_text} "
            f"{row.unit or '单位未知'}；{state}。\n"
            f"制备变量：{variables}；测试条件：{conditions or '未知'}。\n"
            f"原文（{page}，{row.chunk_id}）：{quote}"
        )
    lines.extend(snapshot.warnings)
    lines.extend(handoff.warnings)
    return "\n".join(lines)
