"""Master adapter: independent stage refs -> replayed trial -> cached statistics."""

from __future__ import annotations

import hashlib
import re
from contextlib import nullcontext
from datetime import UTC, datetime

from materials_screening.agent.models import AgentResult

from .fulltext_analysis import FulltextAnalysisProcessor
from .fulltext_preview import _Pause
from .fulltext_tasks import FulltextTask
from .staged_fulltext_engine import StagedFulltextEngine
from .staged_fulltext_evidence import _METRICS
from .staged_fulltext_handoff import render_staged_result, staged_trial_handoff


def scientific_requirements(task):
    # Continuation/confirmation routing is not a new scientific scope.
    instructions = [
        s
        for s in task.user_instructions
        if s.strip().casefold() not in {"继续", "重试", "continue", ""}
    ]
    return task.original_question + "\n" + "\n".join(instructions)


class _StageCalls:
    """Serial scoped provider, same parent budget/deadline/cancellation/counter."""

    def __init__(self, parent, factory):
        self.parent, self.factory, self.provider = parent, factory, None

    def check(self):
        self.parent.check()

    @property
    def calls(self):
        return self.parent.calls

    @calls.setter
    def calls(self, value):
        self.parent.calls = value

    @property
    def budget(self):
        return self.parent.budget

    def generate_structured(self, **kwargs):
        # No concurrent use: FulltextPreview owns the conversation lock until
        # its saved worker exits. Restore preview provider even on failure.
        old_factory, old_provider = self.parent.factory, self.parent.provider
        self.parent.factory, self.parent.provider = self.factory, self.provider
        try:
            return self.parent.generate_structured(**kwargs)
        finally:
            self.provider = self.parent.provider
            self.parent.factory, self.parent.provider = old_factory, old_provider


class StagedFulltextProcessor:
    def __init__(
        self,
        *,
        artifacts,
        stages,
        dataset_factory,
        agent_factory,
        model_profile,
        llm_factory=None,
        max_output_bytes=65536,
        max_output_tokens=4096,
    ):
        self.artifacts, self.stages, self.dataset_factory = (
            artifacts,
            stages,
            dataset_factory,
        )
        self.agent_factory, self.model_profile, self.llm_factory = (
            agent_factory,
            model_profile,
            llm_factory,
        )
        self.max_output_bytes = max_output_bytes
        self.engine = StagedFulltextEngine(
            store=stages,
            model_profile=model_profile,
            max_output_tokens=max_output_tokens,
        )

    def progress_signature(self, task):
        return {
            doc: dict(
                policy=self.stages.load(ref).policy_sha256,
                steps={
                    key: (s.status, s.attempts)
                    for key, s in self.stages.load(ref).steps.items()
                },
            )
            for doc, ref in task.staged_extractions.items()
        }

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
        calls = _StageCalls(calls, self.llm_factory) if self.llm_factory else calls
        reports, tools, phase = [], 0, "extracting"
        statistics_documents = 0

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

        def saved(ref):
            if task.staged_extractions.get(ref.document_id) == ref:
                return
            save(
                staged_extractions={**task.staged_extractions, ref.document_id: ref},
                staged_extraction_history=(*task.staged_extraction_history, ref),
                staged_handoffs={
                    doc: h
                    for doc, h in task.staged_handoffs.items()
                    if doc != ref.document_id
                },
            )

        def save_handoff(doc, handoff):
            if task.staged_handoffs.get(doc) != handoff:
                save(
                    staged_handoffs={**task.staged_handoffs, doc: handoff},
                    staged_handoff_history=(*task.staged_handoff_history, handoff),
                )

        def result(text, code=None, cancelled=False):
            text = text.encode()[: self.max_output_bytes].decode("utf-8", "ignore")
            return AgentResult(
                conversation_id=task.conversation_id,
                user_turn_id=user_turn_id,
                status="cancelled" if cancelled else "error" if code else "completed",
                final_status=None if cancelled else "error" if code else "completed",
                response_text=text,
                model_call_count=calls.calls,
                tool_call_count=tools,
                selected_tools=("literature", "data_analysis")
                if tools
                else ("literature",),
                error={"code": code, "message": text, "retryable": True}
                if code
                else None,
            )

        try:
            if task.figure_review_policy != "disabled":
                raise ValueError("New composite path cannot bypass figure review")
            requirements = scientific_requirements(task)
            metrics = tuple(
                m for m, p in _METRICS.items() if re.search(p, requirements, re.I)
            )
            if not metrics:
                return result(
                    "当前问题的实验指标还不能与已支持的提取项对应；已上传论文与历史结果保留。",
                    "FULLTEXT_STAGED_SCOPE_UNAVAILABLE",
                )
            save(stage="extracting", resume_stage=None)
            selected = [
                doc
                for doc, d in task.preview_decisions.items()
                if d.action == "extract"
            ]
            for doc in selected:
                phase = "extracting"
                index = task.document_ids.index(doc)
                artifact = self.artifacts.get(task.artifact_refs[index])
                preview = next(p for p in task.previews if p.document_id == doc)
                record = self.engine.run_document(
                    task_id=task.task_id,
                    conversation_id=task.conversation_id,
                    document_id=doc,
                    pdf_sha256=artifact.metadata["sha256"],
                    title=preview.title,
                    requirements=requirements,
                    requested_metrics=metrics,
                    chunks=chunks_by_document[doc],
                    calls=calls,
                    on_checkpoint=saved,
                    prior_ref=task.staged_extractions.get(doc),
                    refresh_sources=lambda doc=doc: refresh_sources(doc),
                )
                calls.check()
                handoff = staged_trial_handoff(
                    record,
                    reference=task.staged_extractions[doc],
                    chunks=refresh_sources(doc),
                    dataset_factory=self.dataset_factory,
                    existing=task.staged_handoffs.get(doc),
                )
                save_handoff(doc, handoff)
                phase = "analyzing"
                if handoff.record_count:
                    datasets = self.dataset_factory()
                    frame = datasets.load_dataframe(handoff.dataset_id)
                    if handoff.analysis_id:
                        analysis = datasets.get_analysis(handoff.analysis_id)
                        FulltextAnalysisProcessor._validate_analysis(
                            analysis, frame, handoff.dataset_id
                        )
                        if (
                            hashlib.sha256(
                                analysis.model_dump_json().encode()
                            ).hexdigest()
                            != handoff.analysis_sha256
                        ):
                            raise ValueError("Saved staged statistics changed")
                    else:
                        if self.agent_factory is None:
                            return result(
                                render_staged_result(record, handoff)
                                + "\n统计服务尚未接入，未生成最终建议。",
                                "FULLTEXT_ANALYSIS_UNAVAILABLE",
                            )
                        calls.check()
                        if calls.calls + 3 > calls.budget:
                            raise _Pause()
                        calls.calls += 3
                        agent = self.agent_factory()
                        context = (
                            agent if hasattr(agent, "__enter__") else nullcontext(agent)
                        )
                        with context as runner:
                            outcome = runner.ask(
                                message=f"描述统计 {handoff.dataset_id} "
                                "columns=numeric_value group_by=measurement_context"
                            )
                        tools += outcome.tool_call_count
                        if (
                            outcome.status != "completed"
                            or outcome.final_status != "completed"
                            or outcome.error
                        ):
                            raise ValueError("Staged descriptive agent failed")
                        if (
                            outcome.model_call_count != 3
                            or outcome.tool_call_count != 2
                        ):
                            raise ValueError(
                                "Staged descriptive operation exceeded bound"
                            )
                        ids = tuple(
                            dict.fromkeys(
                                re.findall(
                                    r"analysis-[a-f0-9]{32}", outcome.response_text
                                )
                            )
                        )
                        if len(ids) != 1:
                            raise ValueError("Missing staged analysis reference")
                        analysis = datasets.get_analysis(ids[0])
                        FulltextAnalysisProcessor._validate_analysis(
                            analysis, frame, handoff.dataset_id
                        )
                        if analysis.evidence_id not in outcome.evidence_ids:
                            raise ValueError("Statistics evidence reference mismatch")
                        handoff = handoff.model_copy(
                            update={
                                "analysis_id": analysis.analysis_id,
                                "analysis_sha256": hashlib.sha256(
                                    analysis.model_dump_json().encode()
                                ).hexdigest(),
                            }
                        )
                        save_handoff(doc, handoff)
                    calls.check()
                    statistics_documents += 1
                reports.append(render_staged_result(record, handoff))
            save(stage="finished", resume_stage=None)
            evidence_only = len(reports) - statistics_documents
            if not statistics_documents:
                statistics_note = "本轮仅整理文献证据，未执行描述统计。"
            else:
                statistics_note = (
                    f"已完成描述统计的论文：{statistics_documents}篇；"
                    f"另有{evidence_only}篇仅展示文献证据。"
                    if evidence_only
                    else ""
                ) + (
                    "已完成的描述统计仅描述已核验的来源记录，"
                    "不等于独立重复实验的统计推断。"
                )
            return result(
                "文献分析结果\n\n"
                + "\n\n---\n\n".join(reports)
                + "\n\n"
                + statistics_note
                + "数据库计算结果与这些实验记录保持分开；"
                "未证明可比时，不给出材料优劣排名。"
            )
        except _Pause as exc:
            save(stage="ready_to_resume", resume_stage=phase)
            return result(
                "本批已完成的结果已经保存，后续只处理未完成部分。"
                if not exc.cancelled
                else "已停止处理，成功结果保留，不启动下一请求。",
                "FULLTEXT_ANALYSIS_BUDGET"
                if phase == "analyzing"
                else "FULLTEXT_EXTRACTION_BUDGET",
                cancelled=exc.cancelled,
            )
        except Exception:
            save(stage="failed", resume_stage=phase)
            return result(
                "本次处理尚未完成；已核验的来源记录保留，失败步骤没有自动重复。无需重新上传或改写原问题。",
                "FULLTEXT_ANALYSIS_FAILED"
                if phase == "analyzing"
                else "FULLTEXT_EXTRACTION_FAILED",
            )
