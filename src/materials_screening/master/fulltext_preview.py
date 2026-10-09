"""Evidence-checked, resumable preview stage; never creates measurements."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from threading import Event, Lock
from typing import Any

from materials_screening.agent.models import AgentResult
from materials_screening.llm.errors import LLMError, LLMStructuredOutputError
from materials_screening.sub_agents.literature.preview import (
    PaperPreview,
    PaperPreviewExtractor,
    PreviewBoundaryPolicy,
    infer_preview_boundary_policy,
)
from materials_screening.sub_agents.literature.rag import (
    ChunkRecord,
    PyMuPdfParser,
    infer_pdf_title,
    parse_pdf_chunks,
)

from .artifact_registry import ArtifactRegistry
from .fulltext_tasks import FulltextTask, PreviewDecision, PreviewRetryAttempt


class _Pause(RuntimeError):
    def __init__(self, *, cancelled: bool = False):
        self.cancelled = cancelled


class _ProviderFailure(RuntimeError):
    """Do not let the preview extractor turn transport errors into fallbacks."""


class _PreviewCalls:
    def __init__(self, factory, budget, cancel_event, deadline):
        self.factory, self.budget = factory, budget
        self.cancel_event, self.deadline = cancel_event, deadline
        self.calls = 0
        self.provider = None

    def check(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise _Pause(cancelled=True)
        if time.monotonic() >= self.deadline:
            raise _Pause()

    def generate_structured(self, **kwargs):
        self.check()
        if self.calls >= self.budget:
            raise _Pause()
        if self.provider is None:
            self.provider = self.factory()
        self.check()
        self.calls += 1
        # A combined Master task has separate computed-pure-phase and paper
        # scopes. Never apply an MP-only purity condition to a composite device.
        kwargs["system_prompt"] += (
            "\nThis topic is a combined Master task. Materials Project computed "
            "structure restrictions apply only to database structures. Apply "
            "only explicitly requested literature restrictions to this paper; "
            "do not reject doped films or composite devices merely because MP "
            "queried pure phases. Listed material systems are alternatives unless "
            "the user explicitly requires them together in one paper. Do not "
            "invent a requirement for doping-mechanism explanations. Missing "
            "comparable conditions limit later ranking, not permission to extract "
            "available measurements. Boundary findings cannot create restrictions "
            "that the paper-stage user scope does not contain."
        )
        try:
            return self.provider.generate_structured(**kwargs)
        except LLMStructuredOutputError:
            raise  # Preserve the existing bounded schema-repair behavior.
        except LLMError as exc:
            raise _ProviderFailure("preview provider failed") from exc


def _ordinal_number(value: str) -> int:
    if value.isascii() and value.isdecimal():
        return int(value)
    digits = {char: index for index, char in enumerate("零一二三四五六七八九")}
    digits.update({"〇": 0, "两": 2})
    if value in digits:
        return digits[value]
    if value.count("十") == 1:
        left, right = value.split("十")
        if (not left or left in digits) and (not right or right in digits):
            return (digits[left] if left else 1) * 10 + (digits[right] if right else 0)
    raise ValueError("无法确定论文序号。")


def _selection(task: FulltextTask) -> tuple[tuple[int, ...], bool]:
    selected = tuple(range(len(task.artifact_refs)))
    preview_only = False
    for instruction in task.user_instructions:
        if re.search(r"全部|所有|all papers", instruction, re.I):
            selected = tuple(range(len(task.artifact_refs)))
        numeral = r"[\d零〇一二三四五六七八九十两]+"
        matches = list(
            re.finditer(
                rf"第\s*{numeral}(?:\s*[、,，和及]\s*(?:第\s*)?{numeral})*\s*篇",
                instruction,
            )
        )
        # Do not silently interpret an unsupported ordinal expression as 'all'.
        if re.search(
            r"第[^。；;\n]{0,30}篇",
            re.sub(
                rf"第\s*{numeral}(?:\s*[、,，和及]\s*(?:第\s*)?{numeral})*\s*篇",
                "",
                instruction,
            ),
        ):
            raise ValueError("论文选择表达不明确。")
        positive, negative = [], set()
        for match in matches:
            numbers = tuple(
                dict.fromkeys(
                    _ordinal_number(number)
                    for number in re.findall(numeral, match.group())
                )
            )
            if any(
                number < 1 or number > len(task.artifact_refs) for number in numbers
            ):
                raise ValueError("所选论文序号不在当前任务附件范围内。")
            prefix = re.split(r"[。；;，,\n]", instruction[: match.start()])[-1]
            suffix = re.split(r"[。；;，,\n]", instruction[match.end() :])[0]
            if re.search(r"(?:不要|不用|暂不).*(?:只|仅)", prefix):
                raise ValueError("否定的选择范围需要确认。")
            negation = (
                r"不要|不用|暂不|不需要|无需|不必|不做|不看|不分析|不提取|不预览|"
                r"排除|跳过|除了|除去"
            )
            excluded = re.search(negation, prefix) or re.search(negation, suffix)
            indices = tuple(number - 1 for number in numbers)
            if excluded:
                negative.update(indices)
            else:
                positive.extend(indices)
        if positive:
            selected = tuple(dict.fromkeys(positive))
        selected = tuple(index for index in selected if index not in negative)
        if not selected:
            raise ValueError("当前没有选中的论文。")
        if re.search(r"(?:只|仅)预览", instruction) or (
            not matches
            and re.search(r"(?:暂不|不要|不用)(?:(?:详细|深度)分析|提取)", instruction)
        ):
            preview_only = True
        elif (
            positive
            and re.search(r"分析|提取", instruction)
            or (
                not matches
                and re.search(
                    r"详细分析|深度分析|只分析|提取|确认分析|确认详细分析|开始分析",
                    instruction,
                )
            )
        ):
            preview_only = False
    return selected, preview_only


def _literature_boundary_text(question: str) -> str:
    """Use the explicit paper-search clause, not MP purity mentioned elsewhere."""
    match = re.search(
        r"(?:检索|搜集|搜索|结合)[^。；;\n]*(?:论文|文献|实验研究)[^。；;\n]*", question
    )
    return match.group() if match else ""


def _literature_screening_topic(question: str) -> str:
    """Keep the opening research aim and paper clause, not computed-data steps."""
    scope = _literature_boundary_text(question)
    if not scope:
        return question
    aim = re.split(r"[。\n]", question, maxsplit=1)[0]
    if not re.search(r"查询|检索|数据库|Materials Project|凸包|计算", aim, re.I):
        return aim + "。\n" + scope
    return scope


def _paper_boundary_policy(question: str) -> PreviewBoundaryPolicy:
    # Empty explicit policy is different from legacy None: model findings may
    # not become exclusions without a corroborating paper-stage constraint.
    return (
        infer_preview_boundary_policy(_literature_boundary_text(question))
        or PreviewBoundaryPolicy()
    )


def _unsupported_scope_exclusion(preview: PaperPreview, question: str) -> bool:
    policy = _paper_boundary_policy(question)
    return any(
        finding.severity == "exclude"
        and (
            finding.finding_type == "material_scope"
            and not policy.excluded_material_term_groups
            or finding.finding_type == "route_scope"
            and not policy.allowed_route_terms
        )
        for finding in preview.boundary_findings
    )


def _decision(
    preview: PaperPreview, *, only_preview: bool, question: str
) -> PreviewDecision:
    if only_preview:
        return PreviewDecision(action="hold", reason="用户要求只预览，未启动详细提取。")
    if preview.evidence_quality != "verbatim":
        return PreviewDecision(
            action="clarify", reason="代表引文已降级，论文对象与相关性需要核对。"
        )
    boundaries = infer_preview_boundary_policy(_literature_boundary_text(question))
    hard_exclusion = any(
        item.severity == "exclude" for item in preview.boundary_findings
    )
    if hard_exclusion:
        return PreviewDecision(
            action="exclude" if boundaries is not None else "clarify",
            reason=preview.reason
            if boundaries is not None
            else "预览提出排除，但需确认是否属于文献硬条件，而非数据库纯相限制。",
        )
    if (
        preview.article_type != "research"
        or preview.recommendation == "background_only"
    ):
        return PreviewDecision(action="background", reason=preview.reason)
    if preview.recommendation == "exclude" or preview.topic_relevance == "low":
        return PreviewDecision(
            action="background",
            reason="预览相关性较低，暂不作为目标实验数值来源。" + preview.reason,
        )
    if not re.search(
        r"提取|比较|对比|统计|详细分析|深度分析|extract|compar", question, re.I
    ):
        return PreviewDecision(action="hold", reason="原任务尚未明确深入分析目标。")
    return PreviewDecision(
        action="extract",
        reason="原任务已要求详细核对；预览合格，进入全文提取而非直接统计摘要。",
    )


def _validate_preview(
    preview: PaperPreview, task: FulltextTask, chunks: Sequence[ChunkRecord]
):
    chunk = next((row for row in chunks if row.chunk_id == preview.chunk_id), None)

    def compact(text):
        return "".join(text.split()).casefold()

    if (
        chunk is None
        or preview.generation_status != "model"
        or preview.topic != task.original_question
        or preview.document_id != chunk.document_id
        or preview.source_text_sha256 != chunk.text_sha256
        or (preview.page_from, preview.page_to) != (chunk.page_from, chunk.page_to)
        or not compact(preview.evidence_quote)
        or compact(preview.evidence_quote) not in compact(chunk.text)
        or preview.preview_version
        != PaperPreview.model_fields["preview_version"].default
    ):
        raise ValueError("预览与当前论文源片段不一致。")
    for finding in preview.boundary_findings:
        source = next((row for row in chunks if row.chunk_id == finding.chunk_id), None)
        if source is None or compact(finding.evidence_quote) not in compact(
            source.text
        ):
            raise ValueError("预览边界引用没有对应原文。")


class FulltextPreviewProcessor:
    """Lazy domain adapter; checkpoints every successfully validated paper."""

    def __init__(
        self,
        *,
        artifacts: ArtifactRegistry,
        store_factory: Callable[[], Any],
        rag_factory: Callable[[Any], Any],
        llm_factory: Callable[[], Any],
        parser=None,
        max_pdf_bytes=50 * 1024 * 1024,
        max_output_tokens=2048,
        max_output_bytes=65536,
        extraction_processor=None,
        auto_batches=False,
        batch_seconds=55,
        max_batches=40,
        batch_model_budget=None,
        require_preview_confirmation=False,
    ):
        self.artifacts, self.store_factory = artifacts, store_factory
        self.rag_factory, self.llm_factory = rag_factory, llm_factory
        self.parser = parser or PyMuPdfParser()
        self.max_pdf_bytes, self.max_output_tokens = max_pdf_bytes, max_output_tokens
        self.max_output_bytes = max_output_bytes
        self.extraction_processor = extraction_processor
        if not 0 < batch_seconds <= 3600 or not 1 <= max_batches <= 100:
            raise ValueError("Invalid fulltext batch limits")
        self.auto_batches, self.batch_seconds = auto_batches, batch_seconds
        self.max_batches = max_batches
        if batch_model_budget is not None and not 3 <= batch_model_budget <= 15:
            raise ValueError("Invalid fulltext model budget per batch")
        self.batch_model_budget = batch_model_budget
        self.require_preview_confirmation = require_preview_confirmation
        self._store = None
        self._rag = None
        self._index_lock = Lock()

    @staticmethod
    def retry_token(task, document_id):
        payload = task.model_dump(mode="json")
        return hashlib.sha256(
            json.dumps(
                [payload, document_id], sort_keys=True, ensure_ascii=False
            ).encode()
        ).hexdigest()

    @staticmethod
    def retry_allowed(task, document_id):
        if document_id not in task.document_ids:
            return False
        if (
            task.stage
            not in {
                "waiting_fulltext",
                "previewing",
                "awaiting_clarification",
                "ready_to_resume",
                "failed",
            }
            or task.resume_stage not in {None, "previewing"}
            or task.extraction_snapshots
            or task.snapshot_history
            or task.measurement_handoffs
            or task.fulltext_analysis_ref
            or task.figure_evidence_ref
        ):
            return False
        preview = next((p for p in task.previews if p.document_id == document_id), None)
        return (
            preview is not None
            and (
                preview.evidence_quality != "verbatim"
                or preview.generation_status != "model"
                or _unsupported_scope_exclusion(preview, task.original_question)
            )
        ) or (
            preview is None
            and task.stage == "failed"
            and task.resume_stage == "previewing"
        )

    def regenerate(
        self,
        task,
        *,
        document_id,
        operation_id,
        expected,
        save_task,
        model_budget,
        cancel_event,
        _calls=None,
    ):
        previous_operation = next(
            (a for a in task.preview_retry_history if a.operation_id == operation_id),
            None,
        )
        if previous_operation is not None:
            if (
                previous_operation.document_id != document_id
                or previous_operation.expected != expected
            ):
                raise ValueError("重试操作与原请求不一致。")
            # Interrupted 'started' operations never replay a remote call.
            return task
        if expected != self.retry_token(task, document_id):
            raise ValueError("预览状态已改变，请刷新后重试。")
        if not self.retry_allowed(task, document_id):
            raise ValueError("仅可重试当前任务的失败预览；不能重置提取或已完成任务。")
        if sum(a.document_id == document_id for a in task.preview_retry_history) >= 2:
            raise ValueError("每篇论文最多手动重试2次。")
        if model_budget <= 0:
            raise ValueError("本轮没有预览调用预算。")
        calls = _calls or _PreviewCalls(
            self.llm_factory,
            min(2, model_budget),
            cancel_event,
            time.monotonic() + self.batch_seconds,
        )
        calls.check()
        old = next((p for p in task.previews if p.document_id == document_id), None)
        attempt = PreviewRetryAttempt(
            document_id=document_id,
            operation_id=operation_id,
            expected=expected,
            previous_preview=old,
        )

        def save(**updates):
            nonlocal task
            current = FulltextTask.model_validate(
                {**task.model_dump(), **updates, "updated_at": datetime.now(UTC)}
            )
            save_task(current)
            task = current

        # Durable reservation before model work: crashes cannot replenish quota.
        save(preview_retry_history=(*task.preview_retry_history, attempt))
        proposed = None
        try:
            chunks, title = self._chunks(
                task, task.document_ids.index(document_id), calls
            )
            proposed = PaperPreviewExtractor(
                calls,
                boundary_policy=_paper_boundary_policy(task.original_question),
            ).extract(
                document_id=document_id,
                title=title,
                topic=task.original_question,
                screening_topic=_literature_screening_topic(task.original_question),
                chunks=chunks,
                max_output_tokens=self.max_output_tokens,
            )
            # Re-read the PDF and index after the remote call before committing.
            fresh, _ = self._chunks(task, task.document_ids.index(document_id), calls)
            _validate_preview(proposed, task, fresh)
            if proposed.evidence_quality != "verbatim":
                raise ValueError("unverified proposed quote")
        except Exception as exc:
            reason = (
                "模型引文未通过逐字原文校验，旧预览保留。"
                if proposed is not None and proposed.evidence_quality != "verbatim"
                else "重试取消或时间预算已到，旧预览保留。"
                if isinstance(exc, _Pause)
                else "预览服务或来源核验失败，旧预览保留；未执行详细分析。"
            )
            completed = attempt.model_copy(
                update={
                    "status": "failed",
                    "proposed_preview": proposed,
                    "reason": reason,
                }
            )
            save(preview_retry_history=(*task.preview_retry_history[:-1], completed))
            return task
        completed = attempt.model_copy(
            update={
                "status": "succeeded",
                "proposed_preview": proposed,
                "reason": "新预览通过原文校验，等待重新选择详细分析。",
            }
        )
        previews = {p.document_id: p for p in task.previews}
        previews[document_id] = proposed
        save(
            previews=tuple(previews.values()),
            preview_retry_history=(*task.preview_retry_history[:-1], completed),
            preview_decisions={
                **task.preview_decisions,
                document_id: PreviewDecision(action="hold", reason=completed.reason),
            },
            preview_choice_offsets={
                **task.preview_choice_offsets,
                document_id: len(task.user_instructions),
            },
            stage="awaiting_clarification",
            resume_stage="previewing",
        )
        return task

    def _chunks(self, task, index, calls):
        ref = task.artifact_refs[index]
        path = self.artifacts.resolve_pdf(ref, conversation_id=task.conversation_id)
        if path.stat().st_size > self.max_pdf_bytes:
            raise ValueError("PDF超过全文服务大小限制。")
        doc_id = task.document_ids[index]
        expected = parse_pdf_chunks(path, parser=self.parser, document_id=doc_id)
        with self._index_lock:
            calls.check()
            if self._store is None:
                self._store = self.store_factory()
            if not self._store.document_exists(doc_id):
                if self._rag is None:
                    self._rag = self.rag_factory(self._store)
                calls.check()
                self._rag.ingest((str(path),), None)
        chunks = tuple(self._store.get_document_chunks(doc_id))

        def identity(row):
            return (
                row.chunk_id,
                row.document_id,
                row.page_from,
                row.page_to,
                row.text,
                row.text_sha256,
            )

        if (
            len({row.chunk_id for row in chunks}) != len(chunks)
            or sorted(map(identity, chunks)) != sorted(map(identity, expected))
            or any(
                hashlib.sha256(row.text.encode()).hexdigest() != row.text_sha256
                for row in chunks
            )
        ):
            raise ValueError("已索引片段与当前PDF解析结果不一致；未覆盖旧片段。")
        # Resolve again after parsing/indexing before source reaches a model.
        self.artifacts.resolve_pdf(ref, conversation_id=task.conversation_id)
        return chunks, infer_pdf_title(path) or self.artifacts.get(ref).display_name

    selection = staticmethod(_selection)

    def run(
        self,
        task,
        *,
        save_task,
        model_budget,
        cancel_event,
        user_turn_id,
        progress=None,
    ):
        if self.auto_batches:
            from .fulltext_batches import run_saved_batches

            return run_saved_batches(
                self,
                task,
                save_task=save_task,
                model_budget=model_budget,
                cancel_event=cancel_event,
                user_turn_id=user_turn_id,
                progress=progress,
            )
        return self._run_single_batch(
            task,
            save_task=save_task,
            model_budget=model_budget,
            cancel_event=cancel_event,
            user_turn_id=user_turn_id,
        )

    def _run_single_batch(
        self,
        task: FulltextTask,
        *,
        save_task: Callable[[FulltextTask], None],
        model_budget: int,
        cancel_event: Event | None,
        user_turn_id: str,
    ) -> AgentResult:
        calls = _PreviewCalls(
            self.llm_factory,
            max(0, model_budget),
            cancel_event,
            time.monotonic() + self.batch_seconds,
        )
        reused = 0
        selected = ()
        validated_documents = set()
        awaiting_new_choice = False

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

        def save_rechecked(updated):
            nonlocal task
            save_task(updated)
            task = updated

        def result(message, *, code=None, needs_input=False, cancelled=False):
            from .fulltext_preview_display import preview_report_lines

            lines = preview_report_lines(
                task,
                selected,
                validated_documents,
                awaiting_confirmation=needs_input and not code and not cancelled,
                reused=reused,
            )
            body = "\n".join(lines)
            warnings = ()
            if len((body + message).encode()) > self.max_output_bytes - 256:
                warning = "预览显示已截断，完整预览保存在当前任务；未改变论文选择。"
                limit = max(
                    0, self.max_output_bytes - len((message + warning).encode()) - 32
                )
                body = (
                    body.encode()[:limit].decode("utf-8", errors="ignore")
                    + "\n"
                    + warning
                )
                warnings = (warning,)
            return AgentResult(
                conversation_id=task.conversation_id,
                user_turn_id=user_turn_id,
                status="cancelled" if cancelled else "error" if code else "completed",
                final_status=None
                if cancelled
                else "needs_user_input"
                if needs_input
                else "error"
                if code
                else "completed",
                response_text=body + "\n\n" + message,
                warnings=warnings,
                model_call_count=calls.calls,
                selected_tools=("literature",),
                error={
                    "code": code,
                    "message": message,
                    "retryable": code != "FULLTEXT_EXTRACTION_UNAVAILABLE",
                }
                if code
                else None,
            )

        try:
            selected, only_preview = _selection(task)
            if self.require_preview_confirmation and not any(
                re.search(
                    r"详细分析|深度分析|只分析|提取|确认分析|确认详细分析|开始分析|分析第",
                    text,
                )
                for text in task.user_instructions
            ):
                only_preview = True
        except ValueError:
            save(stage="awaiting_clarification", resume_stage="previewing")
            return result(
                "论文选择为空、超出当前附件范围或表达不明确，请确认论文序号与选择范围。",
                needs_input=True,
            )
        if len(selected) > 20:
            save(stage="awaiting_clarification", resume_stage="previewing")
            return result(
                "一次预览最多20篇，请选择当前任务中的论文子集。", needs_input=True
            )
        try:
            save(stage="previewing", resume_stage=None)
            previews = {row.document_id: row for row in task.previews}
            decisions = {}
            checked_chunks = {}
            for index in selected:
                calls.check()
                doc_id = task.document_ids[index]
                preview = previews.get(doc_id)
                was_cached = preview is not None
                if preview is None and calls.calls >= calls.budget:
                    raise _Pause()
                chunks, title = self._chunks(task, index, calls)
                calls.check()
                if preview is not None:
                    # Never repair a corrupt cache by silently asking the model
                    # to replace it. Scope rechecks require valid original sources.
                    _validate_preview(preview, task, chunks)
                    if _unsupported_scope_exclusion(preview, task.original_question):
                        if not self.retry_allowed(task, doc_id):
                            raise ValueError(
                                "Scope recheck cannot reset extracted work"
                            )
                        # Reserve enough room for the existing two-attempt schema
                        # contract before recording a non-replayable operation.
                        if calls.budget - calls.calls < 2:
                            raise _Pause()
                        attempt_number = sum(
                            a.document_id == doc_id for a in task.preview_retry_history
                        )
                        task = self.regenerate(
                            task,
                            document_id=doc_id,
                            operation_id=f"scope-{doc_id}-{attempt_number}",
                            expected=self.retry_token(task, doc_id),
                            save_task=save_rechecked,
                            model_budget=calls.budget - calls.calls,
                            cancel_event=cancel_event,
                            _calls=calls,
                        )
                        if task.preview_retry_history[-1].status != "succeeded":
                            raise _ProviderFailure("scope recheck did not complete")
                        preview = next(
                            p for p in task.previews if p.document_id == doc_id
                        )
                        # A new preview needs fresh consent under existing rules.
                        # Old previews remain in the bounded retry history.
                        was_cached = False
                        chunks, _ = self._chunks(task, index, calls)
                if preview is None:
                    preview = PaperPreviewExtractor(
                        calls,
                        boundary_policy=_paper_boundary_policy(task.original_question),
                    ).extract(
                        document_id=doc_id,
                        title=title,
                        topic=task.original_question,
                        screening_topic=_literature_screening_topic(
                            task.original_question
                        ),
                        chunks=chunks,
                        max_output_tokens=self.max_output_tokens,
                    )
                _validate_preview(preview, task, chunks)
                checked_chunks[doc_id] = chunks
                validated_documents.add(doc_id)
                if was_cached:
                    reused += 1
                previews[doc_id] = preview
                if self.require_preview_confirmation and not was_cached:
                    save(
                        preview_choice_offsets={
                            **task.preview_choice_offsets,
                            doc_id: len(task.user_instructions),
                        }
                    )
                new_choice = self._new_choice(task, doc_id)
                decisions[doc_id] = _decision(
                    preview,
                    only_preview=only_preview or not new_choice,
                    question=task.original_question,
                )
                if not only_preview and not new_choice:
                    awaiting_new_choice = True
                    decisions[doc_id] = PreviewDecision(
                        action="hold",
                        reason="本篇预览已新生成或重新核验，请确认更新后的分析范围。",
                    )
                save(previews=tuple(previews.values()), preview_decisions=decisions)
                calls.check()  # Preserve a completed response, then honor cancellation.
            actions = {decision.action for decision in decisions.values()}
            if "extract" in actions:
                if self.extraction_processor is not None:
                    return self.extraction_processor.run(
                        task,
                        chunks_by_document=checked_chunks,
                        refresh_sources=lambda doc_id: self._chunks(
                            task, task.document_ids.index(doc_id), calls
                        )[0],
                        calls=calls,
                        save_task=save_task,
                        user_turn_id=user_turn_id,
                    )
                save(stage="ready_to_resume", resume_stage="extracting")
                return result(
                    "合格论文已选定；详细提取处理器尚未接入，未执行实验提取或统计。",
                    code="FULLTEXT_EXTRACTION_UNAVAILABLE",
                )
            save(stage="awaiting_clarification", resume_stage="previewing")
            return result(
                "已按要求只预览，未启动详细提取。"
                if only_preview
                else "预览已完成或重新核验；请确认更新后的预览，再继续详细分析。"
                "尚未执行实验提取或统计。"
                if awaiting_new_choice
                else "当前仅有背景阅读或需核对对象的论文；未执行实验统计。"
                "请确认分析范围或提供更相关全文。",
                needs_input=True,
            )
        except _Pause as exc:
            save(stage="ready_to_resume", resume_stage="previewing")
            return result(
                "已停止，已完成预览保留；没有启动下一篇或实验提取。"
                if exc.cancelled
                else "本轮调用或时间预算已到，已完成预览保留；"
                "继续可处理尚未完成的论文。",
                cancelled=exc.cancelled,
                code=None if exc.cancelled else "FULLTEXT_PREVIEW_BUDGET",
            )
        except Exception:
            save(stage="failed", resume_stage="previewing")
            return result(
                "论文解析、来源核验或预览服务执行失败；已完成部分保留，未执行实验提取或统计。可有界重试，无须改写原科研问题。",
                code="FULLTEXT_PREVIEW_FAILED",
            )

    @staticmethod
    def _new_choice(task, document_id):
        offset = task.preview_choice_offsets.get(document_id)
        if offset is None:
            return True
        # Reuse the same ordinal/negative selection rules on only NEW instructions.
        new = task.model_copy(
            update={"user_instructions": task.user_instructions[offset:]}
        )
        selected, only_preview = _selection(new)
        return (
            not only_preview
            and task.document_ids.index(document_id) in selected
            and any(
                re.search(
                    r"详细分析|深度分析|只分析|提取|分析第|确认分析|开始分析",
                    instruction,
                )
                for instruction in new.user_instructions
            )
        )
