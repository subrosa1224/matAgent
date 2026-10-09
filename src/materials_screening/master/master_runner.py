"""MasterAgentRunner: orchestrates the multi-agent system.

The runner creates/restores conversations in the store, serializes asks per
conversation with a non-blocking lock, builds the ``MasterAgentContext``
right before each invoke, converts the graph state into a safe ``AgentResult``
and maps ``GraphRecursionError`` to a stable error result.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import threading
from datetime import UTC, datetime
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, ConfigDict, Field

from materials_screening.agent.conversation_store import ConversationStore
from materials_screening.agent.errors import AgentConversationError
from materials_screening.agent.model_base import MaterialAgentModel
from materials_screening.agent.models import AgentResult
from materials_screening.agent.settings import AgentSettings
from materials_screening.workflow.context import Clock, IdGenerator

from .artifact_registry import ArtifactRegistry
from .data_analysis_handoffs import DataAnalysisCrossAgentCoordinator
from .fulltext_preview import FulltextPreviewProcessor
from .fulltext_tasks import (
    FulltextTask,
    FulltextTaskSelectionError,
    new_fulltext_task,
    referenced_fulltext_task,
    requests_fulltext_resume,
    requests_fulltext_task,
    select_fulltext_task,
)
from .master_context import MasterAgentContext
from .master_graph_builder import compile_master_graph
from .master_settings import MasterAgentSettings
from .sub_agent_executor import SubAgentExecutor
from .sub_agent_registry import SubAgentRegistry
from .sub_agent_spec import SubAgentSpec


class ConversationView(BaseModel):
    """Safe conversation summary; never transcript, arguments or responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    created_at: str
    updated_at: str
    turn_count: int = Field(default=0, ge=0)
    active_workflow_thread_id: str | None = None
    workflow_threads: tuple[str, ...] = ()


class AgentStreamEvent(BaseModel):
    """Progress event emitted during master agent execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node: str
    message: str
    is_final: bool = False
    result: AgentResult | None = None


_NODE_PROGRESS_MESSAGES: dict[str, str] = {
    "prepare_turn": "正在准备...",
    "call_master_model": "Master 正在分析意图...",
    "execute_sub_agent": "正在委派子Agent执行...",
    "validate_final": "正在验证汇总回答...",
    "finalize_success": "正在完成...",
    "finalize_error": "处理错误中...",
}


class _UuidIdGenerator:
    def new_id(self) -> str:
        return secrets.token_hex(2)


class MasterAgentRunner:
    """Owns the master agent graph and orchestrates one user turn per ask."""

    def __init__(
        self,
        *,
        settings: MasterAgentSettings,
        store: ConversationStore,
        sub_agent_registry: SubAgentRegistry,
        master_model: MaterialAgentModel,
        checkpointer: Any,
        clock: Clock | None = None,
        id_generator: IdGenerator | None = None,
        data_analysis_coordinator: DataAnalysisCrossAgentCoordinator | None = None,
        artifact_registry: ArtifactRegistry | None = None,
        preview_processor: FulltextPreviewProcessor | None = None,
        figure_review_service=None,
        figure_review_enabled: bool = False,
        recursion_limit: int = 16,
    ) -> None:
        if recursion_limit < 1:
            raise ValueError("recursion_limit must be positive")
        self._settings = settings
        self._store = store
        self._sub_agent_registry = sub_agent_registry
        self._master_model = master_model
        self._clock = clock or (lambda: datetime.now(UTC))
        self._id_generator = id_generator or _UuidIdGenerator()
        self._data_analysis_coordinator = data_analysis_coordinator
        self._artifact_registry = artifact_registry
        self._preview_processor = preview_processor
        self._figure_review_service = figure_review_service
        self._figure_review_enabled = figure_review_enabled
        self._recursion_limit = recursion_limit
        self._sub_agent_executor = SubAgentExecutor(
            registry=sub_agent_registry,
            max_output_bytes=settings.master_max_sub_agent_output_bytes,
        )
        self._graph: Any = compile_master_graph(checkpointer=checkpointer)

    # ── Sub-agent registration ────────────────────────────────────────────

    @property
    def sub_agent_registry(self) -> SubAgentRegistry:
        return self._sub_agent_registry

    def register_sub_agent(self, spec: SubAgentSpec) -> None:
        self._sub_agent_registry.register(spec)

    # ── Conversation lifecycle ────────────────────────────────────────────

    def start_conversation(self) -> str:
        for _ in range(32):
            conversation_id = self._id_generator.new_id()
            if not self._store.exists(conversation_id):
                self._store.create(conversation_id)
                return conversation_id
        raise AgentConversationError(
            "CONVERSATION_ID_COLLISION",
            "could not allocate a unique conversation id",
        )

    def get_conversation(self, conversation_id: str) -> ConversationView:
        metadata = self._store.get(conversation_id)
        return ConversationView(
            conversation_id=metadata.conversation_id,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
            turn_count=metadata.turn_count,
            active_workflow_thread_id=metadata.active_workflow_thread_id,
            workflow_threads=self._store.list_workflows(conversation_id),
        )

    def conversation_status(self, conversation_id: str) -> str | None:
        config = self._config(self._thread_id(conversation_id))
        try:
            snapshot = self._graph.get_state(config)
        except Exception:
            return None
        values = snapshot.values or {}
        value = values.get("status")
        return str(value) if value else None

    # ── Ask ───────────────────────────────────────────────────────────────

    def get_fulltext_tasks(self, conversation_id: str) -> tuple[FulltextTask, ...]:
        """Read only server-owned checkpoint tasks, including legacy waiting turns."""
        if not self._store.exists(conversation_id):
            return ()
        state = (
            self._graph.get_state(self._config(self._thread_id(conversation_id))).values
            or {}
        )
        saved = state.get("fulltext_tasks", {})
        if not isinstance(saved, dict):
            raise ValueError("全文任务状态损坏，未自动恢复。")
        tasks = []
        for key, value in saved.items():
            task = FulltextTask.model_validate(value)
            if task.task_id != key or task.conversation_id != conversation_id:
                raise ValueError("全文任务状态不属于当前会话。")
            tasks.append(task)
        if (
            not tasks
            and (state.get("final_draft") or {}).get("status") == "needs_user_input"
        ):
            question = state.get("user_message", "")
            if requests_fulltext_task(question):
                digest = hashlib.sha256(
                    (conversation_id + "|" + question).encode()
                ).hexdigest()[:32]
                tasks.append(
                    new_fulltext_task(
                        conversation_id,
                        question,
                        state=state,
                        task_id="task-fulltext-" + digest,
                    )
                )
        return tuple(tasks)

    def restore_fulltext_context(self, conversation_id: str):
        """Inspect an explicit existing conversation, never create or execute it."""
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,253}", conversation_id) is None:
            raise ValueError("请提供有效的保存会话编号。")
        self.get_conversation(conversation_id)
        lock = self._store.conversation_lock(conversation_id)
        if not lock.acquire(blocking=False):
            raise ValueError("当前会话正在处理，请稍后恢复。")
        try:
            task = select_fulltext_task(
                self.get_fulltext_tasks(conversation_id),
                conversation_id=conversation_id,
            )
            if task is None:
                raise ValueError("当前会话没有待接续的全文任务。")
            if self._artifact_registry is None:
                raise ValueError("全文附件服务尚未配置。")
            names = []
            from materials_screening.sub_agents.literature.rag import (
                PyMuPdfParser,
                parse_pdf_chunks,
            )

            from .fulltext_preview import _validate_preview

            parser = getattr(self._preview_processor, "parser", None) or PyMuPdfParser()
            previews = {p.document_id: p for p in task.previews}
            for index, ref in enumerate(task.artifact_refs):
                path = self._artifact_registry.resolve_pdf(
                    ref, conversation_id=conversation_id
                )
                doc = task.document_ids[index]
                if doc in previews:
                    chunks = parse_pdf_chunks(path, parser=parser, document_id=doc)
                    _validate_preview(previews[doc], task, chunks)
                names.append(self._artifact_registry.get(ref).display_name)
            return task, tuple(names)
        finally:
            lock.release()

    def register_pdf_attachment(self, source: Any, *, conversation_id: str) -> Any:
        """Register an explicit upload for an existing server conversation."""
        if not self._store.exists(conversation_id):
            raise ValueError("请先创建或恢复主控会话，再上传全文。")
        if self._artifact_registry is None:
            raise ValueError("主控未配置安全附件注册器。")
        return self._artifact_registry.register_pdf(
            source, conversation_id=conversation_id
        )

    def _figure_task(self, conversation_id, task_id):
        if self._figure_review_service is None:
            raise ValueError("图证据核对服务未配置。")
        task = next(
            (
                t
                for t in self.get_fulltext_tasks(conversation_id)
                if t.task_id == task_id
            ),
            None,
        )
        if task is None or task.conversation_id != conversation_id:
            raise ValueError("图证据任务不属于当前会话。")
        return task

    def get_figure_review(self, *, conversation_id, task_id):
        lock = self._store.conversation_lock(conversation_id)
        if not lock.acquire(blocking=False):
            raise ValueError("Figure conversation is busy")
        try:
            task = self._figure_task(conversation_id, task_id)
            return self._figure_review_service._load(task, conversation_id)
        finally:
            lock.release()

    def get_figure_image(
        self,
        *,
        conversation_id,
        task_id,
        candidate_id=None,
        document_id=None,
        page=None,
    ):
        """Resolve only an image granted by this server task, after source checks."""
        lock = self._store.conversation_lock(conversation_id)
        if not lock.acquire(blocking=False):
            raise ValueError("Figure conversation is busy")
        try:
            task = self._figure_task(conversation_id, task_id)
            service = self._figure_review_service
            batch = service._load(task, conversation_id)
            if candidate_id is not None and document_id is None and page is None:
                candidate = next(
                    (c for c in batch.candidates if c.candidate_id == candidate_id),
                    None,
                )
                if candidate is None:
                    raise ValueError("Figure candidate outside current task")
                service._candidate_sources(task, candidate)
                return service.store.image_path(candidate.crop_image)
            if candidate_id is not None or document_id is None or page is None:
                raise ValueError("Choose one scoped figure image")
            _, digest = service._pdf(task, document_id)
            view = next(
                (
                    v
                    for v in batch.page_views
                    if v.document_id == document_id and v.page == page
                ),
                None,
            )
            if view is None or view.pdf_sha256 != digest:
                raise ValueError("Figure page outside current task")
            return service.store.image_path(view.image)
        finally:
            lock.release()

    def figure_review_action(
        self, *, conversation_id, task_id, expected, operation_id, action, **parameters
    ):
        from .figure_evidence_contracts import FigureBatchReference, PageRegion

        lock = self._store.conversation_lock(conversation_id)
        if not lock.acquire(blocking=False):
            raise ValueError("Figure conversation is busy")
        try:
            task = self._figure_task(conversation_id, task_id)
            expected = FigureBatchReference.model_validate(
                expected.model_dump() if hasattr(expected, "model_dump") else expected
            )
            service = self._figure_review_service
            args = dict(
                conversation_id=conversation_id,
                expected=expected,
                operation_id=operation_id,
            )
            if action == "add":
                parameters["region"] = PageRegion.model_validate(parameters["region"])
                ref = service.add(task, **args, **parameters)
            elif action == "page":
                ref = service.select_page(task, **args, **parameters)
            elif action == "add_screen":
                parameters["region"] = service.region_from_view(
                    task,
                    conversation_id=conversation_id,
                    document_id=parameters["document_id"],
                    page=parameters["page"],
                    image_sha256=parameters.pop("image_sha256"),
                    region=PageRegion.model_validate(parameters["region"]),
                )
                ref = service.add(task, **args, **parameters)
            elif action == "decide":
                parameters["action"] = parameters.pop("decision")
                ref = service.decide(task, **args, **parameters)
            elif action == "close":
                ref = service.close(task, **args, **parameters)
            else:
                raise ValueError("Unknown structured figure action")
            self._save_fulltext_task(
                FulltextTask.model_validate(
                    {**task.model_dump(), "figure_evidence_ref": ref}
                )
            )
            return service.store.load(ref)
        finally:
            lock.release()

    def regenerate_preview(
        self,
        *,
        conversation_id,
        task_id,
        document_id,
        expected,
        operation_id,
        cancel_event=None,
    ):
        """Explicit structured action; never delegates database/search/extraction."""
        lock = self._store.conversation_lock(conversation_id)
        if not lock.acquire(blocking=False):
            raise ValueError("当前会话正在处理，请稍后刷新。")
        try:
            task = next(
                (
                    t
                    for t in self.get_fulltext_tasks(conversation_id)
                    if t.task_id == task_id
                ),
                None,
            )
            if task is None or task.conversation_id != conversation_id:
                raise ValueError("预览任务不属于当前会话。")
            if self._preview_processor is None:
                raise ValueError("预览服务尚未配置。")
            return self._preview_processor.regenerate(
                task,
                document_id=document_id,
                expected=expected,
                operation_id=operation_id,
                save_task=self._save_fulltext_task,
                model_budget=2,
                cancel_event=cancel_event,
            )
        finally:
            lock.release()

    def _save_fulltext_task(self, task: FulltextTask) -> None:
        config = self._config(self._thread_id(task.conversation_id))
        state = self._graph.get_state(config).values or {}
        tasks = dict(state.get("fulltext_tasks", {}))
        tasks[task.task_id] = task.model_dump(mode="json")
        self._graph.update_state(
            config, {"fulltext_tasks": tasks}, as_node="finalize_success"
        )

    def _remember_fulltext_task(
        self, message: str, result: AgentResult
    ) -> FulltextTask | None:
        if result.status != "completed" or not requests_fulltext_task(message):
            return None
        state = (
            self._graph.get_state(
                self._config(self._thread_id(result.conversation_id))
            ).values
            or {}
        )
        task = new_fulltext_task(result.conversation_id, message, state=state)
        if self._figure_review_enabled:
            task = task.model_copy(
                update={"figure_review_policy": "figure-evidence-review-v1"}
            )
        self._save_fulltext_task(task)
        return task

    def _fulltext_input(
        self,
        message: str,
        conversation_id: str,
        artifact_refs: tuple[str, ...],
        task_id: str | None,
        *,
        record_turn: bool = True,
        cancel_event: threading.Event | None = None,
        model_budget: int | None = None,
        progress=None,
    ) -> AgentResult | None:
        """Authorize and persist binding before any fulltext/model processing.

        Until a processor is wired, stop visibly with a recoverable saved task;
        do not rerun metadata search or pretend that preview/extraction happened.
        """
        if (
            not artifact_refs
            and task_id is None
            and not requests_fulltext_resume(message)
        ):
            return None
        if cancel_event is not None and cancel_event.is_set():
            return AgentResult(
                conversation_id=conversation_id,
                user_turn_id=self._id_generator.new_id(),
                status="cancelled",
                response_text="已停止，未启动附件绑定或全文处理。",
            )
        if len(message.encode("utf-8")) > self._settings.master_max_input_bytes:
            return self._error_result(
                conversation_id,
                "INPUT_TOO_LARGE",
                "user message exceeds the input limit",
            )
        if (
            record_turn
            and self._store.exists(conversation_id)
            and self._store.get(conversation_id).turn_count
            >= self._settings.master_max_conversation_turns
        ):
            return self._error_result(
                conversation_id, "TURN_LIMIT", "conversation reached the turn limit"
            )
        try:
            if (
                len(artifact_refs) > 100
                or any(not isinstance(item, str) for item in artifact_refs)
                or len(set(artifact_refs)) != len(artifact_refs)
            ):
                raise ValueError("附件引用数量无效。")
            for artifact_id in artifact_refs:
                if self._artifact_registry is None:
                    raise ValueError("主控未配置安全附件注册器。")
                self._artifact_registry.resolve_pdf(
                    artifact_id, conversation_id=conversation_id
                )
        except (ValueError, KeyError, OSError):
            return self._error_result(
                conversation_id,
                "FULLTEXT_ATTACHMENT_REJECTED",
                "全文附件无效、内容改变或未获当前会话授权。",
            )
        # A new explicit research request must finish its initial database and
        # metadata stages before an attached PDF enters the continuation task.
        if task_id is None and requests_fulltext_task(message):
            return None
        tasks: tuple[FulltextTask, ...] = ()
        try:
            message_task_id = referenced_fulltext_task(message)
            if task_id is not None and message_task_id not in {None, task_id}:
                raise FulltextTaskSelectionError("消息与指定参数选择了不同全文任务。")
            task_id = task_id or message_task_id
            tasks = self.get_fulltext_tasks(conversation_id)
            task = select_fulltext_task(
                tasks,
                conversation_id=conversation_id,
                task_id=task_id,
            )
        except FulltextTaskSelectionError as exc:
            choices = "\n".join(
                f"- `{item.task_id}`：{item.original_question[:160]}"
                for item in tasks
                if item.stage != "finished"
            )
            return AgentResult(
                conversation_id=conversation_id,
                user_turn_id=self._id_generator.new_id(),
                status="completed",
                final_status="needs_user_input",
                response_text=str(exc) + ("\n\n" + choices if choices else ""),
            )
        except ValueError:
            return self._error_result(
                conversation_id,
                "FULLTEXT_TASK_STATE_INVALID",
                "全文任务状态损坏，未修改或替换原任务。",
            )
        if task is None:
            if artifact_refs or task_id is not None:
                return self._error_result(
                    conversation_id,
                    "FULLTEXT_TASK_NOT_FOUND",
                    "当前没有可接续的全文任务；单独上传论文请使用文献预览。",
                )
            return None
        instructions = task.user_instructions
        if task.stage == "awaiting_figure_review":
            try:
                if self._figure_review_service is None:
                    raise ValueError("Missing figure review service")
                batch = self._figure_review_service._load(task, conversation_id)
            except (ValueError, OSError):
                return self._error_result(
                    conversation_id,
                    "FIGURE_REVIEW_STATE_INVALID",
                    "图证据核对无法安全恢复；旧记录未修改。",
                )
            if batch.status != "closed":
                return AgentResult(
                    conversation_id=conversation_id,
                    user_turn_id=self._id_generator.new_id(),
                    status="completed",
                    final_status="needs_user_input",
                    selected_tools=("literature",),
                    response_text=(
                        "当前等待图中文字核对。请打开图证据核对入口，明确确认或跳过；"
                        "普通“继续”不表示确认，也未执行统计。\n任务：" + task.task_id
                    ),
                )
        if record_turn and message.strip().casefold() not in {
            "",
            "继续",
            "continue",
            "重试",
        }:
            if len(instructions) >= 100:
                return self._error_result(
                    conversation_id,
                    "FULLTEXT_INSTRUCTION_LIMIT",
                    "全文任务补充指令数量超出限制。",
                )
            instructions = (*instructions, message.strip())
        if not artifact_refs and not task.artifact_refs:
            self._save_fulltext_task(
                FulltextTask.model_validate(
                    {
                        **task.model_dump(mode="json"),
                        "user_instructions": instructions,
                        "updated_at": self._clock(),
                    }
                )
            )
            if record_turn:
                self._store.record_turn(conversation_id)
            return AgentResult(
                conversation_id=conversation_id,
                user_turn_id=self._id_generator.new_id(),
                status="completed",
                final_status="needs_user_input",
                response_text="原科研任务已保留，请提供对应论文PDF后继续；未重新检索或执行统计。",
            )
        refs = tuple(dict.fromkeys((*task.artifact_refs, *artifact_refs)))
        if len(refs) > 100:
            return self._error_result(
                conversation_id,
                "FULLTEXT_ATTACHMENT_REJECTED",
                "当前任务附件数量超出限制。",
            )
        # Revalidate previously saved references on every resume as well.
        try:
            if self._artifact_registry is None:
                raise ValueError("Missing artifact registry")
            for reference in refs:
                self._artifact_registry.resolve_pdf(
                    reference, conversation_id=conversation_id
                )
        except (ValueError, KeyError, OSError):
            return self._error_result(
                conversation_id,
                "FULLTEXT_ATTACHMENT_REJECTED",
                "已保存的全文附件不可安全恢复，请重新提供原文件。",
            )
        task = FulltextTask.model_validate(
            {
                **task.model_dump(mode="json"),
                "artifact_refs": refs,
                "document_ids": tuple(
                    "doc-" + reference.removeprefix("artifact-pdf-")
                    for reference in refs
                ),
                "user_instructions": instructions,
                "stage": "ready_to_resume",
                "resume_stage": "previewing",
                "updated_at": self._clock(),
            }
        )
        self._save_fulltext_task(task)
        if record_turn:
            self._store.record_turn(conversation_id)
        if self._preview_processor is not None:
            options = {}
            if getattr(self._preview_processor, "auto_batches", False):
                options["progress"] = progress
            return self._preview_processor.run(
                task,
                save_task=self._save_fulltext_task,
                model_budget=self._settings.master_max_model_calls_per_turn
                if model_budget is None
                else model_budget,
                cancel_event=cancel_event,
                user_turn_id=self._id_generator.new_id(),
                **options,
            )
        return AgentResult(
            conversation_id=conversation_id,
            user_turn_id=self._id_generator.new_id(),
            status="error",
            final_status="error",
            response_text="PDF已安全绑定原科研任务，原问题和既有结果已保留。全文阶段处理器尚未接入，当前未执行预览、提取或实验统计；配置就绪后可继续。",
            error={
                "code": "FULLTEXT_PROCESSOR_UNAVAILABLE",
                "message": "全文阶段处理器尚未接入",
                "retryable": False,
            },
        )

    def ask(
        self,
        *,
        message: str,
        conversation_id: str | None = None,
        artifact_refs: tuple[str, ...] = (),
        task_id: str | None = None,
    ) -> AgentResult:
        resolved_id = (
            conversation_id
            if conversation_id is not None
            else self.start_conversation()
        )
        lock = self._store.conversation_lock(resolved_id)
        if not lock.acquire(blocking=False):
            return self._error_result(
                resolved_id,
                "CONVERSATION_BUSY",
                "another ask is already running for this conversation",
            )
        try:
            handled = self._fulltext_input(message, resolved_id, artifact_refs, task_id)
            if handled is not None:
                return handled
            result = self._ask_locked(message=message, conversation_id=resolved_id)
            task = self._remember_fulltext_task(message, result)
            if artifact_refs and task is not None:
                handled = self._fulltext_input(
                    message,
                    resolved_id,
                    artifact_refs,
                    task.task_id,
                    record_turn=False,
                    model_budget=max(
                        0,
                        self._settings.master_max_model_calls_per_turn
                        - result.model_call_count,
                    ),
                )
                if handled is not None:
                    return self._merge_fulltext_result(result, handled)
            return result
        finally:
            lock.release()

    def _fulltext_input_stream(self, *args, cancel_event, **kwargs):
        if not getattr(self._preview_processor, "auto_batches", False):
            return self._fulltext_input(*args, cancel_event=cancel_event, **kwargs)
        from .fulltext_batches import stream_saved_operation

        def operation(progress):
            return self._fulltext_input(
                *args, cancel_event=cancel_event, progress=progress, **kwargs
            )

        stream = stream_saved_operation(operation, cancel_event)
        try:
            while True:
                try:
                    message = next(stream)
                except StopIteration as completed:
                    return completed.value
                yield AgentStreamEvent(node="fulltext_preview", message=message)
        finally:
            stream.close()

    def ask_stream(
        self,
        *,
        message: str,
        conversation_id: str | None = None,
        cancel_event: threading.Event | None = None,
        artifact_refs: tuple[str, ...] = (),
        task_id: str | None = None,
    ) -> Any:
        cancel_event = cancel_event if cancel_event is not None else threading.Event()
        resolved_id = (
            conversation_id
            if conversation_id is not None
            else self.start_conversation()
        )
        lock = self._store.conversation_lock(resolved_id)
        if not lock.acquire(blocking=False):
            result = self._error_result(
                resolved_id,
                "CONVERSATION_BUSY",
                "another ask is already running for this conversation",
            )
            yield AgentStreamEvent(
                node="runner",
                message="会话正忙：另一个 ask 正在运行",
                is_final=True,
                result=result,
            )
            return
        try:
            if cancel_event is not None and cancel_event.is_set():
                yield AgentStreamEvent(
                    node="runner",
                    message="已停止",
                    is_final=True,
                    result=AgentResult(
                        conversation_id=resolved_id,
                        user_turn_id=self._id_generator.new_id(),
                        status="cancelled",
                        response_text="已停止：本轮未开始执行",
                    ),
                )
                return
            if (
                self._preview_processor is not None
                and not requests_fulltext_task(message)
                and (
                    artifact_refs
                    or task_id is not None
                    or requests_fulltext_resume(message)
                )
            ):
                yield AgentStreamEvent(
                    node="fulltext_preview",
                    message="literature：核验任务与来源，准备全文预览",
                )
            handled = yield from self._fulltext_input_stream(
                message,
                resolved_id,
                artifact_refs,
                task_id,
                cancel_event=cancel_event,
            )
            if handled is not None:
                yield AgentStreamEvent(
                    node="fulltext_preview"
                    if self._preview_processor is not None
                    else "fulltext_task",
                    message="literature：全文阶段结果"
                    if self._preview_processor is not None
                    else "全文任务接续结果",
                    is_final=True,
                    result=handled,
                )
                return
            for event in self._ask_stream_locked(
                message=message,
                conversation_id=resolved_id,
                cancel_event=cancel_event,
            ):
                if event.is_final and event.result is not None:
                    task = self._remember_fulltext_task(message, event.result)
                    if artifact_refs and task is not None:
                        if self._preview_processor is not None:
                            yield AgentStreamEvent(
                                node="fulltext_preview",
                                message="literature：初始阶段已结束，核验附件并继续预览",
                            )
                        handled = yield from self._fulltext_input_stream(
                            message,
                            resolved_id,
                            artifact_refs,
                            task.task_id,
                            record_turn=False,
                            cancel_event=cancel_event,
                            model_budget=max(
                                0,
                                self._settings.master_max_model_calls_per_turn
                                - event.result.model_call_count,
                            ),
                        )
                        if handled is not None:
                            event = event.model_copy(
                                update={
                                    "node": "fulltext_preview"
                                    if self._preview_processor is not None
                                    else event.node,
                                    "message": "literature：全文阶段结果"
                                    if self._preview_processor is not None
                                    else event.message,
                                    "result": self._merge_fulltext_result(
                                        event.result, handled
                                    ),
                                }
                            )
                yield event
        finally:
            lock.release()

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _merge_fulltext_result(
        initial: AgentResult, continuation: AgentResult
    ) -> AgentResult:
        return continuation.model_copy(
            update={
                "response_text": initial.response_text
                + "\n\n"
                + continuation.response_text,
                "model_call_count": initial.model_call_count
                + continuation.model_call_count,
                "tool_call_count": initial.tool_call_count
                + continuation.tool_call_count,
                "selected_tools": tuple(
                    dict.fromkeys(
                        (*initial.selected_tools, *continuation.selected_tools)
                    )
                ),
                "evidence_ids": tuple(
                    dict.fromkeys((*initial.evidence_ids, *continuation.evidence_ids))
                ),
                "warnings": tuple(
                    dict.fromkeys((*initial.warnings, *continuation.warnings))
                ),
                "active_workflow_thread_id": initial.active_workflow_thread_id,
            }
        )

    def _ask_locked(
        self,
        *,
        message: str,
        conversation_id: str,
    ) -> AgentResult:
        if not message.strip():
            return self._error_result(
                conversation_id, "EMPTY_MESSAGE", "message must not be empty"
            )
        if not self._store.exists(conversation_id):
            self._store.create(conversation_id)
        metadata = self._store.get(conversation_id)
        max_turns = self._settings.master_max_conversation_turns
        if metadata.turn_count >= max_turns:
            return self._error_result(
                conversation_id,
                "TURN_LIMIT",
                f"conversation reached the {max_turns}-turn limit",
            )

        agent_settings = AgentSettings(
            agent_max_model_calls_per_turn=(
                self._settings.master_max_model_calls_per_turn
            ),
            agent_max_tool_calls_per_turn=(
                self._settings.master_max_sub_agent_calls_per_turn
            ),
            agent_max_input_bytes=self._settings.master_max_input_bytes,
        )
        context = MasterAgentContext(
            sub_agent_registry=self._sub_agent_registry,
            sub_agent_executor=self._sub_agent_executor,
            master_model=self._master_model,
            settings=agent_settings,
            conversation_store=self._store,
            clock=self._clock,
            id_generator=self._id_generator,
            data_analysis_coordinator=self._data_analysis_coordinator,
        )
        thread_id = self._thread_id(conversation_id)
        config = self._config(thread_id)
        try:
            state = self._graph.invoke(
                {"conversation_id": conversation_id, "user_message": message},
                config=config,
                context=context,
            )
        except GraphRecursionError:
            return self._error_result(
                conversation_id,
                "RECURSION_LIMIT",
                "master agent recursion limit reached",
                user_turn_id=self._checkpointed_user_turn_id(config),
            )

        result = self._to_result(conversation_id, state)
        self._store.record_turn(conversation_id)
        return result

    def _ask_stream_locked(
        self,
        *,
        message: str,
        conversation_id: str,
        cancel_event: threading.Event | None = None,
    ) -> Any:
        if not message.strip():
            result = self._error_result(
                conversation_id, "EMPTY_MESSAGE", "message must not be empty"
            )
            yield AgentStreamEvent(
                node="runner",
                message="消息为空",
                is_final=True,
                result=result,
            )
            return
        if cancel_event is not None and cancel_event.is_set():
            result = AgentResult(
                conversation_id=conversation_id,
                user_turn_id=self._id_generator.new_id(),
                status="cancelled",
                response_text="已停止：本轮未开始执行",
            )
            yield AgentStreamEvent(
                node="runner",
                message="已停止",
                is_final=True,
                result=result,
            )
            return
        if not self._store.exists(conversation_id):
            self._store.create(conversation_id)
        metadata = self._store.get(conversation_id)
        max_turns = self._settings.master_max_conversation_turns
        if metadata.turn_count >= max_turns:
            result = self._error_result(
                conversation_id,
                "TURN_LIMIT",
                f"conversation reached the {max_turns}-turn limit",
            )
            yield AgentStreamEvent(
                node="runner",
                message=f"会话达到 {max_turns} 轮上限",
                is_final=True,
                result=result,
            )
            return

        agent_settings = AgentSettings(
            agent_max_model_calls_per_turn=(
                self._settings.master_max_model_calls_per_turn
            ),
            agent_max_tool_calls_per_turn=(
                self._settings.master_max_sub_agent_calls_per_turn
            ),
            agent_max_input_bytes=self._settings.master_max_input_bytes,
        )
        context = MasterAgentContext(
            sub_agent_registry=self._sub_agent_registry,
            sub_agent_executor=self._sub_agent_executor,
            master_model=self._master_model,
            settings=agent_settings,
            conversation_store=self._store,
            clock=self._clock,
            id_generator=self._id_generator,
            data_analysis_coordinator=self._data_analysis_coordinator,
            cancel_event=cancel_event,
        )
        thread_id = self._thread_id(conversation_id)
        config = self._config(thread_id)

        yielded_messages: set[str] = set()
        try:
            for chunk in self._graph.stream(
                {"conversation_id": conversation_id, "user_message": message},
                config=config,
                context=context,
                stream_mode="updates",
                version="v2",
            ):
                event = self._extract_stream_event(chunk, yielded_messages)
                if event is not None:
                    yield event
        except GraphRecursionError:
            result = self._error_result(
                conversation_id,
                "RECURSION_LIMIT",
                "master agent recursion limit reached",
                user_turn_id=self._checkpointed_user_turn_id(config),
            )
            yield AgentStreamEvent(
                node="runner",
                message="达到递归上限",
                is_final=True,
                result=result,
            )
            return

        snapshot = self._graph.get_state(config)
        state = dict(snapshot.values) if snapshot.values else {}
        result = self._to_result(conversation_id, state)
        self._store.record_turn(conversation_id)

        yield AgentStreamEvent(
            node="runner",
            message="完成",
            is_final=True,
            result=result,
        )

    @staticmethod
    def _extract_stream_event(
        chunk: Any,
        yielded_messages: set[str],
    ) -> AgentStreamEvent | None:
        if not isinstance(chunk, dict):
            return None
        if chunk.get("type") != "updates":
            return None
        data = chunk.get("data")
        if not isinstance(data, dict):
            return None
        for node, update in data.items():
            if not isinstance(update, dict):
                continue
            events = update.get("events")
            if not isinstance(events, list) or not events:
                msg = _NODE_PROGRESS_MESSAGES.get(node)
                if msg is None:
                    continue
                if msg in yielded_messages:
                    continue
                yielded_messages.add(msg)
                return AgentStreamEvent(node=node, message=msg)
            latest = events[-1]
            if not isinstance(latest, dict):
                continue
            msg = str(latest.get("message", ""))
            if not msg:
                continue
            if msg in yielded_messages:
                continue
            yielded_messages.add(msg)
            return AgentStreamEvent(node=node, message=msg)
        return None

    def _to_result(
        self,
        conversation_id: str,
        state: dict[str, Any],
    ) -> AgentResult:
        final_draft = state.get("final_draft")
        warnings = tuple(str(w) for w in (final_draft or {}).get("warnings", []))
        final_status = (
            str(final_draft.get("status"))
            if isinstance(final_draft, dict) and final_draft.get("status")
            else None
        )
        user_turn_id = state.get("user_turn_id") or self._id_generator.new_id()
        # Collect sub-agent names from results as "selected tools"
        current_call_ids = set(state.get("executed_call_ids", []))
        sa_results = (
            result
            for result in state.get("sub_agent_results", [])
            if isinstance(result, dict) and result.get("call_id") in current_call_ids
        )
        sa_names: tuple[str, ...] = tuple(
            r.get("sub_agent_name", "") for r in sa_results if r.get("sub_agent_name")
        )
        return AgentResult(
            conversation_id=conversation_id,
            user_turn_id=str(user_turn_id),
            status=str(state.get("status", "error")),
            final_status=final_status,
            response_text=str(state.get("final_response") or ""),
            active_workflow_thread_id=state.get("active_workflow_thread_id"),
            selected_tools=sa_names,
            tool_call_count=int(state.get("sub_agent_call_count", 0) or 0),
            model_call_count=int(state.get("model_call_count", 0) or 0),
            evidence_ids=tuple(str(item) for item in state.get("evidence_ids", [])),
            warnings=warnings,
            error=state.get("error"),
        )

    def _error_result(
        self,
        conversation_id: str,
        code: str,
        message: str,
        *,
        user_turn_id: str | None = None,
    ) -> AgentResult:
        return AgentResult(
            conversation_id=conversation_id,
            user_turn_id=user_turn_id or self._id_generator.new_id(),
            status="error",
            response_text=message,
            error={"code": code, "message": message, "retryable": False},
        )

    def _checkpointed_user_turn_id(self, config: RunnableConfig) -> str | None:
        try:
            snapshot = self._graph.get_state(config)
        except Exception:
            return None
        values = snapshot.values or {}
        value = values.get("user_turn_id")
        return str(value) if value else None

    @staticmethod
    def _thread_id(conversation_id: str) -> str:
        return f"master_{conversation_id}"

    def _config(self, thread_id: str) -> RunnableConfig:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._recursion_limit,
        }
