"""MaterialAgentRunner (S3.5-M6): start, ask and inspect conversations.

The runner creates/restores conversations in the store, serializes asks per
conversation with a non-blocking lock, builds the ``MaterialAgentContext``
right before each invoke, converts the graph state into a safe ``AgentResult``
and maps ``GraphRecursionError`` to a stable error result. Raw tool arguments,
model responses and reasoning are never exposed.
"""

from __future__ import annotations

import secrets
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, ConfigDict, Field

from materials_screening.agent.context import (
    MaterialAgentContext,
    WorkflowResultReader,
)
from materials_screening.agent.conversation_store import ConversationStore
from materials_screening.agent.errors import AgentConversationError
from materials_screening.agent.graph_builder import compile_agent_graph
from materials_screening.agent.ledger import ToolExecutionLedger
from materials_screening.agent.model_base import MaterialAgentModel
from materials_screening.agent.models import AgentResult
from materials_screening.agent.policy import AgentToolPolicy
from materials_screening.agent.settings import AgentSettings
from materials_screening.agent.tool_executor import ToolExecutor
from materials_screening.agent.tool_registry import AgentToolRegistry
from materials_screening.workflow.context import Clock, IdGenerator
from materials_screening.workflow.runner import WorkflowRunner

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph


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
    """Progress event emitted during agent execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node: str
    message: str
    is_final: bool = False
    result: AgentResult | None = None


_NODE_PROGRESS_MESSAGES: dict[str, str] = {
    "prepare_turn": "正在准备...",
    "call_agent_model": "正在调用 AI 模型...",
    "execute_tools": "正在执行工具...",
    "validate_final": "正在验证最终回答...",
    "finalize_success": "正在完成...",
    "finalize_error": "处理错误中...",
}


class _UuidIdGenerator:
    """Default id generator producing compact, safe, unique ids."""

    def new_id(self) -> str:
        return secrets.token_hex(2)


class MaterialAgentRunner:
    """Owns the agent graph and orchestrates one user turn per ask."""

    def __init__(
        self,
        *,
        settings: AgentSettings,
        store: ConversationStore,
        workflow_runner: WorkflowRunner,
        workflow_result_reader: WorkflowResultReader,
        tool_registry: AgentToolRegistry,
        agent_model: MaterialAgentModel,
        checkpointer: BaseCheckpointSaver[Any],
        clock: Clock | None = None,
        id_generator: IdGenerator | None = None,
        recursion_limit: int = 16,
    ) -> None:
        if recursion_limit < 1:
            raise ValueError("recursion_limit must be positive")
        self._settings = settings
        self._store = store
        self._workflow_runner = workflow_runner
        self._workflow_result_reader = workflow_result_reader
        self._tool_registry = tool_registry
        self._agent_model = agent_model
        self._clock = clock or (lambda: datetime.now(UTC))
        self._id_generator = id_generator or _UuidIdGenerator()
        self._recursion_limit = recursion_limit
        self._tool_executor = ToolExecutor(
            registry=tool_registry,
            policy=AgentToolPolicy(
                max_tool_calls_per_turn=settings.agent_max_tool_calls_per_turn,
                max_workflow_runs_per_turn=settings.agent_max_workflow_runs_per_turn,
                max_argument_bytes=settings.agent_max_argument_bytes,
            ),
            max_argument_bytes=settings.agent_max_argument_bytes,
            max_output_bytes=settings.agent_max_tool_output_bytes,
        )
        self._graph: CompiledStateGraph[Any, Any, Any, Any] = compile_agent_graph(
            checkpointer=checkpointer
        )

    def start_conversation(self) -> str:
        """Create and return a new conversation id; retries on collision."""
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
        """Return a safe conversation view; raises when it does not exist."""
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
        """Last terminal status of the conversation's agent thread, if any."""
        config = self._config(self._thread_id(conversation_id))
        try:
            snapshot = self._graph.get_state(config)
        except Exception:
            return None
        values = snapshot.values or {}
        value = values.get("status")
        return str(value) if value else None

    def ask(
        self,
        *,
        message: str,
        conversation_id: str | None = None,
    ) -> AgentResult:
        """Run one turn; a concurrent ask on the same conversation is busy."""
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
            return self._ask_locked(message=message, conversation_id=resolved_id)
        finally:
            lock.release()

    def ask_stream(
        self,
        *,
        message: str,
        conversation_id: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Any:  # Iterator[AgentStreamEvent] but annotated as Any for runtime compat
        """Run one turn with streaming progress events.

        Yields ``AgentStreamEvent`` for each graph node execution. The final
        yielded event has ``is_final=True`` with the ``result`` field populated.

        ``cancel_event`` enables cooperative cancellation: when set, the turn
        ends at the next node boundary with status ``cancelled`` instead of
        starting further model or tool calls. An in-flight model call or tool
        run completes first; the conversation stays usable afterwards.

        Example::

            final_result = None
            for event in runner.ask_stream(message="..."):
                if event.is_final:
                    final_result = event.result
                else:
                    print(f"[{event.node}] {event.message}")
        """
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
            yield from self._ask_stream_locked(
                message=message,
                conversation_id=resolved_id,
                cancel_event=cancel_event,
            )
        finally:
            lock.release()

    def _ask_stream_locked(
        self,
        *,
        message: str,
        conversation_id: str,
        cancel_event: threading.Event | None = None,
    ) -> Any:  # Iterator[AgentStreamEvent]
        if not message.strip():
            result = self._error_result(
                conversation_id,
                "EMPTY_MESSAGE",
                "message must not be empty",
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
        max_turns = self._settings.agent_max_conversation_turns
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

        ledger = ToolExecutionLedger()
        context = MaterialAgentContext(
            workflow_runner=self._workflow_runner,
            workflow_result_reader=self._workflow_result_reader,
            tool_registry=self._tool_registry,
            tool_executor=self._tool_executor,
            agent_model=self._agent_model,
            ledger=ledger,
            settings=self._settings,
            clock=self._clock,
            id_generator=self._id_generator,
            conversation_links=self._store.list_links(conversation_id),
            cancel_event=cancel_event,
        )
        thread_id = self._thread_id(conversation_id)
        config = self._config(thread_id)

        # Use streaming to get progress updates
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
                "agent recursion limit reached",
                user_turn_id=self._checkpointed_user_turn_id(config),
            )
            yield AgentStreamEvent(
                node="runner",
                message="达到递归上限",
                is_final=True,
                result=result,
            )
            return

        # Get final state
        snapshot = self._graph.get_state(config)
        state = dict(snapshot.values) if snapshot.values else {}

        result = self._to_result(conversation_id, state, ledger)
        active_thread = state.get("active_workflow_thread_id")
        if (
            isinstance(active_thread, str)
            and active_thread
            and not self._store.owns_workflow(conversation_id, active_thread)
        ):
            self._store.link_workflow(conversation_id, active_thread)
            self._store.set_active_thread(conversation_id, active_thread)
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
        """Extract a progress event from a LangGraph v2 updates chunk."""
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
                # Fallback: map node name to default message
                msg = _NODE_PROGRESS_MESSAGES.get(node)
                if msg is None:
                    continue
                if msg in yielded_messages:
                    continue
                yielded_messages.add(msg)
                return AgentStreamEvent(node=node, message=msg)
            # Take the latest (most recent) event
            latest = events[-1]
            if not isinstance(latest, dict):
                continue
            msg = str(latest.get("message", ""))
            if not msg:
                continue
            if msg in yielded_messages:
                continue
            yielded_messages.add(msg)
            return AgentStreamEvent(
                node=node,
                message=msg,
            )
        return None

    def _ask_locked(
        self,
        *,
        message: str,
        conversation_id: str,
    ) -> AgentResult:
        if not message.strip():
            return self._error_result(
                conversation_id,
                "EMPTY_MESSAGE",
                "message must not be empty",
            )
        if not self._store.exists(conversation_id):
            self._store.create(conversation_id)
        metadata = self._store.get(conversation_id)
        max_turns = self._settings.agent_max_conversation_turns
        if metadata.turn_count >= max_turns:
            return self._error_result(
                conversation_id,
                "TURN_LIMIT",
                f"conversation reached the {max_turns}-turn limit",
            )

        ledger = ToolExecutionLedger()
        context = MaterialAgentContext(
            workflow_runner=self._workflow_runner,
            workflow_result_reader=self._workflow_result_reader,
            tool_registry=self._tool_registry,
            tool_executor=self._tool_executor,
            agent_model=self._agent_model,
            ledger=ledger,
            settings=self._settings,
            clock=self._clock,
            id_generator=self._id_generator,
            conversation_links=self._store.list_links(conversation_id),
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
                "agent recursion limit reached",
                user_turn_id=self._checkpointed_user_turn_id(config),
            )

        result = self._to_result(conversation_id, state, ledger)
        active_thread = state.get("active_workflow_thread_id")
        if (
            isinstance(active_thread, str)
            and active_thread
            and not self._store.owns_workflow(conversation_id, active_thread)
        ):
            self._store.link_workflow(conversation_id, active_thread)
            self._store.set_active_thread(conversation_id, active_thread)
        self._store.record_turn(conversation_id)
        return result

    def _to_result(
        self,
        conversation_id: str,
        state: dict[str, Any],
        ledger: ToolExecutionLedger,
    ) -> AgentResult:
        final_draft = state.get("final_draft")
        warnings = tuple(str(w) for w in (final_draft or {}).get("warnings", []))
        final_status = (
            str(final_draft.get("status"))
            if isinstance(final_draft, dict) and final_draft.get("status")
            else None
        )
        user_turn_id = state.get("user_turn_id") or self._id_generator.new_id()
        return AgentResult(
            conversation_id=conversation_id,
            user_turn_id=str(user_turn_id),
            status=str(state.get("status", "error")),
            final_status=final_status,
            response_text=str(state.get("final_response") or ""),
            active_workflow_thread_id=state.get("active_workflow_thread_id"),
            selected_tools=ledger.executed_tool_names(),
            tool_call_count=int(state.get("tool_call_count", 0) or 0),
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
        return f"agent_{conversation_id}"

    def _config(self, thread_id: str) -> RunnableConfig:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._recursion_limit,
        }
