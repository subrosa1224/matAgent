"""Workflow runner (S3-M6): run, get_state, get_history without replay/CLI."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, ConfigDict, ValidationError

from materials_screening.workflow.checkpointer import CheckpointerHandle
from materials_screening.workflow.context import WorkflowContext
from materials_screening.workflow.errors import (
    WorkflowCheckpointError,
    WorkflowErrorCode,
    WorkflowErrorData,
    WorkflowInputError,
    WorkflowInvariantError,
)
from materials_screening.workflow.graph_builder import compile_workflow
from materials_screening.workflow.input_output import (
    WorkflowGraphInput,
    WorkflowInput,
    WorkflowOutput,
)
from materials_screening.workflow.settings import WorkflowSettings
from materials_screening.workflow.state import (
    ArtifactRef,
    WorkflowCheckpointView,
    WorkflowStateView,
    WorkflowStatus,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

_THREAD_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,253}$"
_SAFE_THREAD_ID_LENGTH = 255
_ARTIFACT_REF_FIELDS = (
    "retrieval_ref",
    "filtered_ref",
    "filter_trace_ref",
    "ranked_ref",
    "validation_ref",
    "screening_result_ref",
    "export_manifest_ref",
)


class StreamEvent(BaseModel):
    """Stable node-level stream summary for CLI/app consumers.

    Contains only node, status, counts and message; never full state,
    MaterialRecords, model raw responses or reasoning.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    node: str
    status: str
    counts: dict[str, int]
    message: str


class WorkflowRunner:
    """Owns a compiled graph and checkpointer; builds safe input/config.

    The runner never re-implements node business logic; it only orchestrates
    invoke/stream and converts checkpoints into safe views. ``replay`` and
    the CLI are intentionally not implemented in this milestone.
    """

    def __init__(
        self,
        *,
        settings: WorkflowSettings,
        context: WorkflowContext,
        checkpointer: CheckpointerHandle,
    ) -> None:
        self._settings = settings
        self._context = context
        self._handle = checkpointer
        self._closed = False
        self._last_thread_id: str | None = None
        self._graph: CompiledStateGraph[Any, Any, Any, Any] = compile_workflow(
            checkpointer=checkpointer.checkpointer
        )

    def run(
        self,
        workflow_input: WorkflowInput,
        *,
        stream: bool = False,
        thread_id: str | None = None,
    ) -> WorkflowOutput:
        """Run one workflow; generates run_id/thread_id exactly once."""
        self._ensure_open()
        if thread_id is None:
            run_id = self._new_run_id()
            thread_id = run_id
        else:
            self._validate_thread_id(thread_id)
            if self.thread_exists(thread_id):
                raise WorkflowCheckpointError(
                    f"thread {thread_id!r} already exists; "
                    "use replay or a new thread id"
                )
            run_id = thread_id
        self._last_thread_id = thread_id
        graph_input = self._build_graph_input(workflow_input, run_id)
        config = self._build_config(thread_id)
        try:
            if stream:
                list(
                    self._graph.stream(
                        graph_input.model_dump(mode="json"),
                        config=config,
                        context=self._context,
                        stream_mode="updates",
                        version="v2",
                    )
                )
                snapshot = self._graph.get_state(config)
                state = dict(snapshot.values)
            else:
                state = self._graph.invoke(
                    graph_input.model_dump(mode="json"),
                    config=config,
                    context=self._context,
                )
        except GraphRecursionError as exc:
            return self._failed_output(
                run_id=run_id,
                thread_id=thread_id,
                code=WorkflowErrorCode.WORKFLOW_RECURSION_LIMIT,
                message="workflow recursion limit reached",
                exception_type=type(exc).__name__,
            )
        except WorkflowCheckpointError as exc:
            return self._failed_output(
                run_id=run_id,
                thread_id=thread_id,
                code=WorkflowErrorCode.CHECKPOINT_FAILED,
                message="checkpoint operation failed",
                exception_type=type(exc).__name__,
            )
        return self._build_output(state, run_id, thread_id)

    def stream(
        self,
        workflow_input: WorkflowInput,
        *,
        thread_id: str | None = None,
    ) -> Iterator[StreamEvent]:
        """Yield stable node-level events from ``stream_mode="updates"`` v2."""
        self._ensure_open()
        if thread_id is None:
            run_id = self._new_run_id()
            thread_id = run_id
        else:
            self._validate_thread_id(thread_id)
            if self.thread_exists(thread_id):
                raise WorkflowCheckpointError(
                    f"thread {thread_id!r} already exists; "
                    "use replay or a new thread id"
                )
            run_id = thread_id
        self._last_thread_id = thread_id
        graph_input = self._build_graph_input(workflow_input, run_id)
        config = self._build_config(run_id)
        try:
            for chunk in self._graph.stream(
                graph_input.model_dump(mode="json"),
                config=config,
                context=self._context,
                stream_mode="updates",
                version="v2",
            ):
                event = self._to_stream_event(dict(chunk))
                if event is not None:
                    yield event
        except GraphRecursionError:
            yield StreamEvent(
                node="runner",
                status=WorkflowStatus.FAILED.value,
                counts={},
                message="workflow recursion limit reached",
            )
        except WorkflowCheckpointError:
            yield StreamEvent(
                node="runner",
                status=WorkflowStatus.FAILED.value,
                counts={},
                message="checkpoint operation failed",
            )

    def get_state(self, thread_id: str) -> WorkflowStateView:
        """Return a safe state view; never the raw internal state."""
        self._ensure_open()
        config = self._build_config(thread_id)
        snapshot = self._graph.get_state(config)
        return self._to_state_view(snapshot.values, thread_id)

    def get_output(self, thread_id: str) -> WorkflowOutput:
        """Build the final WorkflowOutput from a thread's checkpoint."""
        self._ensure_open()
        config = self._build_config(thread_id)
        snapshot = self._graph.get_state(config)
        values = snapshot.values if isinstance(snapshot.values, dict) else {}
        run_id = str(values.get("run_id", thread_id))
        return self._build_output(dict(values), run_id, thread_id)

    def thread_exists(self, thread_id: str) -> bool:
        """Return True when the thread already has checkpoints."""
        self._ensure_open()
        config = self._build_config(thread_id)
        snapshot = self._graph.get_state(config)
        return bool(snapshot.values)

    def replay(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
        confirm_remote_calls: bool = False,
    ) -> WorkflowOutput:
        """Replay from an existing checkpoint; never exposes update_state.

        The checkpoint must belong to the thread and its referenced artifacts
        must verify. Replay re-executes nodes after the checkpoint, which may
        re-trigger remote calls, so it requires explicit confirmation.
        """
        self._ensure_open()
        config = self._build_config(thread_id)
        snapshot = self._find_checkpoint(config, checkpoint_id)
        if not confirm_remote_calls:
            raise WorkflowCheckpointError(
                "replay may re-trigger remote calls; pass confirm_remote_calls=True"
            )
        self._verify_replay_artifacts(snapshot.values)
        try:
            state = self._graph.invoke(
                None,
                config=snapshot.config,
                context=self._context,
            )
        except GraphRecursionError as exc:
            return self._failed_output(
                run_id=thread_id,
                thread_id=thread_id,
                code=WorkflowErrorCode.WORKFLOW_RECURSION_LIMIT,
                message="workflow recursion limit reached during replay",
                exception_type=type(exc).__name__,
            )
        except WorkflowCheckpointError as exc:
            return self._failed_output(
                run_id=thread_id,
                thread_id=thread_id,
                code=WorkflowErrorCode.CHECKPOINT_FAILED,
                message="checkpoint operation failed during replay",
                exception_type=type(exc).__name__,
            )
        run_id = str(state.get("run_id", thread_id))
        return self._build_output(state, run_id, thread_id)

    def get_history(
        self,
        thread_id: str,
        limit: int | None = None,
    ) -> tuple[WorkflowCheckpointView, ...]:
        """Return safe checkpoint summaries, newest first."""
        self._ensure_open()
        config = self._build_config(thread_id)
        history = list(
            self._graph.get_state_history(
                config,
                limit=limit
                if limit is not None
                else self._settings.workflow_history_limit,
            )
        )
        return tuple(self._to_checkpoint_view(snapshot) for snapshot in history)

    def close(self) -> None:
        """Close the owned checkpointer handle (idempotent)."""
        if not self._closed:
            self._handle.close()
            self._closed = True

    def __enter__(self) -> WorkflowRunner:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise WorkflowCheckpointError("runner is closed; open a new runner")

    @property
    def last_thread_id(self) -> str | None:
        """Thread id of the most recent run/stream; None before any run."""
        return self._last_thread_id

    @staticmethod
    def _to_stream_event(chunk: dict[str, Any]) -> StreamEvent | None:
        """Project one v2 updates chunk into a safe StreamEvent."""
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
                continue
            event = events[0]
            if not isinstance(event, dict):
                continue
            return StreamEvent(
                node=str(event.get("node", node)),
                status=str(event.get("status", "")),
                counts=WorkflowRunner._int_counts(event.get("metrics")),
                message=str(event.get("message", "")),
            )
        return None

    @staticmethod
    def _int_counts(metrics: object) -> dict[str, int]:
        if not isinstance(metrics, dict):
            return {}
        return {
            str(key): value
            for key, value in metrics.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }

    def _find_checkpoint(self, config: RunnableConfig, checkpoint_id: str) -> Any:
        thread_id = config["configurable"]["thread_id"]
        for snapshot in self._graph.get_state_history(
            config,
            limit=self._settings.workflow_history_limit,
        ):
            current = snapshot.config.get("configurable", {}).get("checkpoint_id")
            if current == checkpoint_id:
                return snapshot
        raise WorkflowCheckpointError(
            f"checkpoint {checkpoint_id!r} not found in thread {thread_id!r}"
        )

    def _verify_replay_artifacts(self, values: Any) -> None:
        if not isinstance(values, dict):
            raise WorkflowCheckpointError("checkpoint values are invalid")
        for field in _ARTIFACT_REF_FIELDS:
            raw_ref = values.get(field)
            if raw_ref is None:
                continue
            try:
                ref = ArtifactRef.model_validate(raw_ref)
                valid = self._context.artifact_store.verify(ref)
            except ValidationError:
                valid = False
            if not valid:
                raise WorkflowCheckpointError(
                    f"artifact {field} is missing or corrupted; refusing replay"
                )

    @staticmethod
    def _new_run_id() -> str:
        return str(uuid.uuid4())

    def _build_graph_input(
        self,
        workflow_input: WorkflowInput,
        run_id: str,
    ) -> WorkflowGraphInput:
        if workflow_input.query is not None:
            return WorkflowGraphInput(
                run_id=run_id,
                workflow_version=self._settings.workflow_version,
                input_mode="query",
                user_query=workflow_input.query,
                output_root=workflow_input.output_root,
                export_cif=workflow_input.export_cif,
            )
        return WorkflowGraphInput(
            run_id=run_id,
            workflow_version=self._settings.workflow_version,
            input_mode="request",
            raw_request=workflow_input.request,
            output_root=workflow_input.output_root,
            export_cif=workflow_input.export_cif,
        )

    def _build_config(self, thread_id: str) -> RunnableConfig:
        self._validate_thread_id(thread_id)
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._settings.workflow_recursion_limit,
            "tags": ["materials-screening", "workflow-v1"],
            "metadata": {
                "workflow_version": self._settings.workflow_version,
            },
        }

    @staticmethod
    def _validate_thread_id(thread_id: str) -> None:
        if not thread_id or len(thread_id) >= _SAFE_THREAD_ID_LENGTH:
            raise WorkflowInputError("thread_id must be non-empty and under 255 chars")
        if not re.fullmatch(_THREAD_ID_PATTERN, thread_id):
            raise WorkflowInputError(f"unsafe thread_id: {thread_id!r}")

    def _build_output(
        self,
        state: dict[str, Any],
        run_id: str,
        thread_id: str,
    ) -> WorkflowOutput:
        raw_status = state.get("status")
        if raw_status is None:
            raise WorkflowInvariantError("workflow state has no status")
        return WorkflowOutput(
            run_id=run_id,
            thread_id=thread_id,
            status=WorkflowStatus(raw_status),
            planner_status=state.get("planner_status"),
            request=state.get("screening_request"),
            retrieved_count=state.get("retrieved_count", 0),
            filtered_count=state.get("filtered_count", 0),
            returned_count=state.get("returned_count", 0),
            validation_passed=state.get("validation_passed"),
            exports=tuple(state.get("exports") or ()),
            warnings=tuple(state.get("warnings") or ()),
            error=state.get("error"),
            clarification_question=state.get("clarification_question"),
        )

    def _failed_output(
        self,
        *,
        run_id: str,
        thread_id: str,
        code: WorkflowErrorCode,
        message: str,
        exception_type: str,
    ) -> WorkflowOutput:
        return WorkflowOutput(
            run_id=run_id,
            thread_id=thread_id,
            status=WorkflowStatus.FAILED,
            planner_status=None,
            request=None,
            retrieved_count=0,
            filtered_count=0,
            returned_count=0,
            validation_passed=None,
            exports=(),
            warnings=(),
            error=WorkflowErrorData(
                code=code,
                node="runner",
                message=message,
                retryable=False,
                exception_type=exception_type,
                occurred_at=self._context.clock().isoformat(),
            ).model_dump(mode="json"),
            clarification_question=None,
        )

    @staticmethod
    def _to_state_view(values: Any, thread_id: str) -> WorkflowStateView:
        if not isinstance(values, dict):
            values = {}
        return WorkflowStateView(
            run_id=values.get("run_id", ""),
            thread_id=thread_id,
            workflow_version=values.get("workflow_version", ""),
            status=values.get("status", ""),
            current_node=values.get("current_node"),
            started_at=values.get("started_at"),
            finished_at=values.get("finished_at"),
            planner_status=values.get("planner_status"),
            retrieved_count=values.get("retrieved_count", 0),
            filtered_count=values.get("filtered_count", 0),
            returned_count=values.get("returned_count", 0),
            validation_passed=values.get("validation_passed"),
            exports=tuple(values.get("exports") or ()),
            warnings=tuple(values.get("warnings") or ()),
            error=values.get("error"),
        )

    @staticmethod
    def _to_checkpoint_view(snapshot: Any) -> WorkflowCheckpointView:
        metadata = snapshot.metadata if snapshot.metadata is not None else {}
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        values = snapshot.values if isinstance(snapshot.values, dict) else {}
        step = metadata.get("step", 0)
        if not isinstance(step, int) or step < 0:
            step = 0
        return WorkflowCheckpointView(
            step=step,
            checkpoint_id=checkpoint_id,
            source=metadata.get("source", ""),
            current_node=values.get("current_node"),
            status=values.get("status", ""),
            next_nodes=tuple(snapshot.next),
            created_at=metadata.get("created_at"),
        )
