"""Execute sub-agent delegation calls (parallel to ToolExecutor)."""

import hashlib
import json
from dataclasses import dataclass

from .sub_agent_registry import SubAgentRegistry
from .sub_agent_spec import SubAgentCall, SubAgentResultEnvelope


@dataclass(frozen=True)
class SubAgentExecutionOutcome:
    """One processed sub-agent invocation result.

    Same pattern as ``ToolExecutionOutcome`` (tool_executor.py:26-35).
    """

    call_id: str
    sub_agent_name: str
    status: str  # "ok" | "error"
    envelope: SubAgentResultEnvelope
    output_json: str  # compact JSON for the master model's transcript


class SubAgentExecutor:
    """Sequentially execute sub-agent delegation calls.

    Pipeline per call:
    1. Resolve delegate function name -> SubAgentSpec
    2. Parse arguments JSON, extract task
    3. Create runner via spec.runner_factory()
    4. Invoke runner.ask(message=task)
    5. Wrap in SubAgentResultEnvelope
    6. Output size check
    """

    def __init__(
        self,
        *,
        registry: SubAgentRegistry,
        max_output_bytes: int = 65536,
    ) -> None:
        self._registry = registry
        self._max_output_bytes = max_output_bytes

    def execute(
        self,
        calls: list[SubAgentCall],
        *,
        parent_conversation_id: str | None = None,
    ) -> list[SubAgentExecutionOutcome]:
        return [
            self._execute_one(call, parent_conversation_id=parent_conversation_id)
            for call in calls
        ]

    def _execute_one(
        self,
        call: SubAgentCall,
        *,
        parent_conversation_id: str | None = None,
    ) -> SubAgentExecutionOutcome:
        # 1. Resolve delegate function -> SubAgentSpec
        spec = self._registry.resolve_by_delegate_function(call.name)
        if spec is None:
            return self._error(
                call,
                spec_name="",
                code="UNKNOWN_SUB_AGENT",
                message=f"unknown sub-agent delegate function: {call.name!r}",
            )

        # 2. Parse arguments JSON, extract task
        try:
            args = json.loads(call.arguments_json)
        except json.JSONDecodeError:
            return self._error(
                call,
                spec_name=spec.name,
                code="INVALID_ARGUMENTS",
                message="sub-agent arguments are not valid JSON",
            )
        if not isinstance(args, dict):
            return self._error(
                call,
                spec_name=spec.name,
                code="INVALID_ARGUMENTS",
                message="sub-agent arguments must be a JSON object",
            )
        task = args.get("task", "")
        if not isinstance(task, str) or not task.strip():
            return self._error(
                call,
                spec_name=spec.name,
                code="MISSING_TASK",
                message="task is required and must be a non-empty string",
            )

        # 3-4. Create runner + invoke
        try:
            runner = spec.runner_factory()
        except Exception as exc:
            return self._error(
                call,
                spec_name=spec.name,
                code="RUNNER_CREATE_FAILED",
                message=f"failed to create sub-agent runner: {exc}",
            )
        try:
            sub_conversation_id = None
            if parent_conversation_id:
                digest = hashlib.sha256(
                    parent_conversation_id.encode("utf-8")
                ).hexdigest()[:24]
                sub_conversation_id = f"sub_{spec.name}_{digest}"
            if sub_conversation_id is None:
                result = runner.ask(message=task)
            else:
                result = runner.ask(
                    message=task,
                    conversation_id=sub_conversation_id,
                )
        except Exception as exc:
            return self._error(
                call,
                spec_name=spec.name,
                code="SUB_AGENT_EXECUTION_FAILED",
                message=f"sub-agent execution failed: {exc}",
            )
        result_status = getattr(result, "status", None)
        final_status = getattr(result, "final_status", None)
        if (
            result_status is not None and result_status != "completed"
        ) or final_status == "error":
            error = getattr(result, "error", None) or {}
            message = (
                error.get("message", "sub-agent did not complete")
                if isinstance(error, dict)
                else "sub-agent did not complete"
            )
            if final_status == "error" and not error:
                message = str(getattr(result, "response_text", ""))[:1000] or (
                    "sub-agent returned an error draft"
                )
            return self._error(
                call,
                spec_name=spec.name,
                code="SUB_AGENT_FAILED",
                message=str(message),
            )

        # 5. Build envelope
        envelope = SubAgentResultEnvelope(
            status="ok",
            sub_agent_name=spec.name,
            call_id=call.call_id,
            task=task,
            response_text=getattr(result, "response_text", ""),
            active_workflow_thread_id=getattr(
                result, "active_workflow_thread_id", None
            ),
            evidence_ids=(
                list(getattr(result, "evidence_ids", ()))
                if hasattr(result, "evidence_ids")
                else []
            ),
            warnings=(
                list(getattr(result, "warnings", ()))
                if hasattr(result, "warnings")
                else []
            ),
        )

        # 6. Output size check
        output_json = json.dumps(
            envelope.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(output_json.encode("utf-8")) > self._max_output_bytes:
            return self._error(
                call,
                spec_name=spec.name,
                code="OUTPUT_TOO_LARGE",
                message="sub-agent output exceeds the size limit",
            )
        return self._outcome(envelope, output_json)

    def _error(
        self,
        call: SubAgentCall,
        *,
        spec_name: str,
        code: str,
        message: str,
    ) -> SubAgentExecutionOutcome:
        envelope = SubAgentResultEnvelope(
            status="error",
            sub_agent_name=spec_name or call.name,
            call_id=call.call_id,
            task="",
            response_text="",
            error={"code": code, "message": message},
        )
        output_json = json.dumps(
            envelope.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return SubAgentExecutionOutcome(
            call_id=call.call_id,
            sub_agent_name=spec_name or call.name,
            status="error",
            envelope=envelope,
            output_json=output_json,
        )

    @staticmethod
    def _outcome(
        envelope: SubAgentResultEnvelope,
        output_json: str,
    ) -> SubAgentExecutionOutcome:
        return SubAgentExecutionOutcome(
            call_id=envelope.call_id,
            sub_agent_name=envelope.sub_agent_name,
            status=envelope.status,
            envelope=envelope,
            output_json=output_json,
        )
