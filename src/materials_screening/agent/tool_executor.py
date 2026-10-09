"""Stage 3.5 deterministic tool executor (S3.5-M4 prep)."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace

from pydantic import ValidationError

from materials_screening.agent.context import AgentToolContext
from materials_screening.agent.errors import (
    AgentInvariantError,
    AgentToolError,
)
from materials_screening.agent.ledger import ToolExecutionLedger
from materials_screening.agent.models import (
    AgentToolCall,
    ToolErrorData,
    ToolResultEnvelope,
    ToolResultStatus,
)
from materials_screening.agent.policy import AgentPolicyError, AgentToolPolicy
from materials_screening.agent.state import MaterialAgentState
from materials_screening.agent.tool_registry import AgentToolRegistry
from materials_screening.errors import (
    InvalidRequestError,
    MaterialsScreeningError,
    RepositoryAuthenticationError,
    RepositoryError,
    RepositoryRateLimitError,
    RepositoryTimeoutError,
)


@dataclass(frozen=True)
class ToolExecutionOutcome:
    """One processed tool call: safe envelope and compact output JSON."""

    call_id: str
    tool_name: str
    evidence_id: str
    status: ToolResultStatus
    envelope: ToolResultEnvelope
    output_json: str


class ToolExecutor:
    """Sequentially execute model-emitted tool calls with full guarding.

    Pipeline per call:
    1. registry lookup
    2. raw argument byte limit
    3. json.loads
    4. pydantic model_validate (input models forbid extra fields)
    5. policy
    6. idempotent ledger reuse for duplicate call ids
    7. execute
    8. ToolResultEnvelope
    9. output size check
    10. evidence registration
    """

    def __init__(
        self,
        *,
        registry: AgentToolRegistry,
        policy: AgentToolPolicy,
        max_argument_bytes: int = 8192,
        max_output_bytes: int = 32768,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._max_argument_bytes = max_argument_bytes
        self._max_output_bytes = max_output_bytes

    def execute(
        self,
        calls: Sequence[AgentToolCall],
        *,
        context: AgentToolContext,
        state: MaterialAgentState,
    ) -> tuple[ToolExecutionOutcome, ...]:
        """Process every call in order; each call gets a matching outcome."""
        return tuple(
            self._execute_one(call, context=context, state=state) for call in calls
        )

    def _execute_one(
        self,
        call: AgentToolCall,
        *,
        context: AgentToolContext,
        state: MaterialAgentState,
    ) -> ToolExecutionOutcome:
        # 1. Registry lookup.
        try:
            tool = self._registry.get(call.name)
        except AgentToolError:
            return self._error(call, "UNKNOWN_TOOL", "unknown or forbidden tool")

        # 2. Raw argument byte limit; the JSON text is never truncated.
        if len(call.arguments_json.encode("utf-8")) > self._max_argument_bytes:
            return self._error(
                call, "ARGUMENT_TOO_LARGE", "tool arguments exceed the size limit"
            )

        # 3. Parse JSON; invalid text is rejected, never truncated or echoed.
        try:
            raw = json.loads(call.arguments_json)
        except json.JSONDecodeError:
            return self._error(
                call, "INVALID_ARGUMENTS", "tool arguments are not valid JSON"
            )
        if not isinstance(raw, dict):
            return self._error(
                call, "INVALID_ARGUMENTS", "tool arguments must be a JSON object"
            )

        # 4. Validate into the tool input model; extra fields are rejected.
        try:
            arguments = tool.input_model.model_validate(raw)
        except ValidationError:
            return self._error(
                call, "INVALID_ARGUMENTS", "tool arguments failed schema validation"
            )

        # 5. Policy; duplicate call ids are resolved idempotently at step 6.
        try:
            self._policy.validate_call(
                tool=tool,
                arguments=arguments,
                call_id=call.call_id,
                conversation_id=context.conversation_id,
                state=state,
                links=context.conversation_links,
                ledger=context.ledger,
            )
        except AgentPolicyError as exc:
            if exc.code == "DUPLICATE_CALL_ID":
                return self._reuse(call, context.ledger)
            return self._error(call, exc.code, str(exc))

        # 7. Execute with a per-call context so each tool records its own id.
        call_context = replace(context, call_id=call.call_id)
        try:
            output = tool.execute(arguments, call_context)
        except AgentToolError as exc:
            code = str(getattr(exc, "code", "TOOL_ERROR"))
            retryable = bool(getattr(exc, "retryable", False))
            return self._error(call, code, str(exc), retryable=retryable)
        except MaterialsScreeningError as exc:
            if isinstance(exc, RepositoryAuthenticationError):
                return self._error(
                    call,
                    "REPOSITORY_AUTHENTICATION",
                    "Materials Project API authentication failed",
                )
            if isinstance(exc, RepositoryRateLimitError):
                return self._error(
                    call,
                    "REPOSITORY_RATE_LIMIT",
                    "Materials Project API rate limit exceeded",
                    retryable=True,
                )
            if isinstance(exc, RepositoryTimeoutError):
                return self._error(
                    call,
                    "REPOSITORY_TIMEOUT",
                    "Materials Project request timed out",
                    retryable=True,
                )
            if isinstance(exc, RepositoryError):
                return self._error(
                    call,
                    "REPOSITORY_CONNECTION",
                    "Materials Project network connection failed; please retry",
                    retryable=True,
                )
            if isinstance(exc, InvalidRequestError):
                return self._error(call, "INVALID_REQUEST", str(exc))
            return self._error(call, "TOOL_ERROR", str(exc))

        # 10. Evidence registration: the tool must record one ledger entry.
        entry = call_context.ledger.get_entry(call.call_id)
        if entry is None:
            raise AgentInvariantError(
                f"tool {tool.name!r} did not register evidence "
                f"for call {call.call_id!r}"
            )
        evidence_id = entry.evidence_id
        if not evidence_id:
            raise AgentInvariantError(
                f"tool {tool.name!r} registered an empty evidence id"
            )

        # 8. Build the safe envelope.
        envelope = ToolResultEnvelope(
            status=ToolResultStatus.OK,
            tool_name=tool.name,
            call_id=call.call_id,
            evidence_id=evidence_id,
            output=output.model_dump(mode="json"),
        )

        # 9. Output size check; oversized output becomes a safe error envelope.
        if len(envelope.to_json().encode("utf-8")) > self._max_output_bytes:
            return self._error(
                call,
                "TOOL_OUTPUT_TOO_LARGE",
                "tool output exceeds the size limit",
                evidence_id=evidence_id,
            )
        return self._outcome(envelope)

    def _reuse(
        self,
        call: AgentToolCall,
        ledger: ToolExecutionLedger,
    ) -> ToolExecutionOutcome:
        """Return the previously recorded result without re-executing."""
        entry = ledger.get_entry(call.call_id)
        if entry is None:
            raise AgentInvariantError(
                f"ledger has no entry for duplicate call {call.call_id!r}"
            )
        if entry.tool_name != call.name:
            return self._error(
                call,
                "DUPLICATE_CALL_ID",
                "call_id was already used by a different tool",
            )
        try:
            output = json.loads(entry.result_json)
        except json.JSONDecodeError:
            raise AgentInvariantError(
                f"stored result for call {call.call_id!r} is not valid JSON"
            ) from None
        if not isinstance(output, dict):
            raise AgentInvariantError(
                f"stored result for call {call.call_id!r} is not a JSON object"
            )
        envelope = ToolResultEnvelope(
            status=ToolResultStatus.OK,
            tool_name=entry.tool_name,
            call_id=call.call_id,
            evidence_id=entry.evidence_id,
            output=output,
        )
        return self._outcome(envelope)

    @staticmethod
    def _error(
        call: AgentToolCall,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        evidence_id: str = "",
    ) -> ToolExecutionOutcome:
        envelope = ToolResultEnvelope(
            status=ToolResultStatus.ERROR,
            tool_name=call.name,
            call_id=call.call_id,
            evidence_id=evidence_id,
            error=ToolErrorData(code=code, message=message, retryable=retryable),
        )
        return ToolExecutor._outcome(envelope)

    @staticmethod
    def _outcome(envelope: ToolResultEnvelope) -> ToolExecutionOutcome:
        return ToolExecutionOutcome(
            call_id=envelope.call_id,
            tool_name=envelope.tool_name,
            evidence_id=envelope.evidence_id,
            status=envelope.status,
            envelope=envelope,
            output_json=envelope.to_json(),
        )
