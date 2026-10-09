"""Final answer validation for the single MaterialAgent (S3.5-M5).

The validator guards the final ``AgentFinalDraft`` before it is shown to the
user. Checks are deterministic and heuristic by design: material IDs are
matched with a configurable pattern (``mp-\\d+``) and task-fact claims are
detected with a small marker list, not a numeric NLP claim parser.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from materials_screening.agent.models import AgentFinalDraft, AgentFinalStatus
from materials_screening.agent.policy import ConversationWorkflowLink

_DEFAULT_ANSWER_MAX_CHARS = 8000
_DEFAULT_MATERIAL_ID_PATTERN = r"mp-\d+"
_THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,253}$")
_WORKFLOW_STATUS_TOOLS = frozenset({"run_screening_workflow"})
_TASK_FACT_MARKERS = (
    "带隙",
    "band gap",
    "hull",
    "排名",
    "rank",
    "分数",
    "score",
    "筛选",
    "工作流",
    "导出",
    "任务状态",
    "completed",
    "no_results",
    "failed",
)
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?:api[_-]?key|secret|password|token)\s*[:=]\s*[^\s,;]{6,}"),
    re.compile(r"authorization\s*[:=]\s*[^\s,;]{4,}", re.IGNORECASE),
    re.compile(r"bearer\s+[A-Za-z0-9._~+/=-]{10,}", re.IGNORECASE),
)


@dataclass(frozen=True)
class EvidenceRecord:
    """One tool result registered as evidence in the current session."""

    evidence_id: str
    tool_name: str
    result_json: str


@dataclass(frozen=True)
class FinalValidationContext:
    """Session facts needed to validate a final draft."""

    conversation_id: str
    user_turn_id: str
    evidence: tuple[EvidenceRecord, ...] = ()
    active_workflow_thread_id: str | None = None
    conversation_links: tuple[ConversationWorkflowLink, ...] = ()


class FinalValidationError(BaseModel):
    """One stable, machine-readable validation error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str


class FinalValidationResult(BaseModel):
    """Aggregated result; ``ok`` is False when any error is present."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    errors: tuple[FinalValidationError, ...] = ()

    @property
    def codes(self) -> tuple[str, ...]:
        """Stable error codes, in validation order."""
        return tuple(error.code for error in self.errors)


class FinalValidator:
    """Deterministic guard for final drafts before user-facing output."""

    def __init__(
        self,
        *,
        max_answer_chars: int = _DEFAULT_ANSWER_MAX_CHARS,
        material_id_pattern: str = _DEFAULT_MATERIAL_ID_PATTERN,
    ) -> None:
        if max_answer_chars < 1:
            raise ValueError("max_answer_chars must be positive")
        self._max_answer_chars = max_answer_chars
        self._material_id = re.compile(material_id_pattern)

    def validate(
        self,
        draft: AgentFinalDraft | Mapping[str, Any],
        context: FinalValidationContext,
    ) -> FinalValidationResult:
        """Validate a draft; all errors are accumulated, never fail-fast."""
        errors: list[FinalValidationError] = []
        parsed = self._validate_schema_and_answer(draft, errors)
        if parsed is None:
            return FinalValidationResult(ok=False, errors=tuple(errors))
        self._validate_secrets(parsed, errors)
        evidence_ids, evidence_material_ids = self._evidence_facts(context)
        self._validate_evidence_ids(parsed, evidence_ids, errors)
        self._validate_referenced_materials(parsed, evidence_material_ids, errors)
        self._validate_answer_material_ids(parsed, errors)
        self._validate_evidence_requirement(parsed, evidence_ids, errors)
        self._validate_thread_ownership(parsed, context, errors)
        self._validate_workflow_status(parsed, context, errors)
        return FinalValidationResult(ok=not errors, errors=tuple(errors))

    def _validate_schema_and_answer(
        self,
        draft: AgentFinalDraft | Mapping[str, Any],
        errors: list[FinalValidationError],
    ) -> AgentFinalDraft | None:
        if isinstance(draft, AgentFinalDraft):
            if len(draft.answer) > self._max_answer_chars:
                errors.append(
                    FinalValidationError(
                        code="ANSWER_TOO_LONG",
                        message=(f"answer exceeds {self._max_answer_chars} characters"),
                    )
                )
            return draft
        if not isinstance(draft, Mapping):
            errors.append(
                FinalValidationError(
                    code="INVALID_FINAL_DRAFT",
                    message="final draft must be an AgentFinalDraft object or dict",
                )
            )
            return None
        answer = draft.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            errors.append(
                FinalValidationError(
                    code="ANSWER_EMPTY",
                    message="final draft answer must not be empty",
                )
            )
        elif len(answer) > self._max_answer_chars:
            errors.append(
                FinalValidationError(
                    code="ANSWER_TOO_LONG",
                    message=(f"answer exceeds {self._max_answer_chars} characters"),
                )
            )
        try:
            return AgentFinalDraft.model_validate(draft)
        except ValidationError:
            errors.append(
                FinalValidationError(
                    code="INVALID_FINAL_DRAFT",
                    message="final draft does not match AgentFinalDraft schema",
                )
            )
            return None

    @staticmethod
    def _validate_secrets(
        parsed: AgentFinalDraft,
        errors: list[FinalValidationError],
    ) -> None:
        texts = [
            parsed.answer,
            *parsed.warnings,
            parsed.follow_up_question or "",
        ]
        if any(pattern.search(text) for text in texts for pattern in _SECRET_PATTERNS):
            errors.append(
                FinalValidationError(
                    code="SUSPECTED_SECRET",
                    message="answer contains a suspected API key or secret",
                )
            )

    @staticmethod
    def _validate_evidence_ids(
        parsed: AgentFinalDraft,
        evidence_ids: set[str],
        errors: list[FinalValidationError],
    ) -> None:
        for evidence_id in parsed.evidence_ids:
            if evidence_id not in evidence_ids:
                errors.append(
                    FinalValidationError(
                        code="EVIDENCE_NOT_FOUND",
                        message=(
                            f"evidence id {evidence_id!r} is not from the "
                            "current session"
                        ),
                    )
                )

    @staticmethod
    def _validate_referenced_materials(
        parsed: AgentFinalDraft,
        evidence_material_ids: set[str],
        errors: list[FinalValidationError],
    ) -> None:
        for material_id in parsed.referenced_material_ids:
            if material_id not in evidence_material_ids:
                errors.append(
                    FinalValidationError(
                        code="MATERIAL_NOT_IN_EVIDENCE",
                        message=(
                            f"referenced material id {material_id!r} is not "
                            "backed by tool evidence"
                        ),
                    )
                )

    def _validate_answer_material_ids(
        self,
        parsed: AgentFinalDraft,
        errors: list[FinalValidationError],
    ) -> None:
        referenced = set(parsed.referenced_material_ids)
        for match in self._material_id.finditer(parsed.answer):
            material_id = match.group(0)
            if material_id not in referenced:
                errors.append(
                    FinalValidationError(
                        code="MATERIAL_ID_MISMATCH",
                        message=(
                            f"material id {material_id!r} in the answer is not "
                            "listed in referenced_material_ids"
                        ),
                    )
                )

    def _validate_evidence_requirement(
        self,
        parsed: AgentFinalDraft,
        evidence_ids: set[str],
        errors: list[FinalValidationError],
    ) -> None:
        if parsed.status is not AgentFinalStatus.COMPLETED:
            # A needs_user_input draft asks a question; it does not claim task
            # facts, so it never requires evidence.
            return
        if parsed.evidence_ids and all(
            evidence_id in evidence_ids for evidence_id in parsed.evidence_ids
        ):
            return
        lowered = parsed.answer.lower()
        has_task_facts = (
            bool(parsed.referenced_material_ids)
            or parsed.active_workflow_thread_id is not None
            or bool(self._material_id.search(parsed.answer))
            or any(marker in lowered for marker in _TASK_FACT_MARKERS)
        )
        if has_task_facts:
            errors.append(
                FinalValidationError(
                    code="TASK_FACT_WITHOUT_EVIDENCE",
                    message=(
                        "task facts in the answer require tool evidence, but "
                        "evidence_ids are missing or invalid"
                    ),
                )
            )

    @staticmethod
    def _validate_thread_ownership(
        parsed: AgentFinalDraft,
        context: FinalValidationContext,
        errors: list[FinalValidationError],
    ) -> None:
        thread_id = parsed.active_workflow_thread_id
        if thread_id is None:
            return
        if not _THREAD_ID_PATTERN.fullmatch(thread_id):
            errors.append(
                FinalValidationError(
                    code="THREAD_ID_INVALID",
                    message="active_workflow_thread_id has an unsafe format",
                )
            )
            return
        owned = (
            {context.active_workflow_thread_id}
            if (context.active_workflow_thread_id is not None)
            else set()
        )
        owned.update(
            link.thread_id
            for link in context.conversation_links
            if link.conversation_id == context.conversation_id
        )
        if thread_id not in owned:
            errors.append(
                FinalValidationError(
                    code="THREAD_OWNERSHIP_DENIED",
                    message=(
                        f"thread {thread_id!r} is not owned by conversation "
                        f"{context.conversation_id!r}"
                    ),
                )
            )

    @staticmethod
    def _validate_workflow_status(
        parsed: AgentFinalDraft,
        context: FinalValidationContext,
        errors: list[FinalValidationError],
    ) -> None:
        if parsed.status is not AgentFinalStatus.COMPLETED:
            return
        for record in context.evidence:
            if record.tool_name not in _WORKFLOW_STATUS_TOOLS:
                continue
            payload = _parse_json_object(record.result_json)
            if payload is None:
                continue
            status = payload.get("status")
            if not isinstance(status, str) or status not in {
                "completed",
                "no_results",
            }:
                errors.append(
                    FinalValidationError(
                        code="WORKFLOW_STATUS_MISMATCH",
                        message=(
                            "draft claims completed while the workflow "
                            "evidence reports a status other than completed "
                            "or no_results"
                        ),
                    )
                )
                return
            if payload.get("validation_passed") is False:
                errors.append(
                    FinalValidationError(
                        code="WORKFLOW_STATUS_MISMATCH",
                        message=("draft claims completed while validation failed"),
                    )
                )
                return

    @staticmethod
    def _evidence_facts(
        context: FinalValidationContext,
    ) -> tuple[set[str], set[str]]:
        """Return session evidence ids and all material ids cited in them."""
        evidence_ids: set[str] = set()
        material_ids: set[str] = set()
        for record in context.evidence:
            evidence_ids.add(record.evidence_id)
            payload = _parse_json_object(record.result_json)
            if payload is not None:
                _collect_material_ids(payload, material_ids)
        return evidence_ids, material_ids


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _collect_material_ids(payload: Any, output: set[str]) -> None:
    """Recursively collect material ids from tool evidence JSON."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"material_id", "material_ids"}:
                if isinstance(value, str):
                    output.add(value)
                elif isinstance(value, list):
                    output.update(item for item in value if isinstance(item, str))
            else:
                _collect_material_ids(value, output)
    elif isinstance(payload, list):
        for value in payload:
            _collect_material_ids(value, output)
