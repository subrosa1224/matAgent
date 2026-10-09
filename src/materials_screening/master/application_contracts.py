"""Strict application contracts for the unified multi-agent surface.

These models belong to the application boundary. They deliberately do not
modify ``SubAgentSpec`` or any domain-agent result model.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SAFE_NAME = r"^[a-z][a-z0-9_]{0,127}$"
_SAFE_ID = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$"


class MultiAgentArtifact(BaseModel):
    """One application-level reference to a domain-owned artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(pattern=_SAFE_ID)
    owner_agent: str = Field(pattern=_SAFE_NAME)
    artifact_type: str = Field(pattern=_SAFE_NAME)
    domain_id: str = Field(pattern=_SAFE_ID)
    display_name: str = Field(min_length=1, max_length=512)
    status: Literal["pending", "ready", "failed"] = "ready"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class UnifiedResultReference(BaseModel):
    """Small stable pointer retained in conversation state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_name: str = Field(pattern=_SAFE_NAME)
    result_type: str = Field(pattern=_SAFE_NAME)
    result_id: str = Field(pattern=_SAFE_ID)


class UnifiedConversationContext(BaseModel):
    """Serializable application context shared by the unified UI and master."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str = Field(pattern=_SAFE_ID)
    active_agent: str | None = Field(default=None, pattern=_SAFE_NAME)
    selected_mode: str = Field(default="auto", pattern=_SAFE_NAME)
    artifacts: tuple[MultiAgentArtifact, ...] = ()
    last_results: dict[str, UnifiedResultReference] = Field(default_factory=dict)
    pending_clarification: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def validate_unique_artifacts(self) -> Self:
        artifact_ids = [item.artifact_id for item in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError(
                "conversation artifacts must have unique artifact_id values"
            )
        for key, reference in self.last_results.items():
            if not re.fullmatch(_SAFE_NAME, key):
                raise ValueError(f"invalid last_results key: {key!r}")
            if key != reference.agent_name:
                raise ValueError("last_results key must match reference.agent_name")
        return self


class UnifiedResultEnvelope(BaseModel):
    """Validated result boundary between orchestration and presentation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_name: str = Field(pattern=_SAFE_NAME)
    status: Literal["completed", "partial", "failed", "needs_clarification"]
    result_type: str = Field(pattern=_SAFE_NAME)
    result: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    follow_up_question: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def validate_terminal_shape(self) -> Self:
        if self.status == "needs_clarification" and not self.follow_up_question:
            raise ValueError("needs_clarification requires follow_up_question")
        if self.status != "needs_clarification" and self.follow_up_question:
            raise ValueError(
                "follow_up_question is only valid for needs_clarification"
            )
        if len(self.artifact_refs) != len(set(self.artifact_refs)):
            raise ValueError("artifact_refs must be unique")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("evidence_refs must be unique")
        if any(not re.fullmatch(_SAFE_ID, value) for value in self.artifact_refs):
            raise ValueError("artifact_refs contain an invalid id")
        if any(not re.fullmatch(_SAFE_ID, value) for value in self.evidence_refs):
            raise ValueError("evidence_refs contain an invalid id")
        return self


class SubAgentUiSpec(BaseModel):
    """Declarative UI metadata for one registered domain sub-agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=_SAFE_NAME)
    display_name: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=1000)
    accepted_artifact_types: tuple[str, ...] = ()
    quick_prompts: tuple[str, ...] = ()
    renderer_name: str = Field(pattern=_SAFE_NAME)

    @model_validator(mode="after")
    def validate_bounded_ui_metadata(self) -> Self:
        if len(self.accepted_artifact_types) > 32:
            raise ValueError("accepted_artifact_types may contain at most 32 values")
        if len(self.quick_prompts) > 12:
            raise ValueError("quick_prompts may contain at most 12 values")
        if len(self.accepted_artifact_types) != len(
            set(self.accepted_artifact_types)
        ):
            raise ValueError("accepted_artifact_types must be unique")
        for artifact_type in self.accepted_artifact_types:
            if not re.fullmatch(_SAFE_NAME, artifact_type):
                raise ValueError(f"invalid artifact type: {artifact_type!r}")
        if any(not prompt.strip() for prompt in self.quick_prompts):
            raise ValueError("quick_prompts must not contain blank prompts")
        return self


class SubAgentUiRegistry:
    """Static, duplicate-safe registry for optional sub-agent UI metadata."""

    def __init__(self, specs: tuple[SubAgentUiSpec, ...] = ()) -> None:
        self._specs: dict[str, SubAgentUiSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: SubAgentUiSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate sub-agent UI spec: {spec.name!r}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> SubAgentUiSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown sub-agent UI spec: {name!r}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def specs(self) -> tuple[SubAgentUiSpec, ...]:
        return tuple(self._specs[name] for name in self.names())
