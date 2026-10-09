"""MA-0 contract tests for the unified multi-agent application boundary."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from materials_screening.master import (
    MultiAgentArtifact,
    SubAgentUiRegistry,
    SubAgentUiSpec,
    UnifiedConversationContext,
    UnifiedResultEnvelope,
    UnifiedResultReference,
)


def _artifact(artifact_id: str = "artifact-pdf-1") -> MultiAgentArtifact:
    return MultiAgentArtifact(
        artifact_id=artifact_id,
        owner_agent="literature",
        artifact_type="pdf_document",
        domain_id="doc-1234",
        display_name="paper.pdf",
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
    )


def _ui_spec(name: str = "literature") -> SubAgentUiSpec:
    return SubAgentUiSpec(
        name=name,
        display_name="Literature Agent",
        description="Search and analyze literature.",
        accepted_artifact_types=("pdf_document",),
        quick_prompts=("分析这篇论文",),
        renderer_name="literature_report",
    )


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            MultiAgentArtifact,
            {
                "artifact_id": "artifact-1",
                "owner_agent": "literature",
                "artifact_type": "pdf_document",
                "domain_id": "doc-1",
                "display_name": "paper.pdf",
                "unexpected": True,
            },
        ),
        (
            UnifiedConversationContext,
            {"conversation_id": "conversation-1", "unexpected": True},
        ),
        (
            UnifiedResultEnvelope,
            {
                "agent_name": "literature",
                "status": "completed",
                "result_type": "literature_report",
                "unexpected": True,
            },
        ),
        (
            SubAgentUiSpec,
            {
                "name": "literature",
                "display_name": "Literature",
                "description": "Literature agent.",
                "renderer_name": "literature_report",
                "unexpected": True,
            },
        ),
    ],
)
def test_contracts_reject_extra_fields(
    model: type[object], payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model(**payload)  # type: ignore[call-arg]


def test_artifact_round_trips_without_domain_assumptions() -> None:
    artifact = _artifact()
    restored = MultiAgentArtifact.model_validate_json(artifact.model_dump_json())

    assert restored == artifact
    assert restored.owner_agent == "literature"
    assert restored.metadata == {}


def test_conversation_rejects_duplicate_artifacts() -> None:
    with pytest.raises(ValidationError, match="unique artifact_id"):
        UnifiedConversationContext(
            conversation_id="conversation-1",
            artifacts=(_artifact(), _artifact()),
        )


def test_conversation_last_result_key_matches_agent() -> None:
    reference = UnifiedResultReference(
        agent_name="literature",
        result_type="literature_report",
        result_id="user-lit-1",
    )
    context = UnifiedConversationContext(
        conversation_id="conversation-1",
        last_results={"literature": reference},
    )
    assert context.last_results["literature"] == reference

    with pytest.raises(ValidationError, match="must match"):
        UnifiedConversationContext(
            conversation_id="conversation-1",
            last_results={"materials_database": reference},
        )


def test_result_envelope_enforces_clarification_shape() -> None:
    envelope = UnifiedResultEnvelope(
        agent_name="literature",
        status="needs_clarification",
        result_type="clarification",
        follow_up_question="请选择需要综合的论文。",
    )
    assert envelope.follow_up_question

    with pytest.raises(ValidationError, match="requires follow_up_question"):
        UnifiedResultEnvelope(
            agent_name="literature",
            status="needs_clarification",
            result_type="clarification",
        )


def test_result_envelope_rejects_duplicate_references() -> None:
    with pytest.raises(ValidationError, match="artifact_refs must be unique"):
        UnifiedResultEnvelope(
            agent_name="literature",
            status="completed",
            result_type="paper_dossier",
            artifact_refs=("artifact-1", "artifact-1"),
        )


def test_ui_spec_is_bounded_and_strict() -> None:
    assert _ui_spec().accepted_artifact_types == ("pdf_document",)
    with pytest.raises(ValidationError, match="must be unique"):
        SubAgentUiSpec(
            name="literature",
            display_name="Literature Agent",
            description="Search and analyze literature.",
            accepted_artifact_types=("pdf_document", "pdf_document"),
            renderer_name="literature_report",
        )


def test_ui_registry_is_sorted_and_duplicate_safe() -> None:
    registry = SubAgentUiRegistry(
        (_ui_spec("literature"), _ui_spec("materials_database"))
    )
    assert registry.names() == ("literature", "materials_database")
    assert registry.get("literature").renderer_name == "literature_report"
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(_ui_spec("literature"))
    with pytest.raises(KeyError, match="unknown"):
        registry.get("missing")
