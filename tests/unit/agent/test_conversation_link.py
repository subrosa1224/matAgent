"""Unit tests for conversation/workflow ownership links (S3.5-M2)."""

import pytest
from pydantic import ValidationError

from materials_screening.agent.policy import ConversationWorkflowLink


class TestConversationWorkflowLink:
    def test_valid(self) -> None:
        link = ConversationWorkflowLink(
            conversation_id="c1",
            thread_id="thread-1",
            created_at="2026-08-06T00:00:00Z",
        )
        assert link.conversation_id == "c1"
        assert link.thread_id == "thread-1"

    def test_frozen_and_extra_rejected(self) -> None:
        assert ConversationWorkflowLink.model_config.get("frozen") is True
        with pytest.raises(ValidationError):
            ConversationWorkflowLink(
                conversation_id="c1",
                thread_id="thread-1",
                created_at="now",
                junk=1,
            )

    @pytest.mark.parametrize(
        "field",
        ["conversation_id", "thread_id", "created_at"],
    )
    def test_empty_fields_rejected(self, field: str) -> None:
        values = {
            "conversation_id": "c1",
            "thread_id": "thread-1",
            "created_at": "now",
        }
        values[field] = ""
        with pytest.raises(ValidationError):
            ConversationWorkflowLink.model_validate(values)
