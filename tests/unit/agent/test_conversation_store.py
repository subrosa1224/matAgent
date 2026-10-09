"""Unit tests for the SQLite conversation store (S3.5-M6)."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from materials_screening.agent.conversation_store import (
    ConversationMetadata,
    ConversationStore,
    SqliteConversationStore,
)
from materials_screening.agent.errors import AgentConversationError


def _tick_clock(start: datetime | None = None) -> Callable[[], datetime]:
    current = start or datetime(2026, 8, 6, 0, 0, tzinfo=UTC)

    def clock() -> datetime:
        nonlocal current
        value = current
        current = current.replace(second=current.second + 1)
        return value

    return clock


def _store(
    db_path: Path,
    **overrides: Any,
) -> SqliteConversationStore:
    values: dict[str, Any] = {"db_path": db_path, "clock": _tick_clock()}
    values.update(overrides)
    return SqliteConversationStore(**values)


class TestCreateAndRead:
    def test_create_and_get_metadata(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "agent_conversations.sqlite")
        store.create("conv_1")

        metadata = store.get("conv_1")

        assert metadata.conversation_id == "conv_1"
        assert metadata.turn_count == 0
        assert metadata.version == 0
        assert metadata.active_workflow_thread_id is None
        assert metadata.created_at == metadata.updated_at
        store.close()

    def test_create_duplicate_rejected(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        store.create("conv_1")

        with pytest.raises(AgentConversationError, match="already exists") as exc:
            store.create("conv_1")

        assert exc.value.code == "CONVERSATION_EXISTS"
        store.close()

    def test_get_missing_rejected(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")

        with pytest.raises(AgentConversationError, match="not found") as exc:
            store.get("conv_missing")

        assert exc.value.code == "CONVERSATION_NOT_FOUND"
        store.close()

    def test_exists(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        store.create("conv_1")

        assert store.exists("conv_1") is True
        assert store.exists("conv_2") is False
        store.close()

    def test_metadata_is_safe_frozen_model(self) -> None:
        metadata = ConversationMetadata(
            conversation_id="c1",
            created_at="2026-08-06T00:00:00+00:00",
            updated_at="2026-08-06T00:00:00+00:00",
        )
        assert metadata.model_config.get("frozen") is True
        assert metadata.model_config.get("extra") == "forbid"


class TestTurnCount:
    def test_record_turn_increments(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        store.create("conv_1")

        first = store.record_turn("conv_1")
        second = store.record_turn("conv_1")

        assert first.turn_count == 1
        assert first.version == 1
        assert second.turn_count == 2
        assert second.version == 2
        store.close()

    def test_record_turn_with_correct_expected_version(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        store.create("conv_1")

        metadata = store.record_turn("conv_1", expected_version=0)

        assert metadata.version == 1
        store.close()

    def test_record_turn_with_stale_version_conflicts(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        store.create("conv_1")
        store.record_turn("conv_1")

        with pytest.raises(
            AgentConversationError, match="does not match expected"
        ) as exc:
            store.record_turn("conv_1", expected_version=0)

        assert exc.value.code == "CONVERSATION_VERSION_CONFLICT"
        store.close()

    def test_record_turn_missing_conversation(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")

        with pytest.raises(AgentConversationError, match="not found") as exc:
            store.record_turn("conv_missing")

        assert exc.value.code == "CONVERSATION_NOT_FOUND"
        store.close()

    def test_record_turn_missing_with_expected_version(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")

        with pytest.raises(AgentConversationError, match="not found") as exc:
            store.record_turn("conv_missing", expected_version=0)

        assert exc.value.code == "CONVERSATION_NOT_FOUND"
        store.close()


class TestWorkflowLinks:
    def test_link_and_list(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        store.create("conv_1")

        store.link_workflow("conv_1", "thread_1")

        assert store.list_workflows("conv_1") == ("thread_1",)
        assert store.owns_workflow("conv_1", "thread_1") is True
        assert store.owns_workflow("conv_1", "thread_other") is False
        store.close()

    def test_link_is_idempotent(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        store.create("conv_1")

        store.link_workflow("conv_1", "thread_1")
        store.link_workflow("conv_1", "thread_1")

        assert store.list_workflows("conv_1") == ("thread_1",)
        store.close()

    def test_cross_conversation_ownership_blocked(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        store.create("conv_a")
        store.create("conv_b")
        store.link_workflow("conv_a", "thread_1")

        assert store.owns_workflow("conv_a", "thread_1") is True
        assert store.owns_workflow("conv_b", "thread_1") is False
        assert store.list_workflows("conv_b") == ()
        store.close()

    def test_list_workflows_ordered_by_created_at(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        store.create("conv_1")

        store.link_workflow("conv_1", "thread_2")
        store.link_workflow("conv_1", "thread_1")

        assert store.list_workflows("conv_1") == ("thread_2", "thread_1")
        store.close()

    def test_list_links_returns_ownership_records(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        store.create("conv_1")
        store.link_workflow("conv_1", "thread_1")
        store.create("conv_2")
        store.link_workflow("conv_2", "thread_other")

        links = store.list_links("conv_1")

        assert len(links) == 1
        assert links[0].conversation_id == "conv_1"
        assert links[0].thread_id == "thread_1"
        assert links[0].created_at
        store.close()

    def test_link_missing_conversation(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")

        with pytest.raises(AgentConversationError, match="not found") as exc:
            store.link_workflow("conv_missing", "thread_1")

        assert exc.value.code == "CONVERSATION_NOT_FOUND"
        store.close()


class TestActiveThread:
    def test_set_and_read_active_thread(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        store.create("conv_1")
        store.link_workflow("conv_1", "thread_1")

        store.set_active_thread("conv_1", "thread_1")

        metadata = store.get("conv_1")
        assert metadata.active_workflow_thread_id == "thread_1"
        assert metadata.version == 2
        store.close()

    def test_set_active_unowned_thread_rejected(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        store.create("conv_1")

        with pytest.raises(AgentConversationError, match="not owned") as exc:
            store.set_active_thread("conv_1", "thread_other")

        assert exc.value.code == "OWNERSHIP_DENIED"
        store.close()

    def test_cross_conversation_active_blocked(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        store.create("conv_a")
        store.create("conv_b")
        store.link_workflow("conv_a", "thread_1")

        with pytest.raises(AgentConversationError, match="not owned") as exc:
            store.set_active_thread("conv_b", "thread_1")

        assert exc.value.code == "OWNERSHIP_DENIED"
        store.close()


class TestPersistence:
    def test_reopen_recovers_state(self, tmp_path: Path) -> None:
        db_path = tmp_path / "agent_conversations.sqlite"
        store = _store(db_path)
        store.create("conv_1")
        store.link_workflow("conv_1", "thread_1")
        store.set_active_thread("conv_1", "thread_1")
        store.record_turn("conv_1")
        store.close()

        reopened = _store(db_path)

        metadata = reopened.get("conv_1")
        assert metadata.turn_count == 1
        assert metadata.active_workflow_thread_id == "thread_1"
        assert reopened.owns_workflow("conv_1", "thread_1") is True
        assert reopened.list_workflows("conv_1") == ("thread_1",)
        reopened.close()

    def test_closed_store_raises(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        store.create("conv_1")
        store.close()

        with pytest.raises(AgentConversationError, match="closed") as exc:
            store.get("conv_1")

        assert exc.value.code == "STORE_CLOSED"

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        with _store(tmp_path / "db.sqlite") as store:
            store.create("conv_1")

        with pytest.raises(AgentConversationError, match="closed"):
            store.get("conv_1")

    def test_deterministic_clock(self, tmp_path: Path) -> None:
        clock = _tick_clock()
        store = SqliteConversationStore(tmp_path / "db.sqlite", clock=clock)
        store.create("conv_1")

        metadata = store.get("conv_1")

        assert metadata.created_at == "2026-08-06T00:00:00+00:00"
        store.close()


class TestConcurrency:
    def test_conversation_lock_rejects_second_holder(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        lock = store.conversation_lock("conv_1")

        assert lock.acquire(blocking=False) is True
        assert lock.acquire(blocking=False) is False
        lock.release()
        assert lock.acquire(blocking=False) is True
        lock.release()
        store.close()

    def test_conversation_lock_is_shared_per_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")

        assert store.conversation_lock("conv_1") is store.conversation_lock("conv_1")
        assert store.conversation_lock("conv_1") is not store.conversation_lock(
            "conv_2"
        )
        store.close()

    def test_conversation_lock_context_manager(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        lock = store.conversation_lock("conv_1")

        with lock:
            assert lock.acquire(blocking=False) is False

        assert lock.acquire(blocking=False) is True
        lock.release()
        store.close()

    def test_protocol_conformance(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")

        assert isinstance(store, ConversationStore)
        store.close()


class TestSafety:
    def test_no_delete_api(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        names = {name for name in dir(store) if not name.startswith("_")}

        assert not any(
            name.startswith(("delete", "remove", "drop", "clear")) for name in names
        )
        store.close()

    def test_schema_has_no_content_columns(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")
        conversation_columns = {
            row[1]
            for row in store._connection.execute(
                "PRAGMA table_info(conversations)"
            ).fetchall()
        }
        link_columns = {
            row[1]
            for row in store._connection.execute(
                "PRAGMA table_info(workflow_links)"
            ).fetchall()
        }

        assert conversation_columns == {
            "conversation_id",
            "created_at",
            "updated_at",
            "turn_count",
            "active_workflow_thread_id",
            "version",
        }
        assert link_columns == {
            "conversation_id",
            "thread_id",
            "created_at",
        }
        forbidden = {"content", "reasoning", "api_key", "response", "result"}
        assert conversation_columns.isdisjoint(forbidden)
        assert link_columns.isdisjoint(forbidden)
        store.close()

    def test_invalid_ids_rejected(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "db.sqlite")

        with pytest.raises(AgentConversationError, match="unsafe format") as exc:
            store.create("bad id!")
        assert exc.value.code == "INVALID_ID"

        store.create("conv_1")
        with pytest.raises(AgentConversationError, match="unsafe format"):
            store.link_workflow("conv_1", "bad thread!")
        store.close()
