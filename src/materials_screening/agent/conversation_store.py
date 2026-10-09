"""SQLite-backed conversation store for the single agent (S3.5-M6).

Stores only safe session metadata: conversation ids, timestamps, turn counts,
workflow thread links and an optimistic concurrency version. API keys,
reasoning, raw model responses and material results are never accepted or
persisted. There is intentionally no delete/remove API.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.agent.errors import AgentConversationError
from materials_screening.agent.policy import ConversationWorkflowLink

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,253}$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    turn_count INTEGER NOT NULL DEFAULT 0,
    active_workflow_thread_id TEXT,
    version INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS workflow_links (
    conversation_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (conversation_id, thread_id),
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);
"""


class ConversationMetadata(BaseModel):
    """Safe conversation metadata; never keys, reasoning or results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    created_at: str
    updated_at: str
    turn_count: int = Field(default=0, ge=0)
    active_workflow_thread_id: str | None = None
    version: int = Field(default=0, ge=0)


class ConversationLock:
    """Per-conversation re-entrant-free lock used to serialize asks.

    Supports non-blocking ``acquire`` so the runner can return
    ``CONVERSATION_BUSY`` instead of blocking on a concurrent ask.
    """

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """Acquire the lock; returns False when already held non-blocking."""
        return self._lock.acquire(blocking=blocking, timeout=timeout)

    def release(self) -> None:
        """Release the lock."""
        self._lock.release()

    def __enter__(self) -> None:
        self.acquire()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()


@runtime_checkable
class ConversationStore(Protocol):
    """Persistence contract for agent conversations (S3.5-M6)."""

    def create(self, conversation_id: str) -> None: ...

    def exists(self, conversation_id: str) -> bool: ...

    def get(self, conversation_id: str) -> ConversationMetadata: ...

    def record_turn(
        self,
        conversation_id: str,
        *,
        expected_version: int | None = None,
    ) -> ConversationMetadata: ...

    def link_workflow(
        self,
        conversation_id: str,
        thread_id: str,
        *,
        expected_version: int | None = None,
    ) -> None: ...

    def set_active_thread(
        self,
        conversation_id: str,
        thread_id: str,
        *,
        expected_version: int | None = None,
    ) -> None: ...

    def owns_workflow(self, conversation_id: str, thread_id: str) -> bool: ...

    def list_workflows(self, conversation_id: str) -> tuple[str, ...]: ...

    def list_links(
        self,
        conversation_id: str,
    ) -> tuple[ConversationWorkflowLink, ...]: ...

    def conversation_lock(self, conversation_id: str) -> ConversationLock: ...

    def close(self) -> None: ...


class SqliteConversationStore:
    """SQLite implementation; safe to reopen for recovery and multi-threaded."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._conversation_locks: dict[str, ConversationLock] = {}
        self._closed = False
        self._connection = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        with self._lock:
            self._connection.executescript(_SCHEMA)

    def create(self, conversation_id: str) -> None:
        """Create a conversation; raises when the id already exists."""
        self._ensure_open()
        self._validate_id(conversation_id, "conversation_id")
        now = self._clock().isoformat()
        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO conversations "
                    "(conversation_id, created_at, updated_at, turn_count, version) "
                    "VALUES (?, ?, ?, 0, 0)",
                    (conversation_id, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise AgentConversationError(
                    "CONVERSATION_EXISTS",
                    f"conversation already exists: {conversation_id!r}",
                ) from exc

    def exists(self, conversation_id: str) -> bool:
        self._ensure_open()
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return row is not None

    def get(self, conversation_id: str) -> ConversationMetadata:
        self._ensure_open()
        with self._lock:
            row = self._connection.execute(
                "SELECT conversation_id, created_at, updated_at, turn_count, "
                "active_workflow_thread_id, version FROM conversations "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise AgentConversationError(
                "CONVERSATION_NOT_FOUND",
                f"conversation {conversation_id!r} not found",
            )
        return ConversationMetadata(
            conversation_id=row["conversation_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            turn_count=row["turn_count"],
            active_workflow_thread_id=row["active_workflow_thread_id"],
            version=row["version"],
        )

    def record_turn(
        self,
        conversation_id: str,
        *,
        expected_version: int | None = None,
    ) -> ConversationMetadata:
        """Increment the turn counter with an optimistic version guard."""
        self._ensure_open()
        self._mutate(
            conversation_id,
            expected_version=expected_version,
            statement=(
                "UPDATE conversations SET turn_count = turn_count + 1, "
                "updated_at = ?, version = version + 1 "
                "WHERE conversation_id = ?"
            ),
            params=(self._clock().isoformat(), conversation_id),
        )
        return self.get(conversation_id)

    def link_workflow(
        self,
        conversation_id: str,
        thread_id: str,
        *,
        expected_version: int | None = None,
    ) -> None:
        """Register conversation -> thread ownership; idempotent per thread."""
        self._ensure_open()
        self._validate_id(conversation_id, "conversation_id")
        self._validate_id(thread_id, "thread_id")
        now = self._clock().isoformat()
        with self._lock:
            self._mutate_locked(
                conversation_id,
                expected_version=expected_version,
                statement=(
                    "UPDATE conversations SET updated_at = ?, "
                    "version = version + 1 WHERE conversation_id = ?"
                ),
                params=(now, conversation_id),
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO workflow_links "
                "(conversation_id, thread_id, created_at) VALUES (?, ?, ?)",
                (conversation_id, thread_id, now),
            )

    def set_active_thread(
        self,
        conversation_id: str,
        thread_id: str,
        *,
        expected_version: int | None = None,
    ) -> None:
        """Set the active thread; the thread must be owned by the conversation."""
        self._ensure_open()
        self._validate_id(conversation_id, "conversation_id")
        self._validate_id(thread_id, "thread_id")
        if not self.owns_workflow(conversation_id, thread_id):
            raise AgentConversationError(
                "OWNERSHIP_DENIED",
                f"thread {thread_id!r} is not owned by conversation "
                f"{conversation_id!r}",
            )
        self._mutate(
            conversation_id,
            expected_version=expected_version,
            statement=(
                "UPDATE conversations SET active_workflow_thread_id = ?, "
                "updated_at = ?, version = version + 1 "
                "WHERE conversation_id = ?"
            ),
            params=(thread_id, self._clock().isoformat(), conversation_id),
        )

    def owns_workflow(self, conversation_id: str, thread_id: str) -> bool:
        self._ensure_open()
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM workflow_links "
                "WHERE conversation_id = ? AND thread_id = ?",
                (conversation_id, thread_id),
            ).fetchone()
        return row is not None

    def list_workflows(self, conversation_id: str) -> tuple[str, ...]:
        self._ensure_open()
        with self._lock:
            rows = self._connection.execute(
                "SELECT thread_id FROM workflow_links "
                "WHERE conversation_id = ? ORDER BY created_at, thread_id",
                (conversation_id,),
            ).fetchall()
        return tuple(str(row["thread_id"]) for row in rows)

    def list_links(
        self,
        conversation_id: str,
    ) -> tuple[ConversationWorkflowLink, ...]:
        """Return the workflow ownership links of one conversation, in order."""
        self._ensure_open()
        with self._lock:
            rows = self._connection.execute(
                "SELECT conversation_id, thread_id, created_at FROM workflow_links "
                "WHERE conversation_id = ? ORDER BY created_at, thread_id",
                (conversation_id,),
            ).fetchall()
        return tuple(
            ConversationWorkflowLink(
                conversation_id=str(row["conversation_id"]),
                thread_id=str(row["thread_id"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        )

    def conversation_lock(self, conversation_id: str) -> ConversationLock:
        """Return the per-conversation lock, creating it on first use."""
        self._ensure_open()
        with self._lock:
            lock = self._conversation_locks.get(conversation_id)
            if lock is None:
                lock = ConversationLock(threading.Lock())
                self._conversation_locks[conversation_id] = lock
            return lock

    def close(self) -> None:
        """Close the underlying connection once; further closes are no-ops."""
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _mutate(
        self,
        conversation_id: str,
        *,
        expected_version: int | None,
        statement: str,
        params: tuple[object, ...],
    ) -> None:
        with self._lock:
            self._mutate_locked(
                conversation_id,
                expected_version=expected_version,
                statement=statement,
                params=params,
            )

    def _mutate_locked(
        self,
        conversation_id: str,
        *,
        expected_version: int | None,
        statement: str,
        params: tuple[object, ...],
    ) -> None:
        if expected_version is not None:
            cursor = self._connection.execute(
                statement + " AND version = ?",
                (*params, expected_version),
            )
            if cursor.rowcount == 0:
                row = self._connection.execute(
                    "SELECT version FROM conversations WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                if row is None:
                    raise AgentConversationError(
                        "CONVERSATION_NOT_FOUND",
                        f"conversation {conversation_id!r} not found",
                    )
                raise AgentConversationError(
                    "CONVERSATION_VERSION_CONFLICT",
                    f"conversation {conversation_id!r} version "
                    f"{row['version']} does not match expected "
                    f"{expected_version}",
                )
            return
        cursor = self._connection.execute(statement, params)
        if cursor.rowcount == 0:
            raise AgentConversationError(
                "CONVERSATION_NOT_FOUND",
                f"conversation {conversation_id!r} not found",
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise AgentConversationError(
                "STORE_CLOSED",
                "conversation store is closed",
            )

    @staticmethod
    def _validate_id(value: str, name: str) -> None:
        if not _ID_PATTERN.fullmatch(value):
            raise AgentConversationError(
                "INVALID_ID",
                f"{name} has an unsafe format: {value!r}",
            )
