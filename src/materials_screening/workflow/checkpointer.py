"""Checkpointer factory and lifecycle handle (S3-M5)."""

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from materials_screening.workflow.errors import WorkflowCheckpointError
from materials_screening.workflow.settings import WorkflowSettings

CheckpointerBackend = Literal["memory", "sqlite"]
_DEFAULT_DB_PATH = Path("data/workflow_checkpoints.sqlite")


def _noop_close() -> None:
    return None


class CheckpointerHandle:
    """Own a checkpointer and its connection; close is explicit and final."""

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver[Any],
        *,
        close: Callable[[], None],
    ) -> None:
        self._checkpointer = checkpointer
        self._close_fn = close
        self._closed = False

    @property
    def checkpointer(self) -> BaseCheckpointSaver[Any]:
        if self._closed:
            raise WorkflowCheckpointError(
                "checkpointer handle is closed; open a new one"
            )
        return self._checkpointer

    def close(self) -> None:
        """Close the underlying connection once; further closes are no-ops."""
        if not self._closed:
            self._close_fn()
            self._closed = True

    def __enter__(self) -> "CheckpointerHandle":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _resolve_db_path(
    *,
    db_path: Path | None,
    settings: WorkflowSettings | None,
    allowed_root: Path | None,
) -> Path:
    if settings is not None and settings.workflow_checkpointer_backend != "sqlite":
        raise WorkflowCheckpointError("settings checkpoint backend must be sqlite")
    if settings is not None:
        candidate = settings.workflow_checkpoint_db
        if (
            db_path is not None
            and db_path.expanduser().resolve() != candidate.expanduser().resolve()
        ):
            raise WorkflowCheckpointError(
                "db_path conflicts with settings workflow_checkpoint_db"
            )
    else:
        candidate = db_path if db_path is not None else _DEFAULT_DB_PATH
    root = (allowed_root if allowed_root is not None else Path("data")).resolve()
    resolved = candidate.expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise WorkflowCheckpointError(
            f"checkpoint db must be inside allowed root {root}"
        )
    return resolved


def create_checkpointer_handle(
    *,
    backend: CheckpointerBackend = "sqlite",
    db_path: Path | None = None,
    settings: WorkflowSettings | None = None,
    allowed_root: Path | None = None,
) -> CheckpointerHandle:
    """Create a handle owning its checkpointer; caller must close it.

    ``memory`` is for tests; ``sqlite`` is the local backend. Only these two
    backends exist (no PostgreSQL/Redis). The SQLite connection stays open
    until ``handle.close()``.
    """
    if backend == "memory":
        return CheckpointerHandle(InMemorySaver(), close=_noop_close)
    if backend == "sqlite":
        resolved = _resolve_db_path(
            db_path=db_path,
            settings=settings,
            allowed_root=allowed_root,
        )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(resolved), check_same_thread=False)
        saver = SqliteSaver(connection)
        return CheckpointerHandle(saver, close=connection.close)
    raise WorkflowCheckpointError(f"unsupported checkpointer backend: {backend!r}")
