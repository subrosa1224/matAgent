"""Unit tests for the checkpointer factory (S3-M5)."""

from pathlib import Path
from typing import TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from materials_screening.workflow.checkpointer import (  # noqa: E402
    CheckpointerHandle,
    create_checkpointer_handle,
)
from materials_screening.workflow.errors import (  # noqa: E402
    WorkflowCheckpointError,
)
from materials_screening.workflow.settings import WorkflowSettings  # noqa: E402


class _State(TypedDict, total=False):
    value: int


def _node(state: _State) -> dict[str, int]:
    return {"value": (state.get("value", 0) or 0) + 1}


def _minimal_graph(checkpointer: object) -> object:
    builder = StateGraph(_State)
    builder.add_node("node", _node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    return builder.compile(checkpointer=checkpointer, name="lifecycle-v1")


class TestCheckpointerBackends:
    def test_memory_backend(self) -> None:
        handle = create_checkpointer_handle(backend="memory")
        assert isinstance(handle.checkpointer, InMemorySaver)
        handle.close()

    def test_memory_handle_as_context_manager(self) -> None:
        with create_checkpointer_handle(backend="memory") as handle:
            assert isinstance(handle, CheckpointerHandle)
            assert isinstance(handle.checkpointer, InMemorySaver)

    def test_sqlite_creates_db_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.sqlite"
        handle = create_checkpointer_handle(
            backend="sqlite",
            db_path=db_path,
            allowed_root=tmp_path,
        )
        assert isinstance(handle.checkpointer, SqliteSaver)
        assert db_path.exists()
        handle.close()

    def test_each_handle_owns_distinct_saver(self, tmp_path: Path) -> None:
        first = create_checkpointer_handle(
            backend="sqlite",
            db_path=tmp_path / "a.sqlite",
            allowed_root=tmp_path,
        )
        second = create_checkpointer_handle(
            backend="sqlite",
            db_path=tmp_path / "b.sqlite",
            allowed_root=tmp_path,
        )
        assert first.checkpointer is not second.checkpointer
        assert (tmp_path / "a.sqlite").exists()
        assert (tmp_path / "b.sqlite").exists()
        first.close()
        second.close()

    def test_unknown_backend_rejected(self) -> None:
        with pytest.raises(WorkflowCheckpointError):
            create_checkpointer_handle(backend="postgres")


class TestConnectionLifecycle:
    def test_connection_stays_open_during_invoke_stream_history(
        self, tmp_path: Path
    ) -> None:
        handle = create_checkpointer_handle(
            backend="sqlite",
            db_path=tmp_path / "checkpoints.sqlite",
            allowed_root=tmp_path,
        )
        graph = _minimal_graph(handle.checkpointer)
        config = {"configurable": {"thread_id": "thread-1"}}
        assert graph.invoke({"value": 0}, config=config) == {"value": 1}

        snapshot = graph.get_state(config)
        assert snapshot.values == {"value": 1}

        history = list(graph.get_state_history(config, limit=10))
        assert history

        streamed = list(
            graph.stream(
                {"value": 0},
                config={"configurable": {"thread_id": "thread-2"}},
                stream_mode="updates",
                version="v2",
            )
        )
        assert any(chunk["type"] == "updates" for chunk in streamed)
        handle.close()

    def test_connection_never_enters_state(self, tmp_path: Path) -> None:
        handle = create_checkpointer_handle(
            backend="sqlite",
            db_path=tmp_path / "checkpoints.sqlite",
            allowed_root=tmp_path,
        )
        graph = _minimal_graph(handle.checkpointer)
        config = {"configurable": {"thread_id": "thread-1"}}
        graph.invoke({"value": 0}, config=config)
        snapshot = graph.get_state(config)
        assert "connection" not in snapshot.values
        assert "checkpointer" not in snapshot.values
        assert "saver" not in snapshot.values
        handle.close()

    def test_close_marks_handle_closed_and_is_idempotent(self) -> None:
        handle = create_checkpointer_handle(backend="memory")
        handle.close()
        handle.close()
        with pytest.raises(WorkflowCheckpointError):
            _ = handle.checkpointer

    def test_sqlite_close_blocks_further_use(self, tmp_path: Path) -> None:
        handle = create_checkpointer_handle(
            backend="sqlite",
            db_path=tmp_path / "checkpoints.sqlite",
            allowed_root=tmp_path,
        )
        handle.close()
        with pytest.raises(WorkflowCheckpointError):
            _ = handle.checkpointer


class TestPathValidation:
    def test_path_outside_allowed_root_rejected(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside.sqlite"
        with pytest.raises(WorkflowCheckpointError):
            create_checkpointer_handle(
                backend="sqlite",
                db_path=outside,
                allowed_root=tmp_path,
            )

    def test_default_root_is_project_data(self, tmp_path: Path) -> None:
        with pytest.raises(WorkflowCheckpointError):
            create_checkpointer_handle(
                backend="sqlite",
                db_path=tmp_path / "checkpoints.sqlite",
            )

    def test_settings_path_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        settings = WorkflowSettings(
            _env_file=None,
            workflow_checkpoint_db=Path("data/custom.sqlite"),
        )
        handle = create_checkpointer_handle(settings=settings)
        assert (tmp_path / "data" / "custom.sqlite").exists()
        handle.close()

    def test_db_path_conflicts_with_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        settings = WorkflowSettings(
            _env_file=None,
            workflow_checkpoint_db=Path("data/a.sqlite"),
        )
        with pytest.raises(WorkflowCheckpointError):
            create_checkpointer_handle(
                settings=settings,
                db_path=Path("data/b.sqlite"),
            )
