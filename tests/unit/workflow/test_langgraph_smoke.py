"""Minimal LangGraph API smoke tests (S3-M1, business-isolated).

Verifies the installed LangGraph API surface (START -> node -> END, Runtime
context, InMemorySaver, thread_id, invoke, stream updates v2, get_state,
get_state_history) without touching Intern, Materials Project or any
stage 1/2 service. Each test builds a fresh graph with a fresh saver.
"""

from typing import TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402
from langgraph.runtime import Runtime  # noqa: E402


class SmokeState(TypedDict, total=False):
    """Minimal TypedDict state used only by this smoke test."""

    value: int
    doubled: int


class SmokeContext(TypedDict):
    """Runtime context injected per invocation."""

    multiplier: int


def _double_node(
    state: SmokeState,
    runtime: Runtime[SmokeContext],
) -> dict[str, int]:
    """Multiply the input value using the runtime context."""
    return {"doubled": state["value"] * runtime.context["multiplier"]}


def _build_graph() -> CompiledStateGraph[
    SmokeState, SmokeContext, SmokeState, SmokeState
]:
    builder = StateGraph(
        SmokeState,
        context_schema=SmokeContext,
    )
    builder.add_node("double", _double_node)
    builder.add_edge(START, "double")
    builder.add_edge("double", END)
    return builder.compile(
        checkpointer=InMemorySaver(),
        name="langgraph-smoke-v1",
    )


def _thread_config(thread_id: str) -> dict[str, object]:
    return {"configurable": {"thread_id": thread_id}}


class TestSmokeInvoke:
    def test_invoke_with_context_and_thread_id(self) -> None:
        graph = _build_graph()
        result = graph.invoke(
            {"value": 21},
            config=_thread_config("invoke-thread"),
            context={"multiplier": 2},
        )
        assert result == {"value": 21, "doubled": 42}

    def test_context_is_scoped_per_thread(self) -> None:
        graph = _build_graph()
        first = graph.invoke(
            {"value": 5},
            config=_thread_config("context-1"),
            context={"multiplier": 3},
        )
        second = graph.invoke(
            {"value": 5},
            config=_thread_config("context-2"),
            context={"multiplier": 10},
        )
        assert first == {"value": 5, "doubled": 15}
        assert second == {"value": 5, "doubled": 50}


class TestSmokeStream:
    def test_stream_updates_v2(self) -> None:
        graph = _build_graph()
        chunks = list(
            graph.stream(
                {"value": 7},
                config=_thread_config("stream-thread"),
                context={"multiplier": 2},
                stream_mode="updates",
                version="v2",
            )
        )
        updates = [chunk for chunk in chunks if chunk["type"] == "updates"]
        assert updates == [
            {
                "type": "updates",
                "ns": (),
                "data": {"double": {"doubled": 14}},
            }
        ]


class TestSmokeState:
    def test_get_state_after_invoke(self) -> None:
        graph = _build_graph()
        config = _thread_config("state-thread")
        graph.invoke(
            {"value": 4},
            config=config,
            context={"multiplier": 3},
        )
        snapshot = graph.get_state(config)
        assert snapshot.values == {"value": 4, "doubled": 12}
        assert snapshot.next == ()
        assert "checkpoint_id" in snapshot.config["configurable"]
        assert "multiplier" not in snapshot.values

    def test_get_state_history_returns_checkpoints(self) -> None:
        graph = _build_graph()
        config = _thread_config("history-thread")
        graph.invoke(
            {"value": 9},
            config=config,
            context={"multiplier": 2},
        )
        history = list(graph.get_state_history(config, limit=10))
        assert len(history) >= 1
        latest = history[0]
        assert latest.values == {"value": 9, "doubled": 18}
        assert "checkpoint_id" in latest.config["configurable"]

    def test_fresh_saver_starts_empty(self) -> None:
        first = _build_graph()
        first.invoke(
            {"value": 1},
            config=_thread_config("shared-thread"),
            context={"multiplier": 2},
        )
        second = _build_graph()
        snapshot = second.get_state(_thread_config("shared-thread"))
        assert snapshot.values == {}
