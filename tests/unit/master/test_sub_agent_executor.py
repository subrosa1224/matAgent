"""Tests for the sub-agent executor."""

import json
from types import SimpleNamespace

from materials_screening.master import (
    SubAgentCall,
    SubAgentExecutor,
    SubAgentRegistry,
    SubAgentResultEnvelope,
    SubAgentSpec,
)


class _MockResult:
    """Duck-typed result matching AgentResult shape."""

    def __init__(
        self,
        response: str,
        evidence: list[str] | None = None,
        active_workflow_thread_id: str | None = None,
    ):
        self.response_text = response
        self.evidence_ids = tuple(evidence or [])
        self.warnings = ()
        self.active_workflow_thread_id = active_workflow_thread_id


def _make_runner_factory(response: str = "mock answer"):
    return lambda: _MockRunner(response)


class _MockRunner:
    def __init__(self, response: str = "mock answer"):
        self._response = response

    def ask(self, *, message: str, conversation_id: str | None = None):  # noqa: ARG002
        return _MockResult(self._response)


def _make_spec(
    name: str = "test_agent",
    response: str = "mock answer",
) -> SubAgentSpec:
    return SubAgentSpec(
        name=name,
        description="test sub-agent",
        system_prompt="test prompt",
        tool_definitions=(),
        runner_factory=_make_runner_factory(response),
    )


class TestSubAgentExecutor:
    def test_completed_turn_with_error_draft_is_a_failed_delegation(self) -> None:
        runner = _MockRunner()
        runner.ask = lambda **_kwargs: SimpleNamespace(
            status="completed",
            final_status="error",
            error=None,
            response_text="统计结果一致性校验失败：分组样本数与总行数不一致。",
            evidence_ids=("evidence-stats",),
            warnings=(),
        )
        spec = SubAgentSpec(
            name="data_analysis",
            description="analysis",
            system_prompt="analysis",
            tool_definitions=(),
            runner_factory=lambda: runner,
        )
        outcome = SubAgentExecutor(registry=SubAgentRegistry([spec])).execute(
            [
                SubAgentCall(
                    call_id="analysis",
                    name="delegate_to_data_analysis",
                    arguments_json='{"task":"describe"}',
                )
            ]
        )[0]
        assert outcome.status == "error"
        assert "统计结果一致性校验失败" in outcome.envelope.error["message"]
        assert outcome.envelope.evidence_ids == []

    def test_parent_conversation_creates_stable_sub_agent_conversation(self) -> None:
        received: list[str | None] = []
        runner = _MockRunner()
        runner.ask = lambda *, message, conversation_id=None: (
            received.append(conversation_id) or _MockResult(message)
        )
        spec = SubAgentSpec(
            name="materials_database",
            description="database",
            system_prompt="database prompt",
            tool_definitions=(),
            runner_factory=lambda: runner,
        )
        executor = SubAgentExecutor(registry=SubAgentRegistry([spec]))
        call = SubAgentCall(
            call_id="call-1",
            name="delegate_to_materials_database",
            arguments_json='{"task":"search"}',
        )

        executor.execute([call], parent_conversation_id="master-44d1")
        executor.execute([call], parent_conversation_id="master-44d1")

        assert received[0] is not None
        assert received == [received[0], received[0]]
        assert received[0].startswith("sub_materials_database_")

    def test_execute_single_ok(self) -> None:
        registry = SubAgentRegistry([_make_spec("agent_a", "hello world")])
        executor = SubAgentExecutor(registry=registry)
        calls = [
            SubAgentCall(
                call_id="call-1",
                name="delegate_to_agent_a",
                arguments_json=json.dumps({"task": "do something"}),
            )
        ]
        outcomes = executor.execute(calls)
        assert len(outcomes) == 1
        assert outcomes[0].status == "ok"
        assert outcomes[0].sub_agent_name == "agent_a"
        assert "hello world" in outcomes[0].output_json

    def test_preserves_active_workflow_thread(self) -> None:
        runner = _MockRunner()
        runner.ask = lambda **_kwargs: _MockResult(
            "screened", active_workflow_thread_id="workflow-1"
        )
        spec = SubAgentSpec(
            name="screening",
            description="screening agent",
            system_prompt="screening prompt",
            tool_definitions=(),
            runner_factory=lambda: runner,
        )
        executor = SubAgentExecutor(registry=SubAgentRegistry([spec]))

        outcome = executor.execute(
            [
                SubAgentCall(
                    call_id="call-1",
                    name="delegate_to_screening",
                    arguments_json='{"task":"screen"}',
                )
            ]
        )[0]

        assert outcome.envelope.active_workflow_thread_id == "workflow-1"

    def test_execute_unknown_delegate(self) -> None:
        registry = SubAgentRegistry([_make_spec("agent_a")])
        executor = SubAgentExecutor(registry=registry)
        calls = [
            SubAgentCall(
                call_id="call-1",
                name="unknown_function",
                arguments_json=json.dumps({"task": "x"}),
            )
        ]
        outcomes = executor.execute(calls)
        assert outcomes[0].status == "error"
        assert "UNKNOWN_SUB_AGENT" in outcomes[0].output_json

    def test_execute_missing_task(self) -> None:
        registry = SubAgentRegistry([_make_spec("agent_a")])
        executor = SubAgentExecutor(registry=registry)
        calls = [
            SubAgentCall(
                call_id="call-1",
                name="delegate_to_agent_a",
                arguments_json=json.dumps({}),
            )
        ]
        outcomes = executor.execute(calls)
        assert outcomes[0].status == "error"
        assert "MISSING_TASK" in outcomes[0].output_json

    def test_execute_empty_task(self) -> None:
        registry = SubAgentRegistry([_make_spec("agent_a")])
        executor = SubAgentExecutor(registry=registry)
        calls = [
            SubAgentCall(
                call_id="call-1",
                name="delegate_to_agent_a",
                arguments_json=json.dumps({"task": ""}),
            )
        ]
        outcomes = executor.execute(calls)
        assert outcomes[0].status == "error"
        assert "MISSING_TASK" in outcomes[0].output_json

    def test_execute_invalid_json(self) -> None:
        registry = SubAgentRegistry([_make_spec("agent_a")])
        executor = SubAgentExecutor(registry=registry)
        calls = [
            SubAgentCall(
                call_id="call-1",
                name="delegate_to_agent_a",
                arguments_json="not valid json",
            )
        ]
        outcomes = executor.execute(calls)
        assert outcomes[0].status == "error"
        assert "INVALID_ARGUMENTS" in outcomes[0].output_json

    def test_execute_non_object_json(self) -> None:
        registry = SubAgentRegistry([_make_spec("agent_a")])
        executor = SubAgentExecutor(registry=registry)
        calls = [
            SubAgentCall(
                call_id="call-1",
                name="delegate_to_agent_a",
                arguments_json=json.dumps(["list", "not", "object"]),
            )
        ]
        outcomes = executor.execute(calls)
        assert outcomes[0].status == "error"
        assert "INVALID_ARGUMENTS" in outcomes[0].output_json

    def test_execute_runner_creation_fails(self) -> None:
        def bad_factory():
            raise ValueError("cannot create runner")

        spec = SubAgentSpec(
            name="bad_agent",
            description="desc",
            system_prompt="prompt",
            tool_definitions=(),
            runner_factory=bad_factory,
        )
        registry = SubAgentRegistry([spec])
        executor = SubAgentExecutor(registry=registry)
        calls = [
            SubAgentCall(
                call_id="call-1",
                name="delegate_to_bad_agent",
                arguments_json=json.dumps({"task": "x"}),
            )
        ]
        outcomes = executor.execute(calls)
        assert outcomes[0].status == "error"
        assert "RUNNER_CREATE_FAILED" in outcomes[0].output_json

    def test_execute_runner_ask_fails(self) -> None:
        class FailingRunner:
            def ask(self, *, message: str):
                raise RuntimeError("boom")

        spec = SubAgentSpec(
            name="fail_agent",
            description="desc",
            system_prompt="prompt",
            tool_definitions=(),
            runner_factory=lambda: FailingRunner(),
        )
        registry = SubAgentRegistry([spec])
        executor = SubAgentExecutor(registry=registry)
        calls = [
            SubAgentCall(
                call_id="call-1",
                name="delegate_to_fail_agent",
                arguments_json=json.dumps({"task": "x"}),
            )
        ]
        outcomes = executor.execute(calls)
        assert outcomes[0].status == "error"
        assert "SUB_AGENT_EXECUTION_FAILED" in outcomes[0].output_json

    def test_envelope_has_correct_shape(self) -> None:
        registry = SubAgentRegistry([_make_spec("agent_a", "answer")])
        executor = SubAgentExecutor(registry=registry)
        calls = [
            SubAgentCall(
                call_id="call-1",
                name="delegate_to_agent_a",
                arguments_json=json.dumps({"task": "test task"}),
            )
        ]
        outcomes = executor.execute(calls)
        env = outcomes[0].envelope
        assert isinstance(env, SubAgentResultEnvelope)
        assert env.status == "ok"
        assert env.sub_agent_name == "agent_a"
        assert env.call_id == "call-1"
        assert env.task == "test task"
        assert env.response_text == "answer"
