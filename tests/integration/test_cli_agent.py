"""Integration tests for the agent CLI commands (S3.5-M6)."""

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from materials_screening.cli import app

_RUNNER = CliRunner()
_CONVERSATION_RE = re.compile(r"Conversation: ([A-Za-z0-9._-]+)")
_WHITELIST_TOOLS = (
    "run_screening_workflow",
    "get_workflow_status",
    "get_workflow_history",
    "get_screening_result",
    "compare_ranked_materials",
)


@pytest.fixture()
def offline_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run CLI with cwd inside tmp so data/ paths stay offline."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _ask(runner_args: list[str], offline_cwd: Path) -> str:
    result = _RUNNER.invoke(app, runner_args)
    assert result.exit_code == 0, result.output
    return result.output


class TestAsk:
    def test_ask_prints_conversation_id_and_safe_response(
        self, offline_cwd: Path
    ) -> None:
        output = _ask(["agent", "ask", "--message", "你好"], offline_cwd)

        match = _CONVERSATION_RE.search(output)
        assert match is not None
        assert "Agent> （离线演示）" in output
        assert "arguments" not in output
        assert "reasoning" not in output

    def test_ask_continues_existing_conversation(self, offline_cwd: Path) -> None:
        first = _ask(["agent", "ask", "--message", "你好"], offline_cwd)
        conversation_id = _CONVERSATION_RE.search(first).group(1)

        second = _ask(
            ["agent", "ask", "--conversation-id", conversation_id, "--message", "继续"],
            offline_cwd,
        )
        show = _ask(["agent", "show", conversation_id], offline_cwd)

        assert _CONVERSATION_RE.search(second).group(1) == conversation_id
        assert "Turns: 2" in show

    def test_ask_requires_message(self, offline_cwd: Path) -> None:
        result = _RUNNER.invoke(app, ["agent", "ask"])

        assert result.exit_code != 0
        assert "Missing option" in result.output

    def test_ask_intern_without_key_fails_offline(
        self, offline_cwd: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("INTERN_API_KEY", raising=False)

        result = _RUNNER.invoke(
            app,
            [
                "agent",
                "ask",
                "--llm-provider",
                "intern",
                "--message",
                "你好",
            ],
        )

        assert result.exit_code == 30
        assert "INTERN_API_KEY is not set" in result.output

    def test_ask_intern_uses_intern_key_not_mp_key(
        self, offline_cwd: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from materials_screening.cli import _build_agent_model

        monkeypatch.setenv("MP_API_KEY", "mp-key-must-not-be-sent-to-intern")
        monkeypatch.setenv("INTERN_API_KEY", "intern-real-key")

        model = _build_agent_model("intern")

        assert model._client.api_key == "intern-real-key"

    def test_ask_intern_without_mp_key_still_builds_model(
        self, offline_cwd: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from materials_screening.cli import _build_agent_model

        monkeypatch.delenv("MP_API_KEY", raising=False)
        monkeypatch.delenv("PMG_MAPI_KEY", raising=False)
        monkeypatch.setenv("INTERN_API_KEY", "intern-real-key")

        model = _build_agent_model("intern")

        assert model._client.api_key == "intern-real-key"


class TestProviderDefaults:
    def test_intern_defaults_to_materials_project(self) -> None:
        from materials_screening.cli import _resolve_provider_defaults

        repository, planner_fixture, materials_fixture = _resolve_provider_defaults(
            "intern",
            None,
            None,
            None,
        )

        assert repository == "materials-project"
        assert planner_fixture is None
        assert materials_fixture is None

    def test_mock_defaults_to_bundled_fixtures(self) -> None:
        from materials_screening.cli import _resolve_provider_defaults

        repository, planner_fixture, materials_fixture = _resolve_provider_defaults(
            "mock",
            None,
            None,
            None,
        )

        assert repository == "mock"
        assert planner_fixture is not None
        assert planner_fixture.name == "planner_cli_fixtures.json"
        assert materials_fixture is not None
        assert materials_fixture.name == "mp_documents.json"

    def test_explicit_values_override_defaults(self) -> None:
        from pathlib import Path

        from materials_screening.cli import _resolve_provider_defaults

        repository, planner_fixture, materials_fixture = _resolve_provider_defaults(
            "mock",
            "materials-project",
            Path("custom_planner.json"),
            Path("custom_materials.json"),
        )

        assert repository == "materials-project"
        assert planner_fixture == Path("custom_planner.json")
        assert materials_fixture == Path("custom_materials.json")


class TestChat:
    def test_chat_turn_then_exit(self, offline_cwd: Path) -> None:
        result = _RUNNER.invoke(
            app,
            ["agent", "chat"],
            input="你好\n/exit\n",
        )

        assert result.exit_code == 0, result.output
        assert "You>" in result.output
        assert "Agent> （离线演示）" in result.output

    def test_chat_supported_commands(self, offline_cwd: Path) -> None:
        result = _RUNNER.invoke(
            app,
            ["agent", "chat"],
            input="/id\n/tools\n/new\n/exit\n",
        )

        assert result.exit_code == 0, result.output
        assert "Conversation:" in result.output
        for tool in _WHITELIST_TOOLS:
            assert tool in result.output
        assert "inputs:" in result.output

    def test_chat_unknown_command_not_executed(self, offline_cwd: Path) -> None:
        result = _RUNNER.invoke(
            app,
            ["agent", "chat"],
            input="/ls\n/rm -rf .\n/exit\n",
        )

        assert result.exit_code == 0, result.output
        assert "Unknown command; supported: /exit /new /id /tools" in result.output
        assert "not recognized" not in result.output

    def test_chat_new_starts_different_conversation(self, offline_cwd: Path) -> None:
        result = _RUNNER.invoke(
            app,
            ["agent", "chat"],
            input="/new\n/new\n/exit\n",
        )

        ids = _CONVERSATION_RE.findall(result.output)
        assert len(ids) == 3
        assert len(set(ids)) == 3

    def test_chat_continues_existing_conversation(self, offline_cwd: Path) -> None:
        first = _ask(["agent", "ask", "--message", "你好"], offline_cwd)
        conversation_id = _CONVERSATION_RE.search(first).group(1)

        result = _RUNNER.invoke(
            app,
            ["agent", "chat", "--conversation-id", conversation_id],
            input="/id\n/exit\n",
        )

        assert result.exit_code == 0, result.output
        assert conversation_id in result.output


class TestShow:
    def test_show_prints_safe_summary(self, offline_cwd: Path) -> None:
        first = _ask(["agent", "ask", "--message", "你好"], offline_cwd)
        conversation_id = _CONVERSATION_RE.search(first).group(1)

        output = _ask(["agent", "show", conversation_id], offline_cwd)

        assert f"Conversation: {conversation_id}" in output
        assert "Turns: 1" in output
        assert "Last status: completed" in output
        assert "arguments" not in output
        assert "api_key" not in output
        assert "reasoning" not in output

    def test_show_missing_conversation_fails(self, offline_cwd: Path) -> None:
        result = _RUNNER.invoke(app, ["agent", "show", "missing"])

        assert result.exit_code != 0
        assert "not found" in result.output


class TestTools:
    def test_tools_shows_whitelist_only(self, offline_cwd: Path) -> None:
        output = _ask(["agent", "tools"], offline_cwd)

        for tool in _WHITELIST_TOOLS:
            assert tool in output
        assert "web_search" not in output
        assert "shell" not in output
        assert "python" not in output


class TestInspect:
    def test_inspect_offline_no_network(self, offline_cwd: Path) -> None:
        result = _RUNNER.invoke(app, ["agent", "inspect"])

        assert result.exit_code == 0, result.output
        assert "model: intern-s2-preview-35b" in result.output
        assert "base url: https://chat.intern-ai.org.cn/api/v1" in result.output
        assert "reasoning effort: none" in result.output
        assert "allow web search: false" in result.output
        assert "langgraph:" in result.output

    def test_inspect_hides_key_value(self, offline_cwd: Path) -> None:
        result = _RUNNER.invoke(app, ["agent", "inspect"])

        assert result.exit_code == 0, result.output
        assert "sk-" not in result.output


class TestOldCliCompatibility:
    def test_version_still_works(self, offline_cwd: Path) -> None:
        result = _RUNNER.invoke(app, ["version"])

        assert result.exit_code == 0, result.output

    def test_workflow_commands_still_work(self, offline_cwd: Path) -> None:
        result = _RUNNER.invoke(app, ["workflow", "draw"])

        assert result.exit_code == 0, result.output
        assert "flowchart" in result.output
