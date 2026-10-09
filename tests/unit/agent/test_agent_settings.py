"""Unit tests for stage 3.5 agent settings (S3.5-M1)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from materials_screening.agent.settings import AgentSettings

_AGENT_ENV_NAMES: tuple[str, ...] = (
    "AGENT_ENABLED",
    "AGENT_VERSION",
    "AGENT_PROMPT_VERSION",
    "AGENT_CHECKPOINTER_BACKEND",
    "AGENT_CHECKPOINT_DB",
    "AGENT_MAX_MODEL_CALLS_PER_TURN",
    "AGENT_MAX_TOOL_CALLS_PER_TURN",
    "AGENT_MAX_WORKFLOW_RUNS_PER_TURN",
    "AGENT_MAX_CONVERSATION_TURNS",
    "AGENT_MAX_ARGUMENT_BYTES",
    "AGENT_MAX_TOOL_OUTPUT_BYTES",
    "AGENT_MAX_INPUT_BYTES",
    "AGENT_REASONING_EFFORT",
    "AGENT_TEMPERATURE",
    "AGENT_STORE_RAW_MODEL_OUTPUT",
    "AGENT_ALLOW_WEB_SEARCH",
    "AGENT_BASE_URL",
    "AGENT_MODEL",
)


def _clear_agent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _AGENT_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


class TestAgentSettingsDefaults:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_agent_env(monkeypatch)
        settings = AgentSettings(_env_file=None)
        assert settings.agent_enabled is True
        assert settings.agent_version == "material-agent-v1"
        assert settings.agent_prompt_version == "material-agent-v1"
        assert settings.agent_checkpointer_backend == "sqlite"
        assert settings.agent_checkpoint_db == Path("data/agent_checkpoints.sqlite")
        assert settings.agent_max_model_calls_per_turn == 4
        assert settings.agent_max_tool_calls_per_turn == 4
        assert settings.agent_max_workflow_runs_per_turn == 1
        assert settings.agent_max_conversation_turns == 20
        assert settings.agent_max_argument_bytes == 8192
        assert settings.agent_max_tool_output_bytes == 32768
        assert settings.agent_max_input_bytes == 122880
        assert settings.agent_reasoning_effort == "none"
        assert settings.agent_temperature == 0.0
        assert settings.agent_store_raw_model_output is False
        assert settings.agent_allow_web_search is False
        assert settings.agent_base_url == "https://chat.intern-ai.org.cn/api/v1"
        assert settings.agent_model == "intern-s2-preview-35b"


class TestAgentSettingsEnvOverride:
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_agent_env(monkeypatch)
        monkeypatch.setenv("AGENT_ENABLED", "false")
        monkeypatch.setenv("AGENT_VERSION", "material-agent-v2")
        monkeypatch.setenv("AGENT_MAX_MODEL_CALLS_PER_TURN", "8")
        monkeypatch.setenv("AGENT_MAX_TOOL_CALLS_PER_TURN", "6")
        monkeypatch.setenv("AGENT_MAX_CONVERSATION_TURNS", "50")
        monkeypatch.setenv("AGENT_REASONING_EFFORT", "low")
        monkeypatch.setenv("AGENT_TEMPERATURE", "0.5")
        monkeypatch.setenv("AGENT_STORE_RAW_MODEL_OUTPUT", "true")
        settings = AgentSettings()
        assert settings.agent_enabled is False
        assert settings.agent_version == "material-agent-v2"
        assert settings.agent_max_model_calls_per_turn == 8
        assert settings.agent_max_tool_calls_per_turn == 6
        assert settings.agent_max_conversation_turns == 50
        assert settings.agent_reasoning_effort == "low"
        assert settings.agent_temperature == 0.5
        assert settings.agent_store_raw_model_output is True


class TestAgentSettingsValidators:
    def test_web_search_cannot_be_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_agent_env(monkeypatch)
        monkeypatch.setenv("AGENT_ALLOW_WEB_SEARCH", "true")
        with pytest.raises(ValidationError):
            AgentSettings()

    @pytest.mark.parametrize(
        "base_url,expected",
        [
            ("https://chat.intern-ai.org.cn/api/v1/", "https://chat.intern-ai.org.cn/api/v1"),
            ("http://localhost:8000/v1", "http://localhost:8000/v1"),
            ("https://custom.llm.example.com/v1", "https://custom.llm.example.com/v1"),
            ("http://localhost:8000", "http://localhost:8000"),
        ],
    )
    def test_base_url_accepted(
        self, monkeypatch: pytest.MonkeyPatch, base_url: str, expected: str
    ) -> None:
        _clear_agent_env(monkeypatch)
        monkeypatch.setenv("AGENT_BASE_URL", base_url)
        assert AgentSettings().agent_base_url == expected

    @pytest.mark.parametrize(
        "model",
        ["intern-s2-preview-35b", "intern-s1", "intern-s1-mini", "my-custom-model-v1"],
    )
    def test_model_accepted(
        self, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        _clear_agent_env(monkeypatch)
        monkeypatch.setenv("AGENT_MODEL", model)
        assert AgentSettings().agent_model == model

    @pytest.mark.parametrize("base_url", ["", "not-a-url"])
    def test_invalid_url_rejected(
        self, monkeypatch: pytest.MonkeyPatch, base_url: str
    ) -> None:
        _clear_agent_env(monkeypatch)
        monkeypatch.setenv("AGENT_BASE_URL", base_url)
        with pytest.raises(ValidationError):
            AgentSettings()

    @pytest.mark.parametrize("model", ["", "   "])
    def test_empty_model_rejected(
        self, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        _clear_agent_env(monkeypatch)
        monkeypatch.setenv("AGENT_MODEL", model)
        with pytest.raises(ValidationError):
            AgentSettings()

    @pytest.mark.parametrize("value", ["0", "11"])
    def test_model_call_bounds(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        _clear_agent_env(monkeypatch)
        monkeypatch.setenv("AGENT_MAX_MODEL_CALLS_PER_TURN", value)
        with pytest.raises(ValidationError):
            AgentSettings()

    @pytest.mark.parametrize("value", ["0", "11"])
    def test_tool_call_bounds(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        _clear_agent_env(monkeypatch)
        monkeypatch.setenv("AGENT_MAX_TOOL_CALLS_PER_TURN", value)
        with pytest.raises(ValidationError):
            AgentSettings()

    def test_workflow_runs_must_be_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_agent_env(monkeypatch)
        monkeypatch.setenv("AGENT_MAX_WORKFLOW_RUNS_PER_TURN", "2")
        with pytest.raises(ValidationError):
            AgentSettings()

    @pytest.mark.parametrize("value", ["0", "101"])
    def test_conversation_turn_bounds(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        _clear_agent_env(monkeypatch)
        monkeypatch.setenv("AGENT_MAX_CONVERSATION_TURNS", value)
        with pytest.raises(ValidationError):
            AgentSettings()

    @pytest.mark.parametrize("value", ["-0.1", "2.1"])
    def test_temperature_bounds(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        _clear_agent_env(monkeypatch)
        monkeypatch.setenv("AGENT_TEMPERATURE", value)
        with pytest.raises(ValidationError):
            AgentSettings()

    def test_checkpoint_db_outside_data_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _clear_agent_env(monkeypatch)
        monkeypatch.setenv("AGENT_CHECKPOINT_DB", str(tmp_path / "a.sqlite"))
        with pytest.raises(ValidationError):
            AgentSettings()

    def test_checkpoint_db_in_data_accepted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _clear_agent_env(monkeypatch)
        monkeypatch.setenv("AGENT_CHECKPOINT_DB", "data/agent_checkpoints.sqlite")
        settings = AgentSettings()
        assert settings.agent_checkpoint_db == Path("data/agent_checkpoints.sqlite")
