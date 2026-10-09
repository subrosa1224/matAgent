"""Unit tests for planner settings."""

import pytest
from pydantic import SecretStr, ValidationError

from materials_screening.planner.settings import Settings


class TestSettingsDefaults:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "LLM_PROVIDER", "INTERN_API_KEY", "INTERN_BASE_URL",
            "INTERN_MODEL", "INTERN_THINKING_MODE", "LLM_TIMEOUT_SECONDS",
            "LLM_MAX_ATTEMPTS", "LLM_MAX_OUTPUT_TOKENS",
        ):
            monkeypatch.delenv(name, raising=False)
        settings = Settings(_env_file=None)
        assert settings.llm_provider == "mock"
        assert settings.intern_api_key is None
        assert settings.intern_base_url == "https://chat.intern-ai.org.cn/api/v1/"
        assert settings.intern_model == "intern-s2-preview-35b"
        assert settings.intern_thinking_mode is True


class TestSettingsEnvOverride:
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "intern")
        monkeypatch.setenv("INTERN_API_KEY", "intern-secret")
        monkeypatch.setenv("INTERN_MODEL", "intern-s2-preview-35b")
        monkeypatch.setenv("INTERN_THINKING_MODE", "false")
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "60")
        settings = Settings()
        assert settings.llm_provider == "intern"
        assert isinstance(settings.intern_api_key, SecretStr)
        assert settings.intern_api_key.get_secret_value() == "intern-secret"
        assert settings.intern_thinking_mode is False
        assert settings.llm_timeout_seconds == 60

    def test_api_key_not_leaked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INTERN_API_KEY", "intern-secret")
        settings = Settings()
        assert "intern-secret" not in repr(settings.intern_api_key)


class TestSettingsValidators:
    def test_base_url_normalized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INTERN_BASE_URL", "https://example.test/api/v1")
        assert Settings().intern_base_url == "https://example.test/api/v1/"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_model_rejected(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("INTERN_MODEL", value)
        with pytest.raises(ValidationError):
            Settings(_env_file=None, intern_model=value)

    @pytest.mark.parametrize("provider", ["deepseek", "anthropic", "qwen"])
    def test_other_providers_rejected(
        self, monkeypatch: pytest.MonkeyPatch, provider: str
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", provider)
        with pytest.raises(ValidationError):
            Settings()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("llm_timeout_seconds", 0),
            ("llm_timeout_seconds", 301),
            ("llm_max_attempts", 0),
            ("llm_max_attempts", 4),
            ("llm_max_output_tokens", 511),
            ("llm_max_output_tokens", 32769),
            ("planner_max_query_chars", 0),
            ("planner_max_query_chars", 20001),
        ],
    )
    def test_bounds(self, field: str, value: int) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **{field: value})
