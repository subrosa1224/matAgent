"""Unit tests for the LLM provider factory (D2-M3)."""

import pytest

from materials_screening.llm.intern_provider import InternProvider
from materials_screening.llm.errors import LLMConfigurationError
from materials_screening.llm.factory import (
    LLMProviderName,
    create_llm_provider,
)
from materials_screening.llm.mock_provider import MockStructuredProvider
from materials_screening.planner.settings import Settings


class TestLLMProviderName:
    def test_values(self) -> None:
        assert LLMProviderName.MOCK == "mock"
        assert LLMProviderName.INTERN == "intern"


class TestCreateLLMProvider:
    def test_default_is_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        settings = Settings(_env_file=None)
        assert isinstance(create_llm_provider(settings), MockStructuredProvider)

    def test_explicit_mock(self) -> None:
        settings = Settings(_env_file=None, llm_provider="mock")
        assert isinstance(create_llm_provider(settings), MockStructuredProvider)

    def test_intern_without_key_raises_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("INTERN_API_KEY", raising=False)
        settings = Settings(_env_file=None, llm_provider="intern")
        with pytest.raises(LLMConfigurationError, match="INTERN_API_KEY"):
            create_llm_provider(settings)

    def test_intern_with_key_returns_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INTERN_API_KEY", "intern-test")
        settings = Settings(_env_file=None, llm_provider="intern")
        provider = create_llm_provider(settings)
        assert isinstance(provider, InternProvider)
