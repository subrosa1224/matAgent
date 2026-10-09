"""LLM provider factory (D2-M3)."""

from enum import StrEnum

from materials_screening.llm.base import StructuredLLM
from materials_screening.llm.intern_provider import InternProvider
from materials_screening.llm.mock_provider import MockStructuredProvider
from materials_screening.planner.settings import Settings


class LLMProviderName(StrEnum):
    MOCK = "mock"
    INTERN = "intern"


def create_llm_provider(settings: Settings) -> StructuredLLM:
    """Create the configured LLM provider without any network access."""
    provider_name = LLMProviderName(settings.llm_provider)
    if provider_name is LLMProviderName.MOCK:
        return MockStructuredProvider()
    return InternProvider(settings)
