"""Planner service orchestration (D2-M3)."""

from collections.abc import Callable
from datetime import UTC, datetime

from materials_screening.llm.base import StructuredLLM
from materials_screening.llm.metadata import build_provider_metadata
from materials_screening.planner.errors import PlannerQueryError
from materials_screening.planner.models import PlannerDraft, PlannerResult
from materials_screening.planner.prompt_builder import (
    PromptBuilder,
    build_user_message,
)
from materials_screening.planner.resolver import PlannerResolver, normalize_query
from materials_screening.planner.settings import Settings

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)


class PlannerService:
    """Coordinate prompt, provider, metadata and resolver for one query."""

    def __init__(
        self,
        settings: Settings,
        provider: StructuredLLM,
        resolver: PlannerResolver | None = None,
        prompt_builder: PromptBuilder | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._resolver = resolver or PlannerResolver()
        self._prompt_builder = prompt_builder or PromptBuilder(
            version=settings.planner_prompt_version
        )
        self._clock = clock or utc_now

    def parse(self, query: str) -> PlannerResult:
        """Validate, normalize, call the provider and resolve the draft."""
        normalized = normalize_query(query)
        self._validate_query(normalized)

        prompt = self._prompt_builder.build()
        response = self._provider.generate_structured(
            system_prompt=prompt.system_prompt,
            user_text=build_user_message(normalized),
            output_model=PlannerDraft,
            schema_name="planner_draft_v1",
            max_output_tokens=self._settings.llm_max_output_tokens,
        )
        metadata = build_provider_metadata(
            response=response,
            prompt_version=prompt.version,
            prompt_hash=prompt.sha256,
            schema_version=self._settings.planner_schema_version,
            reasoning_effort="none",
            retrieved_at=self._clock(),
        )
        return self._resolver.resolve(
            query=normalized,
            draft=response.parsed,
            provider_metadata=metadata,
        )

    def _validate_query(self, normalized: str) -> None:
        if not normalized:
            raise PlannerQueryError("planner query must not be empty")
        if len(normalized) > self._settings.planner_max_query_chars:
            raise PlannerQueryError(
                f"planner query exceeds "
                f"{self._settings.planner_max_query_chars} characters"
            )
