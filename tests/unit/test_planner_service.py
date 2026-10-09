"""Unit tests for PlannerService (D2-M3)."""

from datetime import UTC, datetime

import pytest

from materials_screening.llm.errors import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMServiceUnavailableError,
    LLMStructuredOutputError,
    LLMTimeoutError,
    LLMTruncatedOutputError,
)
from materials_screening.llm.mock_provider import (
    MockStructuredProvider,
    MockErrorKind,
)
from materials_screening.planner.errors import PlannerQueryError
from materials_screening.planner.models import (
    DraftStatus,
    EnergyUnit,
    PlannerDraft,
    PlannerStatus,
)
from materials_screening.planner.service import PlannerService
from materials_screening.planner.settings import Settings

FIXED_TIME = datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC)


def _draft(**overrides: object) -> PlannerDraft:
    values: dict[str, object] = {
        "status": DraftStatus.EXTRACTED,
        "required_elements": [],
        "excluded_elements": [],
        "chemsys": None,
        "formula": None,
        "band_gap_min": None,
        "band_gap_max": None,
        "band_gap_unit": EnergyUnit.UNSPECIFIED,
        "hull_min": None,
        "hull_max": None,
        "hull_unit": "unspecified",
        "density_min": None,
        "density_max": None,
        "density_unit": "unspecified",
        "crystal_system": None,
        "spacegroup_numbers": [],
        "is_metal": None,
        "is_stable": None,
        "theoretical": None,
        "target_band_gap": None,
        "target_band_gap_unit": EnergyUnit.UNSPECIFIED,
        "limit": None,
        "ambiguities": [],
        "unsupported_requirements": [],
        "conflicts": [],
        "assumptions": [],
        "clarification_question": "",
        "evidence": [],
    }
    values.update(overrides)
    return PlannerDraft.model_validate(values)


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "llm_provider": "mock",
        "planner_prompt_version": "planner-v2",
        "planner_schema_version": "planner-draft-v1",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _service(
    provider: MockStructuredProvider,
    settings: Settings | None = None,
) -> PlannerService:
    return PlannerService(
        settings=settings or _settings(),
        provider=provider,
        clock=lambda: FIXED_TIME,
    )


class TestPlannerServiceNormal:
    def test_ready_with_metadata(self) -> None:
        query = "  寻找不含 Pb 的非金属材料  "
        provider = MockStructuredProvider(
            fixtures={
                "寻找不含 Pb 的非金属材料": _draft(
                    excluded_elements=["Pb"], is_metal=False
                )
            }
        )
        result = _service(provider).parse(query)
        assert result.status is PlannerStatus.READY
        assert result.query == "寻找不含 Pb 的非金属材料"
        assert result.request is not None
        assert result.request.excluded_elements == ("Pb",)
        assert result.request.is_metal is False

        metadata = result.provider_metadata
        assert metadata is not None
        assert metadata.provider == "mock"
        assert metadata.model == "mock"
        assert metadata.request_id is not None
        assert metadata.latency_ms == 0
        assert metadata.input_tokens == 0
        assert metadata.output_tokens == 0
        assert metadata.reasoning_tokens == 0
        assert metadata.reasoning_effort == "none"
        assert metadata.prompt_version == "planner-v2"
        assert metadata.prompt_hash
        assert metadata.schema_version == "planner-draft-v1"
        assert metadata.raw_output_sha256
        assert metadata.retrieved_at == FIXED_TIME

    def test_clock_injected(self) -> None:
        provider = MockStructuredProvider(fixtures={"q": _draft()})
        result = _service(provider).parse("q")
        assert result.provider_metadata is not None
        assert result.provider_metadata.retrieved_at == FIXED_TIME

    def test_no_raw_prompt_or_response_in_result(self) -> None:
        provider = MockStructuredProvider(fixtures={"q": _draft()})
        result = _service(provider).parse("q")
        dump = result.model_dump(mode="json")
        text = str(dump)
        assert "结构化信息抽取器" not in text
        assert "<user_query>" not in text
        assert "mock returned" not in text


class TestPlannerServiceValidation:
    def test_empty_query_raises(self) -> None:
        service = _service(MockStructuredProvider())
        with pytest.raises(PlannerQueryError, match="must not be empty"):
            service.parse("   ")

    def test_whitespace_only_raises(self) -> None:
        service = _service(MockStructuredProvider())
        with pytest.raises(PlannerQueryError):
            service.parse("\t\n")

    def test_too_long_query_raises(self) -> None:
        service = _service(
            MockStructuredProvider(),
            settings=_settings(planner_max_query_chars=10),
        )
        with pytest.raises(PlannerQueryError, match="exceeds"):
            service.parse("a" * 11)

    def test_query_at_length_limit_ok(self) -> None:
        query = "a" * 10
        service = _service(
            MockStructuredProvider(fixtures={query: _draft()}),
            settings=_settings(planner_max_query_chars=10),
        )
        result = service.parse(query)
        assert result.status is PlannerStatus.READY

    def test_validation_happens_before_provider(self) -> None:
        service = _service(
            MockStructuredProvider(),
            settings=_settings(planner_max_query_chars=3),
        )
        with pytest.raises(PlannerQueryError):
            service.parse("abcd")


class TestPlannerServiceFourStates:
    def test_ready(self) -> None:
        provider = MockStructuredProvider(fixtures={"q": _draft()})
        assert _service(provider).parse("q").status is PlannerStatus.READY

    def test_needs_clarification(self) -> None:
        provider = MockStructuredProvider(
            fixtures={
                "q": _draft(band_gap_min=5000, band_gap_unit=EnergyUnit.UNSPECIFIED)
            }
        )
        result = _service(provider).parse("q")
        assert result.status is PlannerStatus.NEEDS_CLARIFICATION
        assert result.request is None
        assert result.clarification_question

    def test_invalid(self) -> None:
        provider = MockStructuredProvider(
            fixtures={"q": _draft(required_elements=["Xx"])}
        )
        result = _service(provider).parse("q")
        assert result.status is PlannerStatus.INVALID
        assert result.request is None
        assert result.invalid_reasons

    def test_unsupported(self) -> None:
        provider = MockStructuredProvider(
            fixtures={"q": _draft(unsupported_requirements=["predict"])}
        )
        result = _service(provider).parse("q")
        assert result.status is PlannerStatus.UNSUPPORTED
        assert result.request is None


class TestPlannerServiceProviderErrors:
    def test_provider_error_not_swallowed(self) -> None:
        provider = MockStructuredProvider(error_fixtures={"q": MockErrorKind.TIMEOUT})
        with pytest.raises(LLMTimeoutError):
            _service(provider).parse("q")

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            (MockErrorKind.RATE_LIMIT, LLMRateLimitError),
            (MockErrorKind.AUTHENTICATION, LLMAuthenticationError),
            (MockErrorKind.SERVICE_ERROR, LLMServiceUnavailableError),
            (MockErrorKind.INCOMPLETE_MAX_TOKENS, LLMTruncatedOutputError),
            (MockErrorKind.CONTENT_FILTER, LLMRefusalError),
            (MockErrorKind.EMPTY_OUTPUT, LLMStructuredOutputError),
            (MockErrorKind.INVALID_JSON, LLMStructuredOutputError),
            (MockErrorKind.SCHEMA_MISMATCH, LLMStructuredOutputError),
        ],
    )
    def test_other_provider_errors_not_swallowed(
        self, kind: MockErrorKind, expected: type[Exception]
    ) -> None:
        provider = MockStructuredProvider(error_fixtures={"q": kind})
        with pytest.raises(expected):
            _service(provider).parse("q")
