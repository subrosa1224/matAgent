"""Unit tests for MockStructuredProvider."""

import hashlib
import json
from pathlib import Path

import pytest

from materials_screening.llm.base import StructuredProviderResponse
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
from materials_screening.planner.models import (
    DraftStatus,
    EnergyUnit,
    PlannerDraft,
)
from materials_screening.planner.prompt_builder import build_user_message

FIXTURES_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "planner_mock_fixtures.json"
)


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


def _load_json_fixtures() -> dict[str, PlannerDraft]:
    raw = FIXTURES_PATH.read_text(encoding="utf-8")
    entries = json.loads(raw)
    return {
        entry["query"]: PlannerDraft.model_validate(entry["draft"]) for entry in entries
    }


class TestMockSuccess:
    def test_returns_parsed_draft_for_raw_query(self) -> None:
        draft = _draft(excluded_elements=["Pb"], is_metal=False)
        provider = MockStructuredProvider(fixtures={"不含 Pb": draft})
        response = provider.generate_structured(
            system_prompt="system",
            user_text="不含 Pb",
            output_model=PlannerDraft,
            schema_name="planner_draft_v1",
            max_output_tokens=4096,
        )
        assert isinstance(response, StructuredProviderResponse)
        assert isinstance(response.parsed, PlannerDraft)
        assert response.parsed.excluded_elements == ["Pb"]
        assert response.parsed.is_metal is False
        assert response.provider == "mock"
        assert response.model == "mock"
        assert response.request_id is not None
        assert response.request_id.startswith("mock-")
        assert response.latency_ms == 0
        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.reasoning_tokens == 0

    def test_wrapped_user_message_matches_fixture(self) -> None:
        query = "不含 Pb"
        provider = MockStructuredProvider(fixtures={query: _draft()})
        response = provider.generate_structured(
            system_prompt="system",
            user_text=build_user_message(query),
            output_model=PlannerDraft,
            schema_name="planner_draft_v1",
            max_output_tokens=4096,
        )
        assert response.parsed == _draft()

    def test_raw_output_hash_matches_canonical(self) -> None:
        draft = _draft(excluded_elements=["Pb"])
        provider = MockStructuredProvider(fixtures={"q": draft})
        response = provider.generate_structured(
            system_prompt="s",
            user_text="q",
            output_model=PlannerDraft,
            schema_name="planner_draft_v1",
            max_output_tokens=4096,
        )
        canonical = json.dumps(
            draft.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
        )
        assert (
            response.raw_output_sha256
            == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        )

    def test_deterministic(self) -> None:
        provider = MockStructuredProvider(fixtures={"q": _draft()})
        first = provider.generate_structured(
            system_prompt="s",
            user_text="q",
            output_model=PlannerDraft,
            schema_name="planner_draft_v1",
            max_output_tokens=4096,
        )
        second = provider.generate_structured(
            system_prompt="s",
            user_text="q",
            output_model=PlannerDraft,
            schema_name="planner_draft_v1",
            max_output_tokens=4096,
        )
        assert first == second

    def test_unknown_query_raises(self) -> None:
        provider = MockStructuredProvider()
        with pytest.raises(LLMStructuredOutputError, match="no mock fixture"):
            provider.generate_structured(
                system_prompt="s",
                user_text="未知查询",
                output_model=PlannerDraft,
                schema_name="planner_draft_v1",
                max_output_tokens=4096,
            )

    def test_loads_fixtures_from_json_file(self) -> None:
        fixtures = _load_json_fixtures()
        assert "寻找不含 Pb 的非金属材料" in fixtures
        assert "带隙 1 到 2 eV 的半导体" in fixtures
        provider = MockStructuredProvider(fixtures=fixtures)
        response = provider.generate_structured(
            system_prompt="s",
            user_text="带隙 1 到 2 eV 的半导体",
            output_model=PlannerDraft,
            schema_name="planner_draft_v1",
            max_output_tokens=4096,
        )
        assert response.parsed.band_gap_min == 1.0
        assert response.parsed.band_gap_max == 2.0


class TestMockErrors:
    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            (MockErrorKind.TIMEOUT, LLMTimeoutError),
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
    def test_error_fixture_raises_mapped_error(
        self, kind: MockErrorKind, expected: type[Exception]
    ) -> None:
        provider = MockStructuredProvider(error_fixtures={"q": kind})
        with pytest.raises(expected):
            provider.generate_structured(
                system_prompt="s",
                user_text="q",
                output_model=PlannerDraft,
                schema_name="planner_draft_v1",
                max_output_tokens=4096,
            )

    def test_error_fixture_by_wrapped_message(self) -> None:
        provider = MockStructuredProvider(error_fixtures={"q": MockErrorKind.TIMEOUT})
        with pytest.raises(LLMTimeoutError):
            provider.generate_structured(
                system_prompt="s",
                user_text=build_user_message("q"),
                output_model=PlannerDraft,
                schema_name="planner_draft_v1",
                max_output_tokens=4096,
            )
