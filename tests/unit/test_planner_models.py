"""Unit tests for Stage 2 planner models and LLM errors (D2-M1)."""

import pytest
from pydantic import ValidationError

from materials_screening.llm.base import StructuredProviderResponse
from materials_screening.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMModelNotSupportedError,
    LLMPermissionError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMServiceUnavailableError,
    LLMStructuredOutputError,
    LLMTimeoutError,
    LLMTruncatedOutputError,
)
from materials_screening.llm.metadata import ProviderMetadata
from materials_screening.models import ScreeningRequest
from materials_screening.planner.models import (
    DensityUnit,
    DraftStatus,
    EnergyUnit,
    EvidenceItem,
    HullUnit,
    PlannerDraft,
    PlannerResult,
    PlannerStatus,
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
        "hull_unit": HullUnit.UNSPECIFIED,
        "density_min": None,
        "density_max": None,
        "density_unit": DensityUnit.UNSPECIFIED,
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


def _metadata() -> ProviderMetadata:
    return ProviderMetadata(
        provider="intern",
        model="intern-s2-preview-35b",
        request_id="req-1",
        latency_ms=120,
        input_tokens=100,
        output_tokens=50,
        reasoning_tokens=None,
        reasoning_effort="none",
        prompt_version="planner-v1",
        prompt_hash="abc",
        schema_version="planner-draft-v1",
        raw_output_sha256="def",
    )


class TestEnums:
    def test_draft_status(self) -> None:
        assert DraftStatus.EXTRACTED == "extracted"

    def test_energy_unit_values(self) -> None:
        assert EnergyUnit.EV == "eV"
        assert EnergyUnit.MEV == "meV"
        assert EnergyUnit.UNSPECIFIED == "unspecified"

    def test_hull_unit_values(self) -> None:
        assert HullUnit.EV_PER_ATOM == "eV/atom"
        assert HullUnit.MEV_PER_ATOM == "meV/atom"
        assert HullUnit.UNSPECIFIED == "unspecified"

    def test_density_unit_values(self) -> None:
        assert DensityUnit.G_PER_CM3 == "g/cm3"
        assert DensityUnit.KG_PER_M3 == "kg/m3"
        assert DensityUnit.UNSPECIFIED == "unspecified"

    def test_planner_status_values(self) -> None:
        assert PlannerStatus.READY == "ready"
        assert PlannerStatus.NEEDS_CLARIFICATION == "needs_clarification"
        assert PlannerStatus.INVALID == "invalid"
        assert PlannerStatus.UNSUPPORTED == "unsupported"


class TestPlannerDraft:
    def test_valid_draft(self) -> None:
        draft = _draft(
            excluded_elements=["Pb", "Cd"],
            band_gap_min=1.2,
            band_gap_max=2.0,
            band_gap_unit=EnergyUnit.EV,
            hull_max=0.05,
            hull_unit=HullUnit.EV_PER_ATOM,
            is_metal=False,
            limit=10,
            evidence=[EvidenceItem(quote="带隙 1.2 到 2.0 eV")],
        )
        assert draft.status is DraftStatus.EXTRACTED
        assert draft.excluded_elements == ["Pb", "Cd"]
        assert draft.band_gap_min == 1.2
        assert draft.hull_max == 0.05
        assert draft.limit == 10
        assert draft.evidence[0].quote == "带隙 1.2 到 2.0 eV"

    @pytest.mark.parametrize(
        "field",
        [
            "status",
            "band_gap_unit",
            "hull_unit",
            "density_unit",
            "spacegroup_numbers",
            "evidence",
        ],
    )
    def test_all_fields_required(self, field: str) -> None:
        data = _draft().model_dump()
        del data[field]
        with pytest.raises(ValidationError):
            PlannerDraft.model_validate(data)

    def test_extra_field_rejected(self) -> None:
        data = _draft().model_dump()
        data["reasoning"] = "hidden"
        with pytest.raises(ValidationError):
            PlannerDraft.model_validate(data)

    def test_invalid_enum_value_rejected(self) -> None:
        data = _draft().model_dump()
        data["band_gap_unit"] = "volt"
        with pytest.raises(ValidationError):
            PlannerDraft.model_validate(data)

    def test_frozen(self) -> None:
        assert PlannerDraft.model_config.get("frozen") is True

    def test_no_trusted_screening_request_field(self) -> None:
        assert "request" not in PlannerDraft.model_fields
        assert "ScreeningRequest" not in PlannerDraft.model_fields

    def test_no_reasoning_or_chain_of_thought_fields(self) -> None:
        for forbidden in (
            "reasoning",
            "thoughts",
            "chain_of_thought",
            "database_query",
            "tool_calls",
            "candidate_materials",
        ):
            assert forbidden not in PlannerDraft.model_fields


class TestEvidenceItem:
    def test_valid(self) -> None:
        assert EvidenceItem(quote="不含 Pb").quote == "不含 Pb"

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceItem.model_validate({"quote": "x", "junk": 1})

    def test_frozen(self) -> None:
        assert EvidenceItem.model_config.get("frozen") is True


class TestPlannerResult:
    def test_ready_requires_request(self) -> None:
        result = PlannerResult(
            status=PlannerStatus.READY,
            query="找非金属材料",
            request=ScreeningRequest(is_metal=False),
            provider_metadata=_metadata(),
        )
        assert result.status is PlannerStatus.READY
        assert result.request is not None

    def test_ready_without_request_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must include a request"):
            PlannerResult(status=PlannerStatus.READY, query="q")

    @pytest.mark.parametrize(
        "status",
        [
            PlannerStatus.NEEDS_CLARIFICATION,
            PlannerStatus.INVALID,
            PlannerStatus.UNSUPPORTED,
        ],
    )
    def test_non_ready_with_request_rejected(self, status: PlannerStatus) -> None:
        with pytest.raises(ValidationError, match="must not include a request"):
            PlannerResult(
                status=status,
                query="q",
                request=ScreeningRequest(),
            )

    def test_needs_clarification_requires_question(self) -> None:
        with pytest.raises(ValidationError, match="clarification question"):
            PlannerResult(status=PlannerStatus.NEEDS_CLARIFICATION, query="q")

    def test_needs_clarification_valid(self) -> None:
        result = PlannerResult(
            status=PlannerStatus.NEEDS_CLARIFICATION,
            query="q",
            clarification_question="请指定带隙范围",
        )
        assert result.request is None
        assert result.clarification_question == "请指定带隙范围"

    def test_invalid_and_unsupported_valid_without_request(self) -> None:
        invalid = PlannerResult(
            status=PlannerStatus.INVALID,
            query="q",
            invalid_reasons=("negative_band_gap",),
            conflicts=("band_gap_both_required_and_excluded",),
        )
        unsupported = PlannerResult(
            status=PlannerStatus.UNSUPPORTED,
            query="q",
            unsupported_requirements=("predict_synthesis",),
        )
        assert invalid.request is None
        assert unsupported.request is None

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PlannerResult.model_validate(
                {
                    "status": PlannerStatus.INVALID,
                    "query": "q",
                    "invalid_reasons": ("negative_band_gap",),
                    "junk": 1,
                }
            )

    def test_frozen(self) -> None:
        assert PlannerResult.model_config.get("frozen") is True


class TestProviderMetadata:
    def test_valid(self) -> None:
        metadata = _metadata()
        assert metadata.provider == "intern"
        assert metadata.reasoning_effort == "none"

    def test_extra_field_rejected(self) -> None:
        data = _metadata().model_dump()
        data["api_key"] = "secret"
        with pytest.raises(ValidationError):
            ProviderMetadata.model_validate(data)

    def test_frozen(self) -> None:
        assert ProviderMetadata.model_config.get("frozen") is True


class TestStructuredProviderResponse:
    def test_valid_with_planner_draft(self) -> None:
        draft = _draft()
        response = StructuredProviderResponse[PlannerDraft](
            parsed=draft,
            provider="intern",
            model="intern-s2-preview-35b",
            request_id="req-1",
            latency_ms=120,
            input_tokens=100,
            output_tokens=50,
            reasoning_tokens=None,
            raw_output_sha256="abc",
        )
        assert response.parsed == draft
        assert response.model == "intern-s2-preview-35b"

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StructuredProviderResponse[PlannerDraft].model_validate(
                {
                    "parsed": _draft().model_dump(),
                    "provider": "intern",
                    "model": "intern-s2-preview-35b",
                    "request_id": None,
                    "latency_ms": 1,
                    "input_tokens": None,
                    "output_tokens": None,
                    "reasoning_tokens": None,
                    "raw_output_sha256": None,
                    "junk": 1,
                }
            )

    def test_frozen(self) -> None:
        assert StructuredProviderResponse.model_config.get("frozen") is True


class TestLLMErrors:
    @pytest.mark.parametrize(
        "error_cls",
        [
            LLMConfigurationError,
            LLMAuthenticationError,
            LLMPermissionError,
            LLMRateLimitError,
            LLMTimeoutError,
            LLMConnectionError,
            LLMServiceUnavailableError,
            LLMModelNotSupportedError,
            LLMRefusalError,
            LLMTruncatedOutputError,
            LLMStructuredOutputError,
        ],
    )
    def test_all_errors_are_llm_errors(self, error_cls: type[Exception]) -> None:
        assert issubclass(error_cls, LLMError)

    def test_error_carries_message(self) -> None:
        with pytest.raises(LLMStructuredOutputError, match="invalid json"):
            raise LLMStructuredOutputError("invalid json")

    def test_llm_error_is_exception(self) -> None:
        assert issubclass(LLMError, Exception)
