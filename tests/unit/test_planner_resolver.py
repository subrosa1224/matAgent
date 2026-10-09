"""Unit tests for the deterministic planner resolver (D2-M2)."""

import pytest

from materials_screening.llm.metadata import ProviderMetadata
from materials_screening.models import CrystalSystem
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
from materials_screening.planner.resolver import (
    PlannerResolver,
    normalize_query,
)
from materials_screening.planner.rules import (
    AmbiguityCode,
    ConflictCode,
    InvalidCode,
    StabilityRule,
    UnsupportedCode,
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


def _full_draft() -> PlannerDraft:
    return _draft(
        excluded_elements=["Cd", "Hg", "Pb"],
        band_gap_min=1.2,
        band_gap_max=2.0,
        band_gap_unit=EnergyUnit.EV,
        hull_min=0.0,
        hull_max=0.05,
        hull_unit=HullUnit.EV_PER_ATOM,
        is_metal=False,
        limit=10,
    )


class TestNormalizeQuery:
    def test_trims_whitespace(self) -> None:
        assert normalize_query("  寻找材料  ") == "寻找材料"

    def test_nfc_normalization(self) -> None:
        composed = "café"
        decomposed = "cafe\u0301"
        assert normalize_query(composed) == normalize_query(decomposed)

    def test_empty(self) -> None:
        assert normalize_query("   ") == ""


class TestResolverBasics:
    def test_empty_query_invalid(self) -> None:
        result = PlannerResolver().resolve("   ", _draft())
        assert result.status is PlannerStatus.INVALID
        assert InvalidCode.EMPTY_QUERY.value in result.invalid_reasons
        assert result.request is None

    def test_full_draft_ready(self) -> None:
        result = PlannerResolver().resolve("寻找半导体", _full_draft())
        assert result.status is PlannerStatus.READY
        assert result.request is not None
        assert result.request.excluded_elements == ("Cd", "Hg", "Pb")
        assert result.request.band_gap_ev is not None
        assert result.request.band_gap_ev.min == 1.2
        assert result.request.band_gap_ev.max == 2.0
        assert result.request.energy_above_hull_ev_atom is not None
        assert result.request.energy_above_hull_ev_atom.max == 0.05
        assert result.request.is_metal is False
        assert result.request.limit == 10

    def test_query_normalized_in_result(self) -> None:
        result = PlannerResolver().resolve("  寻找材料  ", _draft())
        assert result.query == "寻找材料"

    def test_provider_metadata_passthrough(self) -> None:
        result = PlannerResolver().resolve("q", _draft(), provider_metadata=_metadata())
        assert result.provider_metadata is not None
        assert result.provider_metadata.request_id == "req-1"

    @pytest.mark.parametrize(
        ("status", "extra"),
        [
            (PlannerStatus.INVALID, {"invalid_reasons": ("x",)}),
            (PlannerStatus.UNSUPPORTED, {}),
            (
                PlannerStatus.NEEDS_CLARIFICATION,
                {"clarification_question": "?"},
            ),
        ],
    )
    def test_non_ready_has_no_request(
        self, status: PlannerStatus, extra: dict[str, str]
    ) -> None:
        result = PlannerResult(status=status, query="q", **extra)
        assert result.request is None


class TestElements:
    def test_invalid_element_invalid(self) -> None:
        result = PlannerResolver().resolve("q", _draft(required_elements=["Xx"]))
        assert result.status is PlannerStatus.INVALID
        assert InvalidCode.INVALID_ELEMENT.value in result.invalid_reasons

    def test_elements_normalized(self) -> None:
        result = PlannerResolver().resolve(
            "q", _draft(required_elements=["pb", "Pb", "O"])
        )
        assert result.status is PlannerStatus.READY
        assert result.request is not None
        assert result.request.required_elements == ("O", "Pb")

    def test_required_excluded_overlap_invalid(self) -> None:
        result = PlannerResolver().resolve(
            "q",
            _draft(
                required_elements=["Pb"],
                excluded_elements=["Pb"],
            ),
        )
        assert result.status is PlannerStatus.INVALID
        assert ConflictCode.REQUIRED_AND_EXCLUDED_ELEMENT.value in result.conflicts


class TestChemsys:
    def test_chemsys_normalized(self) -> None:
        result = PlannerResolver().resolve("q", _draft(chemsys="O-Li-Fe"))
        assert result.status is PlannerStatus.READY
        assert result.request is not None
        assert result.request.chemsys == "Fe-Li-O"

    def test_invalid_chemsys_invalid(self) -> None:
        result = PlannerResolver().resolve("q", _draft(chemsys="Xx-Fe"))
        assert result.status is PlannerStatus.INVALID
        assert InvalidCode.INVALID_CHEMSYS.value in result.invalid_reasons


class TestUnitConversion:
    def test_band_mev_to_ev(self) -> None:
        result = PlannerResolver().resolve(
            "q",
            _draft(
                band_gap_min=500,
                band_gap_max=1000,
                band_gap_unit=EnergyUnit.MEV,
            ),
        )
        assert result.status is PlannerStatus.READY
        assert result.request is not None
        assert result.request.band_gap_ev is not None
        assert result.request.band_gap_ev.min == 0.5
        assert result.request.band_gap_ev.max == 1.0

    def test_hull_mev_per_atom_to_ev_per_atom(self) -> None:
        result = PlannerResolver().resolve(
            "q",
            _draft(
                hull_min=0,
                hull_max=50,
                hull_unit=HullUnit.MEV_PER_ATOM,
            ),
        )
        assert result.request is not None
        assert result.request.energy_above_hull_ev_atom is not None
        assert result.request.energy_above_hull_ev_atom.max == 0.05

    def test_density_kg_per_m3_to_g_per_cm3(self) -> None:
        result = PlannerResolver().resolve(
            "q",
            _draft(
                density_min=500,
                density_max=5200,
                density_unit=DensityUnit.KG_PER_M3,
            ),
        )
        assert result.request is not None
        assert result.request.density_g_cm3 is not None
        assert result.request.density_g_cm3.min == 0.5
        assert result.request.density_g_cm3.max == 5.2

    def test_unspecified_plausible_assumes_default(self) -> None:
        result = PlannerResolver().resolve(
            "q", _draft(band_gap_min=1.5, band_gap_unit=EnergyUnit.UNSPECIFIED)
        )
        assert result.status is PlannerStatus.READY
        assert result.request is not None
        assert result.request.band_gap_ev is not None
        assert result.request.band_gap_ev.min == 1.5
        assert AmbiguityCode.UNIT_UNSPECIFIED_ASSUMED.value in result.assumptions

    def test_unspecified_implausible_needs_clarification(self) -> None:
        result = PlannerResolver().resolve(
            "q", _draft(band_gap_min=5000, band_gap_unit=EnergyUnit.UNSPECIFIED)
        )
        assert result.status is PlannerStatus.NEEDS_CLARIFICATION
        assert AmbiguityCode.UNIT_UNSPECIFIED_IMPLAUSIBLE.value in result.ambiguities
        assert result.clarification_question
        assert result.request is None

    def test_negative_band_gap_invalid(self) -> None:
        result = PlannerResolver().resolve(
            "q",
            _draft(band_gap_min=-1, band_gap_unit=EnergyUnit.EV),
        )
        assert result.status is PlannerStatus.INVALID
        assert InvalidCode.NEGATIVE_BAND_GAP.value in result.invalid_reasons

    def test_negative_hull_invalid(self) -> None:
        result = PlannerResolver().resolve(
            "q", _draft(hull_max=-0.1, hull_unit=HullUnit.EV_PER_ATOM)
        )
        assert result.status is PlannerStatus.INVALID
        assert InvalidCode.NEGATIVE_HULL.value in result.invalid_reasons

    def test_zero_density_invalid(self) -> None:
        result = PlannerResolver().resolve(
            "q", _draft(density_min=0, density_unit=DensityUnit.G_PER_CM3)
        )
        assert result.status is PlannerStatus.INVALID
        assert InvalidCode.NON_POSITIVE_DENSITY.value in result.invalid_reasons

    def test_negative_target_invalid(self) -> None:
        result = PlannerResolver().resolve(
            "q",
            _draft(target_band_gap=-0.5, target_band_gap_unit=EnergyUnit.EV),
        )
        assert result.status is PlannerStatus.INVALID
        assert InvalidCode.NEGATIVE_TARGET_BAND_GAP.value in result.invalid_reasons


class TestRangeConflicts:
    def test_band_gap_min_above_max(self) -> None:
        result = PlannerResolver().resolve(
            "q",
            _draft(
                band_gap_min=2.0,
                band_gap_max=1.0,
                band_gap_unit=EnergyUnit.EV,
            ),
        )
        assert result.status is PlannerStatus.INVALID
        assert ConflictCode.BAND_GAP_MIN_ABOVE_MAX.value in result.conflicts

    def test_hull_min_above_max(self) -> None:
        result = PlannerResolver().resolve(
            "q",
            _draft(
                hull_min=0.1,
                hull_max=0.05,
                hull_unit=HullUnit.EV_PER_ATOM,
            ),
        )
        assert result.status is PlannerStatus.INVALID
        assert ConflictCode.HULL_MIN_ABOVE_MAX.value in result.conflicts

    def test_density_min_above_max(self) -> None:
        result = PlannerResolver().resolve(
            "q",
            _draft(
                density_min=5.0,
                density_max=2.0,
                density_unit=DensityUnit.G_PER_CM3,
            ),
        )
        assert result.status is PlannerStatus.INVALID
        assert ConflictCode.DENSITY_MIN_ABOVE_MAX.value in result.conflicts

    def test_target_above_range(self) -> None:
        result = PlannerResolver().resolve(
            "q",
            _draft(
                band_gap_min=1.0,
                band_gap_max=2.0,
                band_gap_unit=EnergyUnit.EV,
                target_band_gap=2.5,
                target_band_gap_unit=EnergyUnit.EV,
            ),
        )
        assert result.status is PlannerStatus.INVALID
        assert ConflictCode.TARGET_OUTSIDE_BAND_GAP.value in result.conflicts

    def test_target_below_range(self) -> None:
        result = PlannerResolver().resolve(
            "q",
            _draft(
                band_gap_min=1.0,
                band_gap_max=2.0,
                band_gap_unit=EnergyUnit.EV,
                target_band_gap=0.5,
                target_band_gap_unit=EnergyUnit.EV,
            ),
        )
        assert result.status is PlannerStatus.INVALID
        assert ConflictCode.TARGET_OUTSIDE_BAND_GAP.value in result.conflicts

    def test_target_within_range_ready(self) -> None:
        result = PlannerResolver().resolve(
            "q",
            _draft(
                band_gap_min=1.0,
                band_gap_max=2.0,
                band_gap_unit=EnergyUnit.EV,
                target_band_gap=1.5,
                target_band_gap_unit=EnergyUnit.EV,
            ),
        )
        assert result.status is PlannerStatus.READY
        assert result.request is not None
        assert result.request.target_band_gap_ev == 1.5


class TestSpacegroupAndCrystalSystem:
    def test_spacegroup_out_of_range_invalid(self) -> None:
        result = PlannerResolver().resolve("q", _draft(spacegroup_numbers=[231]))
        assert result.status is PlannerStatus.INVALID
        assert InvalidCode.INVALID_SPACEGROUP.value in result.invalid_reasons

    def test_spacegroup_normalized(self) -> None:
        result = PlannerResolver().resolve("q", _draft(spacegroup_numbers=[225, 2, 2]))
        assert result.status is PlannerStatus.READY
        assert result.request is not None
        assert result.request.spacegroup_numbers == (2, 225)

    def test_crystal_system_canonical(self) -> None:
        result = PlannerResolver().resolve("q", _draft(crystal_system="Cubic"))
        assert result.status is PlannerStatus.READY
        assert result.request is not None
        assert result.request.crystal_system is CrystalSystem.CUBIC

    def test_crystal_system_invalid(self) -> None:
        result = PlannerResolver().resolve("q", _draft(crystal_system="cubic"))
        assert result.status is PlannerStatus.INVALID
        assert InvalidCode.INVALID_CRYSTAL_SYSTEM.value in result.invalid_reasons


class TestLimit:
    def test_limit_zero_invalid(self) -> None:
        result = PlannerResolver().resolve("q", _draft(limit=0))
        assert result.status is PlannerStatus.INVALID
        assert InvalidCode.INVALID_LIMIT.value in result.invalid_reasons

    def test_limit_above_100_invalid(self) -> None:
        result = PlannerResolver().resolve("q", _draft(limit=101))
        assert result.status is PlannerStatus.INVALID
        assert InvalidCode.INVALID_LIMIT.value in result.invalid_reasons

    def test_limit_defaults_to_10(self) -> None:
        result = PlannerResolver().resolve("q", _draft())
        assert result.status is PlannerStatus.READY
        assert result.request is not None
        assert result.request.limit == 10

    def test_limit_kept(self) -> None:
        result = PlannerResolver().resolve("q", _draft(limit=5))
        assert result.request is not None
        assert result.request.limit == 5


class TestStabilityRules:
    def test_stable_maps_to_is_stable(self) -> None:
        result = PlannerResolver().resolve("稳定材料", _draft(is_stable=True))
        assert result.status is PlannerStatus.READY
        assert result.request is not None
        assert result.request.is_stable is True
        assert StabilityRule.STABLE_AS_IS_STABLE.value in result.assumptions

    def test_stable_does_not_imply_hull_zero(self) -> None:
        result = PlannerResolver().resolve(
            "稳定材料",
            _draft(is_stable=True, hull_max=0, hull_unit=HullUnit.EV_PER_ATOM),
        )
        assert result.status is PlannerStatus.READY
        assert result.request is not None
        assert result.request.energy_above_hull_ev_atom is None
        assert AmbiguityCode.STABLE_NOT_HULL_ZERO.value in result.ambiguities

    def test_hull_zero_with_evidence_kept(self) -> None:
        result = PlannerResolver().resolve(
            "稳定材料",
            _draft(
                is_stable=True,
                hull_max=0,
                hull_unit=HullUnit.EV_PER_ATOM,
                evidence=[EvidenceItem(quote="能量高于凸包为 0")],
            ),
        )
        assert result.status is PlannerStatus.READY
        assert result.request is not None
        assert result.request.energy_above_hull_ev_atom is not None
        assert result.request.energy_above_hull_ev_atom.max == 0


class TestAmbiguityKeywords:
    def test_application_goal_not_converted_to_threshold(self) -> None:
        result = PlannerResolver().resolve(
            "适合光伏的材料", _draft(band_gap_min=1.2, band_gap_unit=EnergyUnit.EV)
        )
        assert result.status is PlannerStatus.READY
        assert AmbiguityCode.APPLICATION_GOAL.value in result.ambiguities
        assert result.request is not None
        assert result.request.band_gap_ev is not None
        assert result.request.band_gap_ev.min == 1.2

    def test_toxic_or_rare_not_expanded(self) -> None:
        result = PlannerResolver().resolve(
            "不含有毒元素", _draft(excluded_elements=["Pb"])
        )
        assert result.status is PlannerStatus.READY
        assert AmbiguityCode.TOXIC_OR_RARE_ELEMENTS.value in result.ambiguities
        assert result.request is not None
        assert result.request.excluded_elements == ("Pb",)


class TestUnsupported:
    def test_predict_materials_unsupported(self) -> None:
        result = PlannerResolver().resolve("预测新材料", _draft())
        assert result.status is PlannerStatus.UNSUPPORTED
        assert (
            UnsupportedCode.PREDICT_MATERIALS.value in result.unsupported_requirements
        )

    def test_dft_computation_unsupported(self) -> None:
        result = PlannerResolver().resolve("帮我跑DFT计算", _draft())
        assert result.status is PlannerStatus.UNSUPPORTED
        assert UnsupportedCode.DFT_COMPUTATION.value in result.unsupported_requirements

    def test_synthesis_advice_unsupported(self) -> None:
        result = PlannerResolver().resolve("如何合成该材料", _draft())
        assert result.status is PlannerStatus.UNSUPPORTED
        assert UnsupportedCode.SYNTHESIS_ADVICE.value in result.unsupported_requirements

    def test_llm_flagged_unsupported(self) -> None:
        result = PlannerResolver().resolve(
            "q", _draft(unsupported_requirements=["需要预测属性"])
        )
        assert result.status is PlannerStatus.UNSUPPORTED
        assert UnsupportedCode.LLM_FLAGGED.value in result.unsupported_requirements

    def test_application_goal_downgrades_llm_flagged_unsupported(self) -> None:
        result = PlannerResolver().resolve(
            "光伏材料", _draft(unsupported_requirements=["光伏任务"])
        )
        assert result.status is PlannerStatus.READY
        assert AmbiguityCode.APPLICATION_GOAL.value in result.ambiguities
        assert result.unsupported_requirements == ()

    def test_application_goal_keeps_keyword_unsupported(self) -> None:
        result = PlannerResolver().resolve(
            "预测光伏新材料", _draft(unsupported_requirements=["预测新材料"])
        )
        assert result.status is PlannerStatus.UNSUPPORTED
        assert (
            UnsupportedCode.PREDICT_MATERIALS.value in result.unsupported_requirements
        )


class TestStatusPriority:
    def test_invalid_beats_unsupported(self) -> None:
        result = PlannerResolver().resolve(
            "预测新材料", _draft(required_elements=["Xx"])
        )
        assert result.status is PlannerStatus.INVALID

    def test_unsupported_beats_clarification(self) -> None:
        result = PlannerResolver().resolve(
            "预测新材料",
            _draft(band_gap_min=5000, band_gap_unit=EnergyUnit.UNSPECIFIED),
        )
        assert result.status is PlannerStatus.UNSUPPORTED

    def test_clarification_beats_ready(self) -> None:
        result = PlannerResolver().resolve(
            "q",
            _draft(band_gap_min=5000, band_gap_unit=EnergyUnit.UNSPECIFIED),
        )
        assert result.status is PlannerStatus.NEEDS_CLARIFICATION


class TestLLMFlagged:
    def test_llm_flagged_conflict_invalid(self) -> None:
        result = PlannerResolver().resolve(
            "q", _draft(conflicts=["带隙与金属条件矛盾"])
        )
        assert result.status is PlannerStatus.INVALID
        assert ConflictCode.LLM_FLAGGED.value in result.conflicts

    def test_llm_flagged_ambiguity_needs_clarification(self) -> None:
        result = PlannerResolver().resolve("q", _draft(ambiguities=["单位不明确"]))
        assert result.status is PlannerStatus.NEEDS_CLARIFICATION
        assert AmbiguityCode.LLM_FLAGGED.value in result.ambiguities


class TestDeterminism:
    def test_same_input_same_result(self) -> None:
        query = "寻找不含 Pb 的非金属材料"
        draft = _full_draft()
        first = PlannerResolver().resolve(query, draft)
        second = PlannerResolver().resolve(query, draft)
        assert first == second
        assert first.model_dump(mode="json") == second.model_dump(mode="json")
