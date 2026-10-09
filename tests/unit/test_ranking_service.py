"""Unit tests for RankingService (M4)."""

import math

import pytest

from materials_screening.models import (
    CrystalSystem,
    FloatRange,
    MaterialRecord,
    ScoreBreakdown,
    ScreeningRequest,
    SymmetryInfo,
)
from materials_screening.services.ranking_service import (
    BAND_GAP_WEIGHT,
    COMPLETENESS_WEIGHT,
    DIRECT_GAP_WEIGHT,
    STABILITY_WEIGHT,
    RankingService,
    _band_gap_target,
    _sort_key,
)


def _record(**overrides: object) -> MaterialRecord:
    values: dict[str, object] = {
        "source": "materials_project",
        "material_id": "mp-1",
        "formula_pretty": "Fe2O3",
        "elements": ("Fe", "O"),
        "chemsys": "Fe-O",
        "band_gap_ev": 2.0,
        "energy_above_hull_ev_atom": 0.01,
        "formation_energy_ev_atom": -1.5,
        "density_g_cm3": 5.2,
        "is_metal": False,
        "is_gap_direct": True,
        "is_stable": True,
        "theoretical": False,
        "deprecated": False,
        "symmetry": SymmetryInfo(
            crystal_system=CrystalSystem.TRIGONAL,
            symbol="R-3c",
            number=167,
        ),
        "structure_dict": None,
        "provenance": (),
    }
    values.update(overrides)
    return MaterialRecord.model_validate(values)


def _empty_breakdown() -> ScoreBreakdown:
    return ScoreBreakdown(
        stability=0.0,
        band_gap_match=0.0,
        completeness=0.0,
        direct_gap=0.0,
        weighted_stability=0.0,
        weighted_band_gap_match=0.0,
        weighted_completeness=0.0,
        weighted_direct_gap=0.0,
    )


class TestWeights:
    def test_weights_sum_to_one(self) -> None:
        total = (
            STABILITY_WEIGHT + BAND_GAP_WEIGHT + COMPLETENESS_WEIGHT + DIRECT_GAP_WEIGHT
        )
        assert math.isclose(total, 1.0)


class TestBandGapTarget:
    def test_explicit_target_wins(self) -> None:
        request = ScreeningRequest(
            band_gap_ev=FloatRange(min=1.2, max=2.0),
            target_band_gap_ev=1.6,
        )
        assert _band_gap_target(request) == 1.6

    def test_midpoint_when_both_bounds(self) -> None:
        request = ScreeningRequest(band_gap_ev=FloatRange(min=1.2, max=2.0))
        assert _band_gap_target(request) == 1.6

    def test_min_bound_when_only_min(self) -> None:
        request = ScreeningRequest(band_gap_ev=FloatRange(min=1.2))
        assert _band_gap_target(request) == 1.2

    def test_max_bound_when_only_max(self) -> None:
        request = ScreeningRequest(band_gap_ev=FloatRange(max=2.0))
        assert _band_gap_target(request) == 2.0

    def test_no_target_without_range(self) -> None:
        assert _band_gap_target(ScreeningRequest()) is None


class TestStabilityScore:
    def test_hull_zero_with_upper_bound(self) -> None:
        request = ScreeningRequest(
            energy_above_hull_ev_atom=FloatRange(min=0.0, max=0.05)
        )
        ranked = RankingService().rank(
            (_record(energy_above_hull_ev_atom=0.0),), request
        )
        assert ranked[0].score_breakdown.stability == 1.0

    def test_hull_upper_bound_scaled(self) -> None:
        request = ScreeningRequest(
            energy_above_hull_ev_atom=FloatRange(min=0.0, max=0.1)
        )
        ranked = RankingService().rank(
            (_record(energy_above_hull_ev_atom=0.05),), request
        )
        assert ranked[0].score_breakdown.stability == 0.5

    def test_hull_above_upper_bound_clamps_to_zero(self) -> None:
        request = ScreeningRequest(
            energy_above_hull_ev_atom=FloatRange(min=0.0, max=0.1)
        )
        ranked = RankingService().rank(
            (_record(energy_above_hull_ev_atom=0.5),), request
        )
        assert ranked[0].score_breakdown.stability == 0.0

    def test_zero_upper_bound_special_case(self) -> None:
        request = ScreeningRequest(
            energy_above_hull_ev_atom=FloatRange(min=0.0, max=0.0)
        )
        records = (
            _record(material_id="mp-a", energy_above_hull_ev_atom=0.0),
            _record(material_id="mp-b", energy_above_hull_ev_atom=1e-8),
            _record(material_id="mp-c", energy_above_hull_ev_atom=0.01),
        )
        ranked = RankingService().rank(records, request)
        scores = {
            item.record.material_id: item.score_breakdown.stability for item in ranked
        }
        assert scores["mp-a"] == 1.0
        assert scores["mp-b"] == 1.0
        assert scores["mp-c"] == 0.0

    def test_no_hull_condition_uses_exponential(self) -> None:
        request = ScreeningRequest()
        records = (
            _record(material_id="mp-a", energy_above_hull_ev_atom=0.0),
            _record(material_id="mp-b", energy_above_hull_ev_atom=0.05),
        )
        ranked = RankingService().rank(records, request)
        scores = {
            item.record.material_id: item.score_breakdown.stability for item in ranked
        }
        assert scores["mp-a"] == 1.0
        assert math.isclose(scores["mp-b"], math.exp(-1.0), abs_tol=1e-8)

    def test_missing_hull_scores_zero(self) -> None:
        request = ScreeningRequest()
        ranked = RankingService().rank(
            (_record(energy_above_hull_ev_atom=None),), request
        )
        assert ranked[0].score_breakdown.stability == 0.0


class TestBandGapMatch:
    def test_no_user_target_scores_half(self) -> None:
        ranked = RankingService().rank((_record(),), ScreeningRequest())
        assert ranked[0].score_breakdown.band_gap_match == 0.5

    def test_exact_target_scores_one(self) -> None:
        request = ScreeningRequest(
            band_gap_ev=FloatRange(min=1.2, max=2.0),
            target_band_gap_ev=2.0,
        )
        ranked = RankingService().rank((_record(band_gap_ev=2.0),), request)
        assert ranked[0].score_breakdown.band_gap_match == 1.0

    def test_at_tolerance_edge_scores_zero(self) -> None:
        request = ScreeningRequest(
            band_gap_ev=FloatRange(min=1.2, max=2.0),
            target_band_gap_ev=1.6,
        )
        ranked = RankingService().rank((_record(band_gap_ev=1.2),), request)
        assert ranked[0].score_breakdown.band_gap_match == 0.0

    def test_missing_band_gap_with_target_scores_zero(self) -> None:
        request = ScreeningRequest(target_band_gap_ev=1.6)
        ranked = RankingService().rank((_record(band_gap_ev=None),), request)
        assert ranked[0].score_breakdown.band_gap_match == 0.0

    def test_open_range_uses_floor_tolerance(self) -> None:
        request = ScreeningRequest(band_gap_ev=FloatRange(min=1.2))
        ranked = RankingService().rank((_record(band_gap_ev=1.45),), request)
        assert math.isclose(ranked[0].score_breakdown.band_gap_match, 0.0)


class TestCompletenessScore:
    def test_all_present_scores_one(self) -> None:
        record = _record(structure_dict={"lattice": {"a": 1.0}, "sites": []})
        ranked = RankingService().rank((record,), ScreeningRequest())
        assert ranked[0].score_breakdown.completeness == 1.0

    def test_missing_structure_scores_five_sixths(self) -> None:
        ranked = RankingService().rank((_record(),), ScreeningRequest())
        assert ranked[0].score_breakdown.completeness == round(5 / 6, 8)

    def test_minimal_record_scores_zero(self) -> None:
        record = _record(
            band_gap_ev=None,
            energy_above_hull_ev_atom=None,
            formation_energy_ev_atom=None,
            density_g_cm3=None,
            symmetry=None,
            structure_dict=None,
        )
        ranked = RankingService().rank((record,), ScreeningRequest())
        assert ranked[0].score_breakdown.completeness == 0.0


@pytest.mark.parametrize(
    ("is_gap_direct", "expected"),
    [(True, 1.0), (False, 0.0), (None, 0.0)],
)
def test_direct_gap_score(is_gap_direct: bool | None, expected: float) -> None:
    ranked = RankingService().rank(
        (_record(is_gap_direct=is_gap_direct),), ScreeningRequest()
    )
    assert ranked[0].score_breakdown.direct_gap == expected


class TestTotalScore:
    def test_total_matches_weighted_sum_with_precision(self) -> None:
        request = ScreeningRequest(
            energy_above_hull_ev_atom=FloatRange(min=0.0, max=0.1),
            band_gap_ev=FloatRange(min=1.2, max=2.0),
            target_band_gap_ev=2.0,
        )
        record = _record(
            energy_above_hull_ev_atom=0.05,
            band_gap_ev=2.0,
            is_gap_direct=True,
        )
        ranked = RankingService().rank((record,), request)
        breakdown = ranked[0].score_breakdown
        expected = round(
            STABILITY_WEIGHT * 0.5
            + BAND_GAP_WEIGHT * 1.0
            + COMPLETENESS_WEIGHT * round(5 / 6, 8)
            + DIRECT_GAP_WEIGHT * 1.0,
            8,
        )
        assert breakdown.weighted_stability == round(0.45 * 0.5, 8)
        assert breakdown.weighted_band_gap_match == round(0.4 * 1.0, 8)
        assert ranked[0].total_score == expected

    def test_scores_are_bounded_zero_to_one(self) -> None:
        records = (
            _record(material_id="mp-a", energy_above_hull_ev_atom=0.0),
            _record(material_id="mp-b", energy_above_hull_ev_atom=5.0),
            _record(
                material_id="mp-c",
                band_gap_ev=None,
                energy_above_hull_ev_atom=None,
                density_g_cm3=None,
                symmetry=None,
                is_gap_direct=None,
            ),
        )
        ranked = RankingService().rank(records, ScreeningRequest())
        for item in ranked:
            breakdown = item.score_breakdown
            for value in (
                breakdown.stability,
                breakdown.band_gap_match,
                breakdown.completeness,
                breakdown.direct_gap,
                breakdown.weighted_stability,
                breakdown.weighted_band_gap_match,
                breakdown.weighted_completeness,
                breakdown.weighted_direct_gap,
                item.total_score,
            ):
                assert 0.0 <= value <= 1.0

    def test_no_nan_or_inf_in_breakdown_or_total(self) -> None:
        records = (
            _record(material_id="mp-a", energy_above_hull_ev_atom=1e9),
            _record(material_id="mp-b", band_gap_ev=1e9),
            _record(material_id="mp-c", energy_above_hull_ev_atom=None),
            _record(material_id="mp-d", band_gap_ev=None),
        )
        request = ScreeningRequest(
            energy_above_hull_ev_atom=FloatRange(min=0.0, max=0.1),
            band_gap_ev=FloatRange(min=1.2, max=2.0),
            target_band_gap_ev=1.6,
        )
        ranked = RankingService().rank(records, request)
        for item in ranked:
            breakdown = item.score_breakdown
            for value in (
                breakdown.stability,
                breakdown.band_gap_match,
                breakdown.completeness,
                breakdown.direct_gap,
                breakdown.weighted_stability,
                breakdown.weighted_band_gap_match,
                breakdown.weighted_completeness,
                breakdown.weighted_direct_gap,
                item.total_score,
            ):
                assert math.isfinite(value)


class TestScoreConsistency:
    def test_total_score_equals_weighted_breakdown_sum(self) -> None:
        screening_requests = (
            ScreeningRequest(),
            ScreeningRequest(
                band_gap_ev=FloatRange(min=1.2, max=2.0),
                target_band_gap_ev=1.6,
            ),
            ScreeningRequest(energy_above_hull_ev_atom=FloatRange(min=0.0, max=0.05)),
        )
        records = (
            _record(material_id="mp-a"),
            _record(
                material_id="mp-b",
                band_gap_ev=None,
                structure_dict={"lattice": {"a": 1.0}, "sites": []},
            ),
            _record(
                material_id="mp-c",
                energy_above_hull_ev_atom=None,
                is_gap_direct=None,
                density_g_cm3=None,
                symmetry=None,
            ),
        )
        for request in screening_requests:
            ranked = RankingService().rank(records, request)
            for item in ranked:
                breakdown = item.score_breakdown
                assert item.total_score == round(
                    breakdown.weighted_stability
                    + breakdown.weighted_band_gap_match
                    + breakdown.weighted_completeness
                    + breakdown.weighted_direct_gap,
                    8,
                )
                for value in (
                    breakdown.stability,
                    breakdown.band_gap_match,
                    breakdown.completeness,
                    breakdown.direct_gap,
                    breakdown.weighted_stability,
                    breakdown.weighted_band_gap_match,
                    breakdown.weighted_completeness,
                    breakdown.weighted_direct_gap,
                ):
                    assert value == round(value, 8)
                assert breakdown.weighted_stability == round(
                    STABILITY_WEIGHT * breakdown.stability, 8
                )
                assert breakdown.weighted_band_gap_match == round(
                    BAND_GAP_WEIGHT * breakdown.band_gap_match, 8
                )
                assert breakdown.weighted_completeness == round(
                    COMPLETENESS_WEIGHT * breakdown.completeness, 8
                )
                assert breakdown.weighted_direct_gap == round(
                    DIRECT_GAP_WEIGHT * breakdown.direct_gap, 8
                )


class TestTieBreak:
    def test_material_id_breaks_final_tie(self) -> None:
        records = (
            _record(material_id="mp-b"),
            _record(material_id="mp-a"),
        )
        ranked = RankingService().rank(records, ScreeningRequest())
        assert [item.record.material_id for item in ranked] == ["mp-a", "mp-b"]

    def test_sort_key_prefers_lower_hull_on_tie(self) -> None:
        breakdown = _empty_breakdown()
        high_hull = _record(material_id="mp-a", energy_above_hull_ev_atom=0.1)
        low_hull = _record(material_id="mp-b", energy_above_hull_ev_atom=0.0)
        assert _sort_key((low_hull, breakdown), None) < _sort_key(
            (high_hull, breakdown), None
        )

    def test_sort_key_prefers_gap_closer_to_target_on_tie(self) -> None:
        breakdown = _empty_breakdown()
        closer = _record(material_id="mp-a", band_gap_ev=1.4)
        farther = _record(material_id="mp-b", band_gap_ev=2.4)
        assert _sort_key((closer, breakdown), 1.6) < _sort_key(
            (farther, breakdown), 1.6
        )

    def test_higher_total_score_ranks_first(self) -> None:
        request = ScreeningRequest(
            band_gap_ev=FloatRange(min=1.2, max=2.0),
            target_band_gap_ev=2.0,
        )
        records = (
            _record(material_id="mp-low", band_gap_ev=1.2),
            _record(material_id="mp-high", band_gap_ev=2.0),
        )
        ranked = RankingService().rank(records, request)
        assert ranked[0].record.material_id == "mp-high"


class TestDeterminism:
    def test_ranking_is_repeatable(self) -> None:
        records = (
            _record(material_id="mp-1", band_gap_ev=1.3),
            _record(material_id="mp-2", band_gap_ev=2.3),
            _record(material_id="mp-3", band_gap_ev=1.8),
        )
        request = ScreeningRequest(
            band_gap_ev=FloatRange(min=1.2, max=2.4),
            target_band_gap_ev=1.6,
        )
        first = RankingService().rank(records, request)
        second = RankingService().rank(records, request)
        assert [
            (item.record.material_id, item.rank, item.total_score) for item in first
        ] == [(item.record.material_id, item.rank, item.total_score) for item in second]

    def test_input_order_does_not_change_final_order(self) -> None:
        records = (
            _record(material_id="mp-1", band_gap_ev=1.3),
            _record(material_id="mp-2", band_gap_ev=2.3),
            _record(material_id="mp-3", band_gap_ev=1.8),
            _record(material_id="mp-4", band_gap_ev=1.5),
        )
        request = ScreeningRequest(
            band_gap_ev=FloatRange(min=1.2, max=2.4),
            target_band_gap_ev=1.6,
        )
        first = RankingService().rank(records, request)
        second = RankingService().rank(tuple(reversed(records)), request)
        assert [item.record.material_id for item in first] == [
            item.record.material_id for item in second
        ]
        assert [item.rank for item in first] == list(range(1, len(records) + 1))

    def test_does_not_truncate_to_request_limit(self) -> None:
        records = tuple(_record(material_id=f"mp-{index}") for index in range(5))
        request = ScreeningRequest(limit=2)
        ranked = RankingService().rank(records, request)
        assert len(ranked) == 5
