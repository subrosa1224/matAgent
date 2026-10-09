"""Deterministic, explainable ranking service (M4)."""

import math
from collections.abc import Sequence

from materials_screening.models import (
    MaterialRecord,
    RankedMaterial,
    ScoreBreakdown,
    ScreeningRequest,
)

STABILITY_WEIGHT = 0.45
BAND_GAP_WEIGHT = 0.40
COMPLETENESS_WEIGHT = 0.10
DIRECT_GAP_WEIGHT = 0.05

_PRECISION = 8
_HULL_ZERO_TOLERANCE = 1e-8
_HULL_EXP_SCALE = 0.05
_BAND_GAP_TOLERANCE_FLOOR = 0.25


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _round8(value: float) -> float:
    return round(value, _PRECISION)


def _stability_score(record: MaterialRecord, request: ScreeningRequest) -> float:
    hull = record.energy_above_hull_ev_atom
    if hull is None:
        return 0.0
    hull_range = request.energy_above_hull_ev_atom
    if hull_range is not None and hull_range.max is not None:
        upper = hull_range.max
        if upper == 0.0:
            return 1.0 if hull <= _HULL_ZERO_TOLERANCE else 0.0
        return _clamp01(1.0 - hull / upper)
    return math.exp(-hull / _HULL_EXP_SCALE)


def _band_gap_target(request: ScreeningRequest) -> float | None:
    """Select the band-gap target using the documented priority order."""
    if request.target_band_gap_ev is not None:
        return request.target_band_gap_ev
    band_range = request.band_gap_ev
    if band_range is None:
        return None
    if band_range.min is not None and band_range.max is not None:
        return (band_range.min + band_range.max) / 2.0
    if band_range.min is not None:
        return band_range.min
    if band_range.max is not None:
        return band_range.max
    return None


def _band_gap_match(
    record: MaterialRecord,
    request: ScreeningRequest,
    target: float | None,
) -> float:
    if target is None:
        return 0.5
    band_gap = record.band_gap_ev
    if band_gap is None:
        return 0.0
    band_range = request.band_gap_ev
    width = 0.0
    if (
        band_range is not None
        and band_range.min is not None
        and band_range.max is not None
    ):
        width = band_range.max - band_range.min
    tolerance = max(width / 2.0, _BAND_GAP_TOLERANCE_FLOOR)
    return _clamp01(max(0.0, 1.0 - abs(band_gap - target) / tolerance))


def _completeness_score(record: MaterialRecord) -> float:
    fields = (
        record.band_gap_ev,
        record.energy_above_hull_ev_atom,
        record.formation_energy_ev_atom,
        record.density_g_cm3,
        record.symmetry,
        record.structure_dict,
    )
    present_count = sum(1 for value in fields if value is not None)
    return present_count / 6.0


def _direct_gap_score(record: MaterialRecord) -> float:
    return 1.0 if record.is_gap_direct is True else 0.0


def _build_breakdown(
    record: MaterialRecord,
    request: ScreeningRequest,
    target: float | None,
) -> ScoreBreakdown:
    stability = _round8(_stability_score(record, request))
    band_gap_match = _round8(_band_gap_match(record, request, target))
    completeness = _round8(_completeness_score(record))
    direct_gap = _round8(_direct_gap_score(record))
    return ScoreBreakdown(
        stability=stability,
        band_gap_match=band_gap_match,
        completeness=completeness,
        direct_gap=direct_gap,
        weighted_stability=_round8(STABILITY_WEIGHT * stability),
        weighted_band_gap_match=_round8(BAND_GAP_WEIGHT * band_gap_match),
        weighted_completeness=_round8(COMPLETENESS_WEIGHT * completeness),
        weighted_direct_gap=_round8(DIRECT_GAP_WEIGHT * direct_gap),
    )


def _total_score(breakdown: ScoreBreakdown) -> float:
    raw = (
        breakdown.weighted_stability
        + breakdown.weighted_band_gap_match
        + breakdown.weighted_completeness
        + breakdown.weighted_direct_gap
    )
    return _clamp01(_round8(raw))


def _gap_sort_key(record: MaterialRecord, target: float | None) -> float:
    if target is None:
        return 0.0
    if record.band_gap_ev is None:
        return math.inf
    return abs(record.band_gap_ev - target)


def _sort_key(
    item: tuple[MaterialRecord, ScoreBreakdown],
    target: float | None,
) -> tuple[float, float, float, str, str]:
    record, breakdown = item
    hull = record.energy_above_hull_ev_atom
    hull_key = hull if hull is not None else math.inf
    return (
        -_total_score(breakdown),
        hull_key,
        _gap_sort_key(record, target),
        record.material_id,
        record.source,
    )


class RankingService:
    """Pure, stateless ranking service with deterministic tie-breaking."""

    def score(
        self,
        record: MaterialRecord,
        request: ScreeningRequest,
    ) -> tuple[float, ScoreBreakdown]:
        """Return (total_score, breakdown) for a single record."""
        target = _band_gap_target(request)
        breakdown = _build_breakdown(record, request, target)
        return _total_score(breakdown), breakdown

    def rank(
        self,
        records: Sequence[MaterialRecord],
        request: ScreeningRequest,
    ) -> tuple[RankedMaterial, ...]:
        """Rank all records; the caller decides truncation."""
        target = _band_gap_target(request)
        scored: list[tuple[MaterialRecord, ScoreBreakdown]] = [
            (record, _build_breakdown(record, request, target)) for record in records
        ]
        scored.sort(key=lambda item: _sort_key(item, target))
        return tuple(
            RankedMaterial(
                record=record,
                rank=index,
                total_score=_total_score(breakdown),
                score_breakdown=breakdown,
            )
            for index, (record, breakdown) in enumerate(scored, start=1)
        )
