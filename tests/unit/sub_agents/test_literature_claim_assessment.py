from __future__ import annotations

from materials_screening.sub_agents.literature.claim_assessment import (
    assess_claim_against_comparison,
    assess_claim_against_measurement,
)
from materials_screening.sub_agents.literature.comparison import calculate_comparison
from materials_screening.sub_agents.literature.models import (
    ExperimentalMeasurement,
    PaperClaim,
)


def _measurement(
    measurement_id: str, group_id: str, value: float
) -> ExperimentalMeasurement:
    return ExperimentalMeasurement(
        measurement_id=measurement_id,
        group_id=group_id,
        document_id="doc-1",
        metric="photoconversion efficiency",
        value_text=str(value),
        numeric_value=value,
        unit="%",
        source_quote=f"efficiency {value}%",
        chunk_id="chunk-results",
        page_from=6,
        page_to=6,
        source_text_sha256="a" * 64,
    )


def _claim(text: str) -> PaperClaim:
    return PaperClaim(
        claim_id="claim-1",
        document_id="doc-1",
        claim_text=text,
        source_section="abstract",
        source_quote=text,
        chunk_id="chunk-abstract",
        page_from=1,
        page_to=1,
        source_text_sha256="b" * 64,
    )


def test_improvement_claim_is_supported_by_increasing_comparison() -> None:
    comparison = calculate_comparison(
        _measurement("m0", "g0", 1.1), _measurement("m1", "g1", 1.3)
    )
    link = assess_claim_against_comparison(
        _claim("Nb doping improved photoconversion efficiency."), comparison
    )
    assert link.assessment == "supported"


def test_enhancement_without_metric_is_not_overinterpreted() -> None:
    comparison = calculate_comparison(
        _measurement("m0", "baseline", 1.0),
        _measurement("m1", "target", 1.5),
    )
    link = assess_claim_against_comparison(
        _claim("The enhancement was observed relative to undoped TiO2."), comparison
    )

    assert link.assessment == "not_verifiable"
    assert link.assessment_method == "rule"


def test_improvement_claim_is_contradicted_by_decreasing_comparison() -> None:
    comparison = calculate_comparison(
        _measurement("m0", "g0", 1.1), _measurement("m5", "g5", 0.87)
    )
    link = assess_claim_against_comparison(
        _claim("Nb doping improved photoconversion efficiency."), comparison
    )
    assert link.assessment == "contradicted"


def test_claim_without_direction_is_not_overinterpreted() -> None:
    comparison = calculate_comparison(
        _measurement("m0", "g0", 1.1), _measurement("m1", "g1", 1.3)
    )
    link = assess_claim_against_comparison(
        _claim("Photoconversion efficiency was measured."), comparison
    )
    assert link.assessment == "not_verifiable"


def test_direction_does_not_support_an_unmentioned_metric() -> None:
    comparison = calculate_comparison(
        _measurement("m0", "g0", 1.0), _measurement("m1", "g1", 1.5)
    )
    link = assess_claim_against_comparison(
        _claim("The treatment improved porosity."), comparison
    )

    assert link.assessment == "not_verifiable"


def test_exact_metric_value_in_claim_is_supported() -> None:
    measurement = _measurement("m1", "g1", 20.8)
    claim = _claim("The power conversion efficiency reached 20.8%.")
    link = assess_claim_against_measurement(claim, measurement)
    assert link.assessment == "supported"
    assert link.evidence_type == "measurement"


def test_metric_abbreviation_matches_expanded_abstract_term() -> None:
    measurement = _measurement("m-pce", "g1", 20.8).model_copy(update={"metric": "PCE"})
    claim = _claim("The power-conversion efficiency reached 20.8%.")
    assert (
        assess_claim_against_measurement(claim, measurement).assessment == "supported"
    )
