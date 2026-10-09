from __future__ import annotations

from materials_screening.sub_agents.literature.matrix_automation import (
    PendingMatrixExtraction,
)
from materials_screening.sub_agents.literature.matrix_evaluation import (
    _metric,
    evaluate_matrix,
    summarize_matrix_evaluations,
)
from materials_screening.sub_agents.literature.models import (
    ExperimentalGroup,
    ExperimentalMeasurement,
)


def _group(group_id: str, label: str, status: str = "pending") -> ExperimentalGroup:
    return ExperimentalGroup.model_validate(
        {
            "group_id": group_id,
            "document_id": "doc-1",
            "label": label,
            "role": "treatment",
            "material": "TiO2",
            "variables": {"Nb": "1.5 mol%"},
            "source_quote": "Nb 1.5 mol% TiO2",
            "chunk_id": "chunk-1",
            "page_from": 1,
            "page_to": 1,
            "source_text_sha256": "a" * 64,
            "review_status": status,
        }
    )


def _measurement(
    measurement_id: str,
    group_id: str,
    *,
    unit: str = "%",
    status: str = "pending",
) -> ExperimentalMeasurement:
    return ExperimentalMeasurement.model_validate(
        {
            "measurement_id": measurement_id,
            "group_id": group_id,
            "document_id": "doc-1",
            "metric": "PCE",
            "value_text": "1.3",
            "numeric_value": 1.3,
            "unit": unit,
            "source_quote": "PCE 1.3 %",
            "chunk_id": "chunk-1",
            "page_from": 1,
            "page_to": 1,
            "source_text_sha256": "a" * 64,
            "review_status": status,
        }
    )


def _automatic(unit: str = "%") -> PendingMatrixExtraction:
    return PendingMatrixExtraction(
        groups=(_group("predicted", "Nb 1.5"),),
        measurements=(_measurement("predicted-m", "predicted", unit=unit),),
        comparisons=(),
        claims=(),
        claim_evidence_links=(),
        warnings=(),
    )


def test_exact_value_and_unit_pass_all_metrics() -> None:
    result = evaluate_matrix(
        _automatic(),
        document_id="doc-1",
        gold_groups=(_group("gold", "Nb 1.5", "approved"),),
        gold_measurements=(_measurement("gold-m", "gold", status="approved"),),
    )

    assert result.numeric_precision == 1.0
    assert result.key_group_recall == 1.0
    assert result.unit_fidelity == 1.0
    assert result.approved_automatic_records == 0
    assert summarize_matrix_evaluations((result,)).passed is True


def test_wrong_unit_preserves_numeric_precision_but_fails_unit_fidelity() -> None:
    result = evaluate_matrix(
        _automatic("V"),
        document_id="doc-1",
        gold_groups=(_group("gold", "Nb 1.5", "approved"),),
        gold_measurements=(_measurement("gold-m", "gold", status="approved"),),
    )

    assert result.numeric_precision == 1.0
    assert result.unit_fidelity == 0.0
    assert summarize_matrix_evaluations((result,)).passed is False


def test_metric_aliases_are_normalized_without_changing_source_data() -> None:
    assert _metric("Short-circuit current density (Jsc)") == "jsc"
    assert _metric("Jsc") == "jsc"
    assert (
        _metric("Photovoltaic conversion efficiency")
        == "photoconversion efficiency"
    )
    assert _metric("Optical bandgap (Eg)") == "optical bandgap"


def test_human_adjudication_scores_approved_extra_and_ignores_rejected_extra() -> None:
    approved = _measurement("approved-extra", "predicted")
    approved = approved.model_copy(update={"metric": "COL-I timing"})
    rejected = _measurement("rejected-extra", "predicted")
    rejected = rejected.model_copy(update={"metric": "unsupported timing"})
    automatic = PendingMatrixExtraction(
        groups=(_group("predicted", "Nb 1.5"),),
        measurements=(approved, rejected),
        comparisons=(),
        claims=(),
        claim_evidence_links=(),
        warnings=(),
    )

    result = evaluate_matrix(
        automatic,
        document_id="doc-1",
        gold_groups=(_group("gold", "Nb 1.5", "approved"),),
        gold_measurements=(),
        adjudications={
            "approved-extra": "approved",
            "rejected-extra": "rejected",
        },
    )

    assert result.scored_measurements == 1
    assert result.matched_measurements == 1
    assert result.unadjudicated_measurements == 0
    assert result.numeric_precision == 1.0
    assert result.unit_fidelity == 1.0
