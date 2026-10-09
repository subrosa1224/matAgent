from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from materials_screening.data_analysis.models import (
    AnalysisResult,
    ColumnProfile,
    DatasetReference,
    DatasetTransformOperation,
    DatasetTransformRecord,
)
from materials_screening.master.application_contracts import (
    MultiAgentArtifact,
    UnifiedResultEnvelope,
)

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64


def _dataset(**updates: object) -> DatasetReference:
    values: dict[str, object] = {
        "dataset_id": "dataset-source-1",
        "source_artifact_id": "artifact-data-1",
        "display_name": "实验数据.csv",
        "format": "csv",
        "row_count": 12,
        "column_count": 3,
        "schema_fingerprint": _HASH_A,
        "content_fingerprint": _HASH_B,
    }
    values.update(updates)
    return DatasetReference.model_validate(values)


def test_dataset_reference_is_strict_frozen_and_json_round_trips() -> None:
    dataset = _dataset()

    restored = DatasetReference.model_validate_json(dataset.model_dump_json())

    assert restored == dataset
    with pytest.raises(ValidationError):
        DatasetReference.model_validate({**dataset.model_dump(), "path": "secret.csv"})
    with pytest.raises(ValidationError):
        dataset.row_count = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("parent", "operation"),
    [("dataset-parent-1", None), (None, "operation-clean-1")],
)
def test_dataset_reference_requires_complete_derivation_link(
    parent: str | None, operation: str | None
) -> None:
    with pytest.raises(ValidationError, match="require both"):
        _dataset(parent_dataset_id=parent, created_by_operation_id=operation)


def test_dataset_reference_rejects_unsafe_ids_and_self_parent() -> None:
    with pytest.raises(ValidationError):
        _dataset(dataset_id="../../dataset")
    with pytest.raises(ValidationError, match="own parent"):
        _dataset(
            parent_dataset_id="dataset-source-1",
            created_by_operation_id="operation-clean-1",
        )


def test_column_profile_bounds_samples_and_rejects_non_finite_values() -> None:
    profile = ColumnProfile(
        name="conductivity",
        inferred_type="numeric",
        non_null_count=10,
        missing_count=2,
        unique_count=9,
        sample_values=(1.2, 2.3),
        warnings=("存在缺失值",),
    )
    assert profile.sample_values == (1.2, 2.3)

    with pytest.raises(ValidationError):
        ColumnProfile.model_validate(
            {**profile.model_dump(), "sample_values": tuple(range(11))}
        )
    with pytest.raises(ValidationError, match="NaN or infinity"):
        ColumnProfile(
            name="x",
            inferred_type="numeric",
            non_null_count=1,
            missing_count=0,
            unique_count=1,
            sample_values=(math.nan,),
        )


def test_analysis_result_rejects_non_json_or_non_finite_payloads() -> None:
    result = AnalysisResult(
        analysis_id="analysis-describe-1",
        dataset_id="dataset-source-1",
        analysis_type="descriptive",
        method="describe",
        parameters={"columns": ["conductivity"]},
        summary={"mean": 2.5},
        evidence_id="evidence-analysis-1",
    )
    assert AnalysisResult.model_validate_json(result.model_dump_json()) == result

    with pytest.raises(ValidationError, match="NaN or infinity"):
        AnalysisResult.model_validate(
            {**result.model_dump(), "summary": {"mean": math.inf}}
        )
    with pytest.raises(ValidationError, match="unsupported JSON"):
        AnalysisResult(
            analysis_id="analysis-invalid-1",
            dataset_id="dataset-source-1",
            analysis_type="quality",
            method="inspect",
            summary={"bad": object()},
            evidence_id="evidence-analysis-2",
        )


def test_transform_record_requires_new_dataset_and_bounded_operations() -> None:
    operation = DatasetTransformOperation(
        kind="drop_duplicates", parameters={"subset": ["sample_id"]}
    )
    record = DatasetTransformRecord(
        operation_id="operation-clean-1",
        source_dataset_id="dataset-source-1",
        result_dataset_id="dataset-derived-1",
        operations=(operation,),
        rows_before=12,
        rows_after=11,
    )
    assert record.operations == (operation,)

    with pytest.raises(ValidationError, match="new dataset_id"):
        DatasetTransformRecord(
            operation_id="operation-clean-2",
            source_dataset_id="dataset-source-1",
            result_dataset_id="dataset-source-1",
            operations=(operation,),
            rows_before=12,
            rows_after=11,
        )


def test_application_contracts_accept_data_analysis_artifact_and_result() -> None:
    artifact = MultiAgentArtifact(
        artifact_id="artifact-data-1",
        owner_agent="data_analysis",
        artifact_type="dataset_file",
        domain_id="dataset-source-1",
        display_name="experiment.csv",
    )
    envelope = UnifiedResultEnvelope(
        agent_name="data_analysis",
        status="completed",
        result_type="analysis_answer",
        result={"markdown": "分析完成。"},
        artifact_refs=(artifact.artifact_id,),
        evidence_refs=("evidence-analysis-1",),
    )

    assert artifact.owner_agent == "data_analysis"
    assert envelope.result_type == "analysis_answer"
    restored = UnifiedResultEnvelope.model_validate_json(envelope.model_dump_json())
    assert restored == envelope
