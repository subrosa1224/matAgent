from __future__ import annotations

import json
from pathlib import Path

import pytest

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.service import DataAnalysisService


def _service_for_csv(tmp_path: Path) -> tuple[DataAnalysisService, str]:
    source = tmp_path / "quality.csv"
    source.write_text(
        "sample_id,temperature,conductivity,method,constant,date\n"
        "A,800,1,A,yes,2026-01-01\n"
        "B,850,2,A,yes,2026-01-02\n"
        "B,850,2,A,yes,2026-01-02\n"
        "C,900,2,B,yes,2026-01-03\n"
        "D,950,3,B,yes,2026-01-04\n"
        "E,1000,100,B,yes,2026-01-05\n"
        "F,1050,,B,yes,2026-01-06\n",
        encoding="utf-8",
    )
    store = DatasetStore(tmp_path / "private")
    reference = store.register_file(
        source, source_artifact_id="artifact-data-quality"
    )
    return DataAnalysisService(store), reference.dataset_id


def test_inspection_profiles_columns_quality_issues_and_preview(
    tmp_path: Path,
) -> None:
    service, dataset_id = _service_for_csv(tmp_path)

    result = service.inspect_dataset(dataset_id, offset=1, limit=2)

    assert result.dataset.dataset_id == dataset_id
    assert result.preview_offset == 1
    assert len(result.preview_rows) == 2
    assert result.preview_rows[0]["sample_id"] == "B"
    assert result.duplicate_row_count == 1
    profiles = {profile.name: profile for profile in result.columns}
    assert profiles["temperature"].inferred_type == "numeric"
    assert profiles["method"].inferred_type == "categorical"
    assert profiles["date"].inferred_type == "datetime"
    assert profiles["conductivity"].missing_count == 1
    codes = {(issue.code, issue.column) for issue in result.issues}
    assert ("missing_values", "conductivity") in codes
    assert ("constant_column", "constant") in codes
    assert ("iqr_outlier_candidates", "conductivity") in codes
    assert ("duplicate_rows", None) in codes


def test_inspection_converts_nan_to_json_null(tmp_path: Path) -> None:
    service, dataset_id = _service_for_csv(tmp_path)

    result = service.inspect_dataset(dataset_id, offset=6, limit=1)
    encoded = json.loads(result.model_dump_json())

    assert encoded["preview_rows"][0]["conductivity"] is None


def test_inspection_detects_mixed_json_column_types(tmp_path: Path) -> None:
    source = tmp_path / "mixed.json"
    source.write_text('[{"x": 1}, {"x": "unknown"}]', encoding="utf-8")
    store = DatasetStore(tmp_path / "private")
    reference = store.register_file(
        source, source_artifact_id="artifact-data-mixed"
    )

    result = DataAnalysisService(store).inspect_dataset(reference.dataset_id)

    assert any(issue.code == "mixed_types" for issue in result.issues)


@pytest.mark.parametrize(
    ("offset", "limit", "message"),
    [(-1, 20, "non-negative"), (0, 0, "between 1 and 100"), (99, 20, "exceeds")],
)
def test_inspection_rejects_invalid_pagination(
    tmp_path: Path, offset: int, limit: int, message: str
) -> None:
    service, dataset_id = _service_for_csv(tmp_path)

    with pytest.raises(ValueError, match=message):
        service.inspect_dataset(dataset_id, offset=offset, limit=limit)
