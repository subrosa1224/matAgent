from __future__ import annotations

from pathlib import Path

import pytest

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.models import DatasetTransformOperation
from materials_screening.data_analysis.transform import DatasetTransformService


def _transform_service(
    tmp_path: Path,
) -> tuple[DatasetTransformService, DatasetStore, str, bytes]:
    source = tmp_path / "source.csv"
    source.write_text(
        "sample_id,group,value,label,flag,date\n"
        "a,A,1,normal,yes,2026-01-01\n"
        "a,A,1,normal,yes,2026-01-01\n"
        "b,A,,=SUM(A1:A2),no,2026-01-02\n"
        "c,A,3,safe,true,2026-01-03\n"
        "d,B,10,-command,1,2026-01-04\n"
        "e,B,,safe,0,2026-01-05\n"
        "f,B,14,@mention,false,2026-01-06\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    store = DatasetStore(tmp_path / "private")
    reference = store.register_file(
        source, source_artifact_id="artifact-data-transform"
    )
    return DatasetTransformService(store), store, reference.dataset_id, original


def _operation(kind: str, **parameters: object) -> DatasetTransformOperation:
    return DatasetTransformOperation.model_validate(
        {"kind": kind, "parameters": parameters}
    )


def test_transform_creates_immutable_derived_dataset_and_audit_record(
    tmp_path: Path,
) -> None:
    service, store, source_id, original = _transform_service(tmp_path)
    source_path = store.resolve_path(source_id)
    source_hash = store.get(source_id).content_fingerprint

    derived, record, inspection = service.transform_dataset(
        source_id,
        operations=[
            _operation("drop_duplicates", subset=["sample_id"], keep="first"),
            _operation(
                "fill_missing",
                column="value",
                strategy="median",
                group_by="group",
            ),
        ],
        operation_id="operation-clean-main",
    )

    assert derived.dataset_id != source_id
    assert derived.parent_dataset_id == source_id
    assert derived.created_by_operation_id == "operation-clean-main"
    assert derived.row_count == 6
    assert record.rows_before == 7
    assert record.rows_after == 6
    assert store.get_transform(record.operation_id) == record
    assert store.get(source_id).content_fingerprint == source_hash
    assert source_path.read_bytes() == original
    frame = store.load_dataframe(derived.dataset_id)
    values = dict(zip(frame["sample_id"], frame["value"], strict=True))
    assert values["b"] == pytest.approx(2.0)
    assert values["e"] == pytest.approx(12.0)
    assert inspection.dataset == derived
    assert not any(issue.code == "missing_values" for issue in inspection.issues)


def test_transform_escapes_formula_like_text_and_records_warning(
    tmp_path: Path,
) -> None:
    service, store, source_id, _ = _transform_service(tmp_path)

    derived, record, _ = service.transform_dataset(
        source_id,
        operations=[_operation("select_columns", columns=["sample_id", "label"])],
    )

    labels = set(store.load_dataframe(derived.dataset_id)["label"])
    assert "'=SUM(A1:A2)" in labels
    assert "'-command" in labels
    assert "'@mention" in labels
    assert record.warnings
    assert "3" in record.warnings[0]


def test_filter_convert_drop_missing_and_constant_fill_are_whitelisted(
    tmp_path: Path,
) -> None:
    service, store, source_id, _ = _transform_service(tmp_path)

    derived, _, _ = service.transform_dataset(
        source_id,
        operations=[
            _operation("filter_rows", column="group", operator="eq", value="A"),
            _operation("drop_missing", subset=["value"], how="any"),
            _operation("convert_type", column="flag", dtype="boolean"),
            _operation("convert_type", column="date", dtype="datetime"),
            _operation("fill_missing", column="label", strategy="constant", value="x"),
        ],
    )

    frame = store.load_dataframe(derived.dataset_id)
    assert set(frame["group"]) == {"A"}
    assert frame["value"].notna().all()
    assert set(frame["flag"]) == {True}
    assert all(str(value).startswith("2026-01-") for value in frame["date"])


@pytest.mark.parametrize(
    "operation",
    [
        _operation("select_columns", columns=["missing"]),
        _operation("drop_duplicates", subset=["sample_id"], keep="none"),
        _operation("convert_type", column="label", dtype="numeric"),
        _operation("filter_rows", column="value", operator="python", value="x"),
        _operation("fill_missing", column="label", strategy="mean"),
        _operation("drop_missing", how="bad"),
        _operation("select_columns", columns=["sample_id"], unexpected=True),
    ],
)
def test_invalid_transform_is_rejected_before_any_derived_files_are_committed(
    tmp_path: Path, operation: DatasetTransformOperation
) -> None:
    service, _, source_id, _ = _transform_service(tmp_path)
    private_root = tmp_path / "private"

    with pytest.raises(ValueError):
        service.transform_dataset(source_id, operations=[operation])

    datasets = tuple((private_root / "datasets").glob("dataset-*.csv"))
    metadata = tuple((private_root / "metadata").glob("dataset-*.json"))
    transforms_root = private_root / "transforms"
    transforms = (
        tuple(transforms_root.glob("*.json")) if transforms_root.exists() else ()
    )
    assert len(datasets) == 1
    assert len(metadata) == 1
    assert transforms == ()


def test_transform_operations_are_ordered_and_empty_result_is_rejected(
    tmp_path: Path,
) -> None:
    service, _, source_id, _ = _transform_service(tmp_path)

    with pytest.raises(ValueError, match="unknown dataset columns"):
        service.transform_dataset(
            source_id,
            operations=[
                _operation("select_columns", columns=["sample_id"]),
                _operation("fill_missing", column="value", strategy="mean"),
            ],
        )
    with pytest.raises(ValueError, match="empty dataset"):
        service.transform_dataset(
            source_id,
            operations=[
                _operation(
                    "filter_rows", column="group", operator="eq", value="missing"
                )
            ],
        )


def test_duplicate_operation_id_does_not_damage_existing_transform(
    tmp_path: Path,
) -> None:
    service, store, source_id, _ = _transform_service(tmp_path)
    first, first_record, _ = service.transform_dataset(
        source_id,
        operations=[_operation("select_columns", columns=["sample_id", "value"])],
        operation_id="operation-duplicate-test",
    )

    with pytest.raises(ValueError, match="already exists"):
        service.transform_dataset(
            source_id,
            operations=[_operation("select_columns", columns=["sample_id"])],
            operation_id="operation-duplicate-test",
        )

    assert store.get_transform("operation-duplicate-test") == first_record
    assert store.load_dataframe(first.dataset_id).shape == (7, 2)


def test_derived_dataset_and_transform_survive_store_reload(tmp_path: Path) -> None:
    service, store, source_id, _ = _transform_service(tmp_path)
    derived, record, _ = service.transform_dataset(
        source_id,
        operations=[_operation("select_columns", columns=["sample_id", "value"])],
    )

    reloaded = DatasetStore(tmp_path / "private")

    assert reloaded.get(derived.dataset_id) == derived
    assert reloaded.get_transform(record.operation_id) == record
    assert reloaded.load_dataframe(derived.dataset_id).shape == (7, 2)


def test_transform_rejects_missing_operation_list(tmp_path: Path) -> None:
    service, _, source_id, _ = _transform_service(tmp_path)

    with pytest.raises(ValueError, match="between 1 and 32"):
        service.transform_dataset(source_id, operations=[])
