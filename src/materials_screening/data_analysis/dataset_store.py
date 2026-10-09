"""Immutable, private-path storage for validated tabular datasets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from materials_screening.data_analysis.models import (
    AnalysisResult,
    ArtifactFileExtension,
    DataAnalysisArtifact,
    DataAnalysisArtifactType,
    DatasetReference,
    DatasetTransformOperation,
    DatasetTransformRecord,
)
from materials_screening.data_analysis.parser import (
    TabularDataParser,
    schema_fingerprint,
)


class DatasetStore:
    """Persist uploaded datasets while exposing only stable opaque IDs."""

    def __init__(self, root: Path, *, parser: TabularDataParser | None = None) -> None:
        self._root = root.resolve()
        self._dataset_root = (self._root / "datasets").resolve()
        self._metadata_root = (self._root / "metadata").resolve()
        self._analysis_root = (self._root / "analyses").resolve()
        self._transform_root = (self._root / "transforms").resolve()
        self._artifact_root = (self._root / "artifacts").resolve()
        self._artifact_metadata_root = (self._root / "artifact_metadata").resolve()
        self._parser = parser or TabularDataParser()
        self._records: dict[str, tuple[DatasetReference, Path]] = {}
        self._digest_index: dict[str, str] = {}
        self._lock = threading.RLock()
        self._load_metadata()

    def register_file(
        self,
        source: str | Path,
        *,
        source_artifact_id: str,
        display_name: str | None = None,
    ) -> DatasetReference:
        """Validate and copy one source file into immutable private storage."""

        path = Path(source).resolve()
        frame = self._parser.parse(path)
        digest = _sha256_file(path)
        content_fingerprint = f"sha256:{digest}"
        with self._lock:
            existing_id = self._digest_index.get(content_fingerprint)
            if existing_id is not None:
                return self._records[existing_id][0]

            dataset_id = f"dataset-{digest[:24]}"
            suffix = path.suffix.casefold()
            target = (self._dataset_root / f"{dataset_id}{suffix}").resolve()
            metadata_path = (self._metadata_root / f"{dataset_id}.json").resolve()
            self._require_direct_child(target, self._dataset_root)
            self._require_direct_child(metadata_path, self._metadata_root)
            reference = DatasetReference(
                dataset_id=dataset_id,
                source_artifact_id=source_artifact_id,
                display_name=(display_name or path.name)[:512],
                format=suffix.removeprefix("."),  # type: ignore[arg-type]
                row_count=len(frame.index),
                column_count=len(frame.columns),
                schema_fingerprint=schema_fingerprint(frame),
                content_fingerprint=content_fingerprint,
            )
            self._dataset_root.mkdir(parents=True, exist_ok=True)
            self._metadata_root.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                _atomic_copy(path, target)
            _atomic_write_text(metadata_path, reference.model_dump_json(indent=2))
            self._records[dataset_id] = (reference, target)
            self._digest_index[content_fingerprint] = dataset_id
            return reference

    def register_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        source_artifact_id: str,
        display_name: str,
    ) -> DatasetReference:
        """Register bounded structured records without exposing a source path."""

        try:
            content = json.dumps(
                list(records),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "structured records must contain JSON scalar values"
            ) from exc
        staging_root = (self._root / "staging").resolve()
        staging_path = (staging_root / f"records-{uuid.uuid4().hex}.json").resolve()
        self._require_direct_child(staging_path, staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        try:
            _atomic_write_bytes(staging_path, content)
            return self.register_file(
                staging_path,
                source_artifact_id=source_artifact_id,
                display_name=display_name,
            )
        finally:
            staging_path.unlink(missing_ok=True)

    def get(self, dataset_id: str) -> DatasetReference:
        with self._lock:
            record = self._records.get(dataset_id)
            if record is None:
                self._load_metadata()
                record = self._records.get(dataset_id)
        if record is None:
            raise KeyError(f"unknown dataset: {dataset_id!r}")
        return record[0]

    def resolve_path(self, dataset_id: str) -> Path:
        with self._lock:
            record = self._records.get(dataset_id)
            if record is None:
                self._load_metadata()
                record = self._records.get(dataset_id)
        if record is None:
            raise KeyError(f"unknown dataset: {dataset_id!r}")
        path = record[1].resolve()
        self._require_direct_child(path, self._dataset_root)
        if not path.is_file():
            raise ValueError("stored dataset file is missing")
        return path

    def load_dataframe(self, dataset_id: str) -> pd.DataFrame:
        reference = self.get(dataset_id)
        path = self.resolve_path(dataset_id)
        if f"sha256:{_sha256_file(path)}" != reference.content_fingerprint:
            raise ValueError(
                "stored dataset content fingerprint does not match metadata"
            )
        return self._parser.parse(path)

    def save_analysis(self, result: AnalysisResult) -> None:
        """Persist one validated analysis result without exposing its path."""

        self.get(result.dataset_id)
        target = (self._analysis_root / f"{result.analysis_id}.json").resolve()
        self._require_direct_child(target, self._analysis_root)
        with self._lock:
            self._analysis_root.mkdir(parents=True, exist_ok=True)
            if target.exists():
                existing = AnalysisResult.model_validate_json(
                    target.read_text(encoding="utf-8")
                )
                if existing != result:
                    raise ValueError("analysis_id already contains a different result")
                return
            _atomic_write_text(target, result.model_dump_json(indent=2))

    def get_analysis(self, analysis_id: str) -> AnalysisResult:
        target = (self._analysis_root / f"{analysis_id}.json").resolve()
        self._require_direct_child(target, self._analysis_root)
        if not target.is_file():
            raise KeyError(f"unknown analysis: {analysis_id!r}")
        return AnalysisResult.model_validate_json(target.read_text(encoding="utf-8"))

    def save_derived_dataset(
        self,
        frame: pd.DataFrame,
        *,
        source_dataset_id: str,
        operation_id: str,
        operations: tuple[DatasetTransformOperation, ...],
        warnings: tuple[str, ...] = (),
    ) -> tuple[DatasetReference, DatasetTransformRecord]:
        """Atomically persist a derived CSV, metadata and audit record."""

        source = self.get(source_dataset_id)
        if frame.empty:
            raise ValueError("derived dataset must contain at least one row")
        if not 1 <= len(frame.columns) <= 500:
            raise ValueError("derived dataset must contain between 1 and 500 columns")
        dataset_id = f"dataset-{uuid.uuid4().hex}"
        target = (self._dataset_root / f"{dataset_id}.csv").resolve()
        metadata_path = (self._metadata_root / f"{dataset_id}.json").resolve()
        transform_path = (self._transform_root / f"{operation_id}.json").resolve()
        self._require_direct_child(target, self._dataset_root)
        self._require_direct_child(metadata_path, self._metadata_root)
        self._require_direct_child(transform_path, self._transform_root)
        temporary = target.with_name(
            f".{dataset_id}.{uuid.uuid4().hex}.tmp.csv"
        )
        committed: list[Path] = []
        try:
            self._dataset_root.mkdir(parents=True, exist_ok=True)
            self._metadata_root.mkdir(parents=True, exist_ok=True)
            self._transform_root.mkdir(parents=True, exist_ok=True)
            frame.to_csv(temporary, index=False, encoding="utf-8-sig")
            validated = self._parser.parse(temporary)
            content_fingerprint = f"sha256:{_sha256_file(temporary)}"
            reference = DatasetReference(
                dataset_id=dataset_id,
                source_artifact_id=source.source_artifact_id,
                display_name=_derived_display_name(source.display_name),
                format="csv",
                row_count=len(validated.index),
                column_count=len(validated.columns),
                schema_fingerprint=schema_fingerprint(validated),
                content_fingerprint=content_fingerprint,
                parent_dataset_id=source_dataset_id,
                created_by_operation_id=operation_id,
            )
            record = DatasetTransformRecord(
                operation_id=operation_id,
                source_dataset_id=source_dataset_id,
                result_dataset_id=dataset_id,
                operations=operations,
                rows_before=source.row_count,
                rows_after=len(validated.index),
                warnings=warnings,
            )
            with self._lock:
                if transform_path.exists():
                    raise ValueError("operation_id already exists")
                os.replace(temporary, target)
                committed.append(target)
                _atomic_write_text(metadata_path, reference.model_dump_json(indent=2))
                committed.append(metadata_path)
                _atomic_write_text(transform_path, record.model_dump_json(indent=2))
                committed.append(transform_path)
                self._records[dataset_id] = (reference, target)
            return reference, record
        except Exception:
            with self._lock:
                self._records.pop(dataset_id, None)
                for path in reversed(committed):
                    path.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)

    def get_transform(self, operation_id: str) -> DatasetTransformRecord:
        target = (self._transform_root / f"{operation_id}.json").resolve()
        self._require_direct_child(target, self._transform_root)
        if not target.is_file():
            raise KeyError(f"unknown transform operation: {operation_id!r}")
        return DatasetTransformRecord.model_validate_json(
            target.read_text(encoding="utf-8")
        )

    def save_artifact(
        self,
        content: bytes,
        *,
        artifact_type: DataAnalysisArtifactType,
        dataset_id: str,
        analysis_ids: tuple[str, ...] = (),
        display_name: str,
        file_extension: ArtifactFileExtension,
        media_type: str,
    ) -> DataAnalysisArtifact:
        """Persist bounded artifact bytes and return path-free metadata."""

        self.get(dataset_id)
        if not content:
            raise ValueError("artifact content must not be empty")
        if len(content) > 100 * 1024 * 1024:
            raise ValueError("artifact exceeds the 100 MB limit")
        for analysis_id in analysis_ids:
            if self.get_analysis(analysis_id).dataset_id != dataset_id:
                raise ValueError("artifact analysis_ids must belong to dataset_id")
        if file_extension not in {"csv", "json", "md", "png", "docx"}:
            raise ValueError("unsupported artifact file extension")
        artifact_id = f"artifact-data-{uuid.uuid4().hex}"
        target = (self._artifact_root / f"{artifact_id}.{file_extension}").resolve()
        metadata_path = (
            self._artifact_metadata_root / f"{artifact_id}.json"
        ).resolve()
        self._require_direct_child(target, self._artifact_root)
        self._require_direct_child(metadata_path, self._artifact_metadata_root)
        fingerprint = "sha256:" + hashlib.sha256(content).hexdigest()
        artifact = DataAnalysisArtifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            dataset_id=dataset_id,
            analysis_ids=analysis_ids,
            display_name=display_name[:512],
            file_extension=file_extension,
            media_type=media_type,
            size_bytes=len(content),
            content_fingerprint=fingerprint,
        )
        with self._lock:
            self._artifact_root.mkdir(parents=True, exist_ok=True)
            self._artifact_metadata_root.mkdir(parents=True, exist_ok=True)
            try:
                _atomic_write_bytes(target, content)
                _atomic_write_text(metadata_path, artifact.model_dump_json(indent=2))
            except Exception:
                target.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                raise
        return artifact

    def get_artifact(self, artifact_id: str) -> DataAnalysisArtifact:
        target = (self._artifact_metadata_root / f"{artifact_id}.json").resolve()
        self._require_direct_child(target, self._artifact_metadata_root)
        if not target.is_file():
            raise KeyError(f"unknown data-analysis artifact: {artifact_id!r}")
        return DataAnalysisArtifact.model_validate_json(
            target.read_text(encoding="utf-8")
        )

    def resolve_artifact_path(self, artifact_id: str) -> Path:
        artifact = self.get_artifact(artifact_id)
        target = (
            self._artifact_root
            / f"{artifact.artifact_id}.{artifact.file_extension}"
        ).resolve()
        self._require_direct_child(target, self._artifact_root)
        if not target.is_file():
            raise ValueError("stored artifact file is missing")
        if "sha256:" + _sha256_file(target) != artifact.content_fingerprint:
            raise ValueError("stored artifact fingerprint does not match metadata")
        return target

    def _load_metadata(self) -> None:
        if not self._metadata_root.is_dir():
            return
        for metadata_path in sorted(self._metadata_root.glob("dataset-*.json")):
            self._require_direct_child(metadata_path.resolve(), self._metadata_root)
            try:
                reference = DatasetReference.model_validate_json(
                    metadata_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            suffix = f".{reference.format}"
            data_path = (
                self._dataset_root / f"{reference.dataset_id}{suffix}"
            ).resolve()
            if data_path.parent != self._dataset_root or not data_path.is_file():
                continue
            self._records[reference.dataset_id] = (reference, data_path)
            if reference.parent_dataset_id is None:
                self._digest_index[reference.content_fingerprint] = reference.dataset_id

    @staticmethod
    def _require_direct_child(path: Path, root: Path) -> None:
        if path.parent != root:
            raise ValueError("dataset storage path is invalid")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(target: Path, content: str) -> None:
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_bytes(target: Path, content: bytes) -> None:
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _derived_display_name(source_name: str) -> str:
    stem = Path(source_name).stem[:480] or "dataset"
    return f"{stem}-derived.csv"
