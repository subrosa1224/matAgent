"""MaterialSetResolver — resolve a MaterialSetReference into MaterialRecords.

Four resolution paths:
  workflow_thread_ids → FileWorkflowResultReader → Sequence[MaterialRecord]
  material_formulas → Repository.query_by_formula() → Sequence[MaterialRecord]
  material_ids      → Repository.query_by_material_id() → Sequence[MaterialRecord]
  data_file         → DataFileParser.parse() → Sequence[MaterialRecord]

This is a Core-layer service (services/), NOT an Agent-layer component.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from materials_screening.models import MaterialRecord
from materials_screening.parsers.data_file_parser import DataFileParser
from materials_screening.repositories.base import MaterialsRepository
from materials_screening.sub_agents.outlier_detection.models import MaterialSetReference

# Known keys in a screening_result artifact that contain MaterialRecord-like data
_RESULT_KEYS = ("ranked_materials", "records", "candidates")


class MaterialSetResolver:
    """Resolve a MaterialSetReference into a collection of MaterialRecords.

    Resolution paths:

    workflow_thread_ids → WorkflowResultReader → Sequence[MaterialRecord]
    material_formulas → Repository.query_by_formula() → Sequence[MaterialRecord]
    material_ids      → Repository.query_by_material_id() → Sequence[MaterialRecord]
    data_file         → DataFileParser.parse() → Sequence[MaterialRecord]
    """

    def __init__(
        self,
        workflow_reader: Any,  # FileWorkflowResultReader (avoids circular import)
        repository: MaterialsRepository,
        parser: DataFileParser,
        allowed_data_dir: Path = Path("data"),
    ) -> None:
        self._workflow_reader = workflow_reader
        self._repository = repository
        self._parser = parser
        self._allowed_data_dir = allowed_data_dir.resolve()

    def resolve(self, ref: MaterialSetReference) -> Sequence[MaterialRecord]:
        """Resolve records, discarding soft resolution warnings.

        Prefer :meth:`resolve_with_warnings` for user-facing tool calls.
        """
        records, _ = self.resolve_with_warnings(ref)
        return records

    def resolve_with_warnings(
        self,
        ref: MaterialSetReference,
    ) -> tuple[Sequence[MaterialRecord], tuple[str, ...]]:
        """Resolve records together with non-fatal resolution warnings."""
        if ref.workflow_thread_ids is not None:
            return self._resolve_from_workflow_threads(ref.workflow_thread_ids), ()
        if ref.material_formulas is not None:
            return self._resolve_from_formulas(ref.material_formulas)
        if ref.material_ids is not None:
            return self._resolve_from_material_ids(ref.material_ids)
        if ref.data_file is not None:
            return self._resolve_from_file(Path(ref.data_file)), ()
        raise AssertionError("MaterialSetReference validation should prevent this")

    def _resolve_from_workflow_threads(
        self, workflow_thread_ids: tuple[str, ...]
    ) -> Sequence[MaterialRecord]:
        """Read from prior screening workflow artifacts.

        Each workflow thread id reads its screening_result
        artifact and extracts MaterialRecord-like entries.
        """
        records: list[MaterialRecord] = []
        for thread_id in workflow_thread_ids:
            payload = self._workflow_reader.read(thread_id)
            extracted = self._extract_records(payload)
            if not extracted:
                raise ValueError(
                    f"No material records found in workflow result for "
                    f"thread {thread_id!r}"
                )
            records.extend(extracted)
        return tuple(records)

    def _extract_records(self, payload: dict[str, Any]) -> list[MaterialRecord]:
        """Extract MaterialRecord instances from a screening_result dict."""
        for key in _RESULT_KEYS:
            items = payload.get(key)
            if isinstance(items, list) and items:
                return self._parse_record_list(items)
        return []

    def _parse_record_list(self, items: list[dict[str, Any]]) -> list[MaterialRecord]:
        """Parse a list of dicts into MaterialRecords."""
        records: list[MaterialRecord] = []
        for item in items:
            # screening_result stores ranked materials as {record: {...}, rank: ...}
            record_data = item.get("record", item)
            if not isinstance(record_data, dict):
                continue
            try:
                rec = MaterialRecord.model_validate(record_data)
                records.append(rec)
            except Exception:
                continue
        return records

    def _resolve_from_formulas(
        self, formulas: tuple[str, ...]
    ) -> tuple[Sequence[MaterialRecord], tuple[str, ...]]:
        """Query every repository entry/polymorph for each formula."""
        records: list[MaterialRecord] = []
        not_found: list[str] = []
        for formula in formulas:
            matches = self._repository.query_all_by_formula(formula)
            if matches:
                records.extend(matches)
            else:
                not_found.append(formula)
        if not records:
            raise ValueError(
                f"No materials found for formulas: {formulas}"
            )
        warnings = tuple(
            f"No material found for formula {formula!r}; excluded from analysis"
            for formula in not_found
        )
        return tuple(records), warnings

    def _resolve_from_material_ids(
        self, material_ids: tuple[str, ...]
    ) -> tuple[Sequence[MaterialRecord], tuple[str, ...]]:
        """Resolve only the exact database entries explicitly requested."""
        records: list[MaterialRecord] = []
        not_found: list[str] = []
        for material_id in material_ids:
            record = self._repository.query_by_material_id(material_id)
            if record is None:
                not_found.append(material_id)
            else:
                records.append(record)
        if not records:
            raise ValueError(f"No materials found for ids: {material_ids}")
        warnings = tuple(
            f"No material found for id {material_id!r}; excluded from analysis"
            for material_id in not_found
        )
        return tuple(records), warnings

    def _resolve_from_file(self, path: Path) -> Sequence[MaterialRecord]:
        """Parse user-provided CSV/JSON property data file."""
        resolved_path = path.resolve(strict=False)
        if not resolved_path.is_relative_to(self._allowed_data_dir):
            raise ValueError(
                f"Data file must be inside allowed directory: {self._allowed_data_dir}"
            )
        if not resolved_path.is_file():
            raise ValueError(f"Data file not found: {resolved_path}")
        return self._parser.parse(resolved_path)
