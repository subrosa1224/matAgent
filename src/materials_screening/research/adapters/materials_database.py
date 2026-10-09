"""Read-only normalization of MaterialsDatabaseAgent query results."""

from __future__ import annotations

import hashlib
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from materials_screening.chemistry import normalize_chemsys
from materials_screening.research.adapters.models import AdapterBatch
from materials_screening.research.candidate_ledger import (
    CandidateSourceKind,
    EvidenceLocator,
    EvidenceReviewStatus,
    EvidenceSourceKind,
    MaterialCandidate,
    PropertyValue,
    ScreeningEvidence,
)
from materials_screening.research.models import ScreeningProject, SourceRole
from materials_screening.research.property_registry import PropertyRegistry
from materials_screening.services.material_database_service import (
    MaterialDatabaseService,
    project_record,
)
from materials_screening.sub_agents.materials_database.models import (
    OutputField,
    QueryResultReference,
    SearchMaterialsInput,
)

_IDENTITY_AND_PROVENANCE_FIELDS = frozenset(
    {
        "material_id",
        "formula_pretty",
        "elements",
        "chemsys",
        "spacegroup_symbol",
        "structure",
        "structure_hash",
        "structure_fingerprint",
        "source",
        "deprecated",
        "database_version",
        "source_query_ids",
    }
)


class MaterialsDatabaseReadAdapter:
    """Normalize an already-persisted query result; never executes a query."""

    def __init__(self, registry: PropertyRegistry) -> None:
        self._registry = registry

    def normalize(
        self,
        project_id: str,
        result: QueryResultReference,
        *,
        database_version: str | None = None,
    ) -> AdapterBatch:
        candidates: list[MaterialCandidate] = []
        evidence: list[ScreeningEvidence] = []
        warnings: list[str] = list(result.warnings)
        for row in result.materials:
            material_id = str(row.get("material_id") or "").strip()
            formula = str(row.get("formula_pretty") or "").strip()
            chemsys = _row_chemsys(row)
            if not material_id or not formula or not chemsys:
                warnings.append(
                    "database row skipped: missing material identity fields"
                )
                continue
            candidate_id = _id("candidate", project_id, result.query_id, material_id)
            properties: list[PropertyValue] = []
            for key, raw_value in row.items():
                definition = self._registry.resolve(key)
                if definition is None:
                    if (
                        raw_value is not None
                        and key not in _IDENTITY_AND_PROVENANCE_FIELDS
                    ):
                        warnings.append(
                            f"{material_id}: unregistered field {key!r} skipped"
                        )
                    continue
                if raw_value is None:
                    continue
                if key in {"formula_pretty", "elements"}:
                    continue
                evidence_id = _id(
                    "evidence",
                    project_id,
                    result.query_id,
                    material_id,
                    definition.property_id,
                )
                source_role = (
                    SourceRole.DATABASE_CALCULATED
                    if SourceRole.DATABASE_CALCULATED in definition.source_roles
                    else SourceRole.DATABASE_METADATA
                )
                item = ScreeningEvidence(
                    evidence_id=evidence_id,
                    project_id=project_id,
                    candidate_id=candidate_id,
                    source_kind=EvidenceSourceKind.MATERIALS_DATABASE,
                    source_id=result.query_id,
                    locator=EvidenceLocator(
                        query_id=result.query_id,
                        source_material_id=material_id,
                        database_version=database_version,
                    ),
                    property_id=definition.property_id,
                    value=raw_value,
                    unit=definition.canonical_unit,
                    method="Materials Project summary record",
                    conditions={},
                    review_status=EvidenceReviewStatus.VALIDATED,
                    limitations=(definition.comparability_notes,),
                )
                evidence.append(item)
                properties.append(
                    PropertyValue(
                        property_id=definition.property_id,
                        value=raw_value,
                        unit=definition.canonical_unit,
                        source_role=source_role,
                        evidence_id=evidence_id,
                        method=item.method,
                    )
                )
            candidates.append(
                MaterialCandidate(
                    candidate_id=candidate_id,
                    project_id=project_id,
                    source_kind=CandidateSourceKind.MATERIALS_DATABASE,
                    source_material_id=material_id,
                    formula=formula,
                    chemsys=chemsys,
                    structure_fingerprint=_optional_str(
                        row.get("structure_fingerprint") or row.get("structure_hash")
                    ),
                    space_group_number=_space_group(row),
                    property_values=tuple(properties),
                    provenance={
                        "query_id": result.query_id,
                        "database": result.source,
                        "database_version": database_version,
                        "retrieved_at": result.created_at.isoformat(),
                    },
                )
            )
        return AdapterBatch(
            candidates=tuple(candidates),
            evidence=tuple(evidence),
            warnings=tuple(warnings),
        )


class MaterialsDatabaseCandidateSource:
    """Execute one bounded database query and cache its normalized adapter batch."""

    def __init__(
        self,
        service: MaterialDatabaseService,
        registry: PropertyRegistry,
        cache_root: Path,
    ) -> None:
        self._service = service
        self._adapter = MaterialsDatabaseReadAdapter(registry)
        self._cache_root = cache_root.resolve()
        self._lock = threading.RLock()

    def fetch_candidates(
        self, project: ScreeningProject, *, idempotency_key: str
    ) -> AdapterBatch:
        cache_path = self._cache_path(idempotency_key)
        with self._lock:
            if cache_path.is_file():
                return AdapterBatch.model_validate_json(
                    cache_path.read_text(encoding="utf-8")
                )
            scope = project.material_scope
            fields: tuple[OutputField, ...] = (
                "material_id",
                "formula_pretty",
                "elements",
                "chemsys",
                "band_gap_ev",
                "energy_above_hull_ev_atom",
                "formation_energy_ev_atom",
                "density_g_cm3",
                "is_metal",
                "is_gap_direct",
                "is_stable",
                "theoretical",
                "crystal_system",
                "spacegroup_number",
            )
            request = SearchMaterialsInput(
                required_elements=scope.required_elements,
                excluded_elements=scope.excluded_elements,
                chemsys=scope.chemsys,
                formula=scope.formulas[0] if len(scope.formulas) == 1 else None,
                material_ids=scope.material_ids,
                limit=project.resource_limits.candidate_snapshot_limit,
                fields=fields,
            )
            raw = self._service.search(request)
            query_id = str(raw["query_id"])
            records = self._service.store.load_records(query_id)
            materials = tuple(project_record(record, fields) for record in records)
            result = QueryResultReference(
                query_id=query_id,
                source="materials_project",
                matched_count=int(raw["matched_count"]),
                returned_count=len(materials),
                fields=fields,
                materials=materials,
                warnings=tuple(str(item) for item in raw.get("warnings", ())),
                evidence_id=raw.get("evidence_id"),
                created_at=datetime.fromisoformat(str(raw["created_at"])),
            )
            batch = self._adapter.normalize(project.project_id, result)
            _atomic_cache_write(cache_path, batch.model_dump_json(indent=2))
            return batch

    def _cache_path(self, idempotency_key: str) -> Path:
        if not idempotency_key or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for char in idempotency_key
        ):
            raise ValueError("invalid database idempotency key")
        path = (self._cache_root / f"{idempotency_key}.json").resolve()
        if path.parent != self._cache_root:
            raise ValueError("invalid database cache path")
        return path


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _row_chemsys(row: dict[str, Any]) -> str:
    direct = row.get("chemsys")
    if direct:
        return normalize_chemsys(str(direct)) or ""
    elements = row.get("elements")
    if isinstance(elements, list | tuple) and elements:
        return normalize_chemsys("-".join(str(item) for item in elements)) or ""
    return ""


def _space_group(row: dict[str, Any]) -> int | None:
    value = row.get("spacegroup_number")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _atomic_cache_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
