"""Bounded structured handoffs between data analysis and domain agents."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.master.application_feasibility import (
    ApplicationCandidate,
    prioritize_uv_candidates,
)
from materials_screening.services.material_database_service import (
    MaterialDatabaseService,
)
from materials_screening.services.query_result_store import QueryResultStore
from materials_screening.sub_agents.literature.models import (
    ExperimentMatrixDataTable,
)

SourceKind = Literal["user_upload", "materials_database", "literature", "data_analysis"]


class SourcePartition(BaseModel):
    """One immutable source boundary passed into data analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_kind: SourceKind
    source_id: str = Field(min_length=1, max_length=256)
    dataset_id: str = Field(pattern=r"^dataset-[A-Za-z0-9._:-]+$")
    record_count: int = Field(ge=1, le=100_000)
    fields: tuple[str, ...] = Field(min_length=1, max_length=500)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=10_000)
    warnings: tuple[str, ...] = ()


class DataAnalysisHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    handoff_id: str = Field(pattern=r"^handoff-analysis-[0-9a-f]{24}$")
    evidence_id: str = Field(pattern=r"^evidence-[0-9a-f]{24}$")
    task: str = Field(min_length=1, max_length=4000)
    partition: SourcePartition
    target_agent: Literal["data_analysis"] = "data_analysis"


class MaterialLookupHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    handoff_id: str = Field(pattern=r"^handoff-materials-[0-9a-f]{24}$")
    source_dataset_id: str = Field(pattern=r"^dataset-[A-Za-z0-9._:-]+$")
    material_ids: tuple[str, ...] = Field(min_length=1, max_length=100)
    analysis_ids: tuple[str, ...] = Field(default=(), max_length=100)
    target_agent: Literal["materials_database"] = "materials_database"

    @field_validator("material_ids")
    @classmethod
    def validate_material_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)) or any(
            re.fullmatch(r"mp-[A-Za-z0-9-]+", value) is None for value in values
        ):
            raise ValueError("material_ids must be unique Materials Project IDs")
        return values


class SourceSynthesisSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_kind: SourceKind
    source_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=300)
    markdown: str = Field(min_length=1, max_length=20_000)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=10_000)


class SourcePartitionedSynthesis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["completed", "partial", "failed"]
    sections: tuple[SourceSynthesisSection, ...] = Field(max_length=12)
    warnings: tuple[str, ...] = ()


class DataAnalysisCrossAgentCoordinator:
    """Convert only validated structured records across agent boundaries."""

    def __init__(
        self,
        *,
        dataset_store: DatasetStore,
        query_store: QueryResultStore | None = None,
        database_service: MaterialDatabaseService | None = None,
    ) -> None:
        self._datasets = dataset_store
        self._queries = query_store
        self._database = database_service

    def material_query_to_analysis(
        self, query_id: str, *, task: str
    ) -> DataAnalysisHandoff:
        if self._queries is None:
            raise ValueError("materials query store is unavailable")
        records = self._queries.load_records(query_id)
        if not records:
            raise ValueError("materials query snapshot contains no records")
        rows = [_material_row(record) for record in records]
        artifact_id = _artifact_id("database", query_id, rows)
        dataset = self._datasets.register_records(
            rows,
            source_artifact_id=artifact_id,
            display_name=f"{query_id}-materials.json",
        )
        partition = SourcePartition(
            source_kind="materials_database",
            source_id=query_id,
            dataset_id=dataset.dataset_id,
            record_count=len(rows),
            fields=tuple(rows[0]),
            evidence_ids=(query_id,),
        )
        handoff_id = _handoff_id("analysis", partition.model_dump(mode="json"))
        return DataAnalysisHandoff(
            handoff_id=handoff_id,
            evidence_id=_evidence_id(handoff_id),
            task=task,
            partition=partition,
        )

    def material_query_candidates(
        self, query_id: str, *, limit: int = 5, unique_formulas: bool = False
    ) -> tuple[dict[str, Any], ...]:
        """Return a bounded, ordered candidate view for downstream validation."""

        if not 1 <= limit <= 20:
            raise ValueError("candidate limit must be between 1 and 20")
        if self._queries is None:
            raise ValueError("materials query store is unavailable")
        records = self._queries.load_records(query_id)
        if unique_formulas:
            # Select a representative from the complete snapshot BEFORE limiting.
            # Hull energy is only a deterministic retrieval priority, not an
            # experimental-performance score. Raw snapshots/previews are unchanged.
            def key(record: Any) -> tuple[float, str, str]:
                hull = record.energy_above_hull_ev_atom
                energy = float(hull) if hull is not None else math.inf
                if not math.isfinite(energy):
                    energy = math.inf
                return energy, record.formula_pretty, record.material_id

            representatives: dict[str, Any] = {}
            for record in records:
                prior = representatives.get(record.formula_pretty)
                if prior is None or key(record) < key(prior):
                    representatives[record.formula_pretty] = record
            records = sorted(representatives.values(), key=key)
        return tuple(_material_row(record) for record in records[:limit])

    def prioritized_material_query_candidates(
        self, query_id: str, *, limit: int = 5, unique_formulas: bool = False
    ) -> tuple[ApplicationCandidate, ...]:
        """Return a bounded UV-application queue while preserving raw records."""

        if self._queries is None:
            raise ValueError("materials query store is unavailable")
        records = self._queries.load_records(query_id)
        return prioritize_uv_candidates(
            records, limit=limit, unique_formulas=unique_formulas
        )

    def literature_matrix_to_analysis(
        self, table: ExperimentMatrixDataTable, *, task: str
    ) -> DataAnalysisHandoff:
        if any(item.review_status != "approved" for item in table.groups):
            raise ValueError("literature groups must be human-approved")
        if any(item.review_status != "approved" for item in table.measurements):
            raise ValueError("literature measurements must be human-approved")
        groups = {item.group_id: item for item in table.groups}
        rows: list[dict[str, Any]] = []
        skipped = 0
        for measurement in table.measurements:
            group = groups.get(measurement.group_id)
            if group is None:
                raise ValueError("literature measurement references an unknown group")
            if measurement.numeric_value is None:
                skipped += 1
                continue
            row: dict[str, Any] = {
                "document_id": table.document_id,
                "group_id": group.group_id,
                "group_label": group.label,
                "group_role": group.role,
                "material": group.material,
                "measurement_id": measurement.measurement_id,
                "metric": measurement.metric,
                "numeric_value": measurement.numeric_value,
                "unit": measurement.unit,
                "sample_size": measurement.sample_size,
                "evidence_chunk_id": measurement.chunk_id,
                "evidence_page": measurement.page_from,
            }
            row.update(
                {f"variable__{key}": value for key, value in group.variables.items()}
            )
            row.update(
                {f"condition__{key}": value for key, value in group.conditions.items()}
            )
            rows.append(row)
        if not rows:
            raise ValueError("literature matrix has no approved numeric measurements")
        warnings: list[str] = []
        if skipped:
            warnings.append(f"skipped {skipped} non-numeric measurements")
        metric_units: dict[str, set[str]] = {}
        for row in rows:
            metric_units.setdefault(str(row["metric"]), set()).add(str(row["unit"]))
        if any(len(units) > 1 for units in metric_units.values()):
            warnings.append(
                "same metric contains multiple units; values were not converted"
            )
        artifact_id = _artifact_id("literature", table.document_id, rows)
        dataset = self._datasets.register_records(
            rows,
            source_artifact_id=artifact_id,
            display_name=f"{table.document_id}-experiment-matrix.json",
        )
        evidence_ids = tuple(
            dict.fromkeys(str(row["evidence_chunk_id"]) for row in rows)
        )
        partition = SourcePartition(
            source_kind="literature",
            source_id=table.document_id,
            dataset_id=dataset.dataset_id,
            record_count=len(rows),
            fields=tuple(rows[0]),
            evidence_ids=evidence_ids,
            warnings=tuple(warnings),
        )
        handoff_id = _handoff_id("analysis", partition.model_dump(mode="json"))
        return DataAnalysisHandoff(
            handoff_id=handoff_id,
            evidence_id=_evidence_id(handoff_id),
            task=task,
            partition=partition,
        )

    def analysis_to_material_lookup(
        self,
        dataset_id: str,
        *,
        material_id_column: str = "material_id",
        analysis_ids: Sequence[str] = (),
    ) -> MaterialLookupHandoff:
        frame = self._datasets.load_dataframe(dataset_id)
        if material_id_column not in frame.columns:
            raise ValueError(
                "dataset does not contain the requested material ID column"
            )
        raw_ids = [
            str(value).strip() for value in frame[material_id_column].dropna().tolist()
        ]
        invalid = [
            value
            for value in raw_ids
            if re.fullmatch(r"mp-[A-Za-z0-9-]+", value) is None
        ]
        if invalid:
            raise ValueError("dataset contains invalid Materials Project IDs")
        material_ids = tuple(dict.fromkeys(raw_ids))
        if not material_ids:
            raise ValueError("dataset contains no Materials Project IDs")
        if len(material_ids) > 100:
            raise ValueError("material lookup is limited to 100 IDs")
        payload = {
            "dataset_id": dataset_id,
            "material_ids": material_ids,
            "analysis_ids": tuple(analysis_ids),
        }
        return MaterialLookupHandoff(
            handoff_id=_handoff_id("materials", payload),
            source_dataset_id=dataset_id,
            material_ids=material_ids,
            analysis_ids=tuple(analysis_ids),
        )

    def execute_material_lookup(
        self,
        handoff: MaterialLookupHandoff,
        *,
        fields: Sequence[str],
    ) -> SourceSynthesisSection:
        if self._database is None:
            raise ValueError("materials database service is unavailable")
        result = self._database.details(handoff.material_ids, fields)
        materials = result.get("materials", [])
        lines = ["| Material ID | Formula | Properties |", "|---|---|---|"]
        for row in materials:
            properties = ", ".join(
                f"{key}={value}"
                for key, value in row.items()
                if key not in {"material_id", "formula_pretty"}
            )
            lines.append(
                f"| {row.get('material_id', '—')} | "
                f"{row.get('formula_pretty', '—')} | {properties or '—'} |"
            )
        if not materials:
            lines.append("| — | — | 未找到匹配材料 |")
        missing = result.get("missing_material_ids", [])
        if missing:
            lines.extend(("", "未找到：" + "、".join(str(item) for item in missing)))
        return SourceSynthesisSection(
            source_kind="materials_database",
            source_id=handoff.handoff_id,
            title="Materials Project 回查结果",
            markdown="\n".join(lines),
            evidence_ids=handoff.analysis_ids,
        )

    @staticmethod
    def synthesize(
        sections: Sequence[SourceSynthesisSection],
        *,
        warnings: Sequence[str] = (),
    ) -> SourcePartitionedSynthesis:
        if not sections:
            return SourcePartitionedSynthesis(
                status="failed", sections=(), warnings=tuple(warnings)
            )
        status: Literal["completed", "partial", "failed"] = (
            "partial" if warnings else "completed"
        )
        return SourcePartitionedSynthesis(
            status=status,
            sections=tuple(sections),
            warnings=tuple(warnings),
        )


def render_source_partitioned_synthesis(result: SourcePartitionedSynthesis) -> str:
    """Render source sections without merging database and literature claims."""

    labels = {
        "user_upload": "用户数据",
        "materials_database": "材料数据库",
        "literature": "文献实验矩阵",
        "data_analysis": "数据分析",
    }
    lines = ["## 跨 Agent 分区结果"]
    for section in result.sections:
        lines.extend(
            (
                "",
                f"### {section.title}",
                f"> 来源：{labels[section.source_kind]} · `{section.source_id}`",
                "",
                section.markdown,
            )
        )
    if result.warnings:
        lines.extend(("", "### 边界与警告", ""))
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)


def _material_row(record: Any) -> dict[str, Any]:
    symmetry = record.symmetry
    return {
        "material_id": record.material_id,
        "formula_pretty": record.formula_pretty,
        "elements": "-".join(record.elements),
        "chemsys": record.chemsys,
        "band_gap_ev": record.band_gap_ev,
        "energy_above_hull_ev_atom": record.energy_above_hull_ev_atom,
        "formation_energy_ev_atom": record.formation_energy_ev_atom,
        "density_g_cm3": record.density_g_cm3,
        "is_metal": record.is_metal,
        "is_gap_direct": record.is_gap_direct,
        "is_stable": record.is_stable,
        "theoretical": record.theoretical,
        "crystal_system": (
            symmetry.crystal_system.value
            if symmetry is not None and symmetry.crystal_system is not None
            else None
        ),
        "spacegroup_number": symmetry.number if symmetry is not None else None,
        "spacegroup_symbol": symmetry.symbol if symmetry is not None else None,
    }


def _artifact_id(kind: str, source_id: str, rows: Sequence[dict[str, Any]]) -> str:
    digest = _digest({"kind": kind, "source_id": source_id, "rows": rows})
    return f"artifact-cross-{digest[:24]}"


def _handoff_id(kind: str, payload: Any) -> str:
    return f"handoff-{kind}-{_digest(payload)[:24]}"


def _evidence_id(handoff_id: str) -> str:
    return f"evidence-{_digest(handoff_id)[:24]}"


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
