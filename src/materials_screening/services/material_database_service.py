"""Deterministic Materials Project query and analysis services."""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from materials_screening.models import FloatRange, MaterialRecord, ScreeningRequest
from materials_screening.repositories.base import MaterialsRepository
from materials_screening.services.query_result_store import QueryResultStore
from materials_screening.sub_agents.materials_database.models import (
    NUMERIC_FIELDS,
    OUTPUT_FIELDS,
    SearchMaterialsInput,
)


def record_value(record: MaterialRecord, field: str) -> Any:
    if field == "crystal_system":
        return (
            record.symmetry.crystal_system.value
            if record.symmetry and record.symmetry.crystal_system
            else None
        )
    if field == "spacegroup_number":
        return record.symmetry.number if record.symmetry else None
    if field == "spacegroup_symbol":
        return record.symmetry.symbol if record.symmetry else None
    return getattr(record, field, None)


def project_record(record: MaterialRecord, fields: Iterable[str]) -> dict[str, Any]:
    selected = tuple(dict.fromkeys(("material_id", "formula_pretty", *fields)))
    unsupported = set(selected) - OUTPUT_FIELDS
    if unsupported:
        raise ValueError(f"unsupported fields: {sorted(unsupported)}")
    return {field: record_value(record, field) for field in selected}


class MaterialDatabaseService:
    def __init__(
        self, repository: MaterialsRepository, store: QueryResultStore
    ) -> None:
        self.repository = repository
        self.store = store

    def search(self, request: SearchMaterialsInput) -> dict[str, Any]:
        screening = self._screening_request(request)
        retrieval = self.repository.search(screening)
        records = [
            record for record in retrieval.records if self._matches(record, request)
        ]
        for rule in reversed(request.sort):
            records.sort(
                key=lambda item: self._sort_key(record_value(item, rule.field)),
                reverse=rule.direction == "desc",
            )
        matched_count = len(records)
        stored = records[:1000]
        query_id = f"query-{uuid.uuid4().hex}"
        created_at = datetime.now(UTC)
        self.store.save_query(
            query_id,
            {
                "query_id": query_id,
                "source": retrieval.source,
                "database_version": retrieval.database_version,
                "created_at": created_at.isoformat(),
                "request": request.model_dump(mode="json"),
                "matched_count": matched_count,
                "stored_count": len(stored),
                "warnings": list(retrieval.warnings),
            },
            stored,
        )
        shown = stored[: min(request.limit, 20)]
        return {
            "query_id": query_id,
            "source": "materials_project",
            "matched_count": matched_count,
            "returned_count": len(shown),
            "fields": list(request.fields),
            "materials": [project_record(record, request.fields) for record in shown],
            "warnings": list(retrieval.warnings),
            "created_at": created_at.isoformat(),
        }

    def details(
        self, material_ids: Iterable[str], fields: Iterable[str]
    ) -> dict[str, Any]:
        records: list[MaterialRecord] = []
        missing: list[str] = []
        for material_id in tuple(material_ids):
            record = self.repository.query_by_material_id(material_id)
            (records if record else missing).append(record if record else material_id)  # type: ignore[arg-type]
        return {
            "materials": [project_record(record, fields) for record in records],
            "missing_material_ids": missing,
        }

    def page(
        self, query_id: str, offset: int, limit: int, fields: Iterable[str]
    ) -> dict[str, Any]:
        records = self.store.load_records(query_id)
        page = records[offset : offset + limit]
        return {
            "query_id": query_id,
            "offset": offset,
            "limit": limit,
            "total": len(records),
            "materials": [project_record(record, fields) for record in page],
        }

    def compare(
        self,
        material_ids: Iterable[str],
        query_id: str | None,
        properties: Iterable[str],
    ) -> dict[str, Any]:
        props = self._numeric_properties(properties)
        if query_id:
            available = {r.material_id: r for r in self.store.load_records(query_id)}
            ids = tuple(material_ids) or tuple(available)[:20]
            records = [available[mid] for mid in ids if mid in available]
        else:
            records = [
                r
                for mid in material_ids
                if (r := self.repository.query_by_material_id(mid))
            ]
        rows = [project_record(record, props) for record in records]
        summaries: dict[str, Any] = {}
        for prop in props:
            values = [(r.material_id, record_value(r, prop)) for r in records]
            valid = [
                (mid, value) for mid, value in values if isinstance(value, (int, float))
            ]
            summaries[prop] = (
                {
                    "min": min(valid, key=lambda pair: pair[1]),
                    "max": max(valid, key=lambda pair: pair[1]),
                }
                if valid
                else None
            )
        return {"materials": rows, "summaries": summaries}

    def describe(
        self, query_id: str, properties: Iterable[str], include_correlation: bool
    ) -> dict[str, Any]:
        records = self.store.load_records(query_id)
        props = self._numeric_properties(properties)
        result: dict[str, Any] = {}
        for prop in props:
            values = [
                float(v)
                for r in records
                if isinstance((v := record_value(r, prop)), (int, float))
            ]
            result[prop] = self._describe_values(values, len(records))
        correlations: dict[str, float | None] = {}
        if include_correlation:
            for i, left in enumerate(props):
                for right in props[i + 1 :]:
                    pairs = [
                        (float(a), float(b))
                        for r in records
                        if isinstance((a := record_value(r, left)), (int, float))
                        and isinstance((b := record_value(r, right)), (int, float))
                    ]
                    correlations[f"{left}:{right}"] = self._correlation(pairs)
        analysis_id = f"analysis-{uuid.uuid4().hex}"
        payload = {
            "analysis_id": analysis_id,
            "query_id": query_id,
            "analysis_type": "description",
            "statistics": result,
            "correlations": correlations,
        }
        self.store.save_analysis(analysis_id, payload)
        return payload

    def export(
        self,
        query_id: str | None,
        analysis_id: str | None,
        format_: str,
        fields: Iterable[str],
    ) -> dict[str, Any]:
        export_root = self.store.root / "exports"
        export_root.mkdir(parents=True, exist_ok=True)
        ref = query_id or analysis_id
        assert ref is not None
        path = export_root / f"{ref}.{('md' if format_ == 'markdown' else format_)}"
        if query_id:
            rows = [
                project_record(r, fields or OUTPUT_FIELDS)
                for r in self.store.load_records(query_id)
            ]
            self._write_rows(path, rows, format_)
            count = len(rows)
        else:
            payload = self.store.load_analysis(analysis_id or "")
            path.write_text(self._render_payload(payload, format_), "utf-8")
            count = 1
        return {
            "reference_id": ref,
            "format": format_,
            "path": str(path.resolve()),
            "record_count": count,
        }

    @staticmethod
    def _screening_request(request: SearchMaterialsInput) -> ScreeningRequest:
        ranges: dict[str, dict[str, float]] = {}
        booleans: dict[str, bool] = {}
        for item in request.filters:
            if item.field in NUMERIC_FIELDS and item.operator.value in {
                "gte",
                "lte",
                "gt",
                "lt",
            }:
                bound = "min" if item.operator.value in {"gte", "gt"} else "max"
                ranges.setdefault(item.field, {})[bound] = float(item.value)
            elif (
                item.field in {"is_stable", "is_metal", "theoretical"}
                and item.operator.value == "eq"
            ):
                booleans[item.field] = bool(item.value)
        return ScreeningRequest(
            required_elements=request.required_elements,
            excluded_elements=request.excluded_elements,
            chemsys=request.chemsys,
            formula=request.formula,
            material_ids=request.material_ids,
            num_elements=request.num_elements,
            band_gap_ev=FloatRange(**ranges["band_gap_ev"])
            if "band_gap_ev" in ranges
            else None,
            density_g_cm3=FloatRange(**ranges["density_g_cm3"])
            if "density_g_cm3" in ranges
            else None,
            energy_above_hull_ev_atom=FloatRange(**ranges["energy_above_hull_ev_atom"])
            if "energy_above_hull_ev_atom" in ranges
            else None,
            is_stable=booleans.get("is_stable"),
            is_metal=booleans.get("is_metal"),
            theoretical=booleans.get("theoretical"),
            limit=min(request.limit, 100),
        )

    @staticmethod
    def _matches(record: MaterialRecord, request: SearchMaterialsInput) -> bool:
        elements = set(record.elements)
        if request.num_elements is not None and len(elements) != request.num_elements:
            return False
        if request.required_elements and not set(request.required_elements).issubset(
            elements
        ):
            return False
        if request.excluded_elements and set(request.excluded_elements) & elements:
            return False
        if request.formula and record.formula_pretty != request.formula:
            return False
        if request.material_ids and record.material_id not in request.material_ids:
            return False
        if request.chemsys:
            expected_elements = {
                element.strip()
                for element in request.chemsys.split("-")
                if element.strip()
            }
            if set(record.elements) != expected_elements:
                return False
        for item in request.filters:
            actual = record_value(record, item.field)
            expected = item.value
            if actual is None:
                return False
            operator = item.operator.value
            if operator == "eq":
                matched = actual == expected
            elif operator == "gte":
                matched = actual >= expected
            elif operator == "lte":
                matched = actual <= expected
            elif operator == "gt":
                matched = actual > expected
            else:
                matched = actual < expected
            if not matched:
                return False
        return True

    @staticmethod
    def _sort_key(value: Any) -> tuple[bool, Any]:
        return (value is None, value if value is not None else 0)

    @staticmethod
    def _numeric_properties(properties: Iterable[str]) -> tuple[str, ...]:
        props = tuple(dict.fromkeys(properties))
        unsupported = set(props) - NUMERIC_FIELDS
        if unsupported:
            raise ValueError(
                f"properties are not numeric/analysable: {sorted(unsupported)}"
            )
        return props

    @staticmethod
    def _describe_values(values: list[float], total: int) -> dict[str, Any]:
        if not values:
            return {"count": 0, "missing": total}
        ordered = sorted(values)
        q = (
            statistics.quantiles(ordered, n=4, method="inclusive")
            if len(ordered) > 1
            else [ordered[0]] * 3
        )
        return {
            "count": len(values),
            "missing": total - len(values),
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": ordered[0],
            "q1": q[0],
            "median": statistics.median(values),
            "q3": q[2],
            "max": ordered[-1],
        }

    @staticmethod
    def _correlation(pairs: list[tuple[float, float]]) -> float | None:
        if len(pairs) < 2:
            return None
        xs, ys = zip(*pairs, strict=True)
        sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
        if math.isclose(sx, 0) or math.isclose(sy, 0):
            return None
        return float(
            sum(
                (x - statistics.fmean(xs)) * (y - statistics.fmean(ys))
                for x, y in pairs
            )
            / len(pairs)
            / sx
            / sy
        )

    @staticmethod
    def _write_rows(path: Path, rows: list[dict[str, Any]], format_: str) -> None:
        if format_ == "json":
            path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2, default=str), "utf-8"
            )
        elif format_ == "csv":
            fields = list(rows[0]) if rows else []
            with path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        else:
            headers = list(rows[0]) if rows else []
            lines = [
                "| " + " | ".join(headers) + " |",
                "|" + "|".join(["---"] * len(headers)) + "|",
            ]
            lines.extend(
                "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
                for row in rows
            )
            path.write_text("\n".join(lines), "utf-8")

    @staticmethod
    def _render_payload(payload: dict[str, Any], format_: str) -> str:
        if format_ == "json":
            return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        if format_ == "csv":
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(["key", "value"])
            for key, value in payload.items():
                writer.writerow(
                    [key, json.dumps(value, ensure_ascii=False, default=str)]
                )
            return buffer.getvalue()
        return (
            "# Materials analysis result\n\n```json\n"
            + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            + "\n```\n"
        )
