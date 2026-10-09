"""Verified multi-formula cohort, preserving original query snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from pymatgen.core.composition import Composition

from materials_screening.agent.model_base import AgentModelStatus
from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
)
from materials_screening.services.query_result_store import QueryResultStore

from .evidence_delivery import _materials_table, _response
from .formula_screening import FORMULA_SCREENING_FIELDS
from .models import SearchMaterialsInput


def _scope(message):
    text = message.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    match = re.search(r"重点考察\s*([^。；;\n]+)[。；;]", text)
    conditions = re.search(
        r"从\s*Materials\s+Project\s*查询这[^。；;\n]{0,30}?体系中\s*"
        r"非金属\s*[、，,]\s*凸包能不高于\s*(\d+(?:\.\d+)?)\s*"
        r"eV\s*/\s*atom\s*的结构",
        text,
        re.I,
    )
    if not match or not conditions:
        raise ValueError("Unsupported multi-formula screening clause")
    tokens = re.findall(r"(?:[A-Z][a-z]?\d*){2,10}", match[1])
    rest = match[1]
    for token in tokens:
        rest = rest.replace(token, "", 1)
    if (
        rest.strip(" 、，,和及与")
        or not 2 <= len(tokens) <= 10
        or len(set(tokens)) != len(tokens)
    ):
        raise ValueError("Ambiguous target formula list")
    for token in tokens:
        composition = Composition(token, strict=True)
        if composition.is_element or composition.reduced_formula != token:
            raise ValueError("Target must be an explicit reduced compound formula")
    hull = float(conditions[1])
    if not math.isfinite(hull):
        raise ValueError("Invalid hull threshold")
    # Never silently ignore another constraint in the database clause.
    clause = text.split("；", 1)[0].split(";", 1)[0]
    remainder = clause[: conditions.start()] + clause[conditions.end() :]
    if re.search(
        r"不低于|不高于|不大于|小于|大于|高于|低于|[<>≤≥]|不含|排除|限定|必须",
        remainder,
    ):
        raise ValueError("Unparsed database constraint")
    return tuple(tokens), hull


def combine_verified_queries(store, query_ids, message):
    formulas, hull = _scope(message)
    ids = tuple(dict.fromkeys(query_ids))
    if not 2 <= len(ids) <= 10:
        raise ValueError("Invalid source query set")
    expected_systems = {
        "-".join(sorted(str(e) for e in Composition(f).elements)) for f in formulas
    }
    actual_systems, versions, source_hashes = set(), set(), {}
    selected, origins, excluded = {}, {}, []
    for query_id in ids:
        folder = store._safe_folder(query_id)
        raw_meta = (folder / "metadata.json").read_bytes()
        raw_records = (folder / "records.jsonl").read_bytes()
        metadata = json.loads(raw_meta)
        args = SearchMaterialsInput.model_validate(metadata["request"])
        filters = {(f.field, f.operator.value): f.value for f in args.filters}
        system = "-".join(sorted(args.chemsys.split("-"))) if args.chemsys else None
        if args.formula:
            if args.formula not in formulas or system:
                raise ValueError("Source formula outside requested scope")
            system = "-".join(
                sorted(str(e) for e in Composition(args.formula).elements)
            )
        if (
            system not in expected_systems
            or args.required_elements
            or args.excluded_elements
            or args.material_ids
            or args.num_elements is not None
            or args.sort
            or len(args.filters) != 2
            or filters.get(("is_metal", "eq")) is not False
            or type(filters.get(("energy_above_hull_ev_atom", "lte")))
            not in (int, float)
            or filters.get(("energy_above_hull_ev_atom", "lte")) != hull
        ):
            raise ValueError("Source query filters do not match user scope")
        records = store.load_records(query_id)
        if (
            metadata.get("query_id") != query_id
            or metadata.get("source") != "materials_project"
            or metadata.get("stored_count") != len(records)
            or metadata.get("matched_count") != len(records)
        ):
            raise ValueError("Incomplete or unverified source snapshot")
        actual_systems.add(system)
        versions.add(metadata.get("database_version"))
        source_hashes[query_id] = hashlib.sha256(raw_meta + raw_records).hexdigest()
        for record in records:
            if (
                "-".join(sorted(record.chemsys.split("-"))) != system
                or record.is_metal is not False
                or (
                    record.energy_above_hull_ev_atom is None
                    or record.energy_above_hull_ev_atom > hull
                )
            ):
                raise ValueError("Stored source record violates query conditions")
            if record.formula_pretty not in formulas:
                excluded.append(record.material_id)
                continue
            if (
                record.material_id in selected
                and selected[record.material_id] != record
            ):
                raise ValueError("Conflicting records for the same material")
            selected[record.material_id] = record
            origins.setdefault(record.material_id, []).append(query_id)
    if actual_systems != expected_systems or len(versions) != 1 or None in versions:
        raise ValueError("Missing target system or inconsistent database version")
    if not selected:
        raise ValueError("No requested formula candidates")
    rows = tuple(
        sorted(selected.values(), key=lambda r: (r.formula_pretty, r.material_id))
    )
    audit = {
        "source_query_ids": list(ids),
        "source_sha256": source_hashes,
        "requested_formulas": list(formulas),
        "hull_maximum": hull,
        "record_source_query_ids": origins,
        "excluded_material_ids": sorted(set(excluded)),
    }
    digest = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    query_id = "query-cohort-" + digest[:32]
    metadata = {
        "query_id": query_id,
        "source": "materials_project",
        "database_version": next(iter(versions)),
        "request": {"cohort": audit},
        "matched_count": len(rows),
        "stored_count": len(rows),
        **audit,
        "warnings": ["仅组合明确要求的化学式；原查询与其他化学计量记录保留。"],
    }
    folder = store._safe_folder(query_id)
    if folder.exists():
        if (
            json.loads((folder / "metadata.json").read_text("utf-8")) != metadata
            or store.load_records(query_id) != rows
        ):
            raise ValueError("Existing cohort snapshot changed")
    else:
        store.save_query(query_id, metadata, rows)
    return query_id


def explicit_multi_query_calls(message):
    """Parse only the supported complete clause; never guess missing constraints."""
    try:
        formulas, hull = _scope(message)
    except (ValueError, TypeError):
        return None
    searches = []
    for index, formula in enumerate(formulas):
        args = SearchMaterialsInput.model_validate(
            {
                "formula": formula,
                "filters": [
                    {"field": "is_metal", "operator": "eq", "value": False},
                    {
                        "field": "energy_above_hull_ev_atom",
                        "operator": "lte",
                        "value": hull,
                    },
                ],
                "fields": list(FORMULA_SCREENING_FIELDS),
                "limit": 100,
            }
        )
        searches.append(
            AgentFunctionCallItem(
                call_id=f"explicit-multi-query-{index}",
                name="search_materials",
                arguments=args.model_dump_json(),
            )
        )
    return tuple(searches)


def deliver_multi_query_handoff(response, items, message):
    """Only successful current-turn paired tools can create a cohort reference."""
    if response.status != AgentModelStatus.COMPLETED or response.tool_calls:
        return None
    try:
        _scope(message)
        draft = json.loads(response.message_text or "")
        if draft.get("status") not in {"completed", "needs_user_input"}:
            return None
        calls = {x.call_id: x for x in items if isinstance(x, AgentFunctionCallItem)}
        ids, evidence = [], []
        store = QueryResultStore(Path("data/material_queries"))
        for item in items:
            if not isinstance(item, AgentFunctionOutputItem):
                continue
            call = calls.get(item.call_id)
            envelope = json.loads(item.output)
            if call is None or envelope.get("status") != "ok":
                raise ValueError("Missing paired tool or failed query step")
            if call.name != "search_materials":
                continue
            if envelope.get("tool_name") != call.name:
                raise ValueError("Tool identity mismatch")
            query_id = envelope["output"]["query_id"]
            metadata = json.loads(
                (store._safe_folder(query_id) / "metadata.json").read_text("utf-8")
            )
            if SearchMaterialsInput.model_validate_json(
                call.arguments
            ) != SearchMaterialsInput.model_validate(metadata["request"]):
                raise ValueError("Query arguments do not match saved snapshot")
            ids.append(query_id)
            evidence.append(envelope["evidence_id"])
        if len(set(ids)) < 2:
            return None
        query_id = combine_verified_queries(store, ids, message)
        rows = store.load_records(query_id)
        metadata = json.loads(
            (store.root / query_id / "metadata.json").read_text("utf-8")
        )
        material_ids = []
        from materials_screening.services.material_database_service import (
            project_record,
        )

        table = _materials_table(
            [project_record(r, FORMULA_SCREENING_FIELDS) for r in rows],
            list(FORMULA_SCREENING_FIELDS),
            material_ids,
        )
        lines = [
            f"数据库初筛得到 {len(rows)} 个目标结构，"
            "按用户要求核对了化学式、非金属条件和凸包能上限。",
            table,
            "计算带隙与稳定性用于初筛，不等于薄膜透过率或导电性能。",
            "is_stable=false 表示该计算结构不在凸包上，不直接说明无法制备。",
            f"MATERIAL_QUERY_HANDOFF: query_id={query_id}",
        ]
        if metadata["excluded_material_ids"]:
            lines.insert(1, "其他化学计量的结构未纳入本次目标候选，原查询记录保留。")
        return _response(
            "\n\n".join(lines), material_ids, evidence, metadata["warnings"]
        )
    except (ValueError, KeyError, TypeError, OSError):
        return None
