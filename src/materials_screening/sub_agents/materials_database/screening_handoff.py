"""Finish a narrow master-owned screening step only from a verified snapshot.

Standalone details/export/statistics and unparsed constraints retain normal model
behavior. This is a workload boundary, not a higher tool/model call budget.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

from materials_screening.agent.model_base import MaterialAgentRequest
from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
)
from materials_screening.services.material_database_service import project_record
from materials_screening.services.query_result_store import QueryResultStore

from .evidence_delivery import _response, deliver_query_evidence
from .formula_screening import formula_request_matches, parse_formula_screening
from .models import SearchMaterialsInput

SCREENING_HANDOFF_SCOPE = (
    "\nMASTER_SCREENING_SNAPSHOT_ONLY：本步仅按原问题的明确条件筛选并核对查询快照。"
    "筛选成功后立即交接 query_id，不继续逐批查询详情或计算分布；"
    "分组统计、文献检索与应用评估由主控交给后续 Agent。"
    "条件不明确、查询失败或没有候选时如实返回，不放宽条件或宣称全流程完成。"
)
_CLAUSE = re.compile(
    r"(?:请)?从\s*Materials?\s+Project\s*筛选\s*化学体系\s*严格为\s*"
    r"(?P<chemsys>[A-Z][a-z]?(?:\s*-\s*[A-Z][a-z]?){1,9})\s*[、，,]\s*"
    r"凸包能\s*(?:不高于|不大于|小于等于|≤|<=)\s*"
    r"(?P<hull>\d+(?:\.\d+)?)\s*eV\s*/\s*atom\s*的非金属材料"
    r"(?=\s*[，,]\s*(?:分析|并分析))"
)
_OTHER_OPERATION = re.compile(r"详情|导出|排序|离群|异常检测|top[- ]?\d|前\s*\d+", re.I)
_EXTRA_CONSTRAINT = re.compile(
    r"不低于|不高于|不大于|小于|大于|高于|低于|等于|介于|至少|至多|范围为|[<>≤≥]"
)
_FIELDS = {
    "band_gap_ev",
    "density_g_cm3",
    "energy_above_hull_ev_atom",
    "crystal_system",
}


def _constraints(message: str) -> tuple[set[str], float] | None:
    matches = list(_CLAUSE.finditer(message))
    if (
        len(matches) != 1
        or message.count("筛选") != 1
        or _OTHER_OPERATION.search(message)
    ):
        return None
    match = matches[0]
    remaining = message[: match.start()] + message[match.end() :]
    if _EXTRA_CONSTRAINT.search(remaining) or re.search(
        r"要求|限定|必须|不含|排除|仅含|只要|仅限|同时满足|(?:exclude|without)\b",
        remaining,
        re.I,
    ):
        return None
    elements = {e.strip() for e in match["chemsys"].split("-")}
    if len(elements) != len(match["chemsys"].split("-")) or not all(
        Element.is_valid_symbol(e) for e in elements
    ):
        return None
    hull = float(match["hull"])
    return (elements, hull) if math.isfinite(hull) else None


def with_screening_handoff_scope(task: str) -> str:
    """Scope only completely recognized strict-system or explicit-formula filters."""
    return (
        task + SCREENING_HANDOFF_SCOPE
        if _constraints(task) or parse_formula_screening(task)
        else task
    )


def _request_matches(
    args: SearchMaterialsInput, elements: set[str], hull: float
) -> bool:
    if not args.chemsys or set(args.chemsys.split("-")) != elements:
        return False
    if (
        args.required_elements
        or args.excluded_elements
        or args.formula
        or args.material_ids
        or args.num_elements is not None
        or args.sort
    ):
        return False
    filters = {(f.field, f.operator.value): f.value for f in args.filters}
    return (
        len(args.filters) == 2
        and set(filters) == {("is_metal", "eq"), ("energy_above_hull_ev_atom", "lte")}
        and filters[("is_metal", "eq")] is False
        and type(filters[("energy_above_hull_ev_atom", "lte")]) in (int, float)
        and filters[("energy_above_hull_ev_atom", "lte")] == hull
        and _FIELDS.issubset(args.fields)
    )


def scoped_screening_handoff(request: MaterialAgentRequest):
    """Return no early final unless current call, request and all stored rows agree."""
    indexes = [
        i
        for i, x in enumerate(request.input_items)
        if isinstance(x, AgentMessageItem) and x.role == "user"
    ]
    if not indexes:
        return None
    last = indexes[-1]
    message = request.input_items[last].content
    if not message.endswith(SCREENING_HANDOFF_SCOPE):
        return None
    original = message[: -len(SCREENING_HANDOFF_SCOPE)]
    constraints = _constraints(original)
    formula_search = parse_formula_screening(original)
    if constraints is None and formula_search is None:
        return None
    if constraints is not None:
        elements, hull = constraints
    else:
        elements = {str(e) for e in Composition(formula_search.formula).elements}
        hull = next(
            f.value
            for f in formula_search.filters
            if f.field == "energy_above_hull_ev_atom"
        )
    items = request.input_items[last + 1 :]
    calls = [x for x in items if isinstance(x, AgentFunctionCallItem)]
    outputs = [x for x in items if isinstance(x, AgentFunctionOutputItem)]
    if not calls or len(calls) != len(outputs):
        return None
    call_map = {x.call_id: x for x in calls}
    if (
        len(call_map) != len(calls)
        or len({x.call_id for x in outputs}) != len(outputs)
        or {x.call_id for x in outputs} != set(call_map)
        or any(x.name != "search_materials" for x in calls)
    ):
        return None
    try:
        successful = []
        for output in outputs:
            envelope = json.loads(output.output)
            if envelope.get("tool_name") != "search_materials":
                return None
            if envelope.get("status") != "ok":
                if envelope.get("error", {}).get("code") != "INVALID_ARGUMENTS":
                    return None
                continue
            args = SearchMaterialsInput.model_validate_json(
                call_map[output.call_id].arguments
            )
            matches = (
                formula_request_matches(args, formula_search)
                if formula_search is not None
                else _request_matches(args, elements, hull)
            )
            if not matches:
                return None
            successful.append((args, envelope))
        if len(successful) != 1:
            return None
        args, envelope = successful[0]
        payload = envelope["output"]
        query_id = payload["query_id"]
        if not isinstance(query_id, str) or not re.fullmatch(
            r"query-[A-Za-z0-9_-]+", query_id
        ):
            return None
        count = payload["matched_count"]
        if type(count) is not int or not 0 < count <= 1000:
            return None
        if payload.get("source") != "materials_project" or not envelope.get(
            "evidence_id"
        ):
            return None
        folder = Path("data/material_queries") / query_id
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if (
            metadata.get("query_id") != query_id
            or metadata.get("source") != "materials_project"
            or metadata.get("matched_count") != count
            or metadata.get("stored_count") != count
            or SearchMaterialsInput.model_validate(metadata["request"]) != args
        ):
            return None
        rows = QueryResultStore(folder.parent).load_records(query_id)
        if len(rows) != count or len({r.material_id for r in rows}) != count:
            return None
        if not all(
            (
                r.formula_pretty == formula_search.formula
                and set(r.elements) == elements
                if formula_search is not None
                else set(r.elements) == elements
            )
            and r.is_metal is False
            and r.energy_above_hull_ev_atom is not None
            and math.isfinite(r.energy_above_hull_ev_atom)
            and 0 <= r.energy_above_hull_ev_atom <= hull
            and (
                formula_search is None
                or all(
                    r.symmetry is not None and r.symmetry.crystal_system == f.value
                    for f in formula_search.filters
                    if f.field == "crystal_system"
                )
            )
            for r in rows
        ):
            return None
        shown = payload.get("materials")
        if (
            not isinstance(shown, list)
            or not shown
            or payload.get("returned_count") != len(shown)
        ):
            return None
        expected_page = json.loads(
            json.dumps(
                [project_record(r, args.fields) for r in rows[: min(args.limit, 20)]]
            )
        )
        if payload.get("fields") != list(args.fields) or shown != expected_page:
            return None
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    seed = _response("", [], [], [])
    return deliver_query_evidence(seed, items, require_query=True)
