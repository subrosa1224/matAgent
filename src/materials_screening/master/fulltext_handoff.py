"""Explicit snapshot-to-trial staging; never reads a saved topic report."""

from __future__ import annotations

import hashlib
import json
import math
import re

from materials_screening.master.literature_measurement_review import (
    review_measurements,
    source_columns,
)
from materials_screening.sub_agents.literature.matrix import validate_matrix_evidence
from materials_screening.sub_agents.literature.matrix_automation import (
    group_has_ungrounded_boolean_fields,
    measurement_binding_rejection_reason,
)

from .fulltext_snapshots import CheckedChunkStore, ExtractionSnapshot, source_identities
from .fulltext_tasks import MeasurementHandoff


def _reported_scalar(value_text: str, unit: str | None) -> float:
    """Parse one scalar, optionally followed by its exact declared unit.

    Only whitespace and minus typography are normalized; no unit conversion,
    prefix folding, range reduction, or model numeric value fallback.
    """
    text = value_text.replace("−", "-")
    number = r"(?:(?:about|around|approximately|约)\s*)?([+-]?\d+(?:\.\d+)?)"
    match = re.fullmatch(r"\s*" + number + r"\s*", text, re.I)
    if match is None and unit and unit.strip():
        normalized = re.sub(r"\s+", "", unit.replace("−", "-"))
        # Case-sensitive unit match, while approximate qualifiers ignore case.
        prefix = re.match(r"\s*" + number, text, re.I)
        if (
            prefix is not None
            and re.sub(r"\s+", "", text[prefix.end() :]) == normalized
        ):
            match = prefix
    if match is None:
        raise ValueError("Not a reported scalar with a matching declared unit")
    return float(match.group(1))


def snapshot_trial_handoff(
    snapshot: ExtractionSnapshot, *, chunks, dataset_factory, existing=None
):
    if snapshot.status != "complete" or snapshot.matrix is None:
        raise ValueError("Only complete explicit snapshots can stage measurements")
    store = CheckedChunkStore(tuple(chunks))
    if source_identities(chunks) != snapshot.source_identities or any(
        row.document_id != snapshot.document_id
        or hashlib.sha256(row.text.encode()).hexdigest() != row.text_sha256
        for row in chunks
    ):
        raise ValueError("Snapshot sources do not match current checked chunks")
    chunk_map = {row.chunk_id: row for row in chunks}
    groups = {group.group_id: group for group in snapshot.matrix.groups}
    eligible, isolated, isolation_reasons = [], [], {}
    warnings = [
        "本数据集来自本任务的新提取快照，保持 pending，仅供试运行；未经过专家审核。",
        "任务指标覆盖与关键条件可比性仍需分析阶段核对；当前未执行统计或跨样品排名。",
        "测量条目不是独立实验重复；未知测试条件不视为相同条件。",
    ]
    for row in snapshot.matrix.measurements:
        group = groups.get(row.group_id)
        rejection = "invalid_source_evidence"
        try:
            if (
                group is None
                or group.role == "reference"
                or group_has_ungrounded_boolean_fields(group)
            ):
                rejection = "unsafe_or_reference_sample_group"
                raise ValueError("Unsafe sample group")
            validate_matrix_evidence(
                store,
                document_id=snapshot.document_id,
                groups=(group,),
                measurements=(row,),
            )
            if (
                measurement_binding_rejection_reason(row, group, chunks_by_id=chunk_map)
                is not None
            ):
                rejection = "sample_value_binding_rejected"
                raise ValueError("Unsafe sample/value binding")
            # Bounds and ranges are retained in the snapshot, never coerced to
            # midpoint or to a reported scalar in the new trial path.
            rejection = "non_scalar_value"
            value = _reported_scalar(row.value_text, row.unit)
            if not math.isfinite(value) or (
                row.numeric_value is not None and row.numeric_value != value
            ):
                rejection = "inconsistent_numeric_value"
                raise ValueError("Inconsistent scalar")
        except ValueError:
            isolated.append(row.measurement_id)
            isolation_reasons[row.measurement_id] = (rejection,)
            continue
        eligible.append(row)
    review = review_measurements(eligible)
    reviewed_ids = {row.measurement_id for row in review.accepted}
    isolated.extend(
        row.measurement_id for row in eligible if row.measurement_id not in reviewed_ids
    )
    isolation_reasons.update(
        {
            row.measurement_id: ("duplicate_or_conflicting_value",)
            for row in eligible
            if row.measurement_id not in reviewed_ids
        }
    )
    warnings.extend(review.warnings)
    if isolated:
        warnings.append(
            f"隔离 {len(set(isolated))} 条非标量、来源/样品绑定或冲突未解决的测量；"
            "原快照不变。"
        )
    records = []
    for row in review.accepted:
        group = groups[row.group_id]
        value = _reported_scalar(row.value_text, row.unit)
        context = json.dumps(
            {
                "document_id": row.document_id,
                "group_id": row.group_id,
                "metric": row.metric,
                "unit": row.unit,
                "conditions": group.conditions,
                # No condition data is not a common experimental setup.
                "unknown_condition_partition": row.measurement_id
                if not group.conditions
                else None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        records.append(
            {
                "task_id": snapshot.task_id,
                "snapshot_id": snapshot.snapshot_id,
                "attempt_id": snapshot.attempt_id,
                "analysis_mode": "trial",
                "document_id": row.document_id,
                "paper_title": snapshot.title,
                "pdf_sha256": snapshot.pdf_sha256,
                "group_id": group.group_id,
                "group_label": group.label,
                "group_role": group.role,
                "material": group.material,
                "variables": json.dumps(
                    group.variables, ensure_ascii=False, sort_keys=True
                ),
                "conditions": json.dumps(
                    group.conditions, ensure_ascii=False, sort_keys=True
                ),
                "measurement_id": row.measurement_id,
                "metric": row.metric,
                "value_text": row.value_text,
                "numeric_value": value,
                "unit": row.unit,
                "numeric_derivation": "approximate_reported_scalar"
                if re.search(r"about|around|approximately|约", row.value_text, re.I)
                else "reported_scalar",
                "uncertainty_text": row.uncertainty_text,
                "sample_size_reported": row.sample_size,
                "independent_replicates_confirmed": False,
                "review_status": row.review_status,
                "original_review_status": row.review_status,
                "source_quote": row.source_quote,
                "source_chunk_id": row.chunk_id,
                "source_text_sha256": row.source_text_sha256,
                "page_from": row.page_from,
                "page_to": row.page_to,
                "group_source_quote": group.source_quote,
                "group_source_chunk_id": group.chunk_id,
                "group_source_text_sha256": group.source_text_sha256,
                "measurement_context": context,
                **source_columns(review, row.measurement_id),
            }
        )
    dataset_id = None
    if existing is not None:
        if (
            existing.snapshot_id != snapshot.snapshot_id
            or existing.record_count != len(records)
            or existing.isolated_measurement_ids != tuple(dict.fromkeys(isolated))
            or existing.isolation_reasons != isolation_reasons
            or existing.warnings != tuple(dict.fromkeys(warnings))
        ):
            raise ValueError("Snapshot handoff no longer matches its evidence gates")
        if records:
            if existing.dataset_id is None:
                raise ValueError("Handoff has records but no dataset")
            datasets = dataset_factory()
            reference = datasets.get(existing.dataset_id)
            current = json.loads(
                datasets.load_dataframe(existing.dataset_id).to_json(
                    orient="records", force_ascii=False
                )
            )
            if (
                reference.source_artifact_id
                != "artifact-fulltext-" + snapshot.snapshot_id.removeprefix("snapshot-")
                or current != records
            ):
                raise ValueError("Stored trial dataset differs from explicit snapshot")
        elif existing.dataset_id is not None:
            raise ValueError("An empty handoff must not have a dataset")
        return existing
    if records:
        dataset = dataset_factory().register_records(
            records,
            source_artifact_id="artifact-fulltext-"
            + snapshot.snapshot_id.removeprefix("snapshot-"),
            display_name=snapshot.snapshot_id + "-trial-measurements.json",
        )
        dataset_id = dataset.dataset_id
    return MeasurementHandoff(
        snapshot_id=snapshot.snapshot_id,
        dataset_id=dataset_id,
        record_count=len(records),
        isolated_measurement_ids=tuple(dict.fromkeys(isolated)),
        isolation_reasons=isolation_reasons,
        warnings=tuple(dict.fromkeys(warnings)),
    )
