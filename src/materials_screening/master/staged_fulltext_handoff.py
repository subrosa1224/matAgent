"""New composite-source trial handoff; never calls legacy matrix handoff gates."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .fulltext_snapshots import source_identities
from .staged_fulltext_conditions import validate_staged_attribute
from .staged_fulltext_evidence import reported_scalar, validate_composite
from .staged_fulltext_store import StagedExtractionRecord, StagedReference


class StagedHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reference: StagedReference
    dataset_id: str | None = Field(default=None, pattern=r"^dataset-[a-f0-9]{24}$")
    record_count: int = Field(ge=0)
    isolated_measurement_ids: tuple[str, ...] = ()
    review_status: Literal["pending"] = "pending"
    analysis_id: str | None = Field(default=None, pattern=r"^analysis-[a-f0-9]{32}$")
    analysis_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def staged_trial_handoff(record, *, reference, chunks, dataset_factory, existing=None):
    record = StagedExtractionRecord.model_validate(record.model_dump())
    reference = StagedReference.model_validate(reference.model_dump())
    if (
        record.status != "requests_complete"
        or reference.record_id != record.record_id
        or reference.document_id != record.document_id
        or hashlib.sha256(record.model_dump_json(indent=2).encode()).hexdigest()
        != reference.content_sha256
        or source_identities(chunks) != record.source_identities
    ):
        raise ValueError(
            "Only matching source-checked completed stage requests can hand off"
        )
    samples = {s.sample_id: s for s in record.samples}
    candidates = {
        m.measurement_id: dict(
            document_id=record.document_id,
            group_label=samples[m.sample_id].label,
            metric=m.metric,
            unit=m.unit,
        )
        for m in record.measurements
    }
    bindings = defaultdict(list)
    for plan in record.condition_plans:
        for b in plan.bindings:
            validate_staged_attribute(b, candidates, chunks, samples=samples)
            for mid in b.measurement_ids:
                bindings[mid].append(b)
    records, isolated, seen = [], [], set()
    for m in record.measurements:
        sample = samples[m.sample_id]
        validate_composite(sample, m, chunks, samples=samples)
        try:
            value = reported_scalar(m.value_text)
        except ValueError:
            isolated.append(m.measurement_id)
            continue
        # Identical overlapping PDF citations are not independent replicates.
        identity = (
            m.sample_id,
            m.metric,
            m.unit,
            m.value_text,
            m.source.page_from,
            " ".join(m.source.quote.split()),
        )
        if identity in seen:
            isolated.append(m.measurement_id)
            continue
        seen.add(identity)
        by_field = defaultdict(list)
        for b in bindings[m.measurement_id]:
            by_field[(b.kind, b.key)].append(b)
        variables, conditions, states = {}, {}, {}
        for (kind, key), rows in by_field.items():
            values = tuple(dict.fromkeys(b.value_text for b in rows))
            state = "known" if len(values) == 1 else "conflict"
            states[kind + ":" + key] = state
            target = variables if kind == "preparation" else conditions
            target[key] = values[0] if state == "known" else list(values)
        condition_state = (
            "conflict"
            if any(
                k.startswith("condition:") and v == "conflict"
                for k, v in states.items()
            )
            else "known_fields_only"
            if conditions
            else "unknown"
        )
        context = _json(
            dict(
                document_id=record.document_id,
                sample_id=m.sample_id,
                metric=m.metric,
                unit=m.unit,
                conditions=conditions,
                unknown_condition_partition=m.measurement_id,
            )
        )
        records.append(
            dict(
                task_id=record.task_id,
                staged_record_id=record.record_id,
                extraction_contract=record.version,
                document_id=record.document_id,
                paper_title=record.title,
                pdf_sha256=record.pdf_sha256,
                sample_id=m.sample_id,
                group_id=m.sample_id,
                group_label=sample.label,
                material=sample.label,
                material_status=sample.material_status,
                group_role=sample.role,
                measurement_id=m.measurement_id,
                metric=m.metric,
                value_text=m.value_text,
                numeric_value=value,
                unit=m.unit,
                value_qualifier=m.qualifier,
                analysis_mode="trial",
                review_status="pending",
                independent_replicates_confirmed=False,
                variables=_json(variables),
                conditions=_json(conditions),
                condition_status=_json(dict(state=condition_state, fields=states)),
                attribute_sources=_json(
                    [b.model_dump() for b in bindings[m.measurement_id]]
                ),
                comparison_policy=_json(
                    dict(
                        can_rank=False, reason="conditions_not_independently_comparable"
                    )
                ),
                source_quote=m.source.quote,
                source_chunk_id=m.source.chunk_id,
                source_text_sha256=m.source.text_sha256,
                page_from=m.source.page_from,
                page_to=m.source.page_to,
                sample_source_quote=sample.definition.quote,
                sample_source_chunk_id=sample.definition.chunk_id,
                sample_source_text_sha256=sample.definition.text_sha256,
                sample_binding_source=_json(m.sample_binding_source.model_dump()),
                measurement_context=context,
            )
        )
    if existing:
        existing = StagedHandoff.model_validate(existing.model_dump())
        if (
            existing.reference != reference
            or existing.record_count != len(records)
            or existing.isolated_measurement_ids != tuple(isolated)
        ):
            raise ValueError("Saved staged handoff no longer matches replay")
        if records:
            datasets = dataset_factory()
            current = json.loads(
                datasets.load_dataframe(existing.dataset_id).to_json(
                    orient="records", force_ascii=False
                )
            )
            if (
                current != records
                or datasets.get(existing.dataset_id).source_artifact_id
                != "artifact-staged-" + record.record_id[7:]
            ):
                raise ValueError("Staged dataset differs from composite evidence")
        elif existing.dataset_id:
            raise ValueError("Empty staged handoff has dataset")
        return existing
    dataset_id = None
    if records:
        dataset = dataset_factory().register_records(
            records,
            source_artifact_id="artifact-staged-" + record.record_id[7:],
            display_name=record.record_id + "-trial.json",
        )
        dataset_id = dataset.dataset_id
    return StagedHandoff(
        reference=reference,
        dataset_id=dataset_id,
        record_count=len(records),
        isolated_measurement_ids=tuple(isolated),
    )


def render_staged_result(record, handoff):
    """User-facing, scientific boundaries; identifiers belong in diagnostics."""
    samples = {s.sample_id: s for s in record.samples}
    names = {"transmittance": "透过率", "resistivity": "电阻率"}
    attribute_names = {
        "dopant_concentration": "掺杂浓度",
        "doping_concentration": "掺杂浓度",
        "deposition_method": "制备方法",
        "substrate": "基底",
        "temperature": "测试温度",
        "measurement_wavelength": "测量波长",
        "transmittance_wavelength": "透过率测试波长",
        "optical_wavelength": "光学测试波长",
        "wavelength": "测试波长",
        "wavelength_range": "测试波长范围",
        "measurement_wavelength_range": "测量波长范围",
        "transmittance_wavelength_range": "透过率测试波长范围",
        "optical_wavelength_range": "光学测试波长范围",
        "annealing_temperature": "退火温度",
    }
    lines = [
        f"{record.title}",
        "",
        "已核对的实验记录（尚待核对适用条件与专业准确性）：",
    ]
    isolated = set(handoff.isolated_measurement_ids)
    for m in record.measurements:
        nonscalar = m.qualifier in {"bound", "range"}
        if m.measurement_id in isolated and not nonscalar:
            continue
        value = m.value_text
        unresolved_limit = m.qualifier == "bound" and bool(
            re.match(r"^up\s+to\b", value, re.I)
        )
        if m.qualifier == "approximate":
            value = "约" + re.sub(
                r"^(?:around(?:\s+of)?|about|approximately|约|~|≈)\s*",
                "",
                value,
                flags=re.I,
            )
        if m.qualifier == "bound":
            for word, label in (
                ("more than", "大于"),
                ("less than", "小于"),
                ("at least", "不低于"),
                ("at most", "不高于"),
                ("above", "高于"),
                ("below", "低于"),
            ):
                value = re.sub(r"^" + word + r"\s+", label, value, flags=re.I)
        domain = next(
            (
                term
                for term in ("UV–visible", "UV-visible", "visible")
                if term in m.source.quote
            ),
            None,
        )
        display_value = f"{value}{'' if m.unit == '%' else ' '}{m.unit}"
        if unresolved_limit:
            display_value = f"原文限定表述“{display_value}”"
        lines.append(
            f"- {samples[m.sample_id].label}：{names.get(m.metric, m.metric)} "
            + display_value
            + (
                "；原文提到可见光区，具体波长范围尚待核对。"
                if domain == "visible"
                else f"；原文范围为 {domain}，不自动等同于可见光平均透过率。"
                if domain
                else "。"
            )
            + f" 来源：论文第{m.source.page_from}页。"
        )
        if nonscalar:
            lines.append("  原文含范围或数值限定，不按单一数值进行描述统计。")
        if unresolved_limit:
            lines.append("  原文限定含义尚待核对，不能直接解释为精确值或明确上/下界。")
        attrs = [
            b
            for p in record.condition_plans
            for b in p.bindings
            if m.measurement_id in b.measurement_ids
        ]
        for b in attrs:
            lines.append(
                f"  已核对{'制备变量' if b.kind == 'preparation' else '测试条件'}："
                f"{attribute_names.get(b.key, b.key.replace('_', ' '))} "
                f"= {b.value_text}。"
            )
    if not handoff.record_count:
        lines.append("暂未得到可安全用于描述统计的数值；不生成空表统计。")
    lines.extend(["", "仍待核对："])
    for metric, status in record.coverage.items():
        if status == "pending_verification":
            lines.append(
                f"- {names.get(metric, metric)}：原文表格对应、数值或样品归属尚未核验。"
            )
        elif status == "not_found_current_evidence":
            lines.append(
                f"- {names.get(metric, metric)}：本次解析证据未找到可绑定记录，"
                "不代表论文没有数据。"
            )
    lines.append(
        "测试条件尚未证明可比，无法直接排名或据此判定最佳材料。当前是部分结果，不是完整科研结论。"
    )
    return "\n".join(lines)
