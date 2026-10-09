"""Conservative, source-preserving review of literature numeric handoffs.

This does not edit or approve stored measurements. It recognizes a small set of
metric aliases, holds unresolved evidence out of statistics, and never converts
units or chooses a winner between conflicting values.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from materials_screening.sub_agents.literature.matrix import unit_present_in_evidence
from materials_screening.sub_agents.literature.models import ExperimentalMeasurement


@dataclass(frozen=True)
class MeasurementReview:
    accepted: tuple[ExperimentalMeasurement, ...]
    sources: dict[str, tuple[ExperimentalMeasurement, ...]]
    resolutions: dict[str, str]
    warnings: tuple[str, ...]
    markdown: str


def canonical_metric(metric: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", metric.casefold()).strip()
    aliases = {
        "responsivity": "responsivity",
        "responsivity r": "responsivity",
        "gain": "gain",
        "gain g": "gain",
        "response time rise": "response_time_rise",
        "time response rise": "response_time_rise",
        "rise time": "response_time_rise",
        "response time fall": "response_time_fall",
        "time response fall": "response_time_fall",
        "fall time": "response_time_fall",
    }
    return aliases.get(normalized, metric)


def _unit(unit: str | None) -> str:
    # Typography only: keep prefix case, never change scale or dimension.
    text = unicodedata.normalize("NFKC", unit or "").replace("−", "-").replace("⁻", "-")
    return re.sub(r"\s+", "", text)


def _context_key(row: ExperimentalMeasurement) -> tuple[object, ...]:
    return (
        row.document_id,
        row.group_id,
        canonical_metric(row.metric),
        _unit(row.unit),
        row.uncertainty_text,
        row.sample_size,
    )


def _value_key(row: ExperimentalMeasurement) -> object:
    return (
        row.numeric_value if row.numeric_value is not None else row.value_text.strip()
    )


def _same_quote(rows: Sequence[ExperimentalMeasurement]) -> bool:
    if (
        len({(row.chunk_id, row.source_text_sha256, row.source_quote) for row in rows})
        != 1
    ):
        return False
    quote = rows[0].source_quote
    if re.search(r"\b(?:replicates?|both|each|trials?)\b", quote, re.IGNORECASE):
        return False
    if rows[0].numeric_value is None:
        return quote.count(rows[0].value_text) == 1
    numbers = re.findall(
        r"(?<![a-z0-9])(?<!\d\.)[-+]?\d+(?:\.\d+)?(?![a-z0-9]|\.\d)",
        quote,
        re.IGNORECASE,
    )
    return sum(float(number) == rows[0].numeric_value for number in numbers) == 1


def _table_restatement(rows: Sequence[ExperimentalMeasurement]) -> bool:
    # A tightly bounded case: the R/G result pair explicitly points to a table
    # whose own-study row repeats it. Equal values elsewhere are not sufficient.
    if canonical_metric(rows[0].metric) not in {"responsivity", "gain"}:
        return False
    tables: set[str] = set()
    body_tables: set[str] = set()
    for row in rows:
        header = re.match(r"\s*Table\s+(\d+)\b", row.source_quote, re.IGNORECASE)
        if header and re.search(r"\bThis work\b", row.source_quote, re.IGNORECASE):
            own_result = re.search(
                r"(?P<r>\d+(?:\.\d+)?)\s+(?P<g>\d+(?:\.\d+)?)\s+This work\b",
                row.source_quote,
                re.IGNORECASE,
            )
            field = "r" if canonical_metric(row.metric) == "responsivity" else "g"
            if own_result is None or float(own_result[field]) != row.numeric_value:
                return False
            if row.unit and not unit_present_in_evidence(row.unit, row.source_quote):
                return False
            tables.add(header.group(1))
            continue
        supported = False
        for pair in re.finditer(
            r"\bR\s+and\s+G\b.{0,80}?(?P<r>\d+(?:\.\d+)?)\s*"
            r"(?P<unit>.{1,30}?)\s+and\s+(?P<g>\d+(?:\.\d+)?)"
            r"\s*,?\s*respectively\b",
            row.source_quote,
            re.IGNORECASE | re.DOTALL,
        ):
            field = "r" if canonical_metric(row.metric) == "responsivity" else "g"
            if float(pair[field]) != row.numeric_value:
                continue
            if row.unit and not unit_present_in_evidence(row.unit, pair["unit"]):
                continue
            reference = re.search(
                r"\bTable\s+(\d+)\b",
                row.source_quote[pair.end() : pair.end() + 500],
                re.IGNORECASE,
            )
            if reference:
                body_tables.add(reference.group(1))
                supported = True
        if not supported:
            return False
    if not tables or not body_tables or tables != body_tables:
        return False
    # Each row must be accounted for, not just one convenient pair of citations.
    for row in rows:
        if re.match(r"\s*Table\s+\d+\b", row.source_quote, re.IGNORECASE):
            if not re.search(r"\bThis work\b", row.source_quote, re.IGNORECASE):
                return False
        elif not re.search(r"\bR\s+and\s+G\b", row.source_quote, re.IGNORECASE):
            return False
    return True


_PAIRED_TIMES = re.compile(
    r"\b(?P<first>rise|fall)\s*(?P<join>and|/)\s*(?P<second>rise|fall)"
    r"\s+times?\b[^.;]{0,80}?"
    r"(?P<one>\d+(?:\.\d+)?)\s*(?P<unit_one>ms|ns|us|µs|μs|s)?\s*"
    r"(?:and|/)\s*(?P<two>\d+(?:\.\d+)?)\s*"
    r"(?P<unit_two>ms|ns|us|µs|μs|s)\b"
    r"(?P<respectively>\s*,?\s*respectively)?",
    re.IGNORECASE,
)


def _time_conflicts(
    rows: Sequence[ExperimentalMeasurement],
) -> tuple[set[tuple[object, ...]], list[str]]:
    observed: dict[tuple[object, ...], set[float]] = defaultdict(set)
    excerpts: dict[tuple[object, ...], set[str]] = defaultdict(set)
    for row in rows:
        if row.numeric_value is not None and canonical_metric(row.metric) in {
            "response_time_rise",
            "response_time_fall",
        }:
            observed[_context_key(row)].add(row.numeric_value)
    for row in rows:
        for pair in _PAIRED_TIMES.finditer(row.source_quote):
            if pair["first"].casefold() == pair["second"].casefold():
                continue
            if pair["join"].casefold() == "and" and not pair["respectively"]:
                continue
            unit = pair["unit_two"]
            if pair["unit_one"] and _unit(pair["unit_one"]) != _unit(unit):
                continue
            for label, value in (
                (pair["first"], pair["one"]),
                (pair["second"], pair["two"]),
            ):
                key = (
                    row.document_id,
                    row.group_id,
                    f"response_time_{label.casefold()}",
                    _unit(unit),
                    row.uncertainty_text,
                    row.sample_size,
                )
                if key in observed:
                    observed[key].add(float(value))
                    excerpts[key].add(
                        f"第 {row.page_from} 页 [{row.measurement_id}]：{pair.group(0)}"
                    )
    conflicts = {key for key, values in observed.items() if len(values) > 1}
    notes = sorted({excerpt for key in conflicts for excerpt in excerpts[key]})
    return conflicts, notes


def review_measurements(rows: Sequence[ExperimentalMeasurement]) -> MeasurementReview:
    buckets: dict[tuple[object, ...], list[ExperimentalMeasurement]] = defaultdict(list)
    for row in rows:
        buckets[_context_key(row)].append(row)
    time_conflicts, time_notes = _time_conflicts(rows)
    accepted: list[ExperimentalMeasurement] = []
    sources: dict[str, tuple[ExperimentalMeasurement, ...]] = {}
    resolutions: dict[str, str] = {}
    held: list[tuple[ExperimentalMeasurement, str]] = []
    duplicate_count = 0
    for key, bucket in buckets.items():
        if key in time_conflicts or len({_value_key(row) for row in bucket}) > 1:
            held.extend(
                (row, "可能存在值/顺序冲突或未区分的测试条件") for row in bucket
            )
            continue
        resolution = "single_evidence"
        if len(bucket) > 1:
            if _same_quote(bucket):
                resolution = "same_verbatim_evidence"
            elif _table_restatement(bucket):
                resolution = "explicit_table_reference"
            else:
                held.extend((row, "疑似重复，但不能确认同一次测量") for row in bucket)
                continue
            duplicate_count += len(bucket) - 1
        representative = bucket[0].model_copy(
            update={"metric": canonical_metric(bucket[0].metric)}
        )
        accepted.append(representative)
        sources[representative.measurement_id] = tuple(bucket)
        resolutions[representative.measurement_id] = resolution
    warnings = ["统计条目数不是独立实验重复数；不得将重复引用作为新增实验样本。"]
    if duplicate_count:
        warnings.append(
            f"已归并 {duplicate_count} 条有同源或明确表格引用支持的重复证据，"
            "全部来源保留。"
        )
    if held:
        conflict_count = sum("冲突" in reason for _, reason in held)
        suspect_count = len(held) - conflict_count
        if conflict_count:
            warnings.append(
                f"{conflict_count} 条测量存在可能的证据冲突，保留展示但未交给数据分析；"
                "未选择或平均冲突值。"
            )
        if suspect_count:
            warnings.append(
                f"{suspect_count} 条疑似重复测量的实验对应关系不明，"
                "保留展示但未交给数据分析。"
            )
    lines = ["## 数值证据核对", "", "条目代表待审测量摘要，不是独立实验重复数。"]
    if accepted:
        lines.extend(
            (
                "",
                "| 可交接指标 | 原文值与单位 | 来源记录（全部保留） |",
                "|---|---|---|",
            )
        )
        for row in accepted:
            provenance = "; ".join(
                f"{source.measurement_id}（第 {source.page_from} 页）"
                for source in sources[row.measurement_id]
            )
            lines.append(f"| {row.metric} | {_reported_value(row)} | {provenance} |")
    if held:
        lines.extend(("", "### 暂不进入统计的证据", ""))
        for row, reason in held:
            lines.append(
                f"- [{row.measurement_id}] 第 {row.page_from} 页，{row.metric}="
                f"{_reported_value(row)}；{reason}。"
            )
        lines.extend(f"- 顺序核对原文：{note}" for note in time_notes)
    return MeasurementReview(
        tuple(accepted), sources, resolutions, tuple(warnings), "\n".join(lines)
    )


def _reported_value(row: ExperimentalMeasurement) -> str:
    if row.unit is None:
        return row.value_text
    if unit_present_in_evidence(row.unit, row.value_text):
        return row.value_text
    return f"{row.value_text} {row.unit}"


def source_columns(review: MeasurementReview, measurement_id: str) -> dict[str, object]:
    rows = review.sources[measurement_id]
    return {
        "source_measurement_ids": json.dumps([row.measurement_id for row in rows]),
        "source_metrics": json.dumps([row.metric for row in rows], ensure_ascii=False),
        "source_value_texts": json.dumps(
            [row.value_text for row in rows], ensure_ascii=False
        ),
        "source_units": json.dumps([row.unit for row in rows], ensure_ascii=False),
        "source_quotes": json.dumps(
            [row.source_quote for row in rows], ensure_ascii=False
        ),
        "source_chunk_ids": json.dumps([row.chunk_id for row in rows]),
        "source_pages": json.dumps([row.page_from for row in rows]),
        "source_text_sha256s": json.dumps([row.source_text_sha256 for row in rows]),
        "evidence_count": len(rows),
        "duplicate_resolution": review.resolutions[measurement_id],
    }
