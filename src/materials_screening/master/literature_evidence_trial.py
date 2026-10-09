"""A bounded pilot from saved literature evidence to deterministic analysis."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.models import DatasetInspection
from materials_screening.data_analysis.service import DataAnalysisService
from materials_screening.data_analysis.statistics import DataStatisticsService
from materials_screening.master.literature_measurement_review import (
    review_measurements,
    source_columns,
)
from materials_screening.sub_agents.literature.evidence_scope import (
    is_prior_work,
    is_reference_evidence,
)
from materials_screening.sub_agents.literature.matrix_automation import (
    group_has_ungrounded_boolean_fields,
    measurement_binding_rejection_reason,
)
from materials_screening.sub_agents.literature.metric_coverage import (
    RequiredMetricCoverage,
    TaskMetricRequirements,
    check_required_metric_coverage,
    render_required_metric_coverage,
)
from materials_screening.sub_agents.literature.models import (
    ClaimEvidenceLink,
    ExperimentalComparison,
    ExperimentalGroup,
    ExperimentalMeasurement,
    PaperClaim,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord
from materials_screening.sub_agents.literature.user_report import (
    LiteratureUserReport,
)


class LiteratureReportStore(Protocol):
    def load(self, report_id: str) -> LiteratureUserReport | None: ...

    def find_latest(self, topic: str) -> LiteratureUserReport | None: ...


class LiteratureMatrixStore(Protocol):
    def load_matrix(
        self, document_id: str, *, status: str = "approved"
    ) -> tuple[
        list[ExperimentalGroup],
        list[ExperimentalMeasurement],
        list[ExperimentalComparison],
        list[PaperClaim],
        list[ClaimEvidenceLink],
    ]: ...


class LiteratureReportNotFoundError(ValueError):
    """No safely reusable report exists for the exact normalized topic."""


class LiteratureEvidenceTrialResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["completed", "partial", "failed"]
    topic: str = Field(min_length=1, max_length=4000)
    report_id: str
    document_ids: tuple[str, ...]
    dataset_id: str | None = None
    record_count: int = Field(default=0, ge=0)
    analysis_ids: tuple[str, ...] = ()
    report_artifact_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    literature_markdown: str
    analysis_markdown: str
    warnings: tuple[str, ...] = ()
    required_metric_coverage: RequiredMetricCoverage = Field(
        default_factory=RequiredMetricCoverage
    )


class LiteratureEvidenceHandoff(BaseModel):
    """Safe LiteratureAgent output that DataAnalysisAgent can consume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    topic: str = Field(min_length=1, max_length=4000)
    report_id: str
    document_ids: tuple[str, ...]
    dataset_id: str | None = None
    record_count: int = Field(default=0, ge=0)
    evidence_ids: tuple[str, ...] = ()
    literature_markdown: str
    warnings: tuple[str, ...] = ()
    required_metric_coverage: RequiredMetricCoverage = Field(
        default_factory=RequiredMetricCoverage
    )


class LiteratureEvidenceTrialService:
    """Reuse a saved report and stage provenance-bound pilot measurements."""

    def __init__(
        self,
        *,
        report_store: LiteratureReportStore,
        matrix_store: LiteratureMatrixStore,
        dataset_store: DatasetStore,
    ) -> None:
        self._reports = report_store
        self._matrices = matrix_store
        self._datasets = dataset_store

    def run(
        self,
        *,
        topic: str,
        report_id: str | None = None,
        metric_requirements: TaskMetricRequirements | None = None,
    ) -> LiteratureEvidenceTrialResult:
        handoff = self.prepare(
            topic=topic, report_id=report_id, metric_requirements=metric_requirements
        )
        base = LiteratureEvidenceTrialResult(
            status="partial",
            topic=handoff.topic,
            report_id=handoff.report_id,
            document_ids=handoff.document_ids,
            dataset_id=handoff.dataset_id,
            record_count=handoff.record_count,
            evidence_ids=handoff.evidence_ids,
            literature_markdown=handoff.literature_markdown,
            analysis_markdown=(
                "当前没有可安全转换为数值的文献测量记录，数据分析阶段已跳过。"
            ),
            warnings=handoff.warnings,
            required_metric_coverage=handoff.required_metric_coverage,
        )
        if handoff.dataset_id is None:
            return base

        inspection = DataAnalysisService(self._datasets).inspect_dataset(
            handoff.dataset_id, limit=20
        )
        evidence_id = (
            "evidence-"
            + hashlib.sha256(f"{handoff.dataset_id}|descriptive".encode()).hexdigest()[
                :24
            ]
        )
        descriptive = DataStatisticsService(self._datasets).describe_dataset(
            handoff.dataset_id,
            columns=("numeric_value",),
            group_by="measurement_context",
            evidence_id=evidence_id,
        )
        analysis_markdown = _analysis_markdown(inspection, descriptive.summary)
        with_analysis = base.model_copy(
            update={
                "analysis_ids": (descriptive.analysis_id,),
                "analysis_markdown": analysis_markdown,
            }
        )
        report_content = render_literature_evidence_trial(with_analysis)
        artifact = self._datasets.save_artifact(
            report_content.encode("utf-8"),
            artifact_type="analysis_report",
            dataset_id=handoff.dataset_id,
            analysis_ids=(descriptive.analysis_id,),
            display_name="literature-evidence-trial-report.md",
            file_extension="md",
            media_type="text/markdown",
        )
        return with_analysis.model_copy(
            update={"report_artifact_id": artifact.artifact_id}
        )

    def prepare(
        self,
        *,
        topic: str,
        report_id: str | None = None,
        metric_requirements: TaskMetricRequirements | None = None,
    ) -> LiteratureEvidenceHandoff:
        """Reuse only an exact-topic report and register its relevant matrix."""

        clean_topic = " ".join(topic.split())
        if not clean_topic:
            raise ValueError("trial topic must not be blank")
        if metric_requirements is not None:
            metric_requirements.ensure_topic(clean_topic)
        report = (
            self._reports.load(report_id)
            if report_id is not None
            else self._reports.find_latest(clean_topic)
        )
        if report is None:
            raise LiteratureReportNotFoundError(
                "没有找到与该问题完全匹配的已保存文献报告"
            )
        if (
            metric_requirements is not None
            and " ".join(report.topic.split()) != clean_topic
        ):
            raise ValueError("指定文献报告与当前研究问题/指标清单不匹配")

        report, source_chunks = _review_saved_report_sources(report, self._matrices)
        records: list[dict[str, object]] = []
        handed_off_measurements: list[ExperimentalMeasurement] = []
        warnings = [
            "当前没有独立的专家实验数据；本次仅验证公开文献证据链。",
            "文献实验条件来自不同研究，不能当作同一受控实验直接比较。",
            *report.warnings,
        ]
        evidence_ids: list[str] = [
            item.evidence_id for paper in report.papers for item in paper.evidence
        ]
        pending_documents: list[str] = []
        empty_documents: list[str] = []
        irrelevant_measurements = 0
        unsafe_group_measurements = 0
        review_sections: list[str] = []
        for paper in report.papers:
            approved = self._matrices.load_matrix(paper.document_id, status="approved")
            groups, measurements = approved[0], approved[1]
            matrix_status = "approved"
            if not groups and not measurements:
                pending = self._matrices.load_matrix(
                    paper.document_id, status="pending"
                )
                groups, measurements = pending[0], pending[1]
                matrix_status = "pending"
                if groups or measurements:
                    pending_documents.append(paper.document_id)
            if matrix_status == "pending":
                unsafe_group_ids = {
                    group.group_id
                    for group in groups
                    if group_has_ungrounded_boolean_fields(group)
                }
                safe_measurements = [
                    row for row in measurements if row.group_id not in unsafe_group_ids
                ]
                unsafe_group_measurements += len(measurements) - len(safe_measurements)
                measurements = safe_measurements
                group_map = {group.group_id: group for group in groups}
                get_chunks = getattr(self._matrices, "get_document_chunks", None)
                if paper.document_id in source_chunks:
                    chunks = source_chunks[paper.document_id]
                    if chunks is None and measurements:
                        # A failed source read must not become quote-only numeric
                        # validation. Keep the old fail-closed pending handoff.
                        raise ValueError(
                            "原文读取失败，无法安全交接待审核测量；未登记试运行数据。"
                        )
                    chunks = chunks or ()
                else:
                    chunks = tuple(get_chunks(paper.document_id)) if get_chunks else ()
                chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
                safe_binding = []
                for row in measurements:
                    group = group_map.get(row.group_id)
                    reason = (
                        "unknown group"
                        if group is None
                        else measurement_binding_rejection_reason(
                            row, group, chunks_by_id=chunk_map
                        )
                    )
                    if reason is not None:
                        warnings.append(
                            f"已隔离 {row.measurement_id}："
                            "样品/数值绑定或研究归属不明确。"
                        )
                    else:
                        safe_binding.append(row)
                measurements = safe_binding
            numeric_measurements = [
                row
                for row in measurements
                if _numeric_bounds(row.value_text, row.numeric_value) is not None
            ]
            relevant_measurements = [
                row
                for row in numeric_measurements
                if _metric_relevant(clean_topic, row.metric)
            ]
            irrelevant_measurements += len(numeric_measurements) - len(
                relevant_measurements
            )
            review = review_measurements(relevant_measurements)
            if relevant_measurements:
                warnings.extend(review.warnings)
                review_sections.append(review.markdown)
                evidence_ids.extend(row.chunk_id for row in relevant_measurements)
            rows = _measurement_rows(
                document_id=paper.document_id,
                paper_title=paper.title,
                groups=groups,
                measurements=list(review.accepted),
                matrix_status=matrix_status,
            )
            for row in rows:
                row.update(source_columns(review, str(row["measurement_id"])))
            handed_off_ids = {str(row["measurement_id"]) for row in rows}
            handed_off_measurements.extend(
                row for row in review.accepted if row.measurement_id in handed_off_ids
            )
            if not rows:
                empty_documents.append(paper.document_id)
            records.extend(rows)
            evidence_ids.extend(str(row["evidence_chunk_id"]) for row in rows)

        if pending_documents:
            warnings.append(
                f"{len(pending_documents)} 篇论文使用待审核实验矩阵；"
                "所有结果仅供试运行，不能作为定稿结论。"
            )
        if empty_documents:
            warnings.append(
                f"{len(empty_documents)} 篇论文没有可数值化的测量记录，"
                "仍保留在定性文献综合中。"
            )
        if irrelevant_measurements:
            warnings.append(
                f"已排除 {irrelevant_measurements} 条与当前研究问题无关的"
                "光学或发光测量。"
            )
        if unsafe_group_measurements:
            warnings.append(
                f"已隔离 {unsafe_group_measurements} 条分组条件缺少原文支持的"
                "历史待审核测量，未交给数据分析，也未修改或删除原始记录。"
            )

        coverage = check_required_metric_coverage(
            handed_off_measurements,
            metric_requirements.required_metrics if metric_requirements else (),
        )
        if coverage.status == "incomplete":
            warnings.append(
                "必需指标存在缺项：当前关键结果不完整；仅已有记录用于试运行，不自动补提。"
            )

        literature_markdown = "\n\n".join(
            (*review_sections, _literature_markdown(report))
        )
        dataset_id: str | None = None
        if records:
            artifact_digest = hashlib.sha256(report.report_id.encode()).hexdigest()[:24]
            dataset = self._datasets.register_records(
                records,
                source_artifact_id=f"artifact-littrial-{artifact_digest}",
                display_name=f"{report.report_id}-literature-measurements.json",
            )
            dataset_id = dataset.dataset_id
        return LiteratureEvidenceHandoff(
            topic=clean_topic,
            report_id=report.report_id,
            document_ids=tuple(paper.document_id for paper in report.papers),
            dataset_id=dataset_id,
            record_count=len(records),
            evidence_ids=tuple(dict.fromkeys(evidence_ids)),
            literature_markdown=literature_markdown,
            warnings=tuple(dict.fromkeys(warnings)),
            required_metric_coverage=coverage,
        )


def render_literature_evidence_trial(result: LiteratureEvidenceTrialResult) -> str:
    lines = [
        "# 任务5文献实验数据试运行报告",
        "",
        f"- 状态：{'部分完成' if result.status == 'partial' else result.status}",
        f"- 文献报告：`{result.report_id}`",
        f"- 纳入论文：{len(result.document_ids)} 篇",
        f"- 可数值化记录：{result.record_count} 条",
    ]
    if result.dataset_id:
        lines.append(f"- 试运行数据集：`{result.dataset_id}`")
    lines.extend(
        (
            "",
            render_required_metric_coverage(
                result.required_metric_coverage, scope_label="实际交给数据分析的记录"
            ),
        )
    )
    lines.extend(
        (
            "",
            "## 文献 Agent 结果",
            "",
            result.literature_markdown,
            "",
            "## 数据分析 Agent 结果",
            "",
            result.analysis_markdown,
            "",
            "## 结论边界",
            "",
            "本次结果用于验证系统链路，不能替代独立实验数据，"
            "也不能仅凭跨论文观察推断因果关系。",
        )
    )
    if result.warnings:
        lines.extend(("", "## 警告", ""))
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines) + "\n"


def _measurement_rows(
    *,
    document_id: str,
    paper_title: str,
    groups: list[ExperimentalGroup],
    measurements: list[ExperimentalMeasurement],
    matrix_status: str,
) -> list[dict[str, object]]:
    group_by_id = {group.group_id: group for group in groups}
    rows: list[dict[str, object]] = []
    for measurement in measurements:
        group = group_by_id.get(measurement.group_id)
        if group is None:
            continue
        numeric = _numeric_bounds(measurement.value_text, measurement.numeric_value)
        if numeric is None:
            continue
        lower, upper, value, derivation = numeric
        unit = measurement.unit or "unit_not_reported"
        row: dict[str, object] = {
            "document_id": document_id,
            "paper_title": paper_title,
            "group_id": group.group_id,
            "group_label": group.label,
            "group_role": group.role,
            "material": group.material,
            "measurement_id": measurement.measurement_id,
            "metric": measurement.metric,
            "metric_unit": f"{measurement.metric} [{unit}]",
            "measurement_context": (
                f"{document_id} / {group.group_id} / {measurement.metric} [{unit}]"
            ),
            "value_text": measurement.value_text,
            "numeric_value": value,
            "numeric_lower": lower,
            "numeric_upper": upper,
            "numeric_derivation": derivation,
            "unit": measurement.unit,
            "sample_size": measurement.sample_size,
            "review_status": matrix_status,
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
    return rows


def _numeric_bounds(
    value_text: str, numeric_value: float | None
) -> tuple[float, float, float, str] | None:
    if numeric_value is not None and math.isfinite(float(numeric_value)):
        value = float(numeric_value)
        return value, value, value, "reported_scalar"
    text = value_text.strip().replace("−", "-").replace("–", "-").replace("—", "-")
    if re.search(r"\bup\s+to\b|至多|最高|小于|大于|[<>≤≥]", text, re.IGNORECASE):
        return None
    range_match = re.fullmatch(
        r"\s*([+]?[0-9]+(?:\.[0-9]+)?)\s*(?:-|to|~|～)\s*"
        r"([+]?[0-9]+(?:\.[0-9]+)?)(?:\s*[^0-9]*)?",
        text,
        flags=re.IGNORECASE,
    )
    if range_match:
        lower, upper = (float(range_match.group(1)), float(range_match.group(2)))
        if upper < lower:
            return None
        return lower, upper, (lower + upper) / 2, "range_midpoint"
    scalar_match = re.fullmatch(
        r"\s*(?:about|around|approximately|约)?\s*"
        r"([+-]?[0-9]+(?:\.[0-9]+)?)(?:\s*[^0-9]*)?",
        text,
        flags=re.IGNORECASE,
    )
    if scalar_match:
        value = float(scalar_match.group(1))
        return value, value, value, "text_scalar"
    return None


def _review_saved_report_sources(
    report: LiteratureUserReport, matrices: LiteratureMatrixStore
) -> tuple[LiteratureUserReport, dict[str, tuple[ChunkRecord, ...] | None]]:
    """Recheck saved excerpts in memory; never save or approve the old report.

    This checks source location and recognizable attribution, not the scientific
    correctness of an extracted summary. Whitespace-only matching preserves units.
    Any removal invalidates reuse of the old narrative, even without citation tags.
    """
    loader = getattr(matrices, "get_document_chunks", None)
    # None distinguishes a read failure from a successfully loaded empty source.
    sources: dict[str, tuple[ChunkRecord, ...] | None] = {}
    warnings = list(report.warnings)
    papers = []
    removed = 0
    for paper in report.papers:
        if not paper.evidence:
            papers.append(paper)
            continue
        if paper.document_id not in sources:
            try:
                sources[paper.document_id] = tuple(
                    chunk
                    for chunk in (loader(paper.document_id) if loader else ())
                    if chunk.document_id == paper.document_id
                )
            except Exception as exc:
                # Error payloads may contain connection details or source content.
                warnings.append(
                    f"{paper.document_id}: 原文读取失败（{type(exc).__name__}），"
                    "文献仍存在，但本次无法核对原文。"
                )
                sources[paper.document_id] = None
        chunks = sources[paper.document_id] or ()
        retained = []
        for item in paper.evidence:
            excerpt = " ".join(item.source_quote.split())
            matches = [
                chunk
                for chunk in chunks
                if excerpt
                and chunk.page_from <= item.page <= chunk.page_to
                and excerpt in " ".join(chunk.text.split())
            ]
            reason = None
            if not matches:
                reason = "无法核对原文（未在对应论文及页码定位摘录）"
            elif any(
                is_reference_evidence(item.source_quote, chunk, chunks)
                for chunk in matches
            ):
                reason = "参考文献条目，不能作为本论文实验结果"
            elif any(is_prior_work(item.source_quote, chunk.text) for chunk in matches):
                reason = "可识别的他人研究，不能作为本论文实验结果"
            if reason:
                warnings.append(f"已隔离文献证据 {item.evidence_id}：{reason}。")
            else:
                retained.append(item)
        excluded = len(paper.evidence) - len(retained)
        removed += excluded
        papers.append(
            paper.model_copy(
                update={
                    "evidence": tuple(retained),
                    "low_risk_count": sum(
                        item.risk_level == "low" for item in retained
                    ),
                    "medium_risk_count": sum(
                        item.risk_level == "medium" for item in retained
                    ),
                    "excluded_unsafe_count": paper.excluded_unsafe_count + excluded,
                }
            )
        )
    narrative = report.narrative
    if removed and narrative is not None:
        narrative = None
        warnings.append(
            "本次有证据被隔离，旧综合文字暂停复用；仅展示保留证据，不自动重写综合结论。"
        )
    return report.model_copy(
        update={
            "papers": tuple(papers),
            "narrative": narrative,
            "warnings": tuple(warnings),
        }
    ), sources


def _literature_markdown(report: LiteratureUserReport) -> str:
    if report.narrative is not None:
        return report.narrative.markdown
    lines = [
        "自动综合文本未通过校验、尚未生成或已暂停复用，以下仅展示经本次来源核对保留的证据。",
        "下列抽取摘要与原文摘录仍需核对，不代表人工审核通过。",
    ]
    for paper in report.papers:
        lines.extend(("", f"### {paper.title}"))
        if not paper.evidence:
            lines.append(
                "本次没有通过来源核对的定性证据；不代表没有该论文或没有实验结果。"
            )
        for item in paper.evidence:
            lines.extend(
                (
                    "",
                    f"- [{item.evidence_id}] 第 {item.page} 页；"
                    f"风险：{item.risk_level}。",
                    f"  抽取摘要（待核对）：{item.summary}",
                    f"  原文摘录：{item.source_quote[:300]}",
                )
            )
    return "\n".join(lines)


def _metric_relevant(topic: str, metric: str) -> bool:
    optical_markers = (
        "luminescence",
        "photoluminescence",
        "optical",
        "band gap",
        "bandgap",
        "发光",
        "光学",
        "带隙",
    )
    lowered_metric = metric.casefold()
    if not any(marker in lowered_metric for marker in optical_markers):
        return True
    lowered_topic = topic.casefold()
    return any(marker in lowered_topic for marker in optical_markers)


def _analysis_markdown(
    inspection: DatasetInspection, summary: dict[str, object]
) -> str:
    dataset = inspection.dataset
    issues = inspection.issues
    statistics = summary.get("statistics", [])
    lines = [
        f"数据集包含 {dataset.row_count} 条测量记录、{dataset.column_count} 个字段。",
        "统计按“论文+样品+指标+单位”分组，未跨样品或跨论文合并。",
        "统计条目数不是独立实验重复数；多处引用保存在溯源字段中。",
        "",
        "| 论文、样品、指标与单位 | n | 均值 | 中位数 | 最小值 | 最大值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if isinstance(statistics, list):
        for item in statistics:
            if not isinstance(item, dict):
                continue
            lines.append(
                (
                    "| {group} | {count} | {mean} | {median} | {minimum} | {maximum} |"
                ).format(
                    group=item.get("group", "—"),
                    count=item.get("count", "—"),
                    mean=_display(item.get("mean")),
                    median=_display(
                        (item.get("quantiles") or {}).get("0.5")
                        if isinstance(item.get("quantiles"), dict)
                        else None
                    ),
                    minimum=_display(item.get("min")),
                    maximum=_display(item.get("max")),
                )
            )
    lines.extend(
        (
            "",
            f"数据质量提示：{len(issues)} 项。",
            "区间记录使用中点仅用于流程测试，原始区间上下界和原文值均已保留。",
        )
    )
    return "\n".join(lines)


def _display(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
