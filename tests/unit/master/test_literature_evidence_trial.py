from __future__ import annotations

from pathlib import Path

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.master.literature_evidence_trial import (
    LiteratureEvidenceTrialService,
    render_literature_evidence_trial,
)
from materials_screening.sub_agents.literature.models import (
    ExperimentalGroup,
    ExperimentalMeasurement,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord
from materials_screening.sub_agents.literature.user_report import (
    LiteratureUserReport,
    LiteratureUserReportStore,
    UserPaperReport,
    UserReportEvidence,
    UserReportNarrative,
)


class _ReportStore:
    def __init__(self, report: LiteratureUserReport | None) -> None:
        self.report = report

    def load(self, report_id: str) -> LiteratureUserReport | None:
        if self.report is not None and self.report.report_id == report_id:
            return self.report
        return None

    def find_latest(self, topic: str) -> LiteratureUserReport | None:
        if self.report is not None and self.report.topic == topic:
            return self.report
        return None


class _MatrixStore:
    def __init__(
        self,
        groups: list[ExperimentalGroup],
        measurements: list[ExperimentalMeasurement],
    ) -> None:
        self.groups = groups
        self.measurements = measurements

    def load_matrix(self, document_id: str, *, status: str = "approved"):
        del document_id
        if status == "pending":
            return self.groups, self.measurements, [], [], []
        return [], [], [], [], []


def _report() -> LiteratureUserReport:
    return LiteratureUserReport(
        report_id="user-lit-test",
        topic="任务5",
        papers=(
            UserPaperReport(
                document_id="doc-1",
                title="Paper One",
                dossier_status="pending",
                evidence=(),
                low_risk_count=0,
                medium_risk_count=0,
                excluded_high_risk_count=0,
                excluded_unsafe_count=0,
            ),
        ),
        narrative=UserReportNarrative(
            markdown="## 主题概述\n这是已有文献证据综合报告。"
        ),
    )


def _group() -> ExperimentalGroup:
    return ExperimentalGroup(
        group_id="group-1",
        document_id="doc-1",
        label="pH 9",
        role="treatment",
        material="HAp",
        variables={"pH": "9"},
        conditions={"temperature": "200 °C"},
        source_quote="pH 9 produced rods 50-70 nm long.",
        chunk_id="chunk-1",
        page_from=2,
        page_to=2,
        source_text_sha256="a" * 64,
        review_status="pending",
    )


def _measurement() -> ExperimentalMeasurement:
    return ExperimentalMeasurement(
        measurement_id="measurement-1",
        group_id="group-1",
        document_id="doc-1",
        metric="length",
        value_text="50-70 nm",
        numeric_value=None,
        unit="nm",
        source_quote="pH 9 produced rods 50-70 nm long.",
        chunk_id="chunk-1",
        page_from=2,
        page_to=2,
        source_text_sha256="a" * 64,
        review_status="pending",
    )


def test_trial_registers_pending_literature_measurements_with_provenance(
    tmp_path: Path,
) -> None:
    datasets = DatasetStore(tmp_path / "analysis")
    service = LiteratureEvidenceTrialService(
        report_store=_ReportStore(_report()),
        matrix_store=_MatrixStore([_group()], [_measurement()]),
        dataset_store=datasets,
    )

    result = service.run(topic="任务5", report_id="user-lit-test")

    assert result.status == "partial"
    assert result.report_id == "user-lit-test"
    assert result.record_count == 1
    assert result.dataset_id is not None
    assert result.analysis_ids
    assert result.report_artifact_id is not None
    row = datasets.load_dataframe(result.dataset_id).iloc[0]
    assert row["numeric_value"] == 60.0
    assert row["numeric_lower"] == 50.0
    assert row["numeric_upper"] == 70.0
    assert row["numeric_derivation"] == "range_midpoint"
    assert row["review_status"] == "pending"
    assert row["variable__pH"] == "9"
    assert any("待审核" in warning for warning in result.warnings)

    rendered = render_literature_evidence_trial(result)
    assert "已有文献证据综合报告" in rendered
    assert "不能替代独立实验数据" in rendered
    assert result.dataset_id in rendered


def test_trial_keeps_literature_report_when_no_numeric_measurements(
    tmp_path: Path,
) -> None:
    service = LiteratureEvidenceTrialService(
        report_store=_ReportStore(_report()),
        matrix_store=_MatrixStore([_group()], []),
        dataset_store=DatasetStore(tmp_path / "analysis"),
    )

    result = service.run(topic="任务5", report_id="user-lit-test")

    assert result.status == "partial"
    assert result.dataset_id is None
    assert result.record_count == 0
    assert result.analysis_ids == ()
    assert result.report_artifact_id is None
    assert any("没有可数值化" in warning for warning in result.warnings)


def test_trial_can_resolve_latest_report_from_topic(tmp_path: Path) -> None:
    service = LiteratureEvidenceTrialService(
        report_store=_ReportStore(_report()),
        matrix_store=_MatrixStore([_group()], [_measurement()]),
        dataset_store=DatasetStore(tmp_path / "analysis"),
    )

    result = service.run(topic="任务5")

    assert result.report_id == "user-lit-test"


def test_prepare_handoff_registers_dataset_without_running_analysis(
    tmp_path: Path,
) -> None:
    service = LiteratureEvidenceTrialService(
        report_store=_ReportStore(_report()),
        matrix_store=_MatrixStore([_group()], [_measurement()]),
        dataset_store=DatasetStore(tmp_path / "analysis"),
    )

    handoff = service.prepare(topic="任务5", report_id="user-lit-test")

    assert handoff.dataset_id is not None
    assert handoff.record_count == 1
    assert handoff.report_id == "user-lit-test"


def test_handoff_excludes_stale_ungrounded_negative_group(tmp_path: Path) -> None:
    quote = "Graphene/ZnONRs form a Schottky junction; responsivity is 113 AW − 1."
    wrong_group = _group().model_copy(
        update={
            "label": "ZnONR/Si Control",
            "material": "ZnONRs/Si",
            "variables": {"Schottky Junction": "No"},
            "source_quote": quote,
        }
    )
    wrong_measurement = _measurement().model_copy(
        update={"source_quote": quote, "value_text": "113", "numeric_value": 113}
    )
    service = LiteratureEvidenceTrialService(
        report_store=_ReportStore(_report()),
        matrix_store=_MatrixStore([wrong_group], [wrong_measurement]),
        dataset_store=DatasetStore(tmp_path / "analysis"),
    )
    result = service.prepare(topic="任务5")
    assert result.record_count == 0
    assert result.dataset_id is None
    assert any("分组" in warning for warning in result.warnings)


def test_handoff_preserves_evidence_and_warnings_when_narrative_is_rejected(
    tmp_path: Path,
) -> None:
    evidence = UserReportEvidence(
        evidence_id="E-1234567890",
        category="performance_result",
        summary="当前抽取摘要，尚待核对。",
        page=4,
        source_quote="Responsivity was 113 AW − 1.",
        risk_level="medium",
        display_status="check_recommended",
    )
    report = _report().model_copy(
        update={
            "narrative": None,
            "papers": (
                _report().papers[0].model_copy(update={"evidence": (evidence,)}),
            ),
            "warnings": ("自动叙事未通过校验",),
        }
    )

    class SourceMatrixStore(_MatrixStore):
        def get_document_chunks(self, document_id: str):
            return [
                ChunkRecord(
                    chunk_id="chunk-4",
                    document_id=document_id,
                    paper_id=None,
                    page_from=4,
                    page_to=4,
                    text=evidence.source_quote,
                    text_sha256="a" * 64,
                )
            ]

    service = LiteratureEvidenceTrialService(
        report_store=_ReportStore(report),
        matrix_store=SourceMatrixStore([], []),
        dataset_store=DatasetStore(tmp_path / "analysis"),
    )
    result = service.prepare(topic="任务5")
    assert "E-1234567890" in result.literature_markdown
    assert "113 AW − 1" in result.literature_markdown
    assert "第 4 页" in result.literature_markdown
    assert "自动叙事未通过校验" in result.warnings


def test_trial_excludes_unrequested_optical_measurements(tmp_path: Path) -> None:
    optical = _measurement().model_copy(
        update={
            "measurement_id": "measurement-optical",
            "metric": "luminescence lifetime",
            "value_text": "7.2 ns",
            "unit": "ns",
        }
    )
    service = LiteratureEvidenceTrialService(
        report_store=_ReportStore(_report()),
        matrix_store=_MatrixStore([_group()], [_measurement(), optical]),
        dataset_store=DatasetStore(tmp_path / "analysis"),
    )

    result = service.run(topic="任务5", report_id="user-lit-test")

    assert result.record_count == 1
    assert any("研究问题无关" in warning for warning in result.warnings)


def test_report_topic_lookup_skips_legacy_invalid_reports(tmp_path: Path) -> None:
    store = LiteratureUserReportStore(tmp_path)
    store.save(_report())
    (tmp_path / "user-lit-legacy.json").write_text(
        '{"report_id":"user-lit-legacy","topic":"任务5","papers":['
        '{"document_id":"doc-old","title":"old","dossier_status":"pending",'
        '"evidence":[{"category":"materials"}]}]}',
        encoding="utf-8",
    )

    found = store.find_latest("任务5")

    assert found is not None
    assert found.report_id == "user-lit-test"
