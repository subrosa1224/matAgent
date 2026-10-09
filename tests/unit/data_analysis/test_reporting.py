from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from docx import Document

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.reporting import DataAnalysisReportingService
from materials_screening.data_analysis.scientific import ScientificAnalysisBrief
from materials_screening.data_analysis.statistics import DataStatisticsService


def _reporting_service(
    tmp_path: Path,
) -> tuple[DataAnalysisReportingService, DatasetStore, str, str]:
    source = tmp_path / "report.csv"
    source.write_text(
        "sample,group,x,y,z,label\n"
        "a,A,1,2,6,normal\n"
        "b,A,2,4,5,=FORMULA\n"
        "c,A,3,6,4,+FORMULA\n"
        "d,B,4,8,3,-FORMULA\n"
        "e,B,5,10,2,@FORMULA\n"
        "f,B,6,12,1,safe\n",
        encoding="utf-8",
    )
    store = DatasetStore(tmp_path / "private")
    reference = store.register_file(
        source, source_artifact_id="artifact-data-reporting"
    )
    analysis = DataStatisticsService(store).describe_dataset(
        reference.dataset_id, columns=["x", "y"], group_by="group"
    )
    return (
        DataAnalysisReportingService(store),
        store,
        reference.dataset_id,
        analysis.analysis_id,
    )


@pytest.mark.parametrize(
    ("plot_type", "arguments"),
    [
        ("histogram", {"x": "x", "group_by": "group"}),
        ("boxplot", {"y": "y", "group_by": "group"}),
        ("scatter", {"x": "x", "y": "y", "group_by": "group"}),
        ("line", {"x": "x", "y": "y"}),
        ("bar", {"x": "group", "y": "y"}),
        ("heatmap", {"columns": ["x", "y", "z"]}),
    ],
)
def test_six_fixed_plots_create_valid_bounded_png_artifacts(
    tmp_path: Path, plot_type: str, arguments: dict[str, object]
) -> None:
    service, store, dataset_id, _ = _reporting_service(tmp_path)

    artifact = service.create_plot(
        dataset_id,
        plot_type=plot_type,  # type: ignore[arg-type]
        title="Analysis plot",
        **arguments,  # type: ignore[arg-type]
    )

    content = store.resolve_artifact_path(artifact.artifact_id).read_bytes()
    assert content.startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.artifact_type == "analysis_plot"
    assert artifact.media_type == "image/png"
    assert 1_000 < artifact.size_bytes < 5_000_000
    assert plt.get_fignums() == []


def test_plot_rejects_unknown_nonnumeric_or_unbounded_inputs(tmp_path: Path) -> None:
    service, _, dataset_id, _ = _reporting_service(tmp_path)

    with pytest.raises(ValueError, match="must be numeric"):
        service.create_plot(dataset_id, plot_type="histogram", x="label")
    with pytest.raises(ValueError, match="unknown plot column"):
        service.create_plot(dataset_id, plot_type="scatter", x="x", y="missing")
    with pytest.raises(ValueError, match="2 to 20"):
        service.create_plot(dataset_id, plot_type="heatmap", columns=["x"])
    with pytest.raises(ValueError, match="at most 256"):
        service.create_plot(
            dataset_id, plot_type="histogram", x="x", title="x" * 257
        )
    assert plt.get_fignums() == []


@pytest.mark.parametrize("format", ["md", "json", "csv"])
def test_report_formats_are_readable_and_reference_stable_ids(
    tmp_path: Path, format: str
) -> None:
    service, store, dataset_id, analysis_id = _reporting_service(tmp_path)
    plot = service.create_plot(dataset_id, plot_type="scatter", x="x", y="y")

    artifact = service.create_report(
        dataset_id,
        analysis_ids=[analysis_id],
        plot_artifact_ids=[plot.artifact_id],
        format=format,  # type: ignore[arg-type]
    )
    path = store.resolve_artifact_path(artifact.artifact_id)
    text = path.read_text(encoding="utf-8-sig")

    assert analysis_id in text
    assert artifact.analysis_ids == (analysis_id,)
    if format == "md":
        assert "# 数据分析报告" in text
        assert plot.artifact_id in text
    elif format == "json":
        payload = json.loads(text)
        assert payload["dataset_id"] == dataset_id
    else:
        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) == 1
        assert rows[0]["analysis_id"] == analysis_id
        assert json.loads(rows[0]["summary_json"])


def test_scientific_docx_report_prioritizes_narrative_over_internal_ids(
    tmp_path: Path,
) -> None:
    service, store, dataset_id, analysis_id = _reporting_service(tmp_path)
    brief = ScientificAnalysisBrief(
        dataset_id=dataset_id,
        domain="materials_science",
        research_question="不同实验组的 x 指标是否存在具有科研意义的差异？",
        observation_unit="独立材料试样",
        design="independent_groups",
        response_variables=("x",),
        group_variable="group",
        hypothesis="A、B 两组的 x 均值不同",
        practical_thresholds={"x": 1.0},
        units={"x": "a.u."},
        roles_confirmed=True,
    )

    artifact = service.create_report(
        dataset_id,
        analysis_ids=[analysis_id],
        format="docx",
        scientific_brief=brief,
    )
    path = store.resolve_artifact_path(artifact.artifact_id)
    document = Document(path)
    paragraphs = "\n".join(paragraph.text for paragraph in document.paragraphs)

    assert path.read_bytes().startswith(b"PK")
    assert artifact.file_extension == "docx"
    assert "一页式摘要" in paragraphs
    assert brief.research_question in paragraphs
    assert "局限与不确定性" in paragraphs
    assert "建议的下一步" in paragraphs
    assert analysis_id not in paragraphs.split("附录：方法与可复现性记录")[0]


@pytest.mark.parametrize("format", ["csv", "json"])
def test_dataset_export_is_safe_and_json_has_no_nonfinite_values(
    tmp_path: Path, format: str
) -> None:
    service, store, dataset_id, _ = _reporting_service(tmp_path)

    artifact = service.export_dataset(
        dataset_id, format=format  # type: ignore[arg-type]
    )
    text = store.resolve_artifact_path(artifact.artifact_id).read_text(
        encoding="utf-8-sig"
    )

    assert artifact.artifact_type == "dataset_file"
    if format == "csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        labels = {row["label"] for row in rows}
        assert "'=FORMULA" in labels
        assert "'+FORMULA" in labels
        assert "'-FORMULA" in labels
        assert "'@FORMULA" in labels
    else:
        payload = json.loads(text)
        assert len(payload) == 6
        assert payload[0]["sample"] == "a"


def test_artifact_metadata_hides_paths_and_detects_tampering(tmp_path: Path) -> None:
    service, store, dataset_id, _ = _reporting_service(tmp_path)
    artifact = service.create_plot(dataset_id, plot_type="histogram", x="x")
    path = store.resolve_artifact_path(artifact.artifact_id)

    assert str(path) not in artifact.model_dump_json()
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="fingerprint"):
        store.resolve_artifact_path(artifact.artifact_id)
    with pytest.raises(ValueError, match="path is invalid"):
        store.get_artifact("../../secret")


def test_report_rejects_cross_dataset_analysis_and_non_plot_artifact(
    tmp_path: Path,
) -> None:
    service, store, dataset_id, analysis_id = _reporting_service(tmp_path)
    second_source = tmp_path / "second.csv"
    second_source.write_text("x,y\n1,2\n2,4\n", encoding="utf-8")
    second = store.register_file(
        second_source, source_artifact_id="artifact-data-second-report"
    )
    second_analysis = DataStatisticsService(store).describe_dataset(
        second.dataset_id, columns=["x"]
    )
    dataset_export = service.export_dataset(dataset_id, format="csv")

    with pytest.raises(ValueError, match="must belong"):
        service.create_report(
            dataset_id, analysis_ids=[second_analysis.analysis_id]
        )
    with pytest.raises(ValueError, match="analysis_plot"):
        service.create_report(
            dataset_id,
            analysis_ids=[analysis_id],
            plot_artifact_ids=[dataset_export.artifact_id],
        )


def test_report_rejects_empty_or_duplicate_references(tmp_path: Path) -> None:
    service, _, dataset_id, analysis_id = _reporting_service(tmp_path)

    with pytest.raises(ValueError, match="at least one"):
        service.create_report(dataset_id, analysis_ids=[])
    with pytest.raises(ValueError, match="unique"):
        service.create_report(
            dataset_id, analysis_ids=[analysis_id, analysis_id]
        )
