"""Offline wiring checks: same task policy, final handoff rows, and honest reports."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from materials_screening import cli
from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.master.literature_evidence_trial import (
    LiteratureEvidenceHandoff,
    LiteratureEvidenceTrialResult,
    LiteratureEvidenceTrialService,
    render_literature_evidence_trial,
)
from materials_screening.sub_agents.literature import metric_coverage as coverage
from materials_screening.sub_agents.literature.matrix_automation import (
    AutomatedMatrixExtractor,
    MatrixExtractionBatch,
)
from tests.unit.master.test_literature_evidence_trial import (
    _group,
    _MatrixStore,
    _measurement,
    _report,
    _ReportStore,
)
from tests.unit.sub_agents.test_literature_matrix_automation import (
    FakeLlm,
    Store,
    _chunk,
)
from tests.unit.sub_agents.test_matrix_candidate_diagnostics import candidate, group


def policy(**updates):
    return coverage.TaskMetricRequirements.model_validate(
        {
            "topic": "任务5",
            "required_metrics": [{"metric": "length", "unit": "nm"}],
            **updates,
        }
    )


def service(tmp_path, *, rows=None, groups=None, report=None):
    return LiteratureEvidenceTrialService(
        report_store=_ReportStore(_report() if report is None else report),
        matrix_store=_MatrixStore(
            [_group()] if groups is None else groups,
            [_measurement()] if rows is None else rows,
        ),
        dataset_store=DatasetStore(tmp_path / "analysis"),
    )


@pytest.mark.parametrize(
    "payload",
    [
        "not-json-PRIVATE",
        json.dumps({"topic": "任务5", "required_metrics": []}),
        json.dumps({"topic": " ", "required_metrics": [{"metric": "length"}]}),
        json.dumps({"topic": "任务5", "required_metrics": [{"metric": ""}]}),
        json.dumps(
            {
                "topic": "任务5",
                "required_metrics": [{"metric": "length", "expected_value": "PRIVATE"}],
            }
        ),
        json.dumps(
            {
                "topic": "任务5",
                "required_metrics": [{"metric": "length"}],
                "answer": "PRIVATE",
            }
        ),
        json.dumps({"topic": "任务5", "required_metrics": [{"metric": "x"}] * 51}),
        "PRIVATE" * 10000,
        '{"topic":"任务5","topic":"别的任务","required_metrics":[{"metric":"length"}]}',
        '{"topic":"任务5","required_metrics":[{"metric":"length","metric":"other"}]}',
    ],
    ids=[
        "invalid-json",
        "empty",
        "blank-topic",
        "blank-metric",
        "reference-value",
        "answer-field",
        "too-many",
        "oversized",
        "duplicate-topic",
        "duplicate-metric",
    ],
)
def test_bad_policies_are_rejected_without_printing_payload(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        coverage.load_task_metric_requirements(path)
    assert "PRIVATE" not in str(error.value)


def test_policy_loads_bom_and_preserves_explicit_names_and_units(tmp_path):
    path = tmp_path / "policy.json"
    expected = policy()
    path.write_text(expected.model_dump_json(), encoding="utf-8-sig")
    assert coverage.load_task_metric_requirements(path) == expected
    assert expected.required_metrics[0].unit == "nm"


def test_policy_missing_file_reports_safe_error(tmp_path):
    with pytest.raises(ValueError):
        coverage.load_task_metric_requirements(tmp_path / "missing.json")


def test_final_trial_coverage_and_saved_report_share_exact_handoff_ids(tmp_path):
    trial = service(tmp_path)
    result = trial.run(topic="任务5", metric_requirements=policy())
    assert result.required_metric_coverage.status == "covered"
    assert result.status == "partial"  # covered does not approve pending science
    datasets = DatasetStore(tmp_path / "analysis")
    ids = tuple(datasets.load_dataframe(result.dataset_id)["measurement_id"])
    assert result.required_metric_coverage.checks[0].measurement_ids == ids
    text = render_literature_evidence_trial(result)
    assert "必需指标覆盖检查" in text
    assert "实际交给数据分析的记录" in text
    assert "已覆盖" in text
    assert (
        datasets.resolve_artifact_path(result.report_artifact_id).read_text(
            encoding="utf-8"
        )
        == text
    )


def test_missing_requirement_is_explicit_and_does_not_change_data(tmp_path):
    trial = service(tmp_path)
    result = trial.run(
        topic="任务5",
        metric_requirements=policy(
            required_metrics=[{"metric": "discharge capacity after 50 cycles"}]
        ),
    )
    assert result.required_metric_coverage.status == "incomplete"
    assert result.record_count == 1
    assert result.status == "partial"
    text = render_literature_evidence_trial(result)
    assert "缺失" in text
    assert "当前关键结果不完整" in text
    assert "discharge capacity after 50 cycles" in text


def test_empty_trial_reports_missing_without_fabricating_dataset(tmp_path):
    result = service(tmp_path, rows=[]).run(topic="任务5", metric_requirements=policy())
    assert result.required_metric_coverage.status == "incomplete"
    assert result.dataset_id is None
    assert result.analysis_ids == ()
    assert "缺失" in render_literature_evidence_trial(result)


@pytest.mark.parametrize(
    "mode", ["conflict", "wrong_group", "missing_group", "nonnumeric"]
)
def test_rows_not_actually_handed_off_cannot_fill_requirements(tmp_path, mode):
    row = _measurement()
    rows, groups = [row], [_group()]
    if mode == "conflict":
        rows.append(
            row.model_copy(
                update={"measurement_id": "m-conflict", "value_text": "80-90 nm"}
            )
        )
    elif mode == "wrong_group":
        rows = [
            row.model_copy(update={"source_quote": "pH 7 produced rods 50-70 nm long."})
        ]
    elif mode == "missing_group":
        groups = []
    else:
        rows = [
            row.model_copy(update={"value_text": "up to 70 nm", "numeric_value": None})
        ]
    result = service(tmp_path, rows=rows, groups=groups).prepare(
        topic="任务5", metric_requirements=policy()
    )
    assert result.record_count == 0
    assert result.required_metric_coverage.status == "incomplete"


def test_no_policy_and_legacy_models_are_not_checked(tmp_path):
    result = service(tmp_path).run(topic="任务5")
    assert result.required_metric_coverage.status == "not_checked"
    assert "未检查" in render_literature_evidence_trial(result)
    old = result.model_dump(exclude={"required_metric_coverage"})
    assert (
        LiteratureEvidenceTrialResult.model_validate(
            old
        ).required_metric_coverage.status
        == "not_checked"
    )
    handoff = service(tmp_path).prepare(topic="任务5")
    old_handoff = handoff.model_dump(exclude={"required_metric_coverage"})
    assert (
        LiteratureEvidenceHandoff.model_validate(
            old_handoff
        ).required_metric_coverage.status
        == "not_checked"
    )


def test_different_task_policy_fails_before_any_store_is_read(tmp_path):
    trial = service(tmp_path)
    trial._reports = SimpleNamespace(load=lambda _: pytest.fail("read foreign report"))
    with pytest.raises(ValueError, match="不匹配"):
        trial.prepare(topic="另一个任务", report_id="old", metric_requirements=policy())


def test_explicit_report_id_cannot_bypass_policy_topic_check(tmp_path):
    other = _report().model_copy(update={"topic": "其他任务"})
    with pytest.raises(ValueError, match="不匹配"):
        service(tmp_path, report=other).prepare(
            topic="任务5", report_id=other.report_id, metric_requirements=policy()
        )


def test_policy_topic_whitespace_matches_current_normalization(tmp_path):
    result = service(tmp_path).prepare(topic=" 任务5\n", metric_requirements=policy())
    assert result.required_metric_coverage.status == "covered"


def test_renderer_keeps_policy_names_from_injecting_markup():
    required = coverage.RequiredMetric(metric="length\n## forged <script> | *claim*")
    check = coverage.check_required_metric_coverage([], [required])
    text = coverage.render_required_metric_coverage(check)
    assert "\n## forged" not in text
    assert "<script>" not in text
    assert "缺失" in text


def setup_extraction(monkeypatch):
    batch = MatrixExtractionBatch(groups=(group(),), measurements=(candidate(),))
    original = AutomatedMatrixExtractor(FakeLlm(batch), Store()).extract(
        document_id="doc-1", chunks=(_chunk(),)
    )
    seen = []

    def extract(self, **kwargs):
        seen.append(kwargs)
        check = coverage.check_required_metric_coverage(
            original.measurements, kwargs.get("required_metrics", ())
        )
        return replace(
            original,
            diagnostics=original.diagnostics.model_copy(
                update={"required_metric_coverage": check}
            ),
        )

    store = SimpleNamespace(
        get_document_chunks=lambda _: [_chunk()],
        save_experiment_matrix=lambda *args: None,
        save_comparisons=lambda *args: None,
        save_claims_and_links=lambda *args: None,
    )
    settings = SimpleNamespace(
        llm_provider="intern",
        intern_model="fake",
        llm_max_output_tokens=8192,
        literature_extraction_model=None,
    )
    settings.model_copy = lambda **kwargs: settings
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(cli, "_literature_pgvector_store", lambda: store)
    monkeypatch.setattr(cli, "create_llm_provider", lambda _: object())
    monkeypatch.setattr(AutomatedMatrixExtractor, "extract", extract)
    return seen


def test_matrix_cli_loads_same_policy_and_persists_task_binding(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    seen = setup_extraction(monkeypatch)
    path = tmp_path / "policy.json"
    expected = policy()
    path.write_text(expected.model_dump_json(), encoding="utf-8")
    cli.literature_matrix_extract_command(
        "doc-1", show_warnings=False, requirements=path
    )
    assert seen[0]["required_metrics"] == expected.required_metrics
    files = list((tmp_path / "data/literature_extractions").glob("*.json"))
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["metric_requirements"] == expected.model_dump(mode="json")
    assert saved["diagnostics"]["required_metric_coverage"]["status"] == "incomplete"
    assert "缺失" in capsys.readouterr().out  # cannot hide missing requirements


@pytest.mark.parametrize("entry", ["extract", "trial"])
def test_bad_cli_policy_fails_before_model_or_database(monkeypatch, tmp_path, entry):
    path = tmp_path / "bad.json"
    path.write_text('{"PRIVATE": true}', encoding="utf-8")
    monkeypatch.setattr(
        cli, "_literature_pgvector_store", lambda: pytest.fail("database used")
    )
    monkeypatch.setattr(cli, "create_llm_provider", lambda _: pytest.fail("model used"))
    with pytest.raises(cli.typer.Exit):
        if entry == "extract":
            cli.literature_matrix_extract_command(
                "doc-1", show_warnings=False, requirements=path
            )
        else:
            cli.master_evidence_trial_command(
                message="任务5",
                report_id=None,
                data_root=tmp_path / "data",
                output=tmp_path / "report.md",
                requirements=path,
            )
    assert not (tmp_path / "data").exists()


def test_trial_cli_passes_same_policy_and_writes_coverage_report(monkeypatch, tmp_path):
    path = tmp_path / "policy.json"
    expected = policy()
    path.write_text(expected.model_dump_json(), encoding="utf-8")
    trial = service(tmp_path)
    seen = []
    original = trial.run

    def run(**kwargs):
        seen.append(kwargs)
        return original(**kwargs)

    trial.run = run
    monkeypatch.setattr(
        "materials_screening.master.literature_evidence_trial.LiteratureEvidenceTrialService",
        lambda **kwargs: trial,
    )
    monkeypatch.setattr(cli, "_literature_pgvector_store", lambda: object())
    cli.master_evidence_trial_command(
        message="任务5",
        report_id=None,
        data_root=tmp_path / "cli-data",
        output=tmp_path / "report.md",
        requirements=path,
    )
    assert seen[0]["metric_requirements"] == expected
    assert "已覆盖" in (tmp_path / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "command", [["literature", "matrix-extract"], ["master", "evidence-trial"]]
)
def test_cli_help_exposes_optional_requirements_flag(command):
    result = CliRunner().invoke(cli.app, [*command, "--help"])
    assert result.exit_code == 0
    assert "--requirements" in result.stdout


def test_real_cli_argument_parser_routes_policy_without_live_model(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    seen = setup_extraction(monkeypatch)
    path = tmp_path / "policy.json"
    expected = policy()
    path.write_text(expected.model_dump_json(), encoding="utf-8")
    result = CliRunner().invoke(
        cli.app,
        [
            "literature",
            "matrix-extract",
            "doc-1",
            "--requirements",
            str(path),
            "--no-show-warnings",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen[0]["required_metrics"] == expected.required_metrics
    assert "缺失" in result.stdout


def test_real_trial_argument_parser_routes_policy_without_database(
    monkeypatch, tmp_path
):
    path = tmp_path / "policy.json"
    expected = policy()
    path.write_text(expected.model_dump_json(), encoding="utf-8")
    trial = service(tmp_path)
    monkeypatch.setattr(
        "materials_screening.master.literature_evidence_trial.LiteratureEvidenceTrialService",
        lambda **kwargs: trial,
    )
    monkeypatch.setattr(cli, "_literature_pgvector_store", lambda: object())
    result = CliRunner().invoke(
        cli.app,
        [
            "master",
            "evidence-trial",
            "--message",
            "任务5",
            "--requirements",
            str(path),
            "--data-root",
            str(tmp_path / "cli-data"),
            "--output",
            str(tmp_path / "report.md"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "实际交给数据分析的记录" in result.stdout
    assert "状态：已覆盖" in result.stdout


def test_old_matrix_cli_call_uses_no_policy_not_an_option_object(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    seen = setup_extraction(monkeypatch)
    cli.literature_matrix_extract_command("doc-1", show_warnings=False)
    assert seen[0]["required_metrics"] == ()
    assert "未检查" in capsys.readouterr().out


def test_trial_policy_topic_mismatch_fails_before_database(monkeypatch, tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(policy().model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        cli, "_literature_pgvector_store", lambda: pytest.fail("database used")
    )
    with pytest.raises(cli.typer.Exit):
        cli.master_evidence_trial_command(
            message="其他任务",
            report_id=None,
            data_root=tmp_path / "cli-data",
            output=tmp_path / "report.md",
            requirements=path,
        )
    assert not (tmp_path / "cli-data").exists()


def test_final_row_conversion_filter_also_cannot_fill_coverage(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "materials_screening.master.literature_evidence_trial._measurement_rows",
        lambda **kwargs: [],
    )
    result = service(tmp_path).prepare(topic="任务5", metric_requirements=policy())
    assert result.record_count == 0
    assert result.required_metric_coverage.status == "incomplete"
