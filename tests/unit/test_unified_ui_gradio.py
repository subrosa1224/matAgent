"""Behavior tests for the unified multi-agent UI."""

import os
from pathlib import Path
from typing import Any

from materials_screening import unified_ui_gradio as ui
from materials_screening.agent.models import AgentResult
from materials_screening.master import (
    ArtifactRegistry,
    CrossAgentCandidate,
    CrossAgentResult,
    MaterialClue,
    SubAgentUiPlugin,
    SubAgentUiPluginRegistry,
    SubAgentUiSpec,
)
from materials_screening.master.master_runner import AgentStreamEvent
from materials_screening.models import MaterialRecord
from materials_screening.services.query_result_store import QueryResultStore
from materials_screening.sub_agents.literature.models import (
    LiteratureSearchOutput,
    PaperRecord,
)
from materials_screening.sub_agents.literature.store import LiteratureQueryStore


class _MasterRunner:
    def ask_stream(self, *, message: str, conversation_id: str | None) -> Any:
        assert message == "筛选材料"
        assert conversation_id is None
        yield AgentStreamEvent(
            node="runner",
            message="完成",
            is_final=True,
            result=AgentResult(
                conversation_id="master-conversation-1",
                user_turn_id="turn-1",
                status="completed",
                final_status="completed",
                response_text="材料查询结果",
                selected_tools=("delegate_to_materials_database",),
            ),
        )


def test_auto_mode_uses_master_runner(monkeypatch: Any) -> None:
    monkeypatch.setattr(ui, "_master_runner", _MasterRunner())
    updates = list(
        ui._dispatch(
            ui.MODE_AUTO,
            "筛选材料",
            [],
            None,
            ui._initial_state(),
            "",
            2020,
            10,
        )
    )
    history, _trace, state, _message, _files = updates[-1]
    assert history[-1]["content"] == "材料查询结果"
    assert history[0]["content"] == "筛选材料"
    assert state["master_conversation_id"] == "master-conversation-1"
    assert state["last_agent"] == "materials_database"


def test_auto_mode_requires_initialization(monkeypatch: Any) -> None:
    monkeypatch.setattr(ui, "_master_runner", None)
    updates = list(
        ui._dispatch(
            ui.MODE_AUTO,
            "筛选材料",
            [],
            None,
            ui._initial_state(),
            "",
            2020,
            10,
        )
    )
    assert "请先点击" in updates[-1][0][-1]["content"]


def test_database_mode_never_receives_pdf() -> None:
    updates = list(
        ui._dispatch(
            ui.MODE_DATABASE,
            "分析附件",
            [],
            ["paper.pdf"],
            ui._initial_state(),
            "",
            2020,
            10,
        )
    )
    assert "暂不接收附件" in updates[-1][0][-1]["content"]


def test_literature_mode_delegates_to_existing_controller(monkeypatch: Any) -> None:
    def fake_chat(*args: Any, **kwargs: Any) -> Any:
        yield (
            [
                {"role": "user", "content": "预览论文"},
                {"role": "assistant", "content": "论文概览"},
            ],
            "trace",
            {"topic": "", "paths": [], "document_ids": []},
            "",
            None,
        )

    monkeypatch.setattr(ui.literature_ui, "_chat", fake_chat)
    updates = list(
        ui._dispatch(
            ui.MODE_LITERATURE,
            "预览论文",
            [],
            None,
            ui._initial_state(),
            "",
            2020,
            10,
        )
    )
    history, _trace, state, _message, _files = updates[-1]
    assert history[-1]["content"] == "论文概览"
    assert state["last_agent"] == "literature"


def test_ui_builds_with_registered_agent_metadata() -> None:
    assert ui.UI_SPECS.names() == (
        "data_analysis",
        "literature",
        "materials_database",
    )
    assert type(ui._create_ui()).__name__ == "Blocks"


def test_unified_upload_root_uses_literature_ingestion_allowlist(
    monkeypatch: Any, tmp_path: Path
) -> None:
    first = tmp_path / "literature-a"
    second = tmp_path / "literature-b"
    monkeypatch.setenv("LITERATURE_INGEST_ROOTS", f"{first}{os.pathsep}{second}")

    assert ui._literature_artifact_root() == first


def test_attachment_status_lists_selected_pdf_names(tmp_path: Path) -> None:
    first = tmp_path / "paper-one.pdf"
    second = tmp_path / "paper-two.pdf"

    status = ui._attachment_markdown([str(first), str(second)], {})

    assert "已选择 2 篇" in status
    assert "paper-one.pdf" in status
    assert "paper-two.pdf" in status
    assert str(tmp_path) not in status


def test_attachment_status_persists_registered_names_after_send() -> None:
    status = ui._attachment_markdown(
        None,
        {"artifact_names": ["uploaded-paper.pdf"]},
    )

    assert status == "📎 已选择 1 篇：uploaded-paper.pdf"


def test_clear_attachments_removes_selection_but_keeps_analysis_state() -> None:
    state = ui._initial_state()
    state["artifact_ids"] = ["artifact-pdf-1"]
    state["artifact_names"] = ["paper.pdf"]
    state["literature_state"]["document_ids"] = ["doc-1"]

    files, cleared, status = ui._clear_attachments(state)

    assert files is None
    assert cleared["artifact_ids"] == []
    assert cleared["artifact_names"] == []
    assert cleared["literature_state"]["document_ids"] == ["doc-1"]
    assert status == "📎 尚未上传附件"


def test_literature_result_component_pages_saved_query(
    monkeypatch: Any, tmp_path: Path
) -> None:
    store = LiteratureQueryStore(tmp_path / "queries")
    papers = tuple(
        PaperRecord(
            paper_id=f"paper-{index}",
            title=f"Paper {index}",
            doi=f"10.1/{index}",
            year=2025,
            authors=(f"Author {index}",),
            venue="Journal",
            abstract=f"Abstract {index}",
            provenance=(),
        )
        for index in range(1, 13)
    )
    store.save(
        LiteratureSearchOutput(
            query_id="lit-page123",
            provider="unified",
            papers=papers,
            returned_count=len(papers),
        )
    )
    monkeypatch.setattr(ui, "LITERATURE_QUERIES", store)
    state = ui._initial_state()
    conversation = [
        {
            "role": "assistant",
            "content": "统一检索 ID：`lit-page123`",
        }
    ]
    ui._capture_literature_query(conversation, state)

    first_rows, first_details, first_label, state = ui._literature_page(state)
    second_rows, second_details, second_label, state = ui._literature_page(state, 1)

    assert len(first_rows) == 5
    assert first_rows[0][3] == "Paper 1"
    assert "Abstract 1" in first_details
    assert first_label == "第 1 / 3 页 · 共 12 篇"
    assert second_rows[0][3] == "Paper 6"
    assert "Abstract 6" in second_details
    assert second_label == "第 2 / 3 页 · 共 12 篇"


def test_new_conversation_hides_literature_result_panel() -> None:
    outputs = ui._new_conversation_ui(False)

    assert outputs[10]["visible"] is False
    assert outputs[14]["visible"] is False


def test_auto_pdf_uses_artifact_and_hides_internal_path(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.7\ntest")
    registry = ArtifactRegistry(tmp_path / "artifacts")
    monkeypatch.setattr(ui, "ARTIFACTS", registry)

    def fake_chat(
        message: str,
        history: list[dict[str, Any]],
        files: list[str] | None,
        state: dict[str, Any],
        *args: Any,
    ) -> Any:
        assert files and Path(files[0]).is_file()
        state = {**state, "paths": files, "document_ids": ["doc-123"]}
        yield (
            [
                *history,
                {"role": "user", "content": message},
                {"role": "assistant", "content": "预览完成"},
            ],
            "trace",
            state,
            "",
            None,
        )

    monkeypatch.setattr(ui.literature_ui, "_chat", fake_chat)
    updates = list(
        ui._dispatch(
            ui.MODE_AUTO,
            "预览这些论文",
            [],
            [str(source)],
            ui._initial_state(),
            "",
            2020,
            10,
        )
    )
    history, trace, state, _message, _files = updates[-1]
    assert history[-1]["content"] == "预览完成"
    assert state["artifact_ids"]
    assert state["literature_state"]["paths"] == []
    assert str(tmp_path) not in trace


def test_auto_pdf_without_topic_defaults_to_preview(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source = tmp_path / "untitled.pdf"
    source.write_bytes(b"%PDF-1.7\ntest")
    monkeypatch.setattr(ui, "ARTIFACTS", ArtifactRegistry(tmp_path / "artifacts"))

    def fake_chat(message: str, *args: Any) -> Any:
        assert message == "请预览我上传的论文。"
        state = args[2]
        yield (
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "完成"},
            ],
            "trace",
            state,
            "",
            None,
        )

    monkeypatch.setattr(ui.literature_ui, "_chat", fake_chat)
    updates = list(
        ui._dispatch(
            ui.MODE_AUTO,
            "",
            [],
            [str(source)],
            ui._initial_state(),
            "",
            2020,
            10,
        )
    )
    assert updates[-1][2]["last_agent"] == "literature"


def test_auto_csv_registers_dataset_and_routes_to_data_analysis(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source = tmp_path / "measurements.csv"
    source.write_text("sample,value\nA,1\nB,2\n", encoding="utf-8")
    registry = ArtifactRegistry(
        tmp_path / "artifacts",
        dataset_store=ui.DatasetStore(tmp_path / "data-analysis"),
    )
    monkeypatch.setattr(ui, "ARTIFACTS", registry)

    class DataRunner:
        def ask_stream(self, *, message: str, conversation_id: str | None) -> Any:
            assert "数据集分析任务" in message
            assert "dataset-" in message
            assert conversation_id is None
            yield AgentStreamEvent(
                node="runner",
                message="data analysis completed",
                is_final=True,
                result=AgentResult(
                    conversation_id="master-data-1",
                    user_turn_id="turn-data-1",
                    status="completed",
                    final_status="completed",
                    response_text="数据集检查完成：2 行、2 列。",
                    selected_tools=("delegate_to_data_analysis",),
                ),
            )

    monkeypatch.setattr(ui, "_master_runner", DataRunner())
    updates = list(
        ui._dispatch(
            ui.MODE_AUTO,
            "",
            [],
            [str(source)],
            ui._initial_state(),
            "",
            2020,
            10,
        )
    )

    history, _trace, state, _message, _files = updates[-1]
    assert "一键探索分析" in history[-1]["content"]
    assert "2 行 × 2 列" in history[-1]["content"]
    assert state["last_agent"] == "data_analysis"
    assert state["dataset_ids"][0].startswith("dataset-")
    assert state["data_exploration"] is not None
    assert str(tmp_path) not in str(history)


def test_auto_xlsx_reads_first_sheet_and_runs_exploration(
    monkeypatch: Any, tmp_path: Path
) -> None:
    from openpyxl import Workbook

    source = tmp_path / "measurements.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "measurements"
    sheet.append(["sample", "value"])
    sheet.append(["A", 1.0])
    sheet.append(["B", 2.0])
    extra = workbook.create_sheet("ignored")
    extra.append(["not", "used"])
    workbook.save(source)
    monkeypatch.setattr(
        ui,
        "ARTIFACTS",
        ArtifactRegistry(
            tmp_path / "artifacts",
            dataset_store=ui.DatasetStore(tmp_path / "data-analysis"),
        ),
    )

    history, _trace, state, _message, _files = list(
        ui._dispatch(
            ui.MODE_AUTO,
            "",
            [],
            [str(source)],
            ui._initial_state(),
            "",
            2020,
            10,
        )
    )[-1]

    assert "一键探索分析" in history[-1]["content"]
    assert "2 行 × 2 列" in history[-1]["content"]
    assert state["dataset_ids"]


def test_explicit_data_mode_preserves_master_conversation(monkeypatch: Any) -> None:
    received_ids: list[str | None] = []

    class DataRunner:
        def ask_stream(self, *, message: str, conversation_id: str | None) -> Any:
            assert "dataset-12345678" in message
            received_ids.append(conversation_id)
            yield AgentStreamEvent(
                node="runner",
                message="data_analysis completed",
                is_final=True,
                result=AgentResult(
                    conversation_id="master-data-continuous",
                    user_turn_id=f"turn-{len(received_ids)}",
                    status="completed",
                    final_status="completed",
                    response_text="数据检查完成。",
                    selected_tools=("delegate_to_data_analysis",),
                ),
            )

    monkeypatch.setattr(ui, "_master_runner", DataRunner())
    state = ui._initial_state()
    state["dataset_ids"] = ["dataset-12345678"]
    first = list(
        ui._dispatch(
            ui.MODE_DATA_ANALYSIS,
            "检查这份数据",
            [],
            None,
            state,
            "",
            2020,
            10,
        )
    )[-1]
    second = list(
        ui._dispatch(
            ui.MODE_DATA_ANALYSIS,
            "继续检查缺失值",
            first[0],
            None,
            first[2],
            "",
            2020,
            10,
        )
    )[-1]

    assert received_ids == [None, "master-data-continuous"]
    assert second[2]["last_agent"] == "data_analysis"
    assert second[0][-1]["content"] == "数据检查完成。"


def test_invalid_data_upload_can_recover_with_valid_file(
    monkeypatch: Any, tmp_path: Path
) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"not": "rows"}', encoding="utf-8")
    valid = tmp_path / "valid.csv"
    valid.write_text("sample,value\nA,1\n", encoding="utf-8")
    monkeypatch.setattr(
        ui,
        "ARTIFACTS",
        ArtifactRegistry(
            tmp_path / "artifacts",
            dataset_store=ui.DatasetStore(tmp_path / "data-analysis"),
        ),
    )

    class DataRunner:
        def ask_stream(self, *, message: str, conversation_id: str | None) -> Any:
            del message, conversation_id
            yield AgentStreamEvent(
                node="runner",
                message="data analysis completed",
                is_final=True,
                result=AgentResult(
                    conversation_id="master-data-recovered",
                    user_turn_id="turn-recovered",
                    status="completed",
                    final_status="completed",
                    response_text="恢复成功。",
                    selected_tools=("delegate_to_data_analysis",),
                ),
            )

    monkeypatch.setattr(ui, "_master_runner", DataRunner())
    failed = list(
        ui._dispatch(
            ui.MODE_DATA_ANALYSIS,
            "检查数据",
            [],
            [str(invalid)],
            ui._initial_state(),
            "",
            2020,
            10,
        )
    )[-1]
    recovered = list(
        ui._dispatch(
            ui.MODE_DATA_ANALYSIS,
            "检查数据",
            failed[0],
            [str(valid)],
            failed[2],
            "",
            2020,
            10,
        )
    )[-1]

    assert "top-level array" in failed[0][-1]["content"]
    assert recovered[0][-1]["content"] == "恢复成功。"


def test_data_form_runs_statistics_plot_report_and_export(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source = Path("examples/data_analysis/grouped_experiment_demo.csv").resolve()
    store = ui.DatasetStore(tmp_path / "data-analysis")
    registry = ArtifactRegistry(tmp_path / "artifacts", dataset_store=store)
    artifact = registry.register_data_file(source)
    monkeypatch.setattr(ui, "ARTIFACTS", registry)
    state = ui._initial_state()
    state["dataset_ids"] = [artifact.domain_id]

    described = ui._run_data_action(
        "describe",
        state,
        ["hardness_hv", "conductivity_s_cm"],
        "hardness_hv",
        "conductivity_s_cm",
        "method",
        ["A", "B"],
        "welch_t",
        "scatter",
        "pearson",
        "CSV",
    )
    assert "descriptive" in described[0]
    assert described[1]["analysis_ids"]

    tested = ui._run_data_action(
        "test",
        described[1],
        ["hardness_hv"],
        "hardness_hv",
        "conductivity_s_cm",
        "method",
        ["A", "B"],
        "welch_t",
        "scatter",
        "pearson",
        "CSV",
    )
    assert "组间差异分析" in tested[0]

    plotted = ui._run_data_action(
        "plot",
        tested[1],
        ["hardness_hv", "conductivity_s_cm"],
        "temperature_c",
        "conductivity_s_cm",
        "method",
        ["A", "B"],
        "welch_t",
        "scatter",
        "pearson",
        "CSV",
    )
    assert plotted[2] and Path(plotted[2]).is_file()
    assert str(tmp_path) not in plotted[0]

    reported = ui._run_data_action(
        "report",
        plotted[1],
        [],
        None,
        None,
        None,
        [],
        "welch_t",
        "scatter",
        "pearson",
        "CSV",
    )
    assert reported[3] and Path(reported[3]).is_file()
    exported = ui._run_data_action(
        "export",
        reported[1],
        [],
        None,
        None,
        None,
        [],
        "welch_t",
        "scatter",
        "pearson",
        "JSON",
    )
    assert exported[3] and Path(exported[3]).suffix == ".json"


def test_guided_questions_choose_safe_defaults(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source = Path("examples/data_analysis/grouped_experiment_demo.csv").resolve()
    store = ui.DatasetStore(tmp_path / "data-analysis")
    registry = ArtifactRegistry(tmp_path / "artifacts", dataset_store=store)
    artifact = registry.register_data_file(source)
    monkeypatch.setattr(ui, "ARTIFACTS", registry)
    state = ui._initial_state()
    state["dataset_ids"] = [artifact.domain_id]
    bundle = ui.DataExplorationService(
        ui.DataAnalysisService(store), ui.DataStatisticsService(store)
    ).explore(artifact.domain_id)
    state["data_exploration"] = bundle.model_dump(mode="json")
    state["analysis_ids"] = list(bundle.analysis_ids)

    for question, expected in (
        ("这份数据有没有问题？", "一键探索分析"),
        ("先帮我总结一下", "descriptive"),
        ("不同组之间有差异吗？", "组间差异分析"),
        ("哪些字段有关联？", "correlation"),
        ("画一张推荐图", "图表已生成"),
        ("整理成报告", "报告已生成"),
    ):
        result, state, image, download = ui._run_guided_data_action(
            question, state
        )
        assert f"你问：**{question}**" in result
        assert expected in result
        if question == "画一张推荐图":
            assert image and Path(image).is_file()
        if question == "整理成报告":
            assert download and Path(download).is_file()


def test_guided_question_is_answered_inside_chat_history(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source = Path("examples/data_analysis/grouped_experiment_demo.csv").resolve()
    store = ui.DatasetStore(tmp_path / "data-analysis")
    registry = ArtifactRegistry(tmp_path / "artifacts", dataset_store=store)
    artifact = registry.register_data_file(source)
    monkeypatch.setattr(ui, "ARTIFACTS", registry)
    state = ui._initial_state()
    state["dataset_ids"] = [artifact.domain_id]
    state["last_agent"] = "data_analysis"
    bundle = ui.DataExplorationService(
        ui.DataAnalysisService(store), ui.DataStatisticsService(store)
    ).explore(artifact.domain_id)
    state["data_exploration"] = bundle.model_dump(mode="json")
    state["analysis_ids"] = list(bundle.analysis_ids)
    history = ui._with_data_chat_options(
        [{"role": "assistant", "content": "数据上传完成。"}], state
    )
    assert history[-1]["options"] == ui._data_chat_options()

    event = ui.gr.SelectData(
        None,
        {"index": 0, "value": "哪些字段有关联？"},
    )
    answered, updated = ui._answer_data_chat_option(history, state, event)

    assert answered[-2] == {"role": "user", "content": "哪些字段有关联？"}
    assert answered[-1]["role"] == "assistant"
    assert "correlation" in answered[-1]["content"]
    assert answered[-1]["options"] == ui._data_chat_options(
        after="哪些字段有关联？"
    )
    assert answered[-1]["options"] != ui._data_chat_options()
    assert updated["analysis_ids"]


def test_guided_report_ends_repeating_quick_questions(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source = Path("examples/data_analysis/grouped_experiment_demo.csv").resolve()
    store = ui.DatasetStore(tmp_path / "data-analysis")
    registry = ArtifactRegistry(tmp_path / "artifacts", dataset_store=store)
    artifact = registry.register_data_file(source)
    monkeypatch.setattr(ui, "ARTIFACTS", registry)
    bundle = ui.DataExplorationService(
        ui.DataAnalysisService(store), ui.DataStatisticsService(store)
    ).explore(artifact.domain_id)
    state = ui._initial_state()
    state["dataset_ids"] = [artifact.domain_id]
    state["data_exploration"] = bundle.model_dump(mode="json")
    state["analysis_ids"] = list(bundle.analysis_ids)
    state["scientific_brief"] = ui.ScientificAnalysisBrief(
        dataset_id=artifact.domain_id,
        domain="materials_science",
        research_question="不同 method 组的 hardness_hv 是否存在差异？",
        observation_unit="独立材料试样",
        design="independent_groups",
        response_variables=("hardness_hv",),
        group_variable="method",
        roles_confirmed=True,
    ).model_dump(mode="json")
    history = [
        {
            "role": "assistant",
            "content": "要整理报告吗？",
            "options": ui._data_chat_options(after="画一张推荐图"),
        }
    ]
    event = ui.gr.SelectData(
        None,
        {"index": 0, "value": "整理成报告"},
    )

    answered, _state = ui._answer_data_chat_option(history, state, event)

    text_answer = next(
        message
        for message in reversed(answered)
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), str)
    )
    assert "报告已生成" in text_answer["content"]
    assert "options" not in text_answer


def test_group_comparison_asks_for_metric_before_running_test(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source = Path("examples/data_analysis/grouped_experiment_demo.csv").resolve()
    store = ui.DatasetStore(tmp_path / "data-analysis")
    registry = ArtifactRegistry(tmp_path / "artifacts", dataset_store=store)
    artifact = registry.register_data_file(source)
    monkeypatch.setattr(ui, "ARTIFACTS", registry)
    bundle = ui.DataExplorationService(
        ui.DataAnalysisService(store), ui.DataStatisticsService(store)
    ).explore(artifact.domain_id)
    state = ui._initial_state()
    state["dataset_ids"] = [artifact.domain_id]
    state["data_exploration"] = bundle.model_dump(mode="json")
    state["analysis_ids"] = list(bundle.analysis_ids)
    state["scientific_proposal"] = ui.ScientificBriefService(
        store
    ).propose_roles(artifact.domain_id, domain="materials_science").model_dump(
        mode="json"
    )
    initial_analysis_ids = list(state["analysis_ids"])
    history = [
        {
            "role": "assistant",
            "content": "你想继续分析什么？",
            "options": ui._data_chat_options(),
        }
    ]

    ask_event = ui.gr.SelectData(
        None,
        {"index": 0, "value": "不同组之间有差异吗？"},
    )
    clarified, state = ui._answer_data_chat_option(history, state, ask_event)

    assert state["analysis_ids"] == initial_analysis_ids
    assert "分组字段是 `method`" in clarified[-1]["content"]
    assert any(
        option["value"]
        == f"{ui.DATA_GROUP_METRIC_PREFIX}hardness_hv"
        for option in clarified[-1]["options"]
    )

    metric_event = ui.gr.SelectData(
        None,
        {
            "index": 0,
            "value": f"{ui.DATA_GROUP_METRIC_PREFIX}hardness_hv",
        },
    )
    answered, state = ui._answer_data_chat_option(
        clarified, state, metric_event
    )

    answer = answered[-1]["content"]
    assert "不会默认它是独立组实验" in answer
    assert state["analysis_ids"] == initial_analysis_ids
    assert state["scientific_brief"] is None

    for value in (
        f"{ui.DATA_SCIENCE_DESIGN_PREFIX}independent_groups",
        f"{ui.DATA_SCIENCE_GROUP_PREFIX}method",
        f"{ui.DATA_SCIENCE_UNIT_PREFIX}HV",
        ui.DATA_SCIENCE_THRESHOLD_SKIP,
    ):
        answered, state = ui._answer_data_chat_option(
            answered,
            state,
            ui.gr.SelectData(None, {"index": 0, "value": value}),
        )
        assert state["analysis_ids"] == initial_analysis_ids

    assert "当前研究设定" in answered[-1]["content"]
    assert "待你确认" in answered[-1]["content"]
    answered, state = ui._answer_data_chat_option(
        answered,
        state,
        ui.gr.SelectData(
            None, {"index": 0, "value": ui.DATA_SCIENCE_CONFIRM}
        ),
    )
    answer = answered[-1]["content"]
    assert "已确认，可以据此选择统计方法" in answer
    assert "推荐分析方案" in answer
    assert "单因素 ANOVA" in answer
    assert state["analysis_ids"] == initial_analysis_ids

    answered, state = ui._answer_data_chat_option(
        answered,
        state,
        ui.gr.SelectData(
            None,
            {
                "index": 0,
                "value": f"{ui.DATA_SCIENCE_RUN_PREFIX}recommended",
            },
        ),
    )
    answer = answered[-1]["content"]
    assert "科研结论摘要" in answer
    assert "证据强度：较强" in answer
    assert "B（581.1 HV） > A（540.6 HV） > C（521 HV）" in answer
    assert "不能验证随机分组" in answer
    assert "建议的下一步" in answer
    assert "比较指标：`hardness_hv`" in answer
    assert "分组字段：`method`" in answer
    assert "单因素方差分析（ANOVA）" in answer
    assert "assumptions" not in answer
    assert len(state["analysis_ids"]) == len(initial_analysis_ids) + 1


def test_scientific_setup_requires_subject_id_for_paired_design(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source = tmp_path / "paired-ui.csv"
    source.write_text(
        "sample_id,condition,hardness_hv\n"
        "S1,before,500\nS1,after,510\n"
        "S2,before,505\nS2,after,520\n"
        "S3,before,510\nS3,after,530\n"
        "S4,before,515\nS4,after,527\n"
        "S5,before,520\nS5,after,545\n",
        encoding="utf-8",
    )
    store = ui.DatasetStore(tmp_path / "data-analysis")
    registry = ArtifactRegistry(tmp_path / "artifacts", dataset_store=store)
    artifact = registry.register_data_file(source)
    monkeypatch.setattr(ui, "ARTIFACTS", registry)
    state = ui._initial_state()
    state["dataset_ids"] = [artifact.domain_id]
    state["scientific_proposal"] = ui.ScientificBriefService(
        store
    ).propose_roles(artifact.domain_id, domain="materials_science").model_dump(
        mode="json"
    )

    content, _options = ui._begin_scientific_setup(state)
    assert "科研问题" in content
    continued = ui._continue_scientific_setup(
        "同一样品处理前后的硬度是否发生变化？", [], state
    )
    assert continued is not None
    history, state = continued
    assert state["scientific_brief"] is None

    for value in (
        f"{ui.DATA_SCIENCE_RESPONSE_PREFIX}hardness_hv",
        f"{ui.DATA_SCIENCE_DESIGN_PREFIX}paired",
    ):
        history, state = ui._answer_data_chat_option(
            history,
            state,
            ui.gr.SelectData(None, {"index": 0, "value": value}),
        )

    assert state["scientific_brief"] is None
    assert state["scientific_setup"]["step"] == "subject_id"
    assert "样本或受试对象" in history[-1]["content"]
    subject_options = history[-1].get("options") or []
    assert subject_options
    subject_value = subject_options[0]["value"]

    for value in (
        subject_value,
        f"{ui.DATA_SCIENCE_GROUP_PREFIX}condition",
        f"{ui.DATA_SCIENCE_UNIT_PREFIX}HV",
        ui.DATA_SCIENCE_THRESHOLD_SKIP,
        ui.DATA_SCIENCE_CONFIRM,
    ):
        history, state = ui._answer_data_chat_option(
            history,
            state,
            ui.gr.SelectData(None, {"index": 0, "value": value}),
        )

    brief = ui.ScientificAnalysisBrief.model_validate(state["scientific_brief"])
    assert brief.design == "paired"
    assert brief.subject_id_variable
    assert brief.group_variable == "condition"
    assert brief.roles_confirmed is True
    assert "推荐方法：配对 t 检验" in history[-1]["content"]


def test_typed_research_question_continues_inside_chat_without_master(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source = Path("examples/data_analysis/grouped_experiment_demo.csv").resolve()
    store = ui.DatasetStore(tmp_path / "data-analysis")
    registry = ArtifactRegistry(tmp_path / "artifacts", dataset_store=store)
    artifact = registry.register_data_file(source)
    monkeypatch.setattr(ui, "ARTIFACTS", registry)
    monkeypatch.setattr(ui, "_master_runner", None)
    state = ui._initial_state()
    state["dataset_ids"] = [artifact.domain_id]
    state["scientific_proposal"] = ui.ScientificBriefService(
        store
    ).propose_roles(artifact.domain_id, domain="materials_science").model_dump(
        mode="json"
    )
    ui._begin_scientific_setup(state)

    updates = list(
        ui._dispatch(
            ui.MODE_AUTO,
            "不同制备方法是否会改变材料硬度？",
            [],
            None,
            state,
            "",
            2020,
            10,
        )
    )

    history, trace, updated, cleared, _files = updates[-1]
    assert history[-2]["content"] == "不同制备方法是否会改变材料硬度？"
    assert "主要关注哪个测量指标" in history[-1]["content"]
    assert history[-1]["options"]
    assert "尚未初始化" not in trace
    assert updated["scientific_setup"]["step"] == "response"
    assert cleared == ""


def test_upload_with_research_question_enters_scientific_setup_directly(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source = Path("examples/data_analysis/grouped_experiment_demo.csv").resolve()
    store = ui.DatasetStore(tmp_path / "data-analysis")
    registry = ArtifactRegistry(tmp_path / "artifacts", dataset_store=store)
    monkeypatch.setattr(ui, "ARTIFACTS", registry)
    monkeypatch.setattr(ui, "_master_runner", None)

    updates = list(
        ui._dispatch(
            ui.MODE_AUTO,
            "不同制备方法是否会显著影响材料硬度？",
            [],
            [source],
            ui._initial_state(),
            "",
            2020,
            10,
        )
    )

    history, trace, state, _cleared, _files = updates[-1]
    assert "数据已读取：30 行 × 8 列" in history[-1]["content"]
    assert "主要关注哪个测量指标" in history[-1]["content"]
    assert any(
        option["value"]
        == f"{ui.DATA_SCIENCE_RESPONSE_PREFIX}hardness_hv"
        for option in history[-1]["options"]
    )
    assert state["scientific_setup"]["step"] == "response"
    assert state["scientific_setup"]["research_question"].startswith(
        "不同制备方法"
    )
    assert "尚未初始化" not in trace


def test_continuous_relationship_setup_runs_regression(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source = Path("examples/data_analysis/grouped_experiment_demo.csv").resolve()
    store = ui.DatasetStore(tmp_path / "data-analysis")
    registry = ArtifactRegistry(tmp_path / "artifacts", dataset_store=store)
    artifact = registry.register_data_file(source)
    monkeypatch.setattr(ui, "ARTIFACTS", registry)
    state = ui._initial_state()
    state["dataset_ids"] = [artifact.domain_id]
    state["scientific_proposal"] = ui.ScientificBriefService(
        store
    ).propose_roles(artifact.domain_id, domain="materials_science").model_dump(
        mode="json"
    )
    ui._begin_scientific_setup(state)
    continued = ui._continue_scientific_setup(
        "温度与材料硬度是否存在线性关系？", [], state
    )
    assert continued is not None
    history, state = continued

    for value in (
        f"{ui.DATA_SCIENCE_RESPONSE_PREFIX}hardness_hv",
        f"{ui.DATA_SCIENCE_DESIGN_PREFIX}continuous_relationship",
        f"{ui.DATA_SCIENCE_PREDICTOR_PREFIX}temperature_c",
        f"{ui.DATA_SCIENCE_UNIT_PREFIX}HV",
        ui.DATA_SCIENCE_THRESHOLD_SKIP,
        ui.DATA_SCIENCE_CONFIRM,
        f"{ui.DATA_SCIENCE_RUN_PREFIX}recommended",
    ):
        history, state = ui._answer_data_chat_option(
            history,
            state,
            ui.gr.SelectData(None, {"index": 0, "value": value}),
        )

    assert "连续变量关系分析" in history[-1]["content"]
    assert "简单线性回归" in history[-1]["content"]
    assert "仅凭回归不能证明因果关系" in history[-1]["content"]
    result = store.get_analysis(state["analysis_ids"][-1])
    assert result.analysis_type == "regression"


def test_data_form_deduplicates_into_new_dataset(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source = Path("examples/data_analysis/quality_issues_demo.csv").resolve()
    store = ui.DatasetStore(tmp_path / "data-analysis")
    registry = ArtifactRegistry(tmp_path / "artifacts", dataset_store=store)
    artifact = registry.register_data_file(source)
    monkeypatch.setattr(ui, "ARTIFACTS", registry)
    state = ui._initial_state()
    state["dataset_ids"] = [artifact.domain_id]

    outputs = ui._run_deduplicate_action(state, "sample_id")
    derived_id = outputs[1]["dataset_ids"][0]

    assert "12 → 11 行" in outputs[0]
    assert derived_id != artifact.domain_id
    assert store.load_dataframe(artifact.domain_id).shape[0] == 12
    assert store.load_dataframe(derived_id).shape[0] == 11
    assert outputs[5]["visible"] is True


def test_material_query_snapshot_imports_into_data_panel(
    monkeypatch: Any, tmp_path: Path
) -> None:
    query_store = QueryResultStore(tmp_path / "material-queries")
    query_store.save_query(
        "query-ui-cross",
        {"source": "materials_project"},
        (
            MaterialRecord(
                source="materials_project",
                material_id="mp-1",
                formula_pretty="Li2O",
                elements=("Li", "O"),
                band_gap_ev=2.0,
                density_g_cm3=2.1,
            ),
            MaterialRecord(
                source="materials_project",
                material_id="mp-2",
                formula_pretty="MgO",
                elements=("Mg", "O"),
                band_gap_ev=4.0,
                density_g_cm3=3.6,
            ),
        ),
    )
    store = ui.DatasetStore(tmp_path / "data-analysis")
    monkeypatch.setattr(
        ui,
        "ARTIFACTS",
        ArtifactRegistry(tmp_path / "artifacts", dataset_store=store),
    )
    monkeypatch.setattr(ui, "MATERIAL_QUERIES", query_store)

    outputs = ui._import_cross_agent_dataset(
        "materials_database", "query-ui-cross", ui._initial_state()
    )

    assert "materials_database" in outputs[0]
    assert outputs[1]["data_source_kind"] == "materials_database"
    assert outputs[1]["dataset_ids"][0].startswith("dataset-")
    assert outputs[5]["visible"] is True


def test_auto_ordinal_reference_restores_private_artifact_path(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.7\ntest")
    registry = ArtifactRegistry(tmp_path / "artifacts")
    artifact = registry.register_pdf(source)
    monkeypatch.setattr(ui, "ARTIFACTS", registry)
    state = ui._initial_state()
    state["artifact_ids"] = [artifact.artifact_id]
    state["artifact_names"] = [artifact.display_name]

    def fake_chat(
        message: str,
        history: list[dict[str, Any]],
        files: list[str] | None,
        runtime_state: dict[str, Any],
        *args: Any,
    ) -> Any:
        assert message == "深度分析第1篇"
        assert files is None
        assert len(runtime_state["paths"]) == 1
        assert Path(runtime_state["paths"][0]).is_file()
        yield (
            [
                *history,
                {"role": "user", "content": message},
                {"role": "assistant", "content": "深度分析完成"},
            ],
            "trace",
            runtime_state,
            "",
            None,
        )

    monkeypatch.setattr(ui.literature_ui, "_chat", fake_chat)
    updates = list(
        ui._dispatch(
            ui.MODE_AUTO,
            "深度分析第1篇",
            [],
            None,
            state,
            "",
            2020,
            10,
        )
    )
    assert updates[-1][0][-1]["content"] == "深度分析完成"
    assert updates[-1][2]["literature_state"]["paths"] == []


def test_cross_agent_request_uses_typed_coordinator(monkeypatch: Any) -> None:
    class Coordinator:
        def run(self, document_ids: list[str]) -> CrossAgentResult:
            assert document_ids == ["doc-1234567890abcdef12345678"]
            clue = MaterialClue(
                clue_id="clue-1234567890abcdef12345678",
                document_id=document_ids[0],
                paper_title="Paper",
                material_label="TiO2 scaffold",
                required_elements=("O", "Ti"),
                formula_candidates=("TiO2",),
                source_quote="TiO2 scaffold",
                evidence_page=2,
                evidence_chunk_id="chunk-1",
                review_status="approved",
                confidence=1.0,
            )
            return CrossAgentResult(
                status="completed",
                clues=(clue,),
                candidates=(
                    CrossAgentCandidate(
                        clue_id=clue.clue_id,
                        query_id="query-1",
                        material_id="mp-1",
                        formula_pretty="TiO2",
                        is_stable=True,
                    ),
                ),
                query_ids=("query-1",),
            )

    monkeypatch.setattr(ui, "_cross_agent_coordinator", Coordinator())
    state = ui._initial_state()
    state["literature_state"]["document_ids"] = ["doc-1234567890abcdef12345678"]
    updates = list(
        ui._dispatch(
            ui.MODE_AUTO,
            "根据这些论文的组成查询Materials Project候选",
            [],
            None,
            state,
            "",
            2020,
            10,
        )
    )
    assert "论文线索与材料数据库联合结果" in updates[-1][0][-1]["content"]
    assert "mp-1" in updates[-1][0][-1]["content"]


def test_third_agent_registers_without_changing_dispatcher(monkeypatch: Any) -> None:
    def handler(
        message: str,
        conversation: list[dict[str, Any]],
        files: list[Any] | None,
        current: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        del files, args, kwargs
        current["last_agent"] = "test_agent"
        yield (
            [
                *conversation,
                {"role": "user", "content": message},
                {"role": "assistant", "content": "测试 Agent 已响应"},
            ],
            "trace",
            current,
            "",
            None,
        )

    test_plugin = SubAgentUiPlugin(
        spec=SubAgentUiSpec(
            name="test_agent",
            display_name="测试 Agent",
            description="Test plugin.",
            renderer_name="system_message",
        ),
        handler=handler,
    )
    registry = SubAgentUiPluginRegistry((*ui.UI_PLUGINS.plugins(), test_plugin))
    monkeypatch.setattr(ui, "UI_PLUGINS", registry)
    updates = list(
        ui._dispatch("测试 Agent", "你好", [], None, ui._initial_state(), "", 2020, 10)
    )
    assert updates[-1][0][-1]["content"] == "测试 Agent 已响应"
    assert updates[-1][2]["last_agent"] == "test_agent"


def test_plugin_exception_is_recoverable(monkeypatch: Any) -> None:
    def failing_handler(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("private detail")
        yield

    plugin = SubAgentUiPlugin(
        spec=SubAgentUiSpec(
            name="failing_agent",
            display_name="失败 Agent",
            description="Failing plugin.",
            renderer_name="system_message",
        ),
        handler=failing_handler,
    )
    monkeypatch.setattr(ui, "UI_PLUGINS", SubAgentUiPluginRegistry((plugin,)))
    updates = list(
        ui._dispatch("失败 Agent", "执行", [], None, ui._initial_state(), "", 2020, 10)
    )
    assert "当前会话仍可继续" in updates[-1][0][-1]["content"]
    assert "private detail" not in str(updates[-1])


def test_expert_audit_never_exposes_internal_paths() -> None:
    state = ui._initial_state()
    state["artifact_ids"] = ["artifact-pdf-1"]
    state["literature_state"]["paths"] = ["D:/secret/paper.pdf"]
    rendered = ui._audit_markdown(ui.MODE_AUTO, state, True)
    assert "Artifact 数量：1" in rendered
    assert "D:/secret" not in rendered


def test_literature_provider_failure_does_not_blame_search_terms() -> None:
    answer, failed = ui._normalize_literature_response(
        "文献检索子代理调用失败，所有文献提供者都失败了，请修改关键词。"
    )
    assert failed is True
    assert "临时网络故障或接口限流" in answer
    assert "不需要修改当前关键词" in answer
    assert "请修改关键词" not in answer


def test_normal_literature_answer_is_not_rewritten() -> None:
    answer, failed = ui._normalize_literature_response("检索到 8 篇相关论文。")
    assert answer == "检索到 8 篇相关论文。"
    assert failed is False


def test_empty_literature_result_does_not_trigger_clarification_form() -> None:
    answer, failed = ui._normalize_literature_response(
        "文献检索工具返回了无结果。为了更有效地检索，我需要您提供更具体的搜索关键词。"
    )
    assert failed is False
    assert "当前主题已经足够明确" in answer
    assert "需要您提供" not in answer


def test_unified_examples_match_the_proven_standalone_uis() -> None:
    database_prompts = tuple(prompt for _label, prompt in ui.database_ui.QUICK_PROMPTS)
    literature_prompts = tuple(
        prompt for _label, prompt in ui.literature_ui.QUICK_PROMPTS
    )
    assert ui.UI_SPECS.get("materials_database").quick_prompts == database_prompts
    assert ui.UI_SPECS.get("literature").quick_prompts == literature_prompts
    assert len(ui.QUICK_PROMPT_LABELS) == 15
