"""Unified multi-agent UI with MasterAgent routing and manual fallbacks."""

from __future__ import annotations

import html
import os
import re
import socket
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager
from functools import partial
from math import ceil
from pathlib import Path
from typing import Any

import gradio as gr

from materials_screening import literature_ui_gradio as literature_ui
from materials_screening import materials_database_ui_gradio as database_ui
from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.exploration import (
    DataExplorationService,
    ExploratoryAnalysisBundle,
    render_exploration_markdown,
)
from materials_screening.data_analysis.interpretation import (
    ScientificInterpretationService,
    render_scientific_interpretation,
)
from materials_screening.data_analysis.models import DatasetTransformOperation
from materials_screening.data_analysis.reporting import DataAnalysisReportingService
from materials_screening.data_analysis.scientific import (
    ScientificAnalysisBrief,
    ScientificBriefService,
)
from materials_screening.data_analysis.service import DataAnalysisService
from materials_screening.data_analysis.statistics import DataStatisticsService
from materials_screening.data_analysis.transform import DatasetTransformService
from materials_screening.master import (
    ArtifactRegistry,
    DataAnalysisCrossAgentCoordinator,
    LiteratureDatabaseCoordinator,
    SubAgentUiPlugin,
    SubAgentUiPluginRegistry,
    SubAgentUiRegistry,
    SubAgentUiSpec,
    UnifiedResultEnvelope,
    default_renderer_registry,
    stream_master_request,
)
from materials_screening.master.fulltext_tasks import (
    requests_fulltext_resume,
    requests_fulltext_task,
)
from materials_screening.services.query_result_store import QueryResultStore
from materials_screening.sub_agents.literature.models import ExperimentMatrixDataTable
from materials_screening.sub_agents.literature.store import LiteratureQueryStore

MODE_AUTO = "自动判断"
MODE_DATABASE = "材料数据库"
MODE_LITERATURE = "文献与知识"
MODE_DATA_ANALYSIS = "数据分析"

DATA_GUIDED_QUESTIONS = (
    "建立科研分析方案",
    "这份数据有没有问题？",
    "先帮我总结一下",
    "不同组之间有差异吗？",
    "哪些字段有关联？",
    "画一张推荐图",
    "整理成报告",
)
DATA_GROUP_METRIC_PREFIX = "data-group-metric:"
DATA_SCIENCE_RESPONSE_PREFIX = "data-science-response:"
DATA_SCIENCE_DESIGN_PREFIX = "data-science-design:"
DATA_SCIENCE_GROUP_PREFIX = "data-science-group:"
DATA_SCIENCE_SUBJECT_PREFIX = "data-science-subject:"
DATA_SCIENCE_PREDICTOR_PREFIX = "data-science-predictor:"
DATA_SCIENCE_UNIT_PREFIX = "data-science-unit:"
DATA_SCIENCE_RUN_PREFIX = "data-science-run:"
DATA_SCIENCE_THRESHOLD_SKIP = "data-science-threshold:skip"
DATA_SCIENCE_CONFIRM = "data-science-confirm"
DATA_SCIENCE_EDIT_QUESTION = "data-science-edit:question"
DATA_SCIENCE_EDIT_RESPONSE = "data-science-edit:response"
DATA_SCIENCE_EDIT_DESIGN = "data-science-edit:design"
DATA_SCIENCE_EDIT_THRESHOLD = "data-science-edit:threshold"

UI_SPECS = SubAgentUiRegistry(
    (
        SubAgentUiSpec(
            name="materials_database",
            display_name=MODE_DATABASE,
            description="Materials Project 查询、比较、统计和导出。",
            renderer_name="materials_answer",
            quick_prompts=(
                "筛选含 Li、Fe、O，带隙大于 1 eV，凸包上能量不超过 0.05 "
                "eV/atom 的稳定材料，返回材料 ID、化学式、带隙和凸包上能量",
                "查询含 Si 和 O 的稳定材料，按 band_gap_ev 从高到低排序，返回前 10 个",
                "查询 mp-149 的材料详情，返回化学式、元素、带隙、密度、形成能、"
                "凸包上能量、稳定性、晶系和空间群",
                "比较 mp-149、mp-13 和 mp-2534 的带隙、密度、形成能和凸包上能量",
                "统计刚才查询结果中 band_gap_ev 和 density_g_cm3 的分布",
                "对刚才的查询结果执行 band_gap_ev 离群检测，列出离群材料、"
                "检测方法、阈值和样本数",
                "把刚才查询快照中的完整结构化材料结果导出为 CSV",
            ),
        ),
        SubAgentUiSpec(
            name="literature",
            display_name=MODE_LITERATURE,
            description="文献检索、PDF阅读、单篇分析和多篇综合。",
            accepted_artifact_types=("pdf_document",),
            renderer_name="literature_answer",
            quick_prompts=(
                "帮我检索3D打印生物活性玻璃支架孔结构与成骨的近年论文",
                "预览我上传的这些论文",
                "深度分析第1篇论文",
                "综合这些论文并生成主题报告",
            ),
        ),
        SubAgentUiSpec(
            name="data_analysis",
            display_name=MODE_DATA_ANALYSIS,
            description="CSV/JSON/XLSX 数据质量、统计检验、清洗、绘图和报告。",
            accepted_artifact_types=("tabular_dataset",),
            renderer_name="analysis_answer",
            quick_prompts=(
                "检查我上传的数据质量，不要修改数据",
                "对数值字段做描述统计和相关性分析",
                "按分组字段比较目标字段，并报告效应量",
                "生成散点图和 Markdown 分析报告",
            ),
        ),
    )
)

QUICK_PROMPT_LABELS = {
    UI_SPECS.get("materials_database").quick_prompts[0]: "条件筛选",
    UI_SPECS.get("materials_database").quick_prompts[1]: "排序 Top-K",
    UI_SPECS.get("materials_database").quick_prompts[2]: "材料详情",
    UI_SPECS.get("materials_database").quick_prompts[3]: "多材料比较",
    UI_SPECS.get("materials_database").quick_prompts[4]: "描述统计",
    UI_SPECS.get("materials_database").quick_prompts[5]: "离群检测",
    UI_SPECS.get("materials_database").quick_prompts[6]: "结果导出",
    UI_SPECS.get("literature").quick_prompts[0]: "主题检索",
    UI_SPECS.get("literature").quick_prompts[1]: "快速阅读",
    UI_SPECS.get("literature").quick_prompts[2]: "深度分析",
    UI_SPECS.get("literature").quick_prompts[3]: "多文献综合",
    UI_SPECS.get("data_analysis").quick_prompts[0]: "数据质量",
    UI_SPECS.get("data_analysis").quick_prompts[1]: "描述与相关",
    UI_SPECS.get("data_analysis").quick_prompts[2]: "分组检验",
    UI_SPECS.get("data_analysis").quick_prompts[3]: "图表报告",
}

RENDERERS = default_renderer_registry()


def _literature_artifact_root() -> Path:
    """Store uploaded PDFs inside the parser's configured allow-list."""

    raw = os.getenv("LITERATURE_INGEST_ROOTS", "").strip()
    if raw:
        for value in raw.split(os.pathsep):
            if value.strip():
                return Path(value.strip())
    return literature_ui.PDF_ROOT


DATASETS = DatasetStore(Path("data/data_analysis"))
DATA_INSPECTION = DataAnalysisService(DATASETS)
DATA_STATISTICS = DataStatisticsService(DATASETS)
DATA_REPORTING = DataAnalysisReportingService(DATASETS)
DATA_EXPLORATION = DataExplorationService(DATA_INSPECTION, DATA_STATISTICS)
ARTIFACTS = ArtifactRegistry(_literature_artifact_root(), dataset_store=DATASETS)
LITERATURE_QUERIES = LiteratureQueryStore(Path("data/literature_queries"))
MATERIAL_QUERIES = QueryResultStore(Path("data/material_queries"))

_master_runner: Any = None
_master_context: AbstractContextManager[Any] | None = None
_master_lock = threading.Lock()
_cross_agent_coordinator: LiteratureDatabaseCoordinator | None = None

CSS = """
footer { display: none !important; }
body, .gradio-container { background: #ffffff !important; }
.gradio-container { max-width: 100% !important; margin: 0 auto !important;
  width: 100vw !important; padding: 0 !important;
  font-family: "Segoe UI", "Microsoft YaHei", sans-serif !important; }
.gradio-container > .main, .gradio-container main.contain,
.gradio-container .wrap { width: 100% !important; max-width: none !important; }
#multi-header { height: 54px; display: flex; align-items: center;
  padding: 0 22px; border-bottom: 1px solid #ececec; }
#multi-header h1 { margin: 0; color: #202123; font-size: 16px; font-weight: 600; }
#multi-header p { display: none; }
#app-layout { width: 100% !important; max-width: 1800px !important;
  min-width: 1180px !important; flex-wrap: nowrap !important;
  margin: 0 auto !important; padding: 16px 22px 22px !important;
  gap: 16px !important; align-items: stretch !important; }
#left-sidebar { min-width: 300px !important; max-width: 330px !important;
  padding: 12px !important; background: #f7f7f8 !important;
  border-radius: 14px !important; height: calc(100vh - 92px) !important;
  max-height: calc(100vh - 92px) !important;
  overflow-x: hidden !important; overflow-y: auto !important;
  flex: 0 0 300px !important; }
#left-sidebar h3 { margin: 12px 8px 6px !important; color: #6b6c7b;
  font-size: 12px !important; font-weight: 500 !important; }
#right-sidebar { min-width: 300px !important; max-width: 340px !important;
  padding: 14px !important; background: #f7f7f8 !important;
  border-radius: 14px !important; height: calc(100vh - 92px) !important;
  max-height: calc(100vh - 92px) !important;
  overflow-x: hidden !important; overflow-y: auto !important;
  flex: 0 0 300px !important; }
#right-sidebar h3 { margin: 2px 6px 12px !important; color: #343541;
  font-size: 15px !important; }
#right-sidebar strong { display: block; margin: 12px 5px 5px;
  color: #6b6c7b; font-size: 12px; }
.init-status { margin: 4px 4px 8px !important; padding: 8px 10px !important;
  border-radius: 9px !important; background: #fff !important;
  color: #5f6170 !important; font-size: 12px !important; }
#chat-shell { width: auto !important; min-width: 0 !important;
  max-width: 1160px !important; flex: 1 1 auto !important;
  margin: 0 auto !important; padding: 0 8px 22px !important;
  background: transparent !important; border: 0 !important;
  box-shadow: none !important; }
#new-chat { width: 100% !important; border: 1px solid #dedee5 !important;
  box-shadow: none !important; color: #343541 !important; background: #fff !important;
  justify-content: flex-start !important; }
#new-chat:hover { background: #f4f4f4 !important; }
#multi-chat { min-height: 470px !important; border: 0 !important;
  background: transparent !important; box-shadow: none !important;
  overflow-x: hidden !important; }
.bubble-wrap .message-row.bubble.user-row { justify-content: flex-end; }
.bubble-wrap .message-row.bubble.bot-row { justify-content: flex-start; }
.bubble-wrap .user.message { background: #f4f4f4 !important; color: #202123 !important;
  border: none !important; border-radius: 18px !important;
  width: auto !important; max-width: 88% !important; padding: 10px 14px !important; }
.bubble-wrap .user.message * { color: #202123 !important; }
.bubble-wrap .bot.message { background: transparent !important;
  color: #202123 !important;
  border: 0 !important; border-radius: 0 !important;
  max-width: calc(100% - 12px) !important; padding: 14px 4px 22px !important;
  overflow-wrap: anywhere !important;
  box-shadow: none !important; }
.bubble-wrap .bot.message table { display: block !important; max-width: 100% !important;
  overflow-x: auto !important; }
#composer { border: 1px solid #d9d9e3 !important; border-radius: 24px !important;
  padding: 7px 8px 7px 15px !important; gap: 7px !important;
  background: #fff !important;
  box-shadow: 0 2px 12px rgba(0, 0, 0, .08) !important;
  align-items: center !important; }
#composer textarea { border: 0 !important; box-shadow: none !important;
  background: transparent !important; }
#pdf-upload { min-width: 112px !important; max-width: 132px !important;
  width: 120px !important; height: 40px !important; border-radius: 20px !important;
  padding: 0 10px !important; flex: 0 0 120px !important; }
#clear-upload { min-width: 62px !important; max-width: 72px !important;
  width: 68px !important; height: 40px !important; border-radius: 20px !important;
  padding: 0 8px !important; flex: 0 0 68px !important; }
.attachment-status { margin: 8px 8px 4px !important; padding: 7px 11px !important;
  border-radius: 10px !important; background: #f7f7f8 !important;
  color: #565869 !important; font-size: 12px !important; }
.attachment-status p { margin: 0 !important; }
#send-button, #stop-button { min-width: 44px !important; max-width: 64px !important;
  width: 44px !important; height: 40px !important; border-radius: 20px !important;
  font-size: 18px !important; padding: 0 !important; }
.quick-chip { border-radius: 999px !important; border: 1px solid #cfdaea !important;
  background: #fff !important; color: #565869 !important;
  font-size: 12px !important; text-align: left !important;
  white-space: normal !important; height: auto !important;
  min-height: 36px !important; }
button.primary { background: #202123 !important; border-color: #202123 !important; }
.advanced-box { margin-top: 6px !important; border: 0 !important;
  background: transparent !important; color: #6b6c7b !important; }
.trace-card { border: 1px solid #dce4ef; border-radius: 12px; padding: 12px 14px;
  background: #f8fafc; color: #465874; font-size: 13px; }
.trace-step { margin: 0 0 10px; }
.trace-step:last-child { margin-bottom: 0; }
.trace-step b { color: #17243a; }
@media (max-width: 1050px) { #multi-header { padding: 0 14px; } }
"""


def _initial_state() -> dict[str, Any]:
    return {
        "master_conversation_id": None,
        "database_conversation_id": None,
        "literature_state": literature_ui._initial_state(),
        "artifact_ids": [],
        "artifact_names": [],
        "dataset_ids": [],
        "data_exploration": None,
        "scientific_proposal": None,
        "scientific_brief": None,
        "scientific_setup": None,
        "analysis_ids": [],
        "plot_artifact_ids": [],
        "report_artifact_ids": [],
        "literature_query_id": None,
        "material_query_id": None,
        "literature_page": 0,
        "last_agent": None,
    }


def _trace(mode: str, status: str, detail: str = "") -> str:
    selected = html.escape(mode)
    safe_status = html.escape(status)
    safe_detail = html.escape(detail or "等待用户请求")
    return (
        '<div class="trace-card">'
        f'<div class="trace-step"><b>当前模式</b><br>{selected}</div>'
        f'<div class="trace-step"><b>运行状态</b><br>{safe_status}</div>'
        f'<div class="trace-step"><b>执行说明</b><br>{safe_detail}</div>'
        "</div>"
    )


def _close_master_runtime() -> None:
    global _cross_agent_coordinator, _master_context, _master_runner
    with _master_lock:
        context = _master_context
        _master_context = None
        _master_runner = None
        _cross_agent_coordinator = None
        if context is not None:
            context.__exit__(None, None, None)


def _initialize(provider: str, repository: str) -> tuple[str, str]:
    """Initialize both manual controllers and the real Master runner."""

    global _cross_agent_coordinator, _master_context, _master_runner
    from materials_screening.cli import _build_repository, _open_master_runner
    from materials_screening.services.material_database_service import (
        MaterialDatabaseService,
    )
    from materials_screening.services.query_result_store import QueryResultStore

    _close_master_runtime()
    context = _open_master_runner(
        run_root=Path("data/workflow_runs"),
        llm_provider=provider,  # type: ignore[arg-type]
        planner_fixture=None,
        materials_repository=repository,
        materials_fixture=None,
    )
    try:
        runner = context.__enter__()
        status, _database_trace, _conversation_id = database_ui._initialize(
            provider, repository
        )
    except Exception:
        context.__exit__(None, None, None)
        raise
    with _master_lock:
        _master_context = context
        _master_runner = runner
        _cross_agent_coordinator = LiteratureDatabaseCoordinator(
            MaterialDatabaseService(
                _build_repository(repository, None),
                QueryResultStore(Path("data/material_queries")),
            )
        )
    return status, _trace(MODE_AUTO, "已初始化", "MasterAgent 与三个手动模式均可使用")


def _initialize_safe(provider: str, repository: str) -> tuple[str, str]:
    try:
        return _initialize(provider, repository)
    except Exception as exc:
        return (
            "初始化失败，请检查密钥、网络和数据库状态后重试。",
            _trace(MODE_AUTO, "初始化失败", f"错误类型：{type(exc).__name__}"),
        )


def _render_answer(agent_name: str, result_type: str, answer: str) -> str:
    from materials_screening.master.user_answer import render_user_answer

    answer = render_user_answer(answer)
    envelope = UnifiedResultEnvelope(
        agent_name=agent_name,
        status="completed",
        result_type=result_type,
        result={"markdown": answer},
    )
    return RENDERERS.render(envelope)


def _normalize_literature_response(answer: str) -> tuple[str, bool]:
    """Replace misleading provider-failure advice with an actionable message."""

    provider_failure_markers = (
        "所有文献提供者都失败",
        "all literature providers failed",
        "文献检索子代理调用失败",
        "PROVIDER_UNAVAILABLE",
    )
    provider_failed = any(
        marker.casefold() in answer.casefold() for marker in provider_failure_markers
    )
    if provider_failed:
        return (
            "本次文献检索未完成：OpenAlex 与 Semantic Scholar 当前都没有返回"
            "可用结果，通常是临时网络故障或接口限流造成的，并不是检索主题不够"
            "明确。\n\n你不需要修改当前关键词。可以稍后直接重试；如果已经有相关"
            " PDF，也可以上传后继续预览、深度分析和综合。",
            True,
        )
    no_result_markers = (
        "文献检索工具返回了无结果",
        "需要您提供更具体的搜索关键词",
        "为了更有效地检索，我需要您提供",
    )
    if any(marker.casefold() in answer.casefold() for marker in no_result_markers):
        return (
            "本次自动扩展检索没有找到通过相关性筛选的候选文献。当前主题已经"
            "足够明确，不需要再补充年份、玻璃成分或打印工艺。\n\n系统会使用"
            "中英文同义词重新检索；你也可以稍后直接重试，或上传已有 PDF继续"
            "预览和分析。",
            False,
        )
    return answer, False


def _register_files(
    files: list[Any] | None, current: dict[str, Any]
) -> tuple[list[str], str | None]:
    if not files:
        return [], None
    try:
        artifacts = [ARTIFACTS.register_pdf(item) for item in files]
        paths = [str(ARTIFACTS.resolve_path(item.artifact_id)) for item in artifacts]
    except (AttributeError, KeyError, OSError, ValueError) as exc:
        return [], str(exc)
    current["artifact_ids"] = [item.artifact_id for item in artifacts]
    current["artifact_names"] = [item.display_name for item in artifacts]
    return paths, None


def _register_data_files(
    files: list[Any] | None, current: dict[str, Any]
) -> tuple[list[str], str | None]:
    if not files:
        return list(current.get("dataset_ids") or []), None
    try:
        artifacts = [ARTIFACTS.register_data_file(item) for item in files]
    except (AttributeError, KeyError, OSError, ValueError) as exc:
        return [], str(exc)
    current["artifact_ids"] = [item.artifact_id for item in artifacts]
    current["artifact_names"] = [item.display_name for item in artifacts]
    dataset_ids = [item.domain_id for item in artifacts]
    current["dataset_ids"] = dataset_ids
    return dataset_ids, None


def _literature_runtime_state(current: dict[str, Any]) -> dict[str, Any]:
    state = dict(current.get("literature_state") or literature_ui._initial_state())
    state["paths"] = [
        str(ARTIFACTS.resolve_path(item))
        for item in list(current.get("artifact_ids") or [])
        if ARTIFACTS.get(item).owner_agent == "literature"
    ]
    return state


def _safe_literature_state(state: dict[str, Any]) -> dict[str, Any]:
    safe = dict(state)
    safe["paths"] = []
    return safe


def _artifact_detail(current: dict[str, Any], detail: str) -> str:
    names = list(current.get("artifact_names") or [])
    if not names:
        return detail
    visible = "、".join(names[:3])
    if len(names) > 3:
        visible += f" 等 {len(names)} 个"
    return f"{detail}；附件（已就绪）：{visible}"


def _dispatch_literature(
    message: str,
    conversation: list[dict[str, Any]],
    files: list[Any] | None,
    current: dict[str, Any],
    material: str,
    year_from: float,
    limit: float,
    *,
    mode: str,
) -> Iterator[tuple[list[dict[str, Any]], str, dict[str, Any], str, None]]:
    prepared_files: list[str] | None = None
    if files:
        prepared_files, error = _register_files(files, current)
        if error:
            failed = list(conversation)
            failed.append({"role": "user", "content": message.strip()})
            failed.append({"role": "assistant", "content": error})
            yield failed, _trace(mode, "附件被拒绝", error), current, "", None
            return
    runtime_state = _literature_runtime_state(current)
    for history, _lit_trace, new_state, cleared, cleared_files in literature_ui._chat(
        message,
        conversation,
        prepared_files,
        runtime_state,
        material,
        year_from,
        limit,
    ):
        status = "运行中"
        if history and history[-1].get("role") == "assistant":
            content = str(history[-1].get("content", ""))
            content, provider_failed = _normalize_literature_response(content)
            history[-1]["content"] = _render_answer(
                "literature", "literature_answer", content
            )
            if "正在处理，请稍候" not in content:
                status = (
                    "执行失败"
                    if provider_failed or content.startswith("操作未完成")
                    else "已完成"
                )
        current.update(
            literature_state=_safe_literature_state(new_state),
            last_agent="literature",
        )
        yield (
            history,
            _trace(
                mode,
                status,
                _artifact_detail(current, "调用 LiteratureAgent"),
            ),
            current,
            cleared,
            cleared_files,
        )


def _dispatch_database(
    message: str,
    conversation: list[dict[str, Any]],
    files: list[Any] | None,
    current: dict[str, Any],
    material: str,
    year_from: float,
    limit: float,
    *,
    mode: str,
) -> Iterator[tuple[list[dict[str, Any]], str, dict[str, Any], str, None]]:
    del material, year_from, limit
    if files:
        rejected = list(conversation)
        rejected.append({"role": "user", "content": message.strip()})
        rejected.append(
            {
                "role": "assistant",
                "content": (
                    "材料数据库模式暂不接收附件，请切换到对应的文献或数据分析模式。"
                ),
            }
        )
        yield (
            rejected,
            _trace(mode, "附件不适用", "附件未传给数据库子Agent"),
            current,
            "",
            None,
        )
        return
    database_id = current.get("database_conversation_id")
    for history, _db_trace, cleared, new_database_id in database_ui._ask(
        message,
        conversation,
        database_id,
    ):
        if history and history[-1].get("role") == "assistant":
            content = str(history[-1].get("content", ""))
            history[-1]["content"] = _render_answer(
                "materials_database", "materials_answer", content
            )
        current.update(
            database_conversation_id=new_database_id,
            last_agent="materials_database",
        )
        yield (
            history,
            _trace(mode, "运行中", "调用 MaterialsDatabaseAgent"),
            current,
            cleared,
            None,
        )


def _dispatch_data_analysis(
    message: str,
    conversation: list[dict[str, Any]],
    files: list[Any] | None,
    current: dict[str, Any],
    material: str,
    year_from: float,
    limit: float,
    *,
    mode: str,
) -> Iterator[tuple[list[dict[str, Any]], str, dict[str, Any], str, None]]:
    del material, year_from, limit
    dataset_ids, error = _register_data_files(files, current)
    shown_message = message.strip() or "检查我上传的数据质量，不要修改数据。"
    shown = [*conversation, {"role": "user", "content": shown_message}]
    if error:
        yield (
            [*shown, {"role": "assistant", "content": error}],
            _trace(mode, "附件被拒绝", error),
            current,
            "",
            None,
        )
        return
    exploration: ExploratoryAnalysisBundle | None = None
    if files and dataset_ids:
        try:
            store = ARTIFACTS.dataset_store or DATASETS
            exploration = DataExplorationService(
                DataAnalysisService(store), DataStatisticsService(store)
            ).explore(dataset_ids[-1])
        except (KeyError, OSError, ValueError) as exc:
            current["data_eda_error"] = type(exc).__name__
        if exploration is not None:
            current["data_exploration"] = exploration.model_dump(mode="json")
            current["analysis_ids"] = list(exploration.analysis_ids)
            proposal = ScientificBriefService(store).propose_roles(
                dataset_ids[-1], domain="materials_science"
            )
            current["scientific_proposal"] = proposal.model_dump(mode="json")
    if exploration is not None and shown_message.rstrip("。") in {
        "检查我上传的数据质量，不要修改数据",
        "对上传的数据执行一键 EDA",
    }:
        current["last_agent"] = "data_analysis"
        yield (
            [
                *shown,
                {
                    "role": "assistant",
                    "content": _render_answer(
                        "data_analysis",
                        "analysis_answer",
                        render_exploration_markdown(exploration),
                    ),
                },
            ],
            _trace(mode, "已完成", "只读一键 EDA；未修改数据"),
            current,
            "",
            None,
        )
        return
    if exploration is not None and _looks_like_scientific_question(shown_message):
        _begin_scientific_setup(current)
        draft = current["scientific_setup"]
        draft["research_question"] = shown_message
        draft["step"] = "response"
        current["last_agent"] = "data_analysis"
        dataset = exploration.inspection.dataset
        yield (
            [
                *shown,
                {
                    "role": "assistant",
                    "content": (
                        f"数据已读取：{dataset.row_count} 行 × "
                        f"{dataset.column_count} 列。我已把你的话记录为研究问题：\n\n"
                        f"**{shown_message}**\n\n"
                        "这个问题主要关注哪个测量指标？请选择一个主要指标。"
                    ),
                    "options": _response_options(current),
                },
            ],
            _trace(mode, "研究设定中", "已识别科研问题，等待确认主要指标"),
            current,
            "",
            None,
        )
        return
    runner = _master_runner
    if runner is None:
        answer = _render_answer(
            "master", "system_message", "请先点击左侧“初始化”，再发送请求。"
        )
        yield (
            [*shown, {"role": "assistant", "content": answer}],
            _trace(mode, "尚未初始化"),
            current,
            "",
            None,
        )
        return
    internal_prompt = f"数据集分析任务：{shown_message}"
    if dataset_ids and not re.search(r"\bdataset-[A-Za-z0-9._:-]+", shown_message):
        internal_prompt += f"\n已登记数据集：{dataset_ids[-1]}"
    current["last_agent"] = "data_analysis"
    for update in stream_master_request(
        runner,
        message=internal_prompt,
        conversation_id=current.get("master_conversation_id"),
    ):
        if update.conversation_id:
            current["master_conversation_id"] = update.conversation_id
        if update.delegated_agent:
            current["last_agent"] = update.delegated_agent
        visible_history = list(shown)
        if update.is_final and update.result is not None:
            visible_history.append(
                {
                    "role": "assistant",
                    "content": _render_answer(
                        "data_analysis",
                        "analysis_answer",
                        update.result.response_text,
                    ),
                }
            )
        yield (
            visible_history,
            _trace(mode, update.status, f"DataAnalysisAgent：{update.detail}"),
            current,
            "",
            None,
        )


def _looks_like_scientific_question(message: str) -> bool:
    normalized = message.strip().rstrip("。？！?!")
    return len(normalized) >= 5 and any(
        marker in normalized
        for marker in (
            "是否",
            "会不会",
            "有没有差异",
            "有何差异",
            "影响",
            "相关",
            "关系",
            "预测",
            "比较",
        )
    )


UI_PLUGINS = SubAgentUiPluginRegistry(
    (
        SubAgentUiPlugin(
            spec=UI_SPECS.get("materials_database"), handler=_dispatch_database
        ),
        SubAgentUiPlugin(spec=UI_SPECS.get("literature"), handler=_dispatch_literature),
        SubAgentUiPlugin(
            spec=UI_SPECS.get("data_analysis"), handler=_dispatch_data_analysis
        ),
    )
)


def _is_cross_agent_request(message: str) -> bool:
    return bool(
        re.search(r"论文|文献", message, re.IGNORECASE)
        and re.search(r"Materials Project|材料数据库|数据库", message, re.IGNORECASE)
        and re.search(r"组成|元素|候选|查询|筛选", message, re.IGNORECASE)
    )


def _dispatch_cross_agent(
    message: str,
    conversation: list[dict[str, Any]],
    current: dict[str, Any],
) -> Iterator[tuple[list[dict[str, Any]], str, dict[str, Any], str, None]]:
    shown = [*conversation, {"role": "user", "content": message}]
    document_ids = list(
        (current.get("literature_state") or {}).get("document_ids") or []
    )
    coordinator = _cross_agent_coordinator
    if not document_ids:
        answer = "当前没有已分析的论文。请先上传并预览或深度分析PDF。"
        yield (
            [*shown, {"role": "assistant", "content": answer}],
            _trace(MODE_AUTO, "缺少论文上下文", "未执行数据库查询"),
            current,
            "",
            None,
        )
        return
    if coordinator is None:
        answer = "请先点击左侧“初始化”，再执行跨智能体查询。"
        yield (
            [*shown, {"role": "assistant", "content": answer}],
            _trace(MODE_AUTO, "尚未初始化", "未执行数据库查询"),
            current,
            "",
            None,
        )
        return
    yield (
        [
            *shown,
            {"role": "assistant", "content": "正在提取论文材料线索并查询数据库……"},
        ],
        _trace(MODE_AUTO, "跨智能体执行中", "LiteratureAgent → MaterialsDatabaseAgent"),
        current,
        "",
        None,
    )
    result = coordinator.run(document_ids)
    envelope = UnifiedResultEnvelope(
        agent_name="master",
        status="completed" if result.status == "completed" else "partial",
        result_type="cross_agent_answer",
        result=result.model_dump(mode="json"),
        evidence_refs=tuple(clue.evidence_chunk_id for clue in result.clues),
        warnings=result.warnings,
    )
    answer = RENDERERS.render(envelope)
    current["last_agent"] = "master"
    yield (
        [*shown, {"role": "assistant", "content": answer}],
        _trace(
            MODE_AUTO,
            "已完成" if result.status == "completed" else "部分完成",
            f"论文线索 {len(result.clues)} 条；数据库候选 {len(result.candidates)} 条",
        ),
        current,
        "",
        None,
    )


def _attachment_type(files: list[Any]) -> str:
    suffixes = {
        Path(
            str(item if isinstance(item, (str, Path)) else getattr(item, "name", ""))
        ).suffix.casefold()
        for item in files
    }
    if suffixes and suffixes <= {".pdf"}:
        return "pdf"
    if suffixes and suffixes <= {".csv", ".json", ".xlsx"}:
        return "data"
    return "unsupported"


def _routes_to_fulltext_master(
    message: str, *, has_pdf: bool, current: dict[str, Any]
) -> bool:
    """Use server task state; browser artifact history is not continuation authority."""
    if requests_fulltext_task(message):
        return True
    if not has_pdf and not requests_fulltext_resume(message):
        return False
    reader = getattr(_master_runner, "get_fulltext_tasks", None)
    conversation_id = current.get("master_conversation_id")
    if not callable(reader) or not conversation_id:
        return False
    return any(task.stage != "finished" for task in reader(conversation_id))


def _dispatch_master(
    prompt: str,
    conversation: list[dict[str, Any]],
    current: dict[str, Any],
    *,
    files: list[Any] | None = None,
) -> Iterator[tuple[list[dict[str, Any]], str, dict[str, Any], str, None]]:
    """One Master presentation path, with opaque attachment references only."""
    conversation.append({"role": "user", "content": prompt})
    runner = _master_runner
    if runner is None:
        conversation.append(
            {
                "role": "assistant",
                "content": "请先点击左侧“初始化”，再发送请求。",
            }
        )
        yield conversation, _trace(MODE_AUTO, "尚未初始化"), current, "", None
        return
    references = ()
    if files:
        if len(files) > 100:
            raise ValueError("一次最多绑定100篇全文。")
        conversation_id = current.get("master_conversation_id")
        if not conversation_id:
            conversation_id = runner.start_conversation()
            current["master_conversation_id"] = conversation_id
        artifacts = [
            runner.register_pdf_attachment(item, conversation_id=conversation_id)
            for item in files
        ]
        references = tuple(dict.fromkeys(item.artifact_id for item in artifacts))
        current["artifact_ids"] = list(references)
        current["artifact_names"] = list(
            dict.fromkeys(item.display_name for item in artifacts)
        )
    for update in stream_master_request(
        runner,
        message=prompt,
        conversation_id=current.get("master_conversation_id"),
        artifact_refs=references,
    ):
        if update.conversation_id:
            current["master_conversation_id"] = update.conversation_id
        if update.delegated_agent:
            current["last_agent"] = update.delegated_agent
        visible_history = list(conversation)
        if update.is_final and update.result is not None:
            agent_name = update.delegated_agent or "master"
            result_type = {
                "materials_database": "materials_answer",
                "literature": "literature_answer",
                "data_analysis": "analysis_answer",
            }.get(agent_name, "system_message")
            response_text = update.result.response_text
            if agent_name == "literature" and not references:
                response_text, _provider_failed = _normalize_literature_response(
                    response_text
                )
            visible_history.append(
                {
                    "role": "assistant",
                    "content": _render_answer(agent_name, result_type, response_text),
                }
            )
        delegated = (
            UI_SPECS.get(update.delegated_agent).display_name
            if update.delegated_agent
            else "MasterAgent 正在判断"
        )
        yield (
            visible_history,
            _trace(MODE_AUTO, update.status, f"{delegated}：{update.detail}"),
            current,
            "",
            None,
        )


def _dispatch_core(
    mode: str,
    message: str,
    history: list[dict[str, Any]] | None,
    files: list[Any] | None,
    state: dict[str, Any] | None,
    material: str,
    year_from: float,
    limit: float,
) -> Iterator[tuple[list[dict[str, Any]], str, dict[str, Any], str, None]]:
    current = dict(state or _initial_state())
    conversation = list(history or [])
    if (
        not files
        and message.strip()
        and isinstance(current.get("scientific_setup"), dict)
    ):
        continued = _continue_scientific_setup(message.strip(), conversation, current)
        if continued is not None:
            updated_history, current = continued
            current["last_agent"] = "data_analysis"
            yield (
                updated_history,
                _trace(mode, "研究设定中", "逐步确认研究问题与变量角色"),
                current,
                "",
                None,
            )
            return
    if mode == MODE_AUTO:
        attachment_type = _attachment_type(files) if files else None
        if attachment_type in {None, "pdf"} and _routes_to_fulltext_master(
            message.strip(), has_pdf=attachment_type == "pdf", current=current
        ):
            yield from _dispatch_master(
                message.strip() or "继续", conversation, current, files=files
            )
            return
        default_prompt = (
            "请预览我上传的论文。"
            if attachment_type == "pdf"
            else "检查我上传的数据质量，不要修改数据。"
            if attachment_type == "data"
            else ""
        )
        prompt = message.strip() or default_prompt
        if not prompt:
            yield conversation, _trace(mode, "等待输入"), current, "", None
            return
        if files:
            if attachment_type == "pdf":
                yield from _dispatch_literature(
                    prompt,
                    conversation,
                    files,
                    current,
                    material,
                    year_from,
                    limit,
                    mode=mode,
                )
            elif attachment_type == "data":
                yield from _dispatch_data_analysis(
                    prompt,
                    conversation,
                    files,
                    current,
                    material,
                    year_from,
                    limit,
                    mode=mode,
                )
            else:
                answer = (
                    "附件类型不受支持，且不能混合上传。请仅选择 PDF，"
                    "或仅选择 CSV/JSON/XLSX。"
                )
                yield (
                    [
                        *conversation,
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ],
                    _trace(mode, "附件被拒绝", answer),
                    current,
                    "",
                    None,
                )
            return
        if _is_cross_agent_request(prompt):
            yield from _dispatch_cross_agent(prompt, conversation, current)
            return
        artifact_owner = ARTIFACTS.resolve_owner_reference(
            list(current.get("artifact_ids") or []), prompt
        )
        if artifact_owner == "literature":
            yield from _dispatch_literature(
                prompt,
                conversation,
                None,
                current,
                material,
                year_from,
                limit,
                mode=mode,
            )
            return
        if artifact_owner == "data_analysis":
            yield from _dispatch_data_analysis(
                prompt,
                conversation,
                None,
                current,
                material,
                year_from,
                limit,
                mode=mode,
            )
            return
        yield from _dispatch_master(prompt, conversation, current)
        return

    try:
        plugin = UI_PLUGINS.for_mode(mode)
    except KeyError:
        yield (
            [
                *conversation,
                {"role": "user", "content": message.strip()},
                {"role": "assistant", "content": "所选子 Agent 当前不可用。"},
            ],
            _trace(mode, "插件不可用"),
            current,
            "",
            None,
        )
        return
    yield from plugin.handler(
        message,
        conversation,
        files,
        current,
        material,
        year_from,
        limit,
        mode=mode,
    )


def _dispatch(
    mode: str,
    message: str,
    history: list[dict[str, Any]] | None,
    files: list[Any] | None,
    state: dict[str, Any] | None,
    material: str,
    year_from: float,
    limit: float,
) -> Iterator[tuple[list[dict[str, Any]], str, dict[str, Any], str, None]]:
    """Fail closed so one plugin error does not break the whole page session."""

    try:
        yield from _dispatch_core(
            mode,
            message,
            history,
            files,
            state,
            material,
            year_from,
            limit,
        )
    except Exception as exc:
        current = dict(state or _initial_state())
        conversation = list(history or [])
        if message.strip():
            conversation.append({"role": "user", "content": message.strip()})
        conversation.append(
            {
                "role": "assistant",
                "content": "请求未完成，当前会话仍可继续。请重试或切换手动模式。",
            }
        )
        yield (
            conversation,
            _trace(mode, "执行失败", f"错误类型：{type(exc).__name__}"),
            current,
            "",
            None,
        )


def _audit_markdown(mode: str, state: dict[str, Any], enabled: bool) -> str:
    if not enabled:
        return ""
    literature_state = state.get("literature_state") or {}
    conversation_id = str(state.get("master_conversation_id") or "尚未创建")
    return (
        "### 专家审计\n\n"
        f"- 当前模式：`{mode}`\n"
        f"- 最近执行者：`{state.get('last_agent') or 'none'}`\n"
        f"- Master 会话：`{conversation_id}`\n"
        f"- Artifact 数量：{len(state.get('artifact_ids') or [])}\n"
        f"- Dataset 数量：{len(state.get('dataset_ids') or [])}\n"
        f"- 数据来源：`{state.get('data_source_kind') or 'user_upload'}`\n"
        f"- Analysis 数量：{len(state.get('analysis_ids') or [])}\n"
        f"- Plot 数量：{len(state.get('plot_artifact_ids') or [])}\n"
        f"- 文档数量：{len(literature_state.get('document_ids') or [])}\n"
        "- 内部文件路径：不向前端提供"
    )


def _attachment_markdown(
    files: list[Any] | None,
    state: dict[str, Any] | None,
) -> str:
    """Show selected or registered attachments without exposing private paths."""

    names: list[str] = []
    if files:
        for item in files:
            raw = item if isinstance(item, (str, Path)) else getattr(item, "name", "")
            if raw:
                names.append(Path(str(raw)).name)
    if not names:
        names = [str(name) for name in (state or {}).get("artifact_names", [])]
    if not names:
        return "📎 尚未上传附件"
    visible = "、".join(names[:4])
    unit = (
        "篇" if all(Path(name).suffix.casefold() == ".pdf" for name in names) else "个"
    )
    if len(names) > 4:
        visible += f" 等 {len(names)} {unit}"
    return f"📎 已选择 {len(names)} {unit}：{visible}"


def _capture_literature_query(
    conversation: list[dict[str, Any]], current: dict[str, Any]
) -> None:
    for message in reversed(conversation):
        if message.get("role") != "assistant":
            continue
        match = re.search(r"\blit-[A-Za-z0-9]+\b", str(message.get("content", "")))
        if match and LITERATURE_QUERIES.exists(match.group(0)):
            if current.get("literature_query_id") != match.group(0):
                current["literature_page"] = 0
            current["literature_query_id"] = match.group(0)
            return


def _capture_material_query(
    conversation: list[dict[str, Any]], current: dict[str, Any]
) -> None:
    for message in reversed(conversation):
        if message.get("role") != "assistant":
            continue
        match = re.search(r"\bquery-[A-Za-z0-9-]+\b", str(message.get("content", "")))
        if match:
            try:
                MATERIAL_QUERIES.load_records(match.group(0))
            except ValueError:
                continue
            current["material_query_id"] = match.group(0)
            return


def _literature_page(
    state: dict[str, Any] | None, delta: int = 0
) -> tuple[list[list[Any]], str, str, dict[str, Any]]:
    """Render five saved papers without another provider request."""

    current = dict(state or _initial_state())
    query_id = current.get("literature_query_id")
    if not isinstance(query_id, str) or not LITERATURE_QUERIES.exists(query_id):
        return [], "检索完成后，可在这里分页查看论文详情。", "第 0 / 0 页", current
    result = LITERATURE_QUERIES.load(query_id)
    page_size = 5
    page_count = max(1, ceil(len(result.papers) / page_size))
    page = min(max(int(current.get("literature_page", 0)) + delta, 0), page_count - 1)
    current["literature_page"] = page
    selected = result.papers[page * page_size : (page + 1) * page_size]
    relevance = {
        "core": "核心相关",
        "high": "高度相关",
        "extended": "扩展阅读",
        None: "未分级",
    }
    rows: list[list[Any]] = []
    details: list[str] = []
    for index, paper in enumerate(selected, page * page_size + 1):
        rows.append(
            [
                index,
                relevance.get(paper.relevance_level, "未分级"),
                paper.year or "—",
                paper.title,
                paper.doi or "—",
            ]
        )
        details.extend(("", f"#### {index}. {paper.title}"))
        if paper.authors:
            details.append(f"- 作者：{'、'.join(paper.authors)}")
        details.append(f"- 期刊/会议：{paper.venue or '未知'}")
        details.append(f"- DOI：{paper.doi or '无 DOI'}")
        citation_count = (
            paper.cited_by_count if paper.cited_by_count is not None else "未知"
        )
        details.append(f"- 引用数：{citation_count}")
        if paper.landing_page_url:
            details.append(f"- 文献入口：{paper.landing_page_url}")
        details.append(f"- 入选原因：{paper.selection_reason or '主题相关'}")
        details.append(f"- 摘要：{paper.abstract or '数据源未提供'}")
    label = f"第 {page + 1} / {page_count} 页 · 共 {len(result.papers)} 篇"
    return rows, "\n".join(details).strip(), label, current


def _turn_literature_page(
    state: dict[str, Any] | None, delta: int
) -> tuple[list[list[Any]], str, str, dict[str, Any]]:
    return _literature_page(state, delta)


def _clear_attachments(
    state: dict[str, Any] | None,
) -> tuple[None, dict[str, Any], str]:
    """Detach selected files without deleting source files or analysis data."""

    current = dict(state or _initial_state())
    current["artifact_ids"] = []
    current["artifact_names"] = []
    current["dataset_ids"] = []
    current["data_exploration"] = None
    return None, current, "📎 尚未上传附件"


def _data_panel_payload(state: dict[str, Any]) -> tuple[Any, ...]:
    raw = state.get("data_exploration")
    if not isinstance(raw, dict):
        empty = gr.update(choices=[], value=None)
        return (
            [],
            None,
            "上传数据后将在这里显示自动 EDA。",
            gr.update(visible=False),
            empty,
            empty,
            empty,
            empty,
            empty,
        )
    bundle = ExploratoryAnalysisBundle.model_validate(raw)
    inspection = bundle.inspection
    rows = [
        [
            column.name,
            column.inferred_type,
            column.non_null_count,
            column.missing_count,
            column.unique_count,
            "；".join(column.warnings) or "—",
        ]
        for column in inspection.columns
    ]
    numeric = [
        column.name
        for column in inspection.columns
        if column.inferred_type == "numeric" and column.unique_count > 1
    ]
    groupable = [
        column.name for column in inspection.columns if 1 < column.unique_count <= 20
    ]
    all_columns = [column.name for column in inspection.columns]
    dataset = inspection.dataset
    summary = (
        f"当前数据集：`{dataset.dataset_id}` · {dataset.display_name} · "
        f"{dataset.row_count} 行 × {dataset.column_count} 列 · "
        f"质量提示 {len(inspection.issues)} 项"
    )
    first_numeric = numeric[0] if numeric else None
    second_numeric = numeric[1] if len(numeric) > 1 else first_numeric
    first_group = groupable[0] if groupable else None
    return (
        rows,
        list(inspection.preview_rows),
        summary,
        gr.update(visible=True),
        gr.update(choices=numeric, value=numeric[: min(3, len(numeric))]),
        gr.update(choices=numeric, value=first_numeric),
        gr.update(choices=numeric, value=second_numeric),
        gr.update(choices=groupable, value=first_group),
        gr.update(choices=all_columns, value=all_columns[0] if all_columns else None),
    )


def _active_data_store() -> DatasetStore:
    store = ARTIFACTS.dataset_store
    return store if isinstance(store, DatasetStore) else DATASETS


def _data_group_choices(state: dict[str, Any] | None, group_column: str | None) -> Any:
    current = dict(state or _initial_state())
    dataset_ids = list(current.get("dataset_ids") or [])
    if not dataset_ids or not group_column:
        return gr.update(choices=[], value=[])
    frame = _active_data_store().load_dataframe(dataset_ids[-1])
    if group_column not in frame.columns:
        return gr.update(choices=[], value=[])
    values = [
        value.item() if hasattr(value, "item") else value
        for value in frame[group_column].dropna().unique().tolist()[:20]
    ]
    return gr.update(choices=values, value=values[:2])


def _run_guided_data_action(
    question: str | None,
    state: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], Any, Any]:
    """Turn one plain-language choice into a deterministic analysis action."""
    current = dict(state or _initial_state())
    raw = current.get("data_exploration")
    if not isinstance(raw, dict):
        return "请先上传一个数据集，我会先了解它的结构。", current, None, None
    if not question:
        return "请先选择一个你最想知道的问题。", current, None, None

    numeric, group_by, groups = _guided_data_defaults(current)
    if not numeric and not group_by:
        return "这份数据没有可用于当前分析的字段。", current, None, None

    action = "eda"
    columns = numeric[: min(3, len(numeric))]
    x = numeric[0] if numeric else None
    y = numeric[1] if len(numeric) > 1 else None
    plot_type = "scatter" if y else "histogram"
    raw_brief = current.get("scientific_brief")
    scientific_brief = (
        ScientificAnalysisBrief.model_validate(raw_brief)
        if isinstance(raw_brief, dict)
        else None
    )
    if question == "先帮我总结一下":
        action = "describe"
    elif question == "不同组之间有差异吗？":
        if not group_by or len(groups) < 2 or not x:
            return (
                "这份数据没有同时满足“分组字段 + 数值字段”的条件。"
                "你可以选择总结、字段关系或高级设置。",
                current,
                None,
                None,
            )
        action = "test"
    elif question == "哪些字段有关联？":
        if len(numeric) < 2:
            return "至少需要两个可变化的数值字段才能分析关联。", current, None, None
        action = "correlation"
    elif question == "画一张推荐图":
        if not x:
            return "当前没有可绘图的数值字段。", current, None, None
        action = "plot"
        if scientific_brief is not None:
            response = scientific_brief.response_variables[0]
            if scientific_brief.design == "continuous_relationship":
                x, y = scientific_brief.covariates[0], response
                group_by, plot_type = None, "scatter"
            elif scientific_brief.design in {
                "independent_groups",
                "paired",
                "repeated_measures",
            }:
                x, y = response, response
                group_by, plot_type = scientific_brief.group_variable, "boxplot"
            else:
                x, y, group_by, plot_type = response, None, None, "histogram"
    elif question == "整理成报告":
        action = "report"
    elif question != "这份数据有没有问题？":
        return "我还不认识这个问题选项，请重新选择。", current, None, None

    result, updated, image, download = _run_data_action(
        action,
        current,
        columns,
        x,
        y,
        group_by,
        groups[:2],
        "welch_t",
        plot_type,
        "pearson",
        "CSV",
    )
    return f"你问：**{question}**\n\n{result}", updated, image, download


def _guided_data_defaults(
    state: dict[str, Any],
) -> tuple[list[str], str | None, list[Any]]:
    raw = state.get("data_exploration")
    if not isinstance(raw, dict):
        return [], None, []
    inspection = ExploratoryAnalysisBundle.model_validate(raw).inspection
    numeric = [
        column.name
        for column in inspection.columns
        if column.inferred_type == "numeric" and column.unique_count > 1
    ]
    row_count = max(inspection.dataset.row_count, 1)
    group_candidates = sorted(
        (
            column
            for column in inspection.columns
            if 1 < column.unique_count <= 20 and column.unique_count / row_count <= 0.5
        ),
        key=lambda column: column.unique_count,
    )
    group_by = group_candidates[0].name if group_candidates else None
    groups: list[Any] = []
    if group_by:
        frame = _active_data_store().load_dataframe(inspection.dataset.dataset_id)
        groups = [
            value.item() if hasattr(value, "item") else value
            for value in frame[group_by].dropna().unique().tolist()[:8]
        ]
    return numeric, group_by, groups


def _group_metric_options(state: dict[str, Any]) -> list[dict[str, str]]:
    numeric, _group_by, _groups = _guided_data_defaults(state)
    return [
        {"label": column, "value": f"{DATA_GROUP_METRIC_PREFIX}{column}"}
        for column in numeric[:12]
    ]


def _run_group_metric_action(
    column: str, state: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    numeric, group_by, groups = _guided_data_defaults(state)
    if column not in numeric or not group_by or len(groups) < 2:
        return "无法用当前字段完成组间比较，请打开高级设置检查字段。", state
    raw_brief = state.get("scientific_brief")
    if not isinstance(raw_brief, dict):
        brief = ScientificAnalysisBrief(
            dataset_id=str(state.get("dataset_ids", [""])[-1]),
            domain="materials_science",
            research_question=f"不同 {group_by} 组的 {column} 是否存在差异？",
            observation_unit="数据表中的每一行独立观测",
            design="independent_groups",
            response_variables=(column,),
            group_variable=group_by,
            hypothesis=f"至少一个 {group_by} 组的 {column} 不同",
            roles_confirmed=True,
        )
        state["scientific_brief"] = brief.model_dump(mode="json")
    method = "welch_t" if len(groups) == 2 else "anova"
    result, current, _image, _download = _run_data_action(
        "test",
        state,
        [column],
        column,
        None,
        group_by,
        groups,
        method,
        "boxplot",
        "pearson",
        "CSV",
    )
    return result, current


def _science_option(label: str, value: str) -> dict[str, str]:
    return {"label": label, "value": value}


def _scientific_proposal(state: dict[str, Any]) -> dict[str, Any]:
    raw = state.get("scientific_proposal")
    return raw if isinstance(raw, dict) else {}


def _begin_scientific_setup(
    state: dict[str, Any], *, intent: str = "scientific_analysis"
) -> tuple[str, list[dict[str, str]]]:
    state["scientific_brief"] = None
    state["scientific_setup"] = {
        "active": True,
        "step": "research_question",
        "intent": intent,
        "research_question": None,
        "response_variable": None,
        "design": None,
        "group_variable": None,
        "subject_id_variable": None,
        "predictor_variable": None,
        "unit": None,
        "practical_threshold": None,
    }
    return (
        "先用一句话描述你真正想回答的科研问题。\n\n"
        "例如：**不同制备方法是否会改变材料硬度？**\n\n"
        "请直接在对话框里输入，不需要使用专业统计术语。",
        [],
    )


def _response_options(state: dict[str, Any]) -> list[dict[str, str]]:
    candidates = _scientific_proposal(state).get("response_candidates") or []
    return [
        _science_option(str(column), f"{DATA_SCIENCE_RESPONSE_PREFIX}{column}")
        for column in candidates[:12]
    ]


def _design_options() -> list[dict[str, str]]:
    return [
        _science_option(
            "各组是不同样本",
            f"{DATA_SCIENCE_DESIGN_PREFIX}independent_groups",
        ),
        _science_option("同一样本前后配对", f"{DATA_SCIENCE_DESIGN_PREFIX}paired"),
        _science_option(
            "同一样本多次测量",
            f"{DATA_SCIENCE_DESIGN_PREFIX}repeated_measures",
        ),
        _science_option(
            "先做探索，不检验组差异",
            f"{DATA_SCIENCE_DESIGN_PREFIX}exploratory",
        ),
        _science_option(
            "分析连续因素与指标的关系",
            f"{DATA_SCIENCE_DESIGN_PREFIX}continuous_relationship",
        ),
    ]


def _group_options(state: dict[str, Any]) -> list[dict[str, str]]:
    candidates = _scientific_proposal(state).get("group_candidates") or []
    return [
        _science_option(str(column), f"{DATA_SCIENCE_GROUP_PREFIX}{column}")
        for column in candidates[:12]
    ]


def _subject_options(state: dict[str, Any]) -> list[dict[str, str]]:
    candidates = _scientific_proposal(state).get("identifier_candidates") or []
    return [
        _science_option(str(column), f"{DATA_SCIENCE_SUBJECT_PREFIX}{column}")
        for column in candidates[:12]
    ]


def _predictor_options(state: dict[str, Any]) -> list[dict[str, str]]:
    draft = state.get("scientific_setup") or {}
    response = draft.get("response_variable")
    identifiers = set(_scientific_proposal(state).get("identifier_candidates") or [])
    candidates = _scientific_proposal(state).get("covariate_candidates") or []
    return [
        _science_option(str(column), f"{DATA_SCIENCE_PREDICTOR_PREFIX}{column}")
        for column in candidates
        if column != response and column not in identifiers
    ][:12]


def _infer_scientific_unit(column: str) -> str | None:
    lowered = column.casefold()
    suffix_units = (
        ("_hv", "HV"),
        ("_mpa", "MPa"),
        ("_gpa", "GPa"),
        ("_c", "°C"),
        ("_k", "K"),
        ("_percent", "%"),
        ("_pct", "%"),
    )
    return next(
        (unit for suffix, unit in suffix_units if lowered.endswith(suffix)), None
    )


def _unit_options(column: str) -> list[dict[str, str]]:
    inferred = _infer_scientific_unit(column)
    options = []
    if inferred:
        options.append(
            _science_option(
                f"单位是 {inferred}", f"{DATA_SCIENCE_UNIT_PREFIX}{inferred}"
            )
        )
    options.append(_science_option("数据未提供单位", f"{DATA_SCIENCE_UNIT_PREFIX}"))
    return options


def _threshold_prompt() -> tuple[str, list[dict[str, str]]]:
    return (
        "多大的变化才具有实际科研意义？如果已有判断标准，请直接输入一个非负数值；"
        "如果暂时没有，可选择“暂不设置”。",
        [_science_option("暂不设置实际意义阈值", DATA_SCIENCE_THRESHOLD_SKIP)],
    )


def _design_label(value: str | None) -> str:
    return {
        "independent_groups": "不同样本的独立组比较",
        "paired": "同一样本的前后配对比较",
        "repeated_measures": "同一样本的重复测量",
        "exploratory": "探索性分析",
        "continuous_relationship": "连续因素与指标的关系分析",
    }.get(value or "", "尚未选择")


def _research_settings_card(state: dict[str, Any], *, confirmed: bool = False) -> str:
    draft = state.get("scientific_setup") or {}
    response = str(draft.get("response_variable") or "尚未选择")
    threshold = draft.get("practical_threshold")
    unit = str(draft.get("unit") or "未提供")
    threshold_text = (
        "暂未设置" if threshold is None else f"{threshold:g} {unit}".strip()
    )
    role_line = (
        f"- 分组/处理字段：`{draft.get('group_variable')}`\n"
        if draft.get("group_variable")
        else ""
    )
    subject_line = (
        f"- 样本或受试对象 ID：`{draft.get('subject_id_variable')}`\n"
        if draft.get("subject_id_variable")
        else ""
    )
    predictor_line = (
        f"- 连续预测因素：`{draft.get('predictor_variable')}`\n"
        if draft.get("predictor_variable")
        else ""
    )
    status = (
        "已确认，可以据此选择统计方法" if confirmed else "待你确认，尚未运行推断检验"
    )
    return (
        "### 当前研究设定\n\n"
        f"- 研究问题：{draft.get('research_question') or '尚未填写'}\n"
        f"- 主要指标：`{response}`\n"
        f"- 实验设计：{_design_label(draft.get('design'))}\n"
        f"{role_line}{subject_line}{predictor_line}"
        f"- 指标单位：{unit}\n"
        f"- 实际意义阈值：{threshold_text}\n"
        f"- 状态：**{status}**"
    )


def _review_scientific_setup(state: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    draft = state["scientific_setup"]
    draft["step"] = "review"
    return (
        _research_settings_card(state),
        [
            _science_option("确认并制定分析方案", DATA_SCIENCE_CONFIRM),
            _science_option("修改研究问题", DATA_SCIENCE_EDIT_QUESTION),
            _science_option("更换主要指标", DATA_SCIENCE_EDIT_RESPONSE),
            _science_option("更改实验设计", DATA_SCIENCE_EDIT_DESIGN),
            _science_option("修改实际意义阈值", DATA_SCIENCE_EDIT_THRESHOLD),
        ],
    )


def _scientific_structure_error(
    state: dict[str, Any], draft: dict[str, Any]
) -> str | None:
    dataset_ids = list(state.get("dataset_ids") or [])
    if not dataset_ids:
        return "当前没有可用数据集。"
    design = draft.get("design")
    group = draft.get("group_variable")
    subject = draft.get("subject_id_variable")
    frame = _active_data_store().load_dataframe(dataset_ids[-1])
    if design == "independent_groups":
        if not group or group not in frame.columns:
            return "独立组分析必须确认有效的分组字段。"
        if int(frame[group].nunique(dropna=True)) < 2:
            return "分组字段至少需要两个有效组。"
    if design in {"paired", "repeated_measures"}:
        if not group or not subject:
            return "配对或重复测量必须同时确认样本 ID 和条件字段。"
        if group not in frame.columns or subject not in frame.columns:
            return "样本 ID 或条件字段不在当前数据中。"
        working = frame[[subject, group]].dropna()
        if bool(working.duplicated([subject, group]).any()):
            return "同一样本在同一条件下存在多行记录，需要先明确如何汇总。"
        condition_count = int(working[group].nunique())
        if design == "paired" and condition_count != 2:
            return "配对设计必须恰好包含两个条件；当前数据不符合。"
        if design == "repeated_measures" and condition_count < 3:
            return "重复测量设计至少需要三个条件。"
        complete = int(
            (working.groupby(subject)[group].nunique() == condition_count).sum()
        )
        if complete < 2:
            return "少于两个样本具有完整条件记录，不能执行配对或重复测量分析。"
    if design == "continuous_relationship" and not draft.get("predictor_variable"):
        return "连续变量关系分析必须确认一个预测因素。"
    return None


def _continue_scientific_setup(
    message: str,
    history: list[dict[str, Any]],
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    draft = state.get("scientific_setup")
    if not isinstance(draft, dict) or not draft.get("active"):
        return None
    step = draft.get("step")
    if step == "research_question":
        options: list[dict[str, str]] = []
        if len(message.strip()) < 5:
            content = "研究问题还不够完整，请用一句话说明想比较或解释什么。"
        else:
            draft["research_question"] = message.strip()
            draft["step"] = "response"
            options = _response_options(state)
            content = "这个问题主要关注哪个测量指标？请选择一个主要指标。"
            if not options:
                content = "没有识别到可用数值指标，请检查数据字段后再建立方案。"
        return [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": content, "options": options},
        ], state
    if step == "practical_threshold":
        match = re.search(r"(?<![\w.])-?\d+(?:\.\d+)?", message)
        if match is None or float(match.group()) < 0:
            content, options = _threshold_prompt()
            content = "请输入一个非负数值。" + content
        else:
            draft["practical_threshold"] = float(match.group())
            content, options = _review_scientific_setup(state)
        return [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": content, "options": options},
        ], state
    if step == "subject_id":
        candidates = _scientific_proposal(state).get("identifier_candidates") or []
        if message not in candidates:
            return [
                *history,
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "content": "请输入数据中真实存在的样本 ID 字段名。",
                    "options": _subject_options(state),
                },
            ], state
        draft["subject_id_variable"] = message
        draft["step"] = "group"
        return [
            *history,
            {"role": "user", "content": message},
            {
                "role": "assistant",
                "content": "哪个字段表示处理条件、时间点或测量阶段？",
                "options": _group_options(state),
            },
        ], state
    return None


def _data_chat_options(after: str | None = None) -> list[dict[str, str]]:
    labels: tuple[tuple[str, str], ...]
    if after is None:
        labels = (
            ("先建立科研分析方案", "建立科研分析方案"),
            ("先检查数据质量", "这份数据有没有问题？"),
            ("先概括一下数据", "先帮我总结一下"),
            ("比较不同组", "不同组之间有差异吗？"),
            ("看看字段关系", "哪些字段有关联？"),
            ("先画一张图", "画一张推荐图"),
            ("直接整理报告", "整理成报告"),
        )
    elif after == "这份数据有没有问题？":
        labels = (
            ("接着概括数据", "先帮我总结一下"),
            ("再比较不同组", "不同组之间有差异吗？"),
            ("画图直观看看", "画一张推荐图"),
        )
    elif after == "先帮我总结一下":
        labels = (
            ("比较不同组", "不同组之间有差异吗？"),
            ("看看字段关系", "哪些字段有关联？"),
            ("把结果画出来", "画一张推荐图"),
            ("整理成报告", "整理成报告"),
        )
    elif after == "不同组之间有差异吗？":
        labels = (
            ("再看字段关系", "哪些字段有关联？"),
            ("画图看看差异", "画一张推荐图"),
            ("整理成报告", "整理成报告"),
        )
    elif after == "哪些字段有关联？":
        labels = (
            ("把关系画出来", "画一张推荐图"),
            ("再比较不同组", "不同组之间有差异吗？"),
            ("整理成报告", "整理成报告"),
        )
    elif after == "画一张推荐图":
        labels = (
            ("再比较不同组", "不同组之间有差异吗？"),
            ("再看字段关系", "哪些字段有关联？"),
            ("最后整理报告", "整理成报告"),
        )
    else:
        labels = ()
    return [{"label": label, "value": value} for label, value in labels]


def _with_data_chat_options(
    history: list[dict[str, Any]], state: dict[str, Any]
) -> list[dict[str, Any]]:
    if (
        state.get("last_agent") != "data_analysis"
        or not state.get("data_exploration")
        or not history
    ):
        return history
    updated = [dict(message) for message in history]
    setup = state.get("scientific_setup")
    if isinstance(setup, dict) and setup.get("active"):
        return updated
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].get("role") == "assistant":
            if "options" not in updated[index]:
                updated[index]["options"] = _data_chat_options()
            break
    return updated


def _scientific_analysis_plan(
    brief: ScientificAnalysisBrief, state: dict[str, Any]
) -> tuple[str, list[dict[str, str]]]:
    ScientificBriefService.require_confirmed(brief)
    frame = _active_data_store().load_dataframe(brief.dataset_id)
    response = brief.response_variables[0]
    common = (
        "### 推荐分析方案\n\n"
        f"- 回答的问题：{brief.research_question}\n"
        f"- 主要指标：`{response}`\n"
        f"- 显著性阈值：{brief.alpha:g}\n"
    )
    if brief.design == "independent_groups":
        assert brief.group_variable is not None
        group_count = int(frame[brief.group_variable].nunique(dropna=True))
        if group_count == 2:
            method = "Welch t 检验"
            robust = "Mann–Whitney U 检验"
        else:
            method = "单因素 ANOVA；显著后进行 Holm 校正的两两比较"
            robust = "Kruskal–Wallis；显著后进行 Holm 校正的两两比较"
        content = (
            common
            + f"- 数据结构：`{brief.group_variable}` 共 {group_count} 个独立组\n"
            + f"- 推荐方法：{method}\n"
            + "- 输出：各组样本量与均值、p 值、效应量、区间、假设检查和警告\n\n"
            + "选择开始后才会执行计算。若实验条件不满足参数检验，可改用非参数方法。"
        )
        options = [
            _science_option(
                "按推荐方案开始分析",
                f"{DATA_SCIENCE_RUN_PREFIX}recommended",
            ),
            _science_option(
                f"改用{robust}",
                f"{DATA_SCIENCE_RUN_PREFIX}nonparametric",
            ),
        ]
    elif brief.design == "paired":
        assert brief.group_variable is not None
        condition_count = int(frame[brief.group_variable].nunique(dropna=True))
        content = (
            common + f"- 数据结构：同一 `{brief.subject_id_variable}` 在 "
            f"`{brief.group_variable}` 的 {condition_count} 个条件下配对\n"
            + "- 推荐方法：配对 t 检验，并检查配对差值的正态性\n"
            + "- 备选方法：Wilcoxon 符号秩检验\n"
            + "- 缺少任一条件的样本将成对排除，并明确报告数量"
        )
        options = [
            _science_option("开始配对 t 检验", f"{DATA_SCIENCE_RUN_PREFIX}recommended"),
            _science_option(
                "改用 Wilcoxon 检验",
                f"{DATA_SCIENCE_RUN_PREFIX}nonparametric",
            ),
        ]
    elif brief.design == "repeated_measures":
        assert brief.group_variable is not None
        condition_count = int(frame[brief.group_variable].nunique(dropna=True))
        content = (
            common + f"- 数据结构：同一 `{brief.subject_id_variable}` 在 "
            f"`{brief.group_variable}` 的 {condition_count} 个条件下重复测量\n"
            + "- 推荐方法：Friedman 重复测量检验\n"
            + "- 总体显著后：Wilcoxon 两两比较并进行 Holm 校正\n"
            + "- 仅使用条件完整的样本，并报告被排除数量"
        )
        options = [
            _science_option("开始重复测量分析", f"{DATA_SCIENCE_RUN_PREFIX}recommended")
        ]
    elif brief.design == "continuous_relationship":
        predictor = brief.covariates[0]
        content = (
            common
            + f"- 连续预测因素：`{predictor}`\n"
            + "- 推荐方法：简单线性回归\n"
            + "- 输出：斜率及区间、R²、p 值、标准化效应和残差诊断\n\n"
            + "回归结果描述关联，除非研究设计支持，否则不解释为因果关系。"
        )
        options = [
            _science_option("开始线性回归", f"{DATA_SCIENCE_RUN_PREFIX}recommended")
        ]
    else:
        content = (
            common
            + "- 推荐方法：主要指标的描述统计\n"
            + "- 当前设计不运行推断检验，也不会生成因果结论"
        )
        options = [
            _science_option("开始探索性分析", f"{DATA_SCIENCE_RUN_PREFIX}recommended")
        ]
    options.extend(
        (
            _science_option("修改研究问题", DATA_SCIENCE_EDIT_QUESTION),
            _science_option("更改实验设计", DATA_SCIENCE_EDIT_DESIGN),
            _science_option("修改实际意义阈值", DATA_SCIENCE_EDIT_THRESHOLD),
        )
    )
    return content, options


def _run_confirmed_scientific_analysis(
    state: dict[str, Any], *, variant: str
) -> tuple[str, dict[str, Any]]:
    raw = state.get("scientific_brief")
    if not isinstance(raw, dict):
        return "请先确认研究设定。", state
    brief = ScientificAnalysisBrief.model_validate(raw)
    ScientificBriefService.require_confirmed(brief)
    response = brief.response_variables[0]
    statistics = DataStatisticsService(_active_data_store())
    if brief.design == "independent_groups":
        assert brief.group_variable is not None
        frame = _active_data_store().load_dataframe(brief.dataset_id)
        groups = [
            value.item() if hasattr(value, "item") else value
            for value in frame[brief.group_variable].dropna().unique().tolist()
        ]
        method = (
            "mann_whitney"
            if variant == "nonparametric" and len(groups) == 2
            else "kruskal_wallis"
            if variant == "nonparametric"
            else "welch_t"
            if len(groups) == 2
            else "anova"
        )
        result = statistics.run_statistical_test(
            brief.dataset_id,
            method=method,  # type: ignore[arg-type]
            response_column=response,
            group_column=brief.group_variable,
            groups=groups,
            alpha=brief.alpha,
        )
    elif brief.design == "paired":
        result = statistics.run_statistical_test(
            brief.dataset_id,
            method="wilcoxon" if variant == "nonparametric" else "paired_t",
            response_column=response,
            subject_id_column=brief.subject_id_variable,
            condition_column=brief.group_variable,
            alpha=brief.alpha,
        )
    elif brief.design == "repeated_measures":
        result = statistics.run_statistical_test(
            brief.dataset_id,
            method="friedman",
            response_column=response,
            subject_id_column=brief.subject_id_variable,
            condition_column=brief.group_variable,
            alpha=brief.alpha,
        )
    elif brief.design == "continuous_relationship":
        result = statistics.run_linear_regression(
            brief.dataset_id,
            response_column=response,
            predictor_column=brief.covariates[0],
            alpha=brief.alpha,
        )
    else:
        result = statistics.describe_dataset(brief.dataset_id, columns=(response,))
    _remember_id(state, "analysis_ids", result.analysis_id)
    interpretation = ScientificInterpretationService(_active_data_store()).interpret(
        brief, result
    )
    rendered = render_scientific_interpretation(interpretation)
    rendered += "\n\n---\n\n**统计明细**\n\n" + _render_analysis_result(result)
    return rendered, state


def _practical_significance_note(brief: ScientificAnalysisBrief, result: Any) -> str:
    response = brief.response_variables[0]
    threshold = brief.practical_thresholds.get(response)
    unit = brief.units.get(response, "")
    if threshold is None:
        return (
            "### 实际科研意义\n\n"
            "尚未设置实际意义阈值，因此本次只能判断统计证据，"
            "不能自动断言差异在科研或工程上足够重要。"
        )
    observed: float | None = None
    label = "观察到的代表性变化"
    if result.analysis_type == "statistical_test":
        if result.summary.get("mean_difference") is not None:
            observed = abs(float(result.summary["mean_difference"]))
            label = "绝对均值差"
        elif result.summary.get("median_difference") is not None:
            observed = abs(float(result.summary["median_difference"]))
            label = "绝对中位差"
        else:
            rows = (
                result.summary.get("groups") or result.summary.get("conditions") or []
            )
            means = [float(row["mean"]) for row in rows if row.get("mean") is not None]
            if means:
                observed = max(means) - min(means)
                label = "最大组均值跨度"
    elif result.analysis_type == "regression":
        observed = abs(float(result.summary["slope"]))
        label = "预测因素每增加 1 单位时的响应变化"
    if observed is None:
        return (
            "### 实际科研意义\n\n"
            f"已设置阈值 {threshold:g} {unit}，但当前分析不适合自动比较该阈值。"
        )
    verdict = "达到" if observed >= threshold else "未达到"
    return (
        "### 实际科研意义\n\n"
        f"- 判断阈值：{threshold:g} {unit}\n"
        f"- {label}：{observed:.4g} {unit}\n"
        f"- 判断：**{verdict}预设的实际意义阈值**"
    )


def _is_scientific_option(value: str) -> bool:
    return value == "建立科研分析方案" or value.startswith("data-science-")


def _scientific_option_answer(
    value: str, state: dict[str, Any]
) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
    if value == "建立科研分析方案":
        content, options = _begin_scientific_setup(state)
        return content, options, state

    draft = state.get("scientific_setup")
    if not isinstance(draft, dict):
        content, options = _begin_scientific_setup(state)
        return content, options, state

    if value.startswith(DATA_SCIENCE_RESPONSE_PREFIX):
        column = value.removeprefix(DATA_SCIENCE_RESPONSE_PREFIX)
        candidates = _scientific_proposal(state).get("response_candidates") or []
        if column not in candidates:
            return "该指标不在当前数据中，请重新选择。", _response_options(state), state
        draft["response_variable"] = column
        draft["step"] = "design"
        return (
            "这些数据是怎样获得的？请选择最符合实际实验过程的一项。",
            _design_options(),
            state,
        )

    if value.startswith(DATA_SCIENCE_DESIGN_PREFIX):
        design = value.removeprefix(DATA_SCIENCE_DESIGN_PREFIX)
        if design not in {
            "independent_groups",
            "paired",
            "repeated_measures",
            "exploratory",
            "continuous_relationship",
        }:
            return "无法识别这个实验设计，请重新选择。", _design_options(), state
        draft["design"] = design
        draft["group_variable"] = None
        draft["subject_id_variable"] = None
        draft["predictor_variable"] = None
        if design == "independent_groups":
            draft["step"] = "group"
            options = _group_options(state)
            return (
                "哪个字段表示实验组、处理方法或条件？",
                options,
                state,
            )
        if design in {"paired", "repeated_measures"}:
            draft["step"] = "subject_id"
            options = _subject_options(state)
            content = (
                "哪个字段能唯一标识同一个样本或受试对象？"
                if options
                else "未识别到样本 ID 候选。请直接输入数据中的 ID 字段名；"
                "没有 ID 就不能确认配对或重复测量设计。"
            )
            return content, options, state
        if design == "continuous_relationship":
            draft["step"] = "predictor"
            return (
                "哪个连续字段可能与主要指标相关？请选择一个预测因素。",
                _predictor_options(state),
                state,
            )
        draft["step"] = "unit"
        column = str(draft.get("response_variable") or "")
        return f"`{column}` 使用什么单位？", _unit_options(column), state

    if value.startswith(DATA_SCIENCE_GROUP_PREFIX):
        column = value.removeprefix(DATA_SCIENCE_GROUP_PREFIX)
        candidates = _scientific_proposal(state).get("group_candidates") or []
        if column not in candidates:
            return (
                "该分组字段不在候选列表中，请重新选择。",
                _group_options(state),
                state,
            )
        draft["group_variable"] = column
        draft["step"] = "unit"
        response = str(draft.get("response_variable") or "")
        return f"`{response}` 使用什么单位？", _unit_options(response), state

    if value.startswith(DATA_SCIENCE_SUBJECT_PREFIX):
        column = value.removeprefix(DATA_SCIENCE_SUBJECT_PREFIX)
        candidates = _scientific_proposal(state).get("identifier_candidates") or []
        if column not in candidates:
            return (
                "该 ID 字段不在当前数据中，请重新选择。",
                _subject_options(state),
                state,
            )
        draft["subject_id_variable"] = column
        draft["step"] = "group"
        return (
            "哪个字段表示处理条件、时间点或测量阶段？",
            _group_options(state),
            state,
        )

    if value.startswith(DATA_SCIENCE_PREDICTOR_PREFIX):
        column = value.removeprefix(DATA_SCIENCE_PREDICTOR_PREFIX)
        valid = {
            option["value"].removeprefix(DATA_SCIENCE_PREDICTOR_PREFIX)
            for option in _predictor_options(state)
        }
        if column not in valid:
            return "该预测因素不可用，请重新选择。", _predictor_options(state), state
        draft["predictor_variable"] = column
        draft["step"] = "unit"
        response = str(draft.get("response_variable") or "")
        return f"`{response}` 使用什么单位？", _unit_options(response), state

    if value.startswith(DATA_SCIENCE_UNIT_PREFIX):
        draft["unit"] = value.removeprefix(DATA_SCIENCE_UNIT_PREFIX) or None
        draft["step"] = "practical_threshold"
        content, options = _threshold_prompt()
        return content, options, state

    if value == DATA_SCIENCE_THRESHOLD_SKIP:
        draft["practical_threshold"] = None
        content, options = _review_scientific_setup(state)
        return content, options, state

    if value == DATA_SCIENCE_EDIT_QUESTION:
        draft["active"] = True
        draft["step"] = "research_question"
        state["scientific_brief"] = None
        return "请重新输入研究问题。", [], state
    if value == DATA_SCIENCE_EDIT_RESPONSE:
        draft["active"] = True
        draft["step"] = "response"
        state["scientific_brief"] = None
        return "请选择新的主要指标。", _response_options(state), state
    if value == DATA_SCIENCE_EDIT_DESIGN:
        draft["active"] = True
        draft["step"] = "design"
        state["scientific_brief"] = None
        return "请选择正确的实验设计。", _design_options(), state
    if value == DATA_SCIENCE_EDIT_THRESHOLD:
        draft["active"] = True
        draft["step"] = "practical_threshold"
        state["scientific_brief"] = None
        content, options = _threshold_prompt()
        return content, options, state

    if value.startswith(DATA_SCIENCE_RUN_PREFIX):
        variant = value.removeprefix(DATA_SCIENCE_RUN_PREFIX)
        if variant not in {"recommended", "nonparametric"}:
            return "无法识别分析方案，请重新选择。", [], state
        try:
            content, state = _run_confirmed_scientific_analysis(state, variant=variant)
        except (KeyError, OSError, ValueError) as exc:
            return (
                f"分析未执行：{exc}",
                [
                    _science_option("更改实验设计", DATA_SCIENCE_EDIT_DESIGN),
                    _science_option(
                        "改用非参数方法",
                        f"{DATA_SCIENCE_RUN_PREFIX}nonparametric",
                    ),
                ],
                state,
            )
        return (
            content,
            [
                _science_option("画出本次结果", "画一张推荐图"),
                _science_option("整理成科研报告", "整理成报告"),
                _science_option("重新建立研究方案", "建立科研分析方案"),
            ],
            state,
        )

    if value == DATA_SCIENCE_CONFIRM:
        response = str(draft.get("response_variable") or "")
        dataset_ids = list(state.get("dataset_ids") or [])
        if not dataset_ids or not response or not draft.get("design"):
            return "研究设定还不完整，请先补全后再确认。", [], state
        structure_error = _scientific_structure_error(state, draft)
        if structure_error:
            return (
                f"当前设定无法进入统计分析：{structure_error}",
                [
                    _science_option("更改实验设计", DATA_SCIENCE_EDIT_DESIGN),
                    _science_option("更换主要指标", DATA_SCIENCE_EDIT_RESPONSE),
                ],
                state,
            )
        threshold = draft.get("practical_threshold")
        unit = draft.get("unit")
        brief = ScientificAnalysisBrief(
            dataset_id=str(dataset_ids[-1]),
            domain="materials_science",
            research_question=str(draft.get("research_question") or ""),
            observation_unit=(
                "同一样本的重复观测"
                if draft.get("design") in {"paired", "repeated_measures"}
                else "数据表中的每一行独立观测"
            ),
            design=draft["design"],
            response_variables=(response,),
            group_variable=draft.get("group_variable"),
            covariates=(
                (str(draft.get("predictor_variable")),)
                if draft.get("predictor_variable")
                else ()
            ),
            subject_id_variable=draft.get("subject_id_variable"),
            hypothesis=(
                f"不同 {draft.get('group_variable')} 条件下的 {response} 存在差异"
                if draft.get("group_variable")
                else None
            ),
            practical_thresholds=(
                {response: float(threshold)} if threshold is not None else {}
            ),
            units={response: str(unit)} if unit else {},
            roles_confirmed=True,
        )
        state["scientific_brief"] = brief.model_dump(mode="json")
        draft["active"] = False
        draft["step"] = "confirmed"
        plan, options = _scientific_analysis_plan(brief, state)
        content = _research_settings_card(state, confirmed=True)
        content += "\n\n---\n\n" + plan
        return content, options, state

    return "无法识别这个选择，请重新建立科研分析方案。", [], state


def _answer_data_chat_option(
    history: list[dict[str, Any]] | None,
    state: dict[str, Any] | None,
    event: gr.SelectData,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question = str(event.value)
    current_history = [dict(message) for message in (history or [])]
    current = dict(state or _initial_state())
    is_group_metric = question.startswith(DATA_GROUP_METRIC_PREFIX)
    is_scientific = _is_scientific_option(question)
    if (
        question not in DATA_GUIDED_QUESTIONS
        and not is_group_metric
        and not is_scientific
    ):
        return current_history, current
    for index in range(len(current_history) - 1, -1, -1):
        if (
            current_history[index].get("role") == "assistant"
            and "options" in current_history[index]
        ):
            current_history[index].pop("options", None)
            break
    if is_scientific:
        user_label = {
            "建立科研分析方案": "建立科研分析方案",
            DATA_SCIENCE_THRESHOLD_SKIP: "暂不设置实际意义阈值",
            DATA_SCIENCE_CONFIRM: "确认当前研究设定",
            DATA_SCIENCE_EDIT_QUESTION: "修改研究问题",
            DATA_SCIENCE_EDIT_RESPONSE: "更换主要指标",
            DATA_SCIENCE_EDIT_DESIGN: "更改实验设计",
            DATA_SCIENCE_EDIT_THRESHOLD: "修改实际意义阈值",
            f"{DATA_SCIENCE_RUN_PREFIX}recommended": "按推荐方案开始分析",
            f"{DATA_SCIENCE_RUN_PREFIX}nonparametric": "改用非参数方法分析",
        }.get(question)
        if question.startswith(DATA_SCIENCE_DESIGN_PREFIX):
            user_label = "实验设计：" + _design_label(
                question.removeprefix(DATA_SCIENCE_DESIGN_PREFIX)
            )
        if user_label is None:
            for prefix, label in (
                (DATA_SCIENCE_RESPONSE_PREFIX, "主要指标"),
                (DATA_SCIENCE_DESIGN_PREFIX, "实验设计"),
                (DATA_SCIENCE_GROUP_PREFIX, "分组字段"),
                (DATA_SCIENCE_SUBJECT_PREFIX, "样本 ID"),
                (DATA_SCIENCE_PREDICTOR_PREFIX, "连续预测因素"),
                (DATA_SCIENCE_UNIT_PREFIX, "指标单位"),
            ):
                if question.startswith(prefix):
                    user_label = f"{label}：{question.removeprefix(prefix) or '未提供'}"
                    break
        current_history.append(
            {"role": "user", "content": user_label or "继续完善研究设定"}
        )
        content, options, current = _scientific_option_answer(question, current)
        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        if options:
            assistant["options"] = options
        current_history.append(assistant)
        return current_history, current
    if question == "不同组之间有差异吗？":
        current_history.append({"role": "user", "content": question})
        _numeric, group_by, groups = _guided_data_defaults(current)
        metric_options = _group_metric_options(current)
        if not group_by or len(groups) < 2 or not metric_options:
            current_history.append(
                {
                    "role": "assistant",
                    "content": (
                        "我没有找到可直接比较的分组字段和数值指标。"
                        "请打开“数据分析工具（可选）”手动指定。"
                    ),
                }
            )
            return current_history, current
        current_history.append(
            {
                "role": "assistant",
                "content": (
                    f"我识别到分组字段是 `{group_by}`，共有 "
                    f"{len(groups)} 个组。你想比较哪个数值指标？"
                ),
                "options": metric_options,
            }
        )
        return current_history, current
    if is_group_metric:
        column = question.removeprefix(DATA_GROUP_METRIC_PREFIX)
        current_history.append({"role": "user", "content": f"比较指标：{column}"})
        numeric, group_by, groups = _guided_data_defaults(current)
        if column not in numeric or not group_by or len(groups) < 2:
            current_history.append(
                {"role": "assistant", "content": "当前字段不足以建立组间比较方案。"}
            )
            return current_history, current
        current["scientific_brief"] = None
        current["scientific_setup"] = {
            "active": True,
            "step": "design",
            "intent": "group_comparison",
            "research_question": f"不同 {group_by} 组的 {column} 是否存在差异？",
            "response_variable": column,
            "design": None,
            "group_variable": None,
            "subject_id_variable": None,
            "predictor_variable": None,
            "unit": None,
            "practical_threshold": None,
        }
        current_history.append(
            {
                "role": "assistant",
                "content": (
                    f"已将 `{column}` 作为主要指标，但我不会默认它是独立组实验。"
                    "这些数据是怎样获得的？"
                ),
                "options": _design_options(),
            }
        )
        return current_history, current
    if question == "整理成报告" and not isinstance(
        current.get("scientific_brief"), dict
    ):
        current_history.append({"role": "user", "content": question})
        current_history.append(
            {
                "role": "assistant",
                "content": (
                    "科研报告不能只把现有统计结果拼起来。请先明确一个研究问题。"
                    "我会继续确认主要指标、实验设计、样本 ID 和实际意义标准；"
                    "确认后再生成 Word 报告。"
                ),
                "options": [
                    {
                        "label": "建立科研分析方案",
                        "value": "建立科研分析方案",
                    },
                    {"label": "先查看数据质量", "value": "这份数据有没有问题？"},
                ],
            }
        )
        return current_history, current
    current_history.append({"role": "user", "content": question})
    answer, current, image_path, download_path = _run_guided_data_action(
        question, state
    )
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": answer,
    }
    follow_up_options = _data_chat_options(after=question)
    if follow_up_options:
        assistant_message["options"] = follow_up_options
    current_history.append(assistant_message)
    if image_path:
        current_history.append({"role": "assistant", "content": {"path": image_path}})
    if download_path:
        current_history.append(
            {"role": "assistant", "content": {"path": download_path}}
        )
    return current_history, current


def _run_data_action(
    action: str,
    state: dict[str, Any] | None,
    columns: list[str] | None,
    x: str | None,
    y: str | None,
    group_by: str | None,
    groups: list[Any] | None,
    statistical_method: str,
    plot_type: str,
    correlation_method: str,
    export_format: str,
) -> tuple[str, dict[str, Any], Any, Any]:
    current = dict(state or _initial_state())
    dataset_ids = list(current.get("dataset_ids") or [])
    if not dataset_ids:
        return "请先上传并登记一个数据集。", current, None, None
    dataset_id = dataset_ids[-1]
    store = _active_data_store()
    statistics = DataStatisticsService(store)
    reporting = DataAnalysisReportingService(store)
    try:
        if action == "eda":
            bundle = DataExplorationService(
                DataAnalysisService(store), statistics
            ).explore(dataset_id)
            current["data_exploration"] = bundle.model_dump(mode="json")
            current["analysis_ids"] = list(
                dict.fromkeys([*current.get("analysis_ids", []), *bundle.analysis_ids])
            )
            return render_exploration_markdown(bundle), current, None, None
        if action == "describe":
            selected = tuple(columns or ([x] if x else []))
            result = statistics.describe_dataset(
                dataset_id, columns=selected, group_by=group_by or None
            )
            _remember_id(current, "analysis_ids", result.analysis_id)
            return _render_analysis_result(result), current, None, None
        if action == "correlation":
            selected = tuple(columns or [])
            result = statistics.analyze_correlations(
                dataset_id,
                columns=selected,
                method=correlation_method,  # type: ignore[arg-type]
            )
            _remember_id(current, "analysis_ids", result.analysis_id)
            return _render_analysis_result(result), current, None, None
        if action == "test":
            if statistical_method == "paired_t":
                if not x or not y:
                    raise ValueError("配对 t 检验需要选择 X 和 Y 两个数值字段")
                result = statistics.run_statistical_test(
                    dataset_id,
                    method="paired_t",
                    paired_columns=(x, y),
                )
            else:
                result = statistics.run_statistical_test(
                    dataset_id,
                    method=statistical_method,  # type: ignore[arg-type]
                    response_column=x,
                    group_column=group_by,
                    groups=tuple(groups or ()),
                )
                if x and group_by:
                    brief = ScientificAnalysisBrief(
                        dataset_id=dataset_id,
                        domain="materials_science",
                        research_question=(f"不同 {group_by} 组的 {x} 是否存在差异？"),
                        observation_unit="数据表中的每一行独立观测",
                        design="independent_groups",
                        response_variables=(x,),
                        group_variable=group_by,
                        hypothesis=f"至少一个 {group_by} 组的 {x} 不同",
                        roles_confirmed=True,
                    )
                    current["scientific_brief"] = brief.model_dump(mode="json")
            _remember_id(current, "analysis_ids", result.analysis_id)
            return _render_analysis_result(result), current, None, None
        if action == "plot":
            artifact = reporting.create_plot(
                dataset_id,
                plot_type=plot_type,  # type: ignore[arg-type]
                x=x,
                y=y,
                group_by=group_by or None,
                columns=tuple(columns or ()),
                correlation_method=correlation_method,  # type: ignore[arg-type]
            )
            _remember_id(current, "plot_artifact_ids", artifact.artifact_id)
            path = str(store.resolve_artifact_path(artifact.artifact_id))
            return (
                f"图表已生成：`{artifact.artifact_id}`",
                current,
                path,
                None,
            )
        if action == "report":
            analysis_ids = tuple(current.get("analysis_ids") or [])
            if not analysis_ids:
                raise ValueError("请先执行至少一项统计分析")
            raw_brief = current.get("scientific_brief")
            if not isinstance(raw_brief, dict):
                raise ValueError("请先明确研究问题，并完成响应变量、分组和实验设计确认")
            scientific_brief = ScientificAnalysisBrief.model_validate(raw_brief)
            artifact = reporting.create_report(
                dataset_id,
                analysis_ids=analysis_ids,
                plot_artifact_ids=tuple(current.get("plot_artifact_ids") or []),
                format="docx",
                scientific_brief=scientific_brief,
            )
            _remember_id(current, "report_artifact_ids", artifact.artifact_id)
            path = str(store.resolve_artifact_path(artifact.artifact_id))
            return (
                f"科研 Word 报告已生成：`{artifact.artifact_id}`",
                current,
                None,
                path,
            )
        if action == "export":
            format_name = "json" if export_format == "JSON" else "csv"
            artifact = reporting.export_dataset(
                dataset_id,
                format=format_name,  # type: ignore[arg-type]
            )
            _remember_id(current, "report_artifact_ids", artifact.artifact_id)
            path = str(store.resolve_artifact_path(artifact.artifact_id))
            return (
                f"安全数据导出已生成：`{artifact.artifact_id}`",
                current,
                None,
                path,
            )
        raise ValueError("未知的数据分析操作")
    except (KeyError, OSError, ValueError) as exc:
        return f"操作未完成：{exc}", current, None, None


def _run_deduplicate_action(
    state: dict[str, Any] | None, key_column: str | None
) -> tuple[Any, ...]:
    current = dict(state or _initial_state())
    dataset_ids = list(current.get("dataset_ids") or [])
    if not dataset_ids or not key_column:
        return (
            "请先上传数据并选择去重键字段。",
            current,
            *_data_panel_payload(current),
        )
    store = _active_data_store()
    try:
        dataset, transform, _inspection = DatasetTransformService(
            store
        ).transform_dataset(
            dataset_ids[-1],
            operations=(
                DatasetTransformOperation(
                    kind="drop_duplicates",
                    parameters={"subset": [key_column], "keep": "first"},
                ),
            ),
        )
        bundle = DataExplorationService(
            DataAnalysisService(store), DataStatisticsService(store)
        ).explore(dataset.dataset_id)
        current["dataset_ids"] = [dataset.dataset_id]
        current["data_exploration"] = bundle.model_dump(mode="json")
        current["analysis_ids"] = list(bundle.analysis_ids)
        current["plot_artifact_ids"] = []
        current["data_source_kind"] = "data_analysis"
        return (
            f"去重完成：{transform.rows_before} → {transform.rows_after} 行；"
            f"原数据未覆盖，新数据集为 `{dataset.dataset_id}`。",
            current,
            *_data_panel_payload(current),
        )
    except (KeyError, OSError, ValueError) as exc:
        return f"去重未完成：{exc}", current, *_data_panel_payload(current)


def _remember_id(current: dict[str, Any], key: str, value: str) -> None:
    current[key] = list(dict.fromkeys([*current.get(key, []), value]))


def _render_analysis_result(result: Any) -> str:
    lines = [
        f"## {result.analysis_type}",
        "",
        f"分析 ID：`{result.analysis_id}`",
        f"方法：`{result.method}`",
        "",
    ]
    if result.analysis_type == "descriptive":
        lines.extend(
            (
                "| 字段 | 分组 | n | 均值 | 中位数 | 标准差 |",
                "|---|---|---:|---:|---:|---:|",
            )
        )
        for row in result.summary.get("statistics", []):
            lines.append(
                f"| {row.get('column', '—')} | {row.get('group', '全部')} | "
                f"{row.get('count', '—')} | {row.get('mean', '—')} | "
                f"{row.get('median', '—')} | {row.get('std', '—')} |"
            )
    elif result.analysis_type == "correlation":
        lines.extend(("| X | Y | n | 相关系数 |", "|---|---|---:|---:|"))
        for row in result.summary.get("pairs", []):
            if row.get("x") != row.get("y"):
                lines.append(
                    f"| {row.get('x')} | {row.get('y')} | {row.get('n')} | "
                    f"{row.get('coefficient', '—')} |"
                )
        lines.extend(("", "相关性不代表因果关系。"))
    elif result.analysis_type == "statistical_test":
        response = result.parameters.get("response_column")
        group_column = result.parameters.get("group_column")
        p_value = result.summary.get("p_value")
        significant = result.summary.get("significant") is True
        method_label = {
            "welch_t": "Welch t 检验",
            "student_t": "独立样本 t 检验",
            "mann_whitney": "Mann–Whitney U 检验",
            "anova": "单因素方差分析（ANOVA）",
            "kruskal_wallis": "Kruskal–Wallis 检验",
            "paired_t": "配对 t 检验",
            "wilcoxon": "Wilcoxon 符号秩检验",
            "friedman": "Friedman 重复测量检验",
        }.get(result.method, result.method)
        if significant:
            conclusion = (
                "检测到统计学差异。"
                if result.method not in {"anova", "kruskal_wallis", "friedman"}
                else "至少有一个条件或组存在统计学差异。"
            )
        else:
            conclusion = "没有足够证据认为这些组存在统计学差异。"
        lines = [
            "**组间差异分析**",
            "",
            f"比较指标：`{response or '—'}`",
            f"分组字段：`{group_column or '—'}`",
            f"方法：{method_label}",
            "",
            f"**结论：{conclusion}**",
            "",
            f"p 值：{_format_analysis_value(p_value)}（显著性阈值 0.05）",
            "",
        ]
        rows = result.summary.get("groups") or result.summary.get("conditions") or []
        if rows:
            lines.extend(("", "各组或条件估计："))
            for group in rows:
                group_interval = group.get("mean_confidence_interval")
                interval_text = (
                    f"[{_format_analysis_value(group_interval[0])}, "
                    f"{_format_analysis_value(group_interval[1])}]"
                    if group_interval
                    else "—"
                )
                lines.append(
                    f"- {group.get('label', '—')}：n={group.get('n', '—')}，"
                    f"均值={_format_analysis_value(group.get('mean'))}，"
                    f"均值95%区间={interval_text}"
                )
        elif result.summary.get("pairs") is not None:
            lines.append(f"完整配对样本数：{result.summary.get('pairs')}")
        effect = result.summary.get("effect_size") or {}
        if effect:
            lines.extend(
                (
                    "",
                    f"效应量 `{effect.get('name', '—')}`："
                    f"{_format_analysis_value(effect.get('value'))}",
                )
            )
        interval = result.summary.get("confidence_interval")
        if interval:
            lines.append(
                "均值差的 95% 置信区间："
                f"[{_format_analysis_value(interval[0])}, "
                f"{_format_analysis_value(interval[1])}]"
            )
        post_hoc = result.summary.get("post_hoc") or []
        if post_hoc:
            lines.extend(
                (
                    "",
                    "**校正后的两两比较**",
                )
            )
            for comparison in post_hoc:
                left = comparison.get("group_a", comparison.get("condition_a", "—"))
                right = comparison.get("group_b", comparison.get("condition_b", "—"))
                lines.append(
                    f"- {left} vs {right}：原始p="
                    f"{_format_analysis_value(comparison.get('raw_p_value'))}，"
                    "Holm校正p="
                    f"{_format_analysis_value(comparison.get('adjusted_p_value'))}，"
                    f"显著={'是' if comparison.get('significant') else '否'}"
                )
        dropped = result.summary.get("dropped_incomplete_subjects")
        if dropped:
            lines.append(f"因条件不完整而排除的样本数：{dropped}")
        lines.extend(("", f"分析记录：`{result.analysis_id}`"))
    elif result.analysis_type == "regression":
        slope = result.summary.get("slope")
        interval = result.summary.get("slope_confidence_interval") or [None, None]
        p_value = result.summary.get("p_value")
        significant = result.summary.get("significant") is True
        lines = [
            "**连续变量关系分析**",
            "",
            f"响应指标：`{result.parameters.get('response_column', '—')}`",
            f"预测因素：`{result.parameters.get('predictor_column', '—')}`",
            "方法：简单线性回归",
            "",
            (
                "**结论：检测到线性关联的统计证据。**"
                if significant
                else "**结论：没有足够证据认为两者存在线性关联。**"
            ),
            "",
            f"样本数：{result.summary.get('n', '—')}",
            f"斜率：{_format_analysis_value(slope)}",
            "斜率的 95% 置信区间："
            f"[{_format_analysis_value(interval[0])}, "
            f"{_format_analysis_value(interval[1])}]",
            f"R²：{_format_analysis_value(result.summary.get('r_squared'))}",
            f"p 值：{_format_analysis_value(p_value)}",
            "",
            "该结果描述变量间关联；仅凭回归不能证明因果关系。",
            "",
            f"分析记录：`{result.analysis_id}`",
        ]
    else:
        for key, value in result.summary.items():
            lines.append(f"- {key}：{value}")
    if result.warnings:
        lines.extend(("", "**警告**", ""))
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)


def _format_analysis_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _import_cross_agent_dataset(
    source_kind: str,
    identifier: str,
    state: dict[str, Any] | None,
) -> tuple[Any, ...]:
    current = dict(state or _initial_state())
    source_id = identifier.strip()
    if source_kind == "materials_database" and not source_id:
        source_id = str(current.get("material_query_id") or "")
    if source_kind == "literature" and not source_id:
        document_ids = list(
            (current.get("literature_state") or {}).get("document_ids") or []
        )
        source_id = str(document_ids[-1]) if document_ids else ""
    if not source_id:
        return (
            "没有可导入的稳定来源 ID。请先完成数据库查询或文献矩阵审核。",
            current,
            *_data_panel_payload(current),
        )
    store = _active_data_store()
    coordinator = DataAnalysisCrossAgentCoordinator(
        dataset_store=store,
        query_store=MATERIAL_QUERIES,
    )
    try:
        if source_kind == "materials_database":
            handoff = coordinator.material_query_to_analysis(
                source_id, task="通过统一页面继续分析材料数据库快照"
            )
        elif source_kind == "literature":
            from materials_screening.cli import _literature_pgvector_store

            matrix_store = _literature_pgvector_store()
            groups, measurements, comparisons, claims, links = matrix_store.load_matrix(
                source_id, status="approved"
            )
            table = ExperimentMatrixDataTable(
                title="Human-approved experiment matrix",
                document_id=source_id,
                groups=tuple(groups),
                measurements=tuple(measurements),
                comparisons=tuple(comparisons),
                claims=tuple(claims),
                claim_evidence_links=tuple(links),
            )
            handoff = coordinator.literature_matrix_to_analysis(
                table, task="通过统一页面分析已审核文献实验矩阵"
            )
        else:
            raise ValueError("不支持的跨 Agent 数据来源")
        dataset_id = handoff.partition.dataset_id
        bundle = DataExplorationService(
            DataAnalysisService(store), DataStatisticsService(store)
        ).explore(dataset_id)
        current["dataset_ids"] = [dataset_id]
        current["data_exploration"] = bundle.model_dump(mode="json")
        current["analysis_ids"] = list(bundle.analysis_ids)
        current["data_source_kind"] = handoff.partition.source_kind
        current["data_source_id"] = handoff.partition.source_id
        current["last_agent"] = "data_analysis"
        return (
            f"已导入 {handoff.partition.source_kind} 来源："
            f"`{handoff.partition.source_id}`，并完成只读 EDA。",
            current,
            *_data_panel_payload(current),
        )
    except (KeyError, OSError, ValueError) as exc:
        return f"跨 Agent 导入未完成：{exc}", current, *_data_panel_payload(current)


def _dispatch_ui(
    mode: str,
    message: str,
    history: list[dict[str, Any]] | None,
    files: list[Any] | None,
    state: dict[str, Any] | None,
    material: str,
    year_from: float,
    limit: float,
    expert_mode: bool,
) -> Iterator[tuple[Any, ...]]:
    for conversation, trace, current, cleared, cleared_files in _dispatch(
        mode, message, history, files, state, material, year_from, limit
    ):
        _capture_literature_query(conversation, current)
        _capture_material_query(conversation, current)
        page_rows, page_details, page_label, current = _literature_page(current)
        has_literature_result = bool(page_rows)
        data_panel = _data_panel_payload(current)
        yield (
            _with_data_chat_options(conversation, current),
            trace,
            current,
            cleared,
            cleared_files,
            _audit_markdown(mode, current, expert_mode),
            _attachment_markdown(None, current),
            page_rows,
            page_details,
            page_label,
            gr.update(visible=has_literature_result),
            *data_panel,
        )


def _new_conversation() -> tuple[list[Any], str, dict[str, Any], str, None]:
    state = _initial_state()
    return [], _trace(MODE_DATABASE, "新会话"), state, "", None


def _new_conversation_ui(
    expert_mode: bool,
) -> tuple[Any, ...]:
    conversation, trace, state, message, files = _new_conversation()
    return (
        conversation,
        trace,
        state,
        message,
        files,
        _audit_markdown(MODE_AUTO, state, expert_mode),
        _attachment_markdown(None, state),
        [],
        "检索完成后，可在这里分页查看论文详情。",
        "第 0 / 0 页",
        gr.update(visible=False),
        *_data_panel_payload(state),
    )


def _restore_conversation_ui(
    conversation_id: str, expert_mode: bool
) -> tuple[Any, ...]:
    runner = _master_runner
    if runner is None:
        raise gr.Error("请先初始化，再恢复已保存会话。")
    try:
        task, names = runner.restore_fulltext_context(conversation_id.strip())
    except Exception as exc:
        detail = "请核对会话编号及附件是否仍可用；当前页面未被更改。"
        if isinstance(exc, ValueError) and str(exc).startswith(("请", "当前", "任务")):
            detail = str(exc)
        raise gr.Error("恢复未完成：" + detail) from None
    # Build a clean browser binding only after every source/grant was checked.
    # Do not resurrect a previous tab's unrelated datasets or attachment paths.
    payload = list(_new_conversation_ui(expert_mode))
    state = payload[2]
    state.update(
        master_conversation_id=task.conversation_id,
        artifact_ids=list(task.artifact_refs),
        artifact_names=list(names),
        last_agent="literature",
    )
    waiting = (
        task.stage == "awaiting_clarification" and task.resume_stage == "previewing"
    )
    preview_complete = bool(task.artifact_refs) and len(task.previews) == len(
        task.artifact_refs
    )
    guidance = (
        "你可以输入“继续预览全部论文，暂不详细分析”。"
        "已保存预览会在接续时重新核验；完成预览后，再选择需要分析的论文。"
    )
    if waiting and preview_complete:
        guidance = "全部预览已保存，无须重新预览。请确认要详细分析的论文范围。"
    text = (
        f"已恢复这轮科研任务：{len(task.artifact_refs)} 篇全文已绑定，"
        f"{len(task.previews)} 篇有已保存预览，"
        f"{len(task.extraction_snapshots)} 篇有提取进度。\n\n"
        "这是已保存进度，不是完整分析成功报告。恢复没有调用模型，"
        "也没有重新检索或开始详细分析。\n\n" + guidance
    )
    if task.artifact_refs:
        progress = f"已保存 {len(task.previews)}/{len(task.artifact_refs)} 篇全文预览"
        if waiting and preview_complete:
            progress += "，等待你确认分析范围"
        text = progress + "。\n\n" + text
    if task.previews:
        from materials_screening.master.fulltext_preview_display import (
            preview_report_lines,
            render_fulltext_preview,
        )

        raw = "\n".join(
            preview_report_lines(
                task,
                tuple(range(len(task.artifact_refs))),
                {preview.document_id for preview in task.previews},
                awaiting_confirmation=waiting,
            )
        )
        if waiting:
            raw += (
                "\n\n目前仅完成预览，未启动详细分析。确认后可以输入"
                "“确认详细分析全部论文”，或“只分析第1、3篇”。"
            )
        text += "\n\n" + render_fulltext_preview(raw)
    payload[0] = [
        {"role": "user", "content": task.original_question},
        {"role": "assistant", "content": text},
    ]
    payload[1] = _trace(MODE_AUTO, "会话已恢复", "文件和进度保留，等待你的下一步指令")
    payload[5] = _audit_markdown(MODE_AUTO, state, expert_mode)
    payload[6] = _attachment_markdown(None, state)
    return tuple(payload)


def _available_port(preferred: int, *, attempts: int = 20) -> int:
    for port in range(preferred, min(preferred + attempts, 65536)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError("没有可用的本地端口")


def _create_ui() -> gr.Blocks:
    demo = gr.Blocks(title="Materials Multi-Agent")
    with demo:
        gr.HTML(
            "<div id='multi-header'><h1>材料智能助手</h1>"
            "<p>检索材料、阅读论文，或分析表格数据</p></div>"
        )
        session_state = gr.State(value=_initial_state())
        quick_buttons: list[tuple[gr.Button, str]] = []
        with gr.Row(elem_id="app-layout"):
            with gr.Column(scale=1, min_width=300, elem_id="left-sidebar"):
                new_button = gr.Button(
                    "＋ 新对话", size="sm", scale=0, elem_id="new-chat"
                )
                with gr.Accordion("恢复已保存会话", open=False):
                    restore_id = gr.Textbox(
                        label="保存的会话编号",
                        value=os.getenv("MASTER_UI_RESTORE_CONVERSATION", ""),
                    )
                    restore_button = gr.Button("恢复科研任务", size="sm")
                    gr.Markdown("找回问题、已上传全文和进度，不会自动开始分析。")
                gr.Markdown("### Agent 初始化")
                mode = gr.Dropdown(
                    choices=[MODE_AUTO, *UI_PLUGINS.modes()],
                    value=MODE_AUTO,
                    label="工作模式",
                )
                with gr.Row():
                    provider = gr.Dropdown(
                        choices=["intern", "mock"],
                        value="intern",
                        label="基座模型",
                        scale=1,
                    )
                    repository = gr.Dropdown(
                        choices=["materials-project", "mock"],
                        value="materials-project",
                        label="材料数据库",
                        scale=1,
                    )
                init_button = gr.Button("初始化 Agent", variant="primary")
                init_status = gr.Markdown(
                    "正在自动初始化……", elem_classes=["init-status"]
                )
                with gr.Accordion(
                    "文献工具",
                    open=False,
                    elem_classes=["advanced-box"],
                ):
                    gr.Markdown(
                        "PDF、CSV、JSON 或 XLSX 请通过对话输入框左侧的上传按钮添加。"
                    )
                    material = gr.Textbox(label="文献材料约束（可选）")
                    year_from = gr.Number(value=2020, label="文献起始年份")
                    limit = gr.Slider(5, 30, value=15, step=1, label="文献候选数")
                with gr.Accordion(
                    "高级信息",
                    open=False,
                    elem_classes=["advanced-box"],
                ):
                    expert_mode = gr.Checkbox(label="专家审计模式", value=False)
                    trace = gr.HTML(_trace(MODE_DATABASE, "等待初始化"))
                    audit_panel = gr.Markdown(visible=False)
            with gr.Column(scale=5, min_width=0, elem_id="chat-shell"):
                chatbot = gr.Chatbot(
                    label="",
                    show_label=False,
                    height=470,
                    layout="bubble",
                    elem_id="multi-chat",
                    placeholder=(
                        "你可以直接询问材料数据库、检索文献，"
                        "也可以上传 PDF 阅读，或上传 CSV/JSON/XLSX 做数据分析。"
                    ),
                )
                with gr.Accordion(
                    "数据分析工具（可选）", open=False, visible=False
                ) as data_result_panel:
                    gr.Markdown("### 当前数据集")
                    data_summary = gr.Markdown("上传数据后将在这里显示自动 EDA。")
                    with gr.Accordion("查看字段和数据预览", open=False):
                        data_fields = gr.Dataframe(
                            headers=[
                                "字段",
                                "类型",
                                "有效值",
                                "缺失值",
                                "唯一值",
                                "提示",
                            ],
                            datatype=[
                                "str",
                                "str",
                                "number",
                                "number",
                                "number",
                                "str",
                            ],
                            value=[],
                            interactive=False,
                            wrap=True,
                        )
                        data_preview = gr.JSON(value=None)
                    with gr.Accordion("高级分析设置", open=False):
                        data_columns = gr.Dropdown(
                            choices=[],
                            multiselect=True,
                            label="数值字段",
                        )
                        with gr.Row():
                            data_x = gr.Dropdown(choices=[], label="X / 响应字段")
                            data_y = gr.Dropdown(choices=[], label="Y 字段")
                            data_group = gr.Dropdown(
                                choices=[], label="分组字段（可选）"
                            )
                        data_groups = gr.Dropdown(
                            choices=[],
                            multiselect=True,
                            label="参与比较的组",
                        )
                        with gr.Row():
                            data_statistical_method = gr.Dropdown(
                                choices=[
                                    "student_t",
                                    "welch_t",
                                    "paired_t",
                                    "mann_whitney",
                                    "anova",
                                    "kruskal_wallis",
                                ],
                                value="welch_t",
                                label="统计检验",
                            )
                            data_correlation_method = gr.Dropdown(
                                choices=["pearson", "spearman"],
                                value="pearson",
                                label="相关方法",
                            )
                            data_plot_type = gr.Dropdown(
                                choices=[
                                    "histogram",
                                    "boxplot",
                                    "scatter",
                                    "line",
                                    "bar",
                                    "heatmap",
                                ],
                                value="scatter",
                                label="图表类型",
                            )
                        data_export_format = gr.Radio(
                            choices=["CSV", "JSON"],
                            value="CSV",
                            label="导出格式",
                        )
                        with gr.Row():
                            data_key = gr.Dropdown(
                                choices=[],
                                label="去重键字段",
                                scale=3,
                            )
                            data_deduplicate_button = gr.Button(
                                "按键去重并新建数据集",
                                size="sm",
                                scale=2,
                            )
                        with gr.Row():
                            data_eda_button = gr.Button("一键 EDA", size="sm")
                            data_describe_button = gr.Button("描述统计", size="sm")
                            data_correlation_button = gr.Button("相关性", size="sm")
                            data_test_button = gr.Button("显著性检验", size="sm")
                        with gr.Row():
                            data_plot_button = gr.Button("生成图表", size="sm")
                            data_report_button = gr.Button("生成报告", size="sm")
                            data_export_button = gr.Button("导出数据", size="sm")
                        data_action_result = gr.Markdown("尚未执行高级操作。")
                        data_plot_preview = gr.Image(label="图表预览", type="filepath")
                        data_download = gr.File(label="报告或数据下载")
                    with gr.Accordion("跨 Agent 数据导入", open=False):
                        cross_source_kind = gr.Radio(
                            choices=["materials_database", "literature"],
                            value="materials_database",
                            label="来源",
                        )
                        cross_source_id = gr.Textbox(
                            label="query_id / document_id（留空使用最近结果）"
                        )
                        cross_import_button = gr.Button("导入并执行只读 EDA", size="sm")
                with (
                    gr.Group(visible=False) as literature_result_panel,
                    gr.Accordion("文献检索结果", open=True),
                ):
                    literature_table = gr.Dataframe(
                        headers=["#", "相关性", "年份", "论文题目", "DOI"],
                        datatype=["number", "str", "str", "str", "str"],
                        value=[],
                        interactive=False,
                        wrap=True,
                    )
                    literature_details = gr.Markdown(
                        "检索完成后，可在这里分页查看论文详情。"
                    )
                    with gr.Row():
                        previous_page = gr.Button("← 上一页", size="sm")
                        literature_page_label = gr.Markdown("第 0 / 0 页")
                        next_page = gr.Button("下一页 →", size="sm")
                from materials_screening.master.figure_evidence_ui import (
                    mount_figure_review,
                )

                figure_refresh, figure_outputs = mount_figure_review(
                    session_state, chatbot, lambda: _master_runner
                )
                from materials_screening.master.preview_retry_ui import (
                    mount_preview_retry,
                )

                preview_refresh, preview_outputs = mount_preview_retry(
                    session_state, lambda: _master_runner
                )
                attachment_status = gr.Markdown(
                    "📎 尚未上传附件", elem_classes=["attachment-status"]
                )
                with gr.Row(elem_id="composer"):
                    files = gr.UploadButton(
                        "📎 上传附件",
                        file_count="multiple",
                        file_types=[".pdf", ".csv", ".json", ".xlsx"],
                        type="filepath",
                        elem_id="pdf-upload",
                    )
                    clear_files_button = gr.Button(
                        "× 清除", size="sm", scale=0, elem_id="clear-upload"
                    )
                    message = gr.Textbox(
                        label="",
                        show_label=False,
                        placeholder="输入你的问题……",
                        scale=8,
                        lines=1,
                        max_lines=6,
                    )
                    send_button = gr.Button(
                        "↑", variant="primary", scale=0, elem_id="send-button"
                    )
                    stop_button = gr.Button(
                        "■", variant="stop", scale=0, elem_id="stop-button"
                    )
            with gr.Column(scale=1, min_width=300, elem_id="right-sidebar"):
                gr.Markdown("### 功能示例")
                for plugin in UI_PLUGINS.plugins():
                    spec = plugin.spec
                    gr.Markdown(f"**{spec.display_name}**")
                    for prompt in spec.quick_prompts:
                        quick_buttons.append(
                            (
                                gr.Button(
                                    QUICK_PROMPT_LABELS.get(prompt, prompt),
                                    size="sm",
                                    elem_classes=["quick-chip"],
                                ),
                                prompt,
                            )
                        )
        inputs = [
            mode,
            message,
            chatbot,
            files,
            session_state,
            material,
            year_from,
            limit,
            expert_mode,
        ]
        outputs = [
            chatbot,
            trace,
            session_state,
            message,
            files,
            audit_panel,
            attachment_status,
            literature_table,
            literature_details,
            literature_page_label,
            literature_result_panel,
            data_fields,
            data_preview,
            data_summary,
            data_result_panel,
            data_columns,
            data_x,
            data_y,
            data_group,
            data_key,
        ]
        send_event = send_button.click(_dispatch_ui, inputs, outputs)
        submit_event = message.submit(_dispatch_ui, inputs, outputs)
        send_event.then(figure_refresh, session_state, figure_outputs)
        send_event.then(preview_refresh, session_state, preview_outputs)
        submit_event.then(figure_refresh, session_state, figure_outputs)
        submit_event.then(preview_refresh, session_state, preview_outputs)
        stop_button.click(fn=None, cancels=[send_event, submit_event])
        data_group.change(
            _data_group_choices,
            [session_state, data_group],
            data_groups,
        )
        chatbot.option_select(
            _answer_data_chat_option,
            [chatbot, session_state],
            [chatbot, session_state],
        )
        data_action_inputs = [
            session_state,
            data_columns,
            data_x,
            data_y,
            data_group,
            data_groups,
            data_statistical_method,
            data_plot_type,
            data_correlation_method,
            data_export_format,
        ]
        data_action_outputs = [
            data_action_result,
            session_state,
            data_plot_preview,
            data_download,
        ]
        for button, action in (
            (data_eda_button, "eda"),
            (data_describe_button, "describe"),
            (data_correlation_button, "correlation"),
            (data_test_button, "test"),
            (data_plot_button, "plot"),
            (data_report_button, "report"),
            (data_export_button, "export"),
        ):
            button.click(
                partial(_run_data_action, action),
                data_action_inputs,
                data_action_outputs,
            )
        data_deduplicate_button.click(
            _run_deduplicate_action,
            [session_state, data_key],
            [
                data_action_result,
                session_state,
                data_fields,
                data_preview,
                data_summary,
                data_result_panel,
                data_columns,
                data_x,
                data_y,
                data_group,
                data_key,
            ],
        )
        cross_import_button.click(
            _import_cross_agent_dataset,
            [cross_source_kind, cross_source_id, session_state],
            [
                data_action_result,
                session_state,
                data_fields,
                data_preview,
                data_summary,
                data_result_panel,
                data_columns,
                data_x,
                data_y,
                data_group,
                data_key,
            ],
        )
        init_button.click(
            _initialize_safe, [provider, repository], [init_status, trace]
        )

        def restore_saved(conversation_id, expert):
            return (*_restore_conversation_ui(conversation_id, expert), MODE_AUTO)

        def restore_on_load(conversation_id, expert):
            if not conversation_id.strip():
                return tuple(gr.skip() for _ in [*outputs, mode])
            return restore_saved(conversation_id, expert)

        restore_event = restore_button.click(
            restore_saved, [restore_id, expert_mode], [*outputs, mode]
        )
        restore_event.then(figure_refresh, session_state, figure_outputs)
        restore_event.then(preview_refresh, session_state, preview_outputs)
        initialized = demo.load(
            _initialize_safe, [provider, repository], [init_status, trace]
        )
        restored = initialized.then(
            restore_on_load, [restore_id, expert_mode], [*outputs, mode]
        )
        restored.then(figure_refresh, session_state, figure_outputs)
        restored.then(preview_refresh, session_state, preview_outputs)
        new_event = new_button.click(
            _new_conversation_ui,
            expert_mode,
            [
                chatbot,
                trace,
                session_state,
                message,
                files,
                audit_panel,
                attachment_status,
                literature_table,
                literature_details,
                literature_page_label,
                literature_result_panel,
                data_fields,
                data_preview,
                data_summary,
                data_result_panel,
                data_columns,
                data_x,
                data_y,
                data_group,
                data_key,
            ],
        )
        new_event.then(figure_refresh, session_state, figure_outputs)
        new_event.then(preview_refresh, session_state, preview_outputs)
        files.upload(
            _attachment_markdown,
            [files, session_state],
            attachment_status,
        )
        clear_files_button.click(
            _clear_attachments,
            session_state,
            [files, session_state, attachment_status],
        )
        previous_page.click(
            lambda state: _turn_literature_page(state, -1),
            session_state,
            [
                literature_table,
                literature_details,
                literature_page_label,
                session_state,
            ],
        )
        next_page.click(
            lambda state: _turn_literature_page(state, 1),
            session_state,
            [
                literature_table,
                literature_details,
                literature_page_label,
                session_state,
            ],
        )
        expert_mode.change(
            lambda enabled: gr.update(visible=enabled),
            expert_mode,
            audit_panel,
        )
        for button, prompt in quick_buttons:
            button.click(lambda value=prompt: value, None, message)
    return demo


def main() -> None:
    preferred = int(os.getenv("GRADIO_SERVER_PORT", "8501"))
    port = _available_port(preferred)
    if port != preferred:
        print(f"端口 {preferred} 已占用，自动使用 http://127.0.0.1:{port}")
    _create_ui().launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
    )


if __name__ == "__main__":
    main()
