"""Offline stage-3.5 acceptance: mock agent eval + demos (S3.5-M8).

Run: ``uv run python scripts/run_agent_eval_acceptance.py``

Everything is fully offline: the agent model is a compliant scripted mock, the
workflow runs against a mock planner and mock repository, and all data lands
in a temporary directory. No DeepSeek, Materials Project or other network.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from materials_screening.agent.conversation_store import SqliteConversationStore
from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentModel,
    MaterialAgentRequest,
    MaterialAgentResponse,
)
from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
)
from materials_screening.agent.runner import MaterialAgentRunner
from materials_screening.agent.settings import AgentSettings
from materials_screening.agent.tool_registry import (
    ALLOWED_TOOL_NAMES,
    AgentToolRegistry,
)
from materials_screening.agent_tools import (
    CompareRankedMaterialsTool,
    GetScreeningResultTool,
    GetWorkflowHistoryTool,
    GetWorkflowStatusTool,
    RunScreeningWorkflowTool,
)
from materials_screening.agent_tools.result_reader import FileWorkflowResultReader
from materials_screening.evaluation.agent_evaluator import (
    AgentEvaluator,
    RecordingAgentModel,
    load_agent_cases,
)
from materials_screening.evaluation.agent_report import metrics_to_markdown
from materials_screening.llm.mock_provider import MockDeepSeekProvider
from materials_screening.models import (
    MaterialRecord,
    PropertyProvenance,
    PropertyValueType,
)
from materials_screening.planner.models import (
    DraftStatus,
    EnergyUnit,
    HullUnit,
    PlannerDraft,
)
from materials_screening.planner.service import PlannerService
from materials_screening.planner.settings import Settings
from materials_screening.repositories.materials_project import (
    PROVENANCE_PROPERTIES,
)
from materials_screening.repositories.mock import MockMaterialsRepository
from materials_screening.services.filter_service import FilterService
from materials_screening.services.ranking_service import RankingService
from materials_screening.services.validation_service import ValidationService
from materials_screening.workflow.artifact_store import FileRunArtifactStore
from materials_screening.workflow.checkpointer import create_checkpointer_handle
from materials_screening.workflow.context import WorkflowContext
from materials_screening.workflow.export_adapter import WorkflowExportAdapter
from materials_screening.workflow.runner import WorkflowRunner
from materials_screening.workflow.settings import WorkflowSettings

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = REPO_ROOT / "tests" / "eval" / "single_agent_eval.jsonl"
FIXED_QUERY = "筛选半导体材料"
FIXED_CLOCK = datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


def _records() -> tuple[MaterialRecord, ...]:
    def record(
        material_id: str,
        *,
        band_gap: float,
        hull: float,
        formation: float,
        density: float,
        direct_gap: bool,
        stable: bool,
    ) -> MaterialRecord:
        values: dict[str, object] = {
            "source": "mock",
            "material_id": material_id,
            "formula_pretty": material_id,
            "elements": ("Si", "O"),
            "chemsys": "Si-O",
            "band_gap_ev": band_gap,
            "energy_above_hull_ev_atom": hull,
            "formation_energy_ev_atom": formation,
            "density_g_cm3": density,
            "is_metal": False,
            "is_gap_direct": direct_gap,
            "is_stable": stable,
            "theoretical": False,
            "symmetry": {
                "crystal_system": "Cubic",
                "symbol": "Fm-3m",
                "number": 225,
            },
            "structure_dict": None,
        }
        provenance = tuple(
            PropertyProvenance(
                property_name=name,
                source="materials_project",
                source_material_id=material_id,
                value_type=PropertyValueType.DFT_CALCULATED,
                database_version="fixture-v1",
                retrieved_at=FIXED_CLOCK,
            )
            for name in PROVENANCE_PROPERTIES
            if values.get(name) is not None
        )
        return MaterialRecord.model_validate({**values, "provenance": provenance})

    return (
        record(
            "mp-1",
            band_gap=1.5,
            hull=0.0,
            formation=-2.0,
            density=3.5,
            direct_gap=True,
            stable=True,
        ),
        record(
            "mp-2",
            band_gap=1.6,
            hull=0.1,
            formation=-1.8,
            density=3.2,
            direct_gap=False,
            stable=True,
        ),
        record(
            "mp-3",
            band_gap=1.4,
            hull=0.2,
            formation=-1.5,
            density=2.9,
            direct_gap=False,
            stable=False,
        ),
        record(
            "mp-4",
            band_gap=1.7,
            hull=0.05,
            formation=-1.9,
            density=3.0,
            direct_gap=True,
            stable=True,
        ),
    )


def _planner_service() -> PlannerService:
    draft = PlannerDraft.model_validate(
        {
            "status": DraftStatus.EXTRACTED,
            "required_elements": [],
            "excluded_elements": [],
            "chemsys": None,
            "formula": None,
            "band_gap_min": 1.0,
            "band_gap_max": 2.0,
            "band_gap_unit": EnergyUnit.EV,
            "hull_min": 0.0,
            "hull_max": 0.5,
            "hull_unit": HullUnit.EV_PER_ATOM,
            "density_min": None,
            "density_max": None,
            "density_unit": "unspecified",
            "crystal_system": None,
            "spacegroup_numbers": [],
            "is_metal": None,
            "is_stable": None,
            "theoretical": None,
            "target_band_gap": None,
            "target_band_gap_unit": EnergyUnit.UNSPECIFIED,
            "limit": 10,
            "ambiguities": [],
            "unsupported_requirements": [],
            "conflicts": [],
            "assumptions": [],
            "clarification_question": "",
            "evidence": [],
        }
    )
    return PlannerService(
        settings=Settings(
            _env_file=None,
            llm_provider="mock",
            planner_prompt_version="planner-v2",
            planner_schema_version="planner-draft-v1",
            deepseek_reasoning_effort="none",
        ),
        provider=MockDeepSeekProvider(fixtures={FIXED_QUERY: draft}),
    )


def _registry() -> AgentToolRegistry:
    return AgentToolRegistry(
        (
            RunScreeningWorkflowTool(),
            GetWorkflowStatusTool(),
            GetWorkflowHistoryTool(),
            GetScreeningResultTool(),
            CompareRankedMaterialsTool(),
        )
    )


class _IdGenerator:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.count = 0

    def new_id(self) -> str:
        self.count += 1
        return f"{self.prefix}{self.count}"


class _CompliantMock:
    """Scripted mock that follows the eval expectations.

    Actions: ``("tool", name, arguments)`` emits one function call;
    ``("draft", kind)`` emits a final draft where ``kind`` is one of
    ``evidence`` (references the latest tool evidence from the transcript),
    ``concept`` (general knowledge, no evidence), ``refusal`` (safe refusal),
    ``no_task`` (honest no-active-task answer) or ``clarify``.
    """

    def __init__(self, actions: list[tuple[Any, ...]]) -> None:
        self._actions = list(actions)
        self._index = 0

    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        action = self._actions[self._index]
        self._index += 1
        if action[0] == "tool":
            _, name, arguments = action
            return MaterialAgentResponse(
                status=AgentModelStatus.COMPLETED,
                output_items=(
                    AgentFunctionCallItem(
                        call_id=f"call_{self._index}",
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                ),
                request_id="mock",
                provider="mock",
                model="mock",
            )
        draft = self._draft(request, str(action[1]))
        return MaterialAgentResponse(
            status=AgentModelStatus.COMPLETED,
            output_items=(
                AgentMessageItem(
                    role="assistant",
                    content=json.dumps(draft, ensure_ascii=False),
                ),
            ),
            request_id="mock",
            provider="mock",
            model="mock",
        )

    @staticmethod
    def _draft(request: MaterialAgentRequest, kind: str) -> dict[str, Any]:
        if kind == "clarify":
            return {
                "status": "needs_user_input",
                "answer": "请提供更具体的筛选条件。",
                "active_workflow_thread_id": None,
                "referenced_material_ids": [],
                "evidence_ids": [],
                "warnings": [],
                "follow_up_question": "请提供更具体的筛选条件。",
            }
        answers = {
            "evidence": "筛选完成，推荐候选与直连模式一致。",
            "concept": "无机半导体是材料科学的重要领域。",
            "refusal": "无法访问该任务。",
            "no_task": "当前没有进行中的任务。",
        }
        evidence_ids: list[str] = []
        if kind == "evidence":
            for item in reversed(request.input_items):
                if not isinstance(item, AgentFunctionOutputItem):
                    continue
                try:
                    envelope = json.loads(item.output)
                except json.JSONDecodeError:
                    continue
                if isinstance(envelope, dict) and envelope.get("evidence_id"):
                    evidence_ids.append(str(envelope["evidence_id"]))
                    break
        return {
            "status": "completed",
            "answer": answers.get(kind, answers["concept"]),
            "active_workflow_thread_id": None,
            "referenced_material_ids": [],
            "evidence_ids": evidence_ids,
            "warnings": [],
            "follow_up_question": None,
        }


def _make_agent_runner(
    work_dir: Path,
    agent_model: MaterialAgentModel,
) -> tuple[MaterialAgentRunner, SqliteConversationStore, WorkflowRunner]:
    run_root = Path("data/workflow_runs")
    workflow_context = WorkflowContext(
        planner_service=_planner_service(),
        materials_repository=MockMaterialsRepository(records=_records()),
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=WorkflowExportAdapter(run_root),
        artifact_store=FileRunArtifactStore(run_root),
        clock=lambda: FIXED_CLOCK,
        id_generator=_IdGenerator("w-"),
    )
    workflow_runner = WorkflowRunner(
        settings=WorkflowSettings(_env_file=None),
        context=workflow_context,
        checkpointer=create_checkpointer_handle(backend="memory"),
    )
    store = SqliteConversationStore(
        Path("data/agent_conversations.sqlite"),
        clock=lambda: FIXED_CLOCK,
    )
    agent_runner = MaterialAgentRunner(
        settings=AgentSettings(_env_file=None),
        store=store,
        workflow_runner=workflow_runner,
        workflow_result_reader=FileWorkflowResultReader(run_root),
        tool_registry=_registry(),
        agent_model=agent_model,
        checkpointer=InMemorySaver(),
        clock=lambda: FIXED_CLOCK,
        id_generator=_IdGenerator("a-"),
    )
    return agent_runner, store, workflow_runner


def _print_agent_result(result: Any, label: str) -> None:
    print(f"\n[{label}] status={result.status} final={result.final_status}")
    print(f"  tools={list(result.selected_tools)} evidence={list(result.evidence_ids)}")
    print(f"  active_thread={result.active_workflow_thread_id}")
    print(f"  agent> {result.response_text}")
    if result.error is not None:
        print(f"  error={result.error.get('code')}")


def _run_mock_eval(work_dir: Path) -> None:
    print("\n==== Mock Agent eval (representative cases) ====")
    all_cases = load_agent_cases(EVAL_FILE)
    by_id = {case.id: case for case in all_cases}
    selected_ids = (
        "agent_screening_001",
        "agent_screening_007",
        "agent_status_001",
        "agent_compare_001",
        "agent_general_001",
        "agent_clarify_001",
        "agent_safety_001",
    )
    cases = [by_id[case_id] for case_id in selected_ids]
    actions = []
    for case in cases:
        for turn in case.turns:
            expected = turn.expected_tool
            if expected is None:
                if turn.expected_final_status == "needs_user_input":
                    actions.append(("draft", "clarify"))
                else:
                    actions.append(("draft", "concept"))
            elif expected == "run_screening_workflow":
                actions.append(
                    ("tool", "run_screening_workflow", {"query": FIXED_QUERY})
                )
                actions.append(("draft", "evidence"))
            elif expected == "get_workflow_status":
                actions.append(("tool", "get_workflow_status", {}))
                actions.append(("draft", "evidence"))
            elif expected == "get_workflow_history":
                actions.append(("tool", "get_workflow_history", {}))
                actions.append(("draft", "evidence"))
            elif expected == "get_screening_result":
                actions.append(("tool", "get_screening_result", {}))
                actions.append(("draft", "evidence"))
            elif expected == "compare_ranked_materials":
                actions.append(
                    (
                        "tool",
                        "compare_ranked_materials",
                        {"material_ids": ["mp-1", "mp-2"]},
                    )
                )
                actions.append(("draft", "evidence"))

    harnesses: list[tuple[SqliteConversationStore, WorkflowRunner]] = []

    def factory(recorder: RecordingAgentModel) -> MaterialAgentRunner:
        recorder.attach(_CompliantMock(actions))
        runner, store, workflow_runner = _make_agent_runner(work_dir, recorder)
        harnesses.append((store, workflow_runner))
        return runner

    evaluator = AgentEvaluator(factory)
    metrics, results = evaluator.evaluate_with_details(cases)
    print(metrics_to_markdown(metrics))
    failed = [
        result.id
        for result in results
        if result.error_type is not None
        or (result.multi_turn and result.multi_turn_success is False)
        or any(
            not observation.tool_selection_ok
            or observation.final_status != observation.expected_final_status
            or observation.unauthorized_tool_used
            for observation in result.turns
        )
    ]
    print(f"Failed cases: {failed if failed else 'none'}")
    for store, workflow_runner in harnesses:
        store.close()
        workflow_runner.close()


def _run_demos(work_dir: Path) -> None:
    print("\n==== Demos ====")

    single = _CompliantMock(
        [
            ("tool", "run_screening_workflow", {"query": FIXED_QUERY}),
            ("draft", "evidence"),
        ]
    )
    runner, store, workflow_runner = _make_agent_runner(work_dir, single)
    result = runner.ask(message="寻找不含 Pb 的半导体", conversation_id="demo_single")
    _print_agent_result(result, "single-turn screening")
    store.close()
    workflow_runner.close()

    multi = _CompliantMock(
        [
            ("tool", "run_screening_workflow", {"query": FIXED_QUERY}),
            ("draft", "evidence"),
            ("tool", "get_screening_result", {}),
            ("draft", "evidence"),
        ]
    )
    runner, store, workflow_runner = _make_agent_runner(work_dir, multi)
    first = runner.ask(message="寻找不含 Pb 的半导体", conversation_id="demo_multi")
    second = runner.ask(message="为什么第一名排名最高？", conversation_id="demo_multi")
    _print_agent_result(first, "multi-turn turn 1")
    _print_agent_result(second, "multi-turn result explanation")
    store.close()
    workflow_runner.close()

    no_active = _CompliantMock(
        [
            ("tool", "get_workflow_status", {}),
            ("draft", "no_task"),
        ]
    )
    runner, store, workflow_runner = _make_agent_runner(work_dir, no_active)
    result = runner.ask(message="当前任务状态如何？", conversation_id="demo_no_active")
    _print_agent_result(result, "no active task")
    store.close()
    workflow_runner.close()

    # Cross-conversation rejection: thread_a belongs to demo_owner.
    owner_runner, owner_store, owner_workflow = _make_agent_runner(
        work_dir, _CompliantMock([("draft", "concept")])
    )
    owner_runner.ask(message="开始任务", conversation_id="demo_owner")
    owner_store.link_workflow("demo_owner", "thread_a")
    cross = _CompliantMock(
        [
            ("tool", "get_workflow_status", {"thread_id": "thread_a"}),
            ("draft", "refusal"),
        ]
    )
    runner, store, workflow_runner = _make_agent_runner(work_dir, cross)
    result = runner.ask(message="查看另一个会话的任务", conversation_id="demo_intruder")
    _print_agent_result(result, "cross-conversation rejected")
    for s, w in ((owner_store, owner_workflow), (store, workflow_runner)):
        s.close()
        w.close()


def _confirmations() -> None:
    print("\n==== Confirmations ====")
    print(f"whitelist tools ({len(ALLOWED_TOOL_NAMES)}): {sorted(ALLOWED_TOOL_NAMES)}")
    assert len(ALLOWED_TOOL_NAMES) == 5
    assert "web_search" not in ALLOWED_TOOL_NAMES
    try:
        AgentSettings(_env_file=None, agent_allow_web_search=True)
        raise AssertionError("web search must stay disabled")
    except ValueError:
        print("web search: settings validator rejects enable")
    print("all confirmations passed")


def main() -> None:
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as raw_dir:
        work_dir = Path(raw_dir)
        try:
            os.chdir(work_dir)
            _run_mock_eval(work_dir)
            _run_demos(work_dir)
            _confirmations()
        finally:
            os.chdir(original_cwd)
    print("\nOffline acceptance finished: no network calls were made.")


if __name__ == "__main__":
    main()
