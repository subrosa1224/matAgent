"""Direct (stage 1) / workflow (stage 3) / agent (stage 3.5) equivalence.

The same natural-language query and the same mock material fixtures must
produce identical candidate ids, ranks, total scores, score breakdowns and
validation results through all three execution paths. The agent is only
allowed to change wording, conversation ids and evidence/run metadata; the
underlying screening facts must not change. Assertions are strict: no
tolerances and no relaxed field subsets.
"""

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pymatgen.core.structure import Structure

pytest.importorskip("langgraph")

from materials_screening.agent.conversation_store import (  # noqa: E402
    SqliteConversationStore,
)
from materials_screening.agent.mock_model import (  # noqa: E402
    MockAgentTurn,
    MockMaterialAgentModel,
    MockToolCall,
)
from materials_screening.agent.runner import MaterialAgentRunner  # noqa: E402
from materials_screening.agent.settings import AgentSettings  # noqa: E402
from materials_screening.agent.tool_registry import AgentToolRegistry  # noqa: E402
from materials_screening.agent_tools import (  # noqa: E402
    CompareRankedMaterialsTool,
    GetScreeningResultTool,
    GetWorkflowHistoryTool,
    GetWorkflowStatusTool,
    RunScreeningWorkflowTool,
)
from materials_screening.agent_tools.result_reader import (  # noqa: E402
    FileWorkflowResultReader,
)
from materials_screening.llm.mock_provider import MockStructuredProvider  # noqa: E402
from materials_screening.models import (  # noqa: E402
    FloatRange,
    MaterialRecord,
    PropertyProvenance,
    PropertyValueType,
    ScreeningRequest,
)
from materials_screening.planner.models import (  # noqa: E402
    DraftStatus,
    EnergyUnit,
    HullUnit,
    PlannerDraft,
)
from materials_screening.planner.service import PlannerService  # noqa: E402
from materials_screening.planner.settings import Settings  # noqa: E402
from materials_screening.repositories.materials_project import (  # noqa: E402
    PROVENANCE_PROPERTIES,
)
from materials_screening.repositories.mock import MockMaterialsRepository  # noqa: E402
from materials_screening.services.export_service import ExportService  # noqa: E402
from materials_screening.services.filter_service import FilterService  # noqa: E402
from materials_screening.services.ranking_service import RankingService  # noqa: E402
from materials_screening.services.screening_service import (  # noqa: E402
    ScreeningService,
)
from materials_screening.services.validation_service import (  # noqa: E402
    ValidationService,
)
from materials_screening.workflow.artifact_store import (  # noqa: E402
    FileRunArtifactStore,
)
from materials_screening.workflow.checkpointer import (  # noqa: E402
    create_checkpointer_handle,
)
from materials_screening.workflow.context import WorkflowContext  # noqa: E402
from materials_screening.workflow.export_adapter import (  # noqa: E402
    WorkflowExportAdapter,
)
from materials_screening.workflow.input_output import WorkflowInput  # noqa: E402
from materials_screening.workflow.runner import WorkflowRunner  # noqa: E402
from materials_screening.workflow.settings import WorkflowSettings  # noqa: E402
from materials_screening.workflow.state import ArtifactRef  # noqa: E402

QUERY = "带隙 1 到 2 eV，hull 0 到 0.5 eV，最多 10 个候选"
FIXED_CLOCK = datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


def _request() -> ScreeningRequest:
    return ScreeningRequest(
        band_gap_ev=FloatRange(min=1.0, max=2.0),
        energy_above_hull_ev_atom=FloatRange(min=0.0, max=0.5),
        is_metal=None,
        limit=10,
    )


def _structure_dict() -> dict[str, object]:
    structure = Structure(
        lattice=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Fe", "O"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    return structure.as_dict()


def _record(
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
        "structure_dict": _structure_dict(),
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


def _records() -> tuple[MaterialRecord, ...]:
    return (
        _record(
            "mp-1",
            band_gap=1.5,
            hull=0.0,
            formation=-2.0,
            density=3.5,
            direct_gap=True,
            stable=True,
        ),
        _record(
            "mp-2",
            band_gap=1.6,
            hull=0.1,
            formation=-1.8,
            density=3.2,
            direct_gap=False,
            stable=True,
        ),
        _record(
            "mp-3",
            band_gap=1.4,
            hull=0.2,
            formation=-1.5,
            density=2.9,
            direct_gap=False,
            stable=False,
        ),
        _record(
            "mp-4",
            band_gap=1.7,
            hull=0.05,
            formation=-1.9,
            density=3.0,
            direct_gap=True,
            stable=True,
        ),
    )


def _parse_csv(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text)))


def _direct_result(
    tmp_path: Path,
    request: ScreeningRequest,
    records: tuple[MaterialRecord, ...],
) -> tuple[dict[str, Any], list[list[str]]]:
    service = ScreeningService(
        repository=MockMaterialsRepository(records=records),
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=ExportService(),
        clock=lambda: FIXED_CLOCK,
    )
    run_output = service.run(request, tmp_path / "direct", include_cif=False)
    result = run_output.result
    csv_path = tmp_path / "direct" / result.metadata.run_id / "candidates.csv"
    return result.model_dump(mode="json"), _parse_csv(
        csv_path.read_text(encoding="utf-8-sig")
    )


class _FakePlanner:
    def parse(self, query: str) -> object:
        raise AssertionError("planner must not be called in request mode")


class _IdGenerator:
    def __init__(self) -> None:
        self.count = 0

    def new_id(self) -> str:
        self.count += 1
        return f"id-{self.count}"


class _AgentIdGenerator:
    def __init__(self) -> None:
        self.count = 0

    def new_id(self) -> str:
        self.count += 1
        return f"id_{self.count}"


def _workflow_result(
    tmp_path: Path,
    request: ScreeningRequest,
    records: tuple[MaterialRecord, ...],
) -> tuple[dict[str, Any], list[list[str]]]:
    run_root = tmp_path / "workflow"
    context = WorkflowContext(
        planner_service=_FakePlanner(),
        materials_repository=MockMaterialsRepository(records=records),
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=WorkflowExportAdapter(run_root),
        artifact_store=FileRunArtifactStore(run_root),
        clock=lambda: FIXED_CLOCK,
        id_generator=_IdGenerator(),
    )
    runner = WorkflowRunner(
        settings=WorkflowSettings(_env_file=None),
        context=context,
        checkpointer=create_checkpointer_handle(backend="memory"),
    )
    output = runner.run(
        WorkflowInput(
            request=request.model_dump(mode="json"),
            output_root=str(run_root),
            export_cif=False,
        )
    )
    store = FileRunArtifactStore(run_root)
    manifest_path = run_root / output.thread_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_ref = ArtifactRef.model_validate(manifest["artifacts"]["screening_result"])
    result_payload = store.get_json(result_ref)
    csv_path = run_root / output.thread_id / "exports" / "candidates.csv"
    runner.close()
    return result_payload, _parse_csv(csv_path.read_text(encoding="utf-8-sig"))


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
    provider = MockStructuredProvider(fixtures={QUERY: draft})
    return PlannerService(
        settings=Settings(
            _env_file=None,
            llm_provider="mock",
            planner_prompt_version="planner-v2",
            planner_schema_version="planner-draft-v1",
        ),
        provider=provider,
    )


def _agent_tool_registry() -> AgentToolRegistry:
    return AgentToolRegistry(
        (
            RunScreeningWorkflowTool(),
            GetWorkflowStatusTool(),
            GetWorkflowHistoryTool(),
            GetScreeningResultTool(),
            CompareRankedMaterialsTool(),
        )
    )


def _agent_result(
    tmp_path: Path,
    records: tuple[MaterialRecord, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], list[list[str]], str]:
    """Run the agent with the same query/fixtures and return the facts."""
    from langgraph.checkpoint.memory import InMemorySaver

    monkeypatch.chdir(tmp_path)
    run_root = Path("data/workflow_runs")
    workflow_context = WorkflowContext(
        planner_service=_planner_service(),
        materials_repository=MockMaterialsRepository(records=records),
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=WorkflowExportAdapter(run_root),
        artifact_store=FileRunArtifactStore(run_root),
        clock=lambda: FIXED_CLOCK,
        id_generator=_IdGenerator(),
    )
    workflow_runner = WorkflowRunner(
        settings=WorkflowSettings(_env_file=None),
        context=workflow_context,
        checkpointer=create_checkpointer_handle(backend="memory"),
    )

    run_call = MockToolCall(
        call_id="call_1",
        name="run_screening_workflow",
        arguments=json.dumps({"query": QUERY}, ensure_ascii=False),
    )
    final_draft = {
        "status": "completed",
        "answer": "筛选完成，推荐候选与直连模式一致。",
        "active_workflow_thread_id": None,
        "referenced_material_ids": [],
        # prepare consumes id_1/id_2; the run tool evidence is id_3.
        "evidence_ids": ["id_3"],
        "warnings": [],
        "follow_up_question": None,
    }
    model = MockMaterialAgentModel(
        script=[
            MockAgentTurn(tool_calls=(run_call,)),
            MockAgentTurn.final_draft(final_draft),
        ]
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
        tool_registry=_agent_tool_registry(),
        agent_model=model,
        checkpointer=InMemorySaver(),
        clock=lambda: FIXED_CLOCK,
        id_generator=_AgentIdGenerator(),
    )
    try:
        result = agent_runner.ask(message=QUERY, conversation_id="agent_conv")
    finally:
        store.close()
        workflow_runner.close()

    assert result.status == "completed", result.error
    thread_id = result.active_workflow_thread_id
    assert thread_id is not None
    run_dir = run_root / thread_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    result_ref = ArtifactRef.model_validate(manifest["artifacts"]["screening_result"])
    payload = FileRunArtifactStore(run_root).get_json(result_ref)
    csv_text = (run_dir / "exports" / "candidates.csv").read_text(encoding="utf-8-sig")
    return payload, _parse_csv(csv_text), result.response_text


def _ranked_facts(payload: dict[str, Any]) -> tuple[list[Any], ...]:
    ranked = payload["ranked_materials"]
    return (
        [item["record"]["material_id"] for item in ranked],
        [item["rank"] for item in ranked],
        [item["total_score"] for item in ranked],
        [item["score_breakdown"] for item in ranked],
    )


class TestDirectWorkflowAgentEquivalence:
    def test_candidate_ids_ranks_scores_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        request = _request()
        records = _records()
        direct, _ = _direct_result(tmp_path, request, records)
        workflow, _ = _workflow_result(tmp_path, request, records)
        agent, _, _ = _agent_result(tmp_path, records, monkeypatch)

        direct_facts = _ranked_facts(direct)
        workflow_facts = _ranked_facts(workflow)
        agent_facts = _ranked_facts(agent)

        assert agent_facts == direct_facts
        assert agent_facts == workflow_facts

    def test_validation_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        request = _request()
        records = _records()
        direct, _ = _direct_result(tmp_path, request, records)
        workflow, _ = _workflow_result(tmp_path, request, records)
        agent, _, _ = _agent_result(tmp_path, records, monkeypatch)

        assert agent["validation"] == direct["validation"] == workflow["validation"]
        assert agent["validation"]["passed"] is True

    def test_counts_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        request = _request()
        records = _records()
        direct, _ = _direct_result(tmp_path, request, records)
        workflow, _ = _workflow_result(tmp_path, request, records)
        agent, _, _ = _agent_result(tmp_path, records, monkeypatch)

        assert (
            agent["retrieved_count"]
            == direct["retrieved_count"]
            == workflow["retrieved_count"]
            == 4
        )
        assert (
            agent["passed_filter_count"]
            == direct["passed_filter_count"]
            == workflow["passed_filter_count"]
            == 4
        )
        assert len(agent["ranked_materials"]) == len(direct["ranked_materials"])

    def test_csv_core_data_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        request = _request()
        records = _records()
        direct, direct_csv = _direct_result(tmp_path, request, records)
        workflow, workflow_csv = _workflow_result(tmp_path, request, records)
        agent, agent_csv, _ = _agent_result(tmp_path, records, monkeypatch)

        assert agent_csv == direct_csv == workflow_csv
        assert agent_csv[0] == direct_csv[0] == workflow_csv[0]

    def test_agent_only_changes_presentation_and_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        request = _request()
        records = _records()
        direct, _ = _direct_result(tmp_path, request, records)
        workflow, _ = _workflow_result(tmp_path, request, records)
        agent, _, response_text = _agent_result(tmp_path, records, monkeypatch)

        # The agent's wording is its own presentation.
        assert response_text == "筛选完成，推荐候选与直连模式一致。"
        # Run/thread and evidence metadata are allowed to differ.
        assert agent["metadata"]["run_id"] != direct["metadata"]["run_id"]
        assert agent["metadata"]["run_id"] != workflow["metadata"]["run_id"]
        # The candidate facts themselves are already asserted identical above;
        # this test pins that nothing beyond presentation/metadata changed.
        assert _ranked_facts(agent) == _ranked_facts(direct)

    def test_agent_planner_resolves_to_same_request(self) -> None:
        result = _planner_service().parse(QUERY)

        assert result.request == _request()
