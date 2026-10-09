"""Real end-to-end: DeepSeek Agent -> run -> DeepSeek Planner -> MP -> verified.

Run: ``RUN_REAL_DEEPSEEK_AGENT_TESTS=1 uv run python
scripts/e2e_real_agent_workflow.py``

This makes REAL billing/network requests: the DeepSeek agent model, the
DeepSeek workflow planner and the Materials Project repository. Exactly one
narrow query is executed, sequentially, with no batch and no re-run. All
artifacts, checkpoints, conversation state and the non-sensitive report land
in a temporary directory outside the repository.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from materials_screening.agent.conversation_store import SqliteConversationStore
from materials_screening.agent.deepseek_model import DeepSeekAgentModel
from materials_screening.agent.model_base import MaterialAgentModel
from materials_screening.agent.runner import MaterialAgentRunner
from materials_screening.agent.settings import AgentSettings
from materials_screening.agent.tool_registry import AgentToolRegistry
from materials_screening.agent_tools import (
    CompareRankedMaterialsTool,
    GetScreeningResultTool,
    GetWorkflowHistoryTool,
    GetWorkflowStatusTool,
    RunScreeningWorkflowTool,
)
from materials_screening.agent_tools.result_reader import FileWorkflowResultReader
from materials_screening.llm.deepseek_provider import DeepSeekProvider
from materials_screening.planner.service import PlannerService
from materials_screening.planner.settings import Settings
from materials_screening.repositories.materials_project import (
    MaterialsProjectRepository,
)
from materials_screening.services.filter_service import FilterService
from materials_screening.services.ranking_service import RankingService
from materials_screening.services.validation_service import ValidationService
from materials_screening.workflow.artifact_store import FileRunArtifactStore
from materials_screening.workflow.checkpointer import create_checkpointer_handle
from materials_screening.workflow.context import WorkflowContext
from materials_screening.workflow.export_adapter import WorkflowExportAdapter
from materials_screening.workflow.runner import WorkflowRunner
from materials_screening.workflow.settings import WorkflowSettings
from materials_screening.workflow.state import ArtifactRef

QUERY = "寻找带隙 1.0 到 2.0 eV、不含 Pb 的半导体材料"
MIN_BAND_GAP = 1.0
MAX_BAND_GAP = 2.0


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _IdGenerator:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.count = 0

    def new_id(self) -> str:
        self.count += 1
        return f"{self.prefix}{self.count}"


class _RecordingModel:
    """Records requests/responses; only safe metadata is reported."""

    def __init__(self, inner: MaterialAgentModel) -> None:
        self._inner = inner
        self.requests: list[Any] = []
        self.responses: list[Any] = []

    def generate(self, request: Any) -> Any:
        self.requests.append(request)
        response = self._inner.generate(request)
        self.responses.append(response)
        return response


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


def _extract_run_envelope(recorder: _RecordingModel) -> dict[str, Any]:
    """Parse the run_screening_workflow result envelope from the transcript."""
    for request in reversed(recorder.requests):
        for item in request.input_items:
            if not hasattr(item, "type") or item.type != "function_call_output":
                continue
            try:
                envelope = json.loads(item.output)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(envelope, dict)
                and envelope.get("tool_name") == "run_screening_workflow"
            ):
                output = envelope.get("output")
                if isinstance(output, dict):
                    return output
    return {}


def _check_hard_constraints(payload: dict[str, Any]) -> dict[str, Any]:
    """Verify ranked candidates satisfy the query's hard constraints."""
    ranked = payload.get("ranked_materials", [])
    checks = {
        "validation_passed": payload.get("validation", {}).get("passed") is True,
        "ranked_count": len(ranked),
        "materials": [],
    }
    for item in ranked:
        record = item.get("record", {})
        elements = record.get("elements", [])
        band_gap = record.get("band_gap_ev")
        checks["materials"].append(
            {
                "material_id": record.get("material_id"),
                "band_gap_ev": band_gap,
                "band_gap_in_range": (
                    band_gap is not None and MIN_BAND_GAP <= band_gap <= MAX_BAND_GAP
                ),
                "no_pb": "Pb" not in elements,
            }
        )
    checks["all_band_gaps_in_range"] = all(
        entry["band_gap_in_range"] for entry in checks["materials"]
    )
    checks["no_pb_in_any"] = all(entry["no_pb"] for entry in checks["materials"])
    checks["constraints_satisfied"] = (
        checks["validation_passed"]
        and checks["ranked_count"] > 0
        and checks["all_band_gaps_in_range"]
        and checks["no_pb_in_any"]
    )
    return checks


def main() -> int:
    load_dotenv(Path(".env"))
    if os.getenv("RUN_REAL_DEEPSEEK_AGENT_TESTS") != "1":
        print("refusing: RUN_REAL_DEEPSEEK_AGENT_TESTS=1 is required")
        return 2
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    mp_key = os.getenv("MP_API_KEY", "").strip()
    if not deepseek_key:
        print("refusing: DEEPSEEK_API_KEY is not set")
        return 2
    if not mp_key:
        print("refusing: MP_API_KEY is not set")
        return 2

    print("== acknowledgment ==")
    print("This run makes REAL DeepSeek agent + DeepSeek planner + Materials")
    print("Project requests; exactly one narrow query, sequential, no batch.")
    print(f"query: {QUERY}")

    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as raw_dir:
        work_dir = Path(raw_dir)
        try:
            os.chdir(work_dir)
            run_root = Path("data/workflow_runs")

            planner_settings = Settings(
                _env_file=None,
                llm_provider="deepseek",
                deepseek_api_key=SecretStr(deepseek_key),
                llm_max_attempts=1,
            )
            planner_service = PlannerService(
                settings=planner_settings,
                provider=DeepSeekProvider(planner_settings),
            )
            repository = MaterialsProjectRepository(
                api_key=mp_key,
                max_attempts=1,
            )
            workflow_context = WorkflowContext(
                planner_service=planner_service,
                materials_repository=repository,
                filter_service=FilterService(),
                ranking_service=RankingService(),
                validation_service=ValidationService(),
                export_service=WorkflowExportAdapter(run_root),
                artifact_store=FileRunArtifactStore(run_root),
                clock=_utc_now,
                id_generator=_IdGenerator("w-"),
            )
            workflow_runner = WorkflowRunner(
                settings=WorkflowSettings(_env_file=None),
                context=workflow_context,
                checkpointer=create_checkpointer_handle(backend="memory"),
            )

            agent_settings = AgentSettings(
                _env_file=None, agent_max_model_calls_per_turn=4
            )
            real_model = DeepSeekAgentModel(
                agent_settings,
                api_key=SecretStr(deepseek_key),
            )
            recorder = _RecordingModel(real_model)
            store = SqliteConversationStore(
                Path("data/agent_conversations.sqlite"),
                clock=_utc_now,
            )
            agent_runner = MaterialAgentRunner(
                settings=agent_settings,
                store=store,
                workflow_runner=workflow_runner,
                workflow_result_reader=FileWorkflowResultReader(run_root),
                tool_registry=_registry(),
                agent_model=recorder,
                checkpointer=InMemorySaver(),
                clock=_utc_now,
                id_generator=_IdGenerator("a-"),
            )
            try:
                result = agent_runner.ask(message=QUERY, conversation_id="e2e_real")
            finally:
                store.close()
                workflow_runner.close()

            print("\n== agent result ==")
            print(f"status={result.status} final={result.final_status}")
            print(f"tools={list(result.selected_tools)}")
            print(f"conversation_id={result.conversation_id}")
            print(f"active_workflow_thread_id={result.active_workflow_thread_id}")
            print(f"evidence_ids={list(result.evidence_ids)}")
            print(f"agent> {result.response_text[:600]}")
            if result.error is not None:
                print(f"error={result.error.get('code')}")

            run_envelope = _extract_run_envelope(recorder)
            thread_id = str(
                result.active_workflow_thread_id or run_envelope.get("thread_id") or ""
            )
            manifest_path = run_root / thread_id / "manifest.json"
            result_payload: dict[str, Any] = {}
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                ref = ArtifactRef.model_validate(
                    manifest["artifacts"]["screening_result"]
                )
                result_payload = FileRunArtifactStore(run_root).get_json(ref)
            checks = _check_hard_constraints(result_payload)

            report = {
                "query": QUERY,
                "conversation_id": result.conversation_id,
                "workflow_thread_id": thread_id,
                "run_envelope": {
                    key: run_envelope.get(key)
                    for key in (
                        "status",
                        "planner_status",
                        "retrieved_count",
                        "filtered_count",
                        "returned_count",
                        "validation_passed",
                    )
                },
                "model_calls": [
                    {
                        "request_id": response.request_id,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "reasoning_tokens": response.reasoning_tokens,
                        "latency_ms": response.latency_ms,
                    }
                    for response in recorder.responses
                ],
                "final_evidence_ids": list(result.evidence_ids),
                "constraint_checks": checks,
            }
            report_path = work_dir / "e2e_report.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\nreport written (non-sensitive) to: {report_path}")
            print("hard constraint checks:", json.dumps(checks, ensure_ascii=False))
            print(f"final evidence ids: {list(result.evidence_ids)}")

            failures = []
            if result.status != "completed" or result.final_status != "completed":
                failures.append(f"agent final status: {result.final_status}")
            if not result.evidence_ids:
                failures.append("final answer has no evidence")
            if run_envelope.get("planner_status") != "ready":
                failures.append(f"planner status: {run_envelope.get('planner_status')}")
            if not checks["constraints_satisfied"]:
                failures.append("hard constraints not satisfied")
            if failures:
                print("\n[FAIL] " + "; ".join(failures))
                return 1
            print("\nend-to-end PASS: verified result + evidence + hard constraints")
            return 0
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
