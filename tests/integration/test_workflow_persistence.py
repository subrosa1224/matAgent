"""SQLite persistence and restart recovery (S3-M6)."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from materials_screening.models import (  # noqa: E402
    MaterialRecord,
    PropertyProvenance,
    PropertyValueType,
)
from materials_screening.repositories.base import RetrievalResult  # noqa: E402
from materials_screening.services.filter_service import FilterService  # noqa: E402
from materials_screening.services.ranking_service import RankingService  # noqa: E402
from materials_screening.services.validation_service import (
    ValidationService,  # noqa: E402
)
from materials_screening.workflow.artifact_store import (  # noqa: E402
    FileRunArtifactStore,
)
from materials_screening.workflow.checkpointer import (  # noqa: E402
    CheckpointerHandle,
    create_checkpointer_handle,
)
from materials_screening.workflow.context import WorkflowContext  # noqa: E402
from materials_screening.workflow.errors import (  # noqa: E402
    WorkflowCheckpointError,
)
from materials_screening.workflow.export_adapter import (  # noqa: E402
    WorkflowExportAdapter,
)
from materials_screening.workflow.input_output import WorkflowInput  # noqa: E402
from materials_screening.workflow.runner import WorkflowRunner  # noqa: E402
from materials_screening.workflow.settings import WorkflowSettings  # noqa: E402
from materials_screening.workflow.state import ArtifactRef, WorkflowStatus  # noqa: E402

_CHECKPOINT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _fixed_clock() -> datetime:
    return datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


class _RecordsRepository:
    def __init__(self, records: tuple[MaterialRecord, ...]) -> None:
        self._records = records

    def search(self, request: object) -> RetrievalResult:
        return RetrievalResult(
            source="mock",
            database_version="fixture-v1",
            retrieved_at=_fixed_clock(),
            records=self._records,
        )

    def healthcheck(self) -> bool:
        return True


class _FakePlanner:
    def parse(self, query: str) -> object:
        raise AssertionError("planner must not be called in request mode")


class _IdGenerator:
    def __init__(self) -> None:
        self.count = 0

    def new_id(self) -> str:
        self.count += 1
        return f"id-{self.count}"


def _record(material_id: str, band_gap: float) -> MaterialRecord:
    provenance = tuple(
        PropertyProvenance(
            property_name=name,
            source="materials_project",
            source_material_id=material_id,
            value_type=PropertyValueType.DFT_CALCULATED,
            database_version="fixture-v1",
            retrieved_at=_fixed_clock(),
        )
        for name in ("band_gap_ev", "energy_above_hull_ev_atom", "is_metal")
    )
    return MaterialRecord(
        source="mock",
        material_id=material_id,
        formula_pretty=material_id,
        elements=("Si",),
        band_gap_ev=band_gap,
        energy_above_hull_ev_atom=0.0,
        is_metal=False,
        provenance=provenance,
    )


def _build_context(run_root: Path) -> WorkflowContext:
    return WorkflowContext(
        planner_service=_FakePlanner(),
        materials_repository=_RecordsRepository(
            (_record("mp-1", 1.5), _record("mp-2", 1.6))
        ),
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=WorkflowExportAdapter(run_root),
        artifact_store=FileRunArtifactStore(run_root),
        clock=_fixed_clock,
        id_generator=_IdGenerator(),
    )


def _open_runner(
    *,
    tmp_path: Path,
    db_path: Path,
    run_root: Path,
) -> tuple[WorkflowRunner, CheckpointerHandle]:
    handle = create_checkpointer_handle(
        backend="sqlite",
        db_path=db_path,
        allowed_root=tmp_path,
    )
    runner = WorkflowRunner(
        settings=WorkflowSettings(_env_file=None),
        context=_build_context(run_root),
        checkpointer=handle,
    )
    return runner, handle


def _request_input(run_root: Path) -> WorkflowInput:
    return WorkflowInput(
        request={
            "limit": 10,
            "band_gap_ev": {"min": 1.0, "max": 2.0},
        },
        output_root=str(run_root),
    )


class TestSqliteRestartRecovery:
    def test_full_workflow_survives_restart(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.sqlite"
        run_root = tmp_path / "runs"

        # 1. Run the full mock workflow on a tmp SQLite checkpointer.
        first_runner, first_handle = _open_runner(
            tmp_path=tmp_path,
            db_path=db_path,
            run_root=run_root,
        )
        first_saver = first_handle.checkpointer
        output = first_runner.run(_request_input(run_root))
        assert output.status is WorkflowStatus.COMPLETED
        assert output.retrieved_count == 2
        thread_id = output.thread_id

        # 2. Close the runner and its connection.
        first_runner.close()
        with pytest.raises(WorkflowCheckpointError):
            _ = first_handle.checkpointer

        # 3. New checkpointer, graph and runner against the same DB file.
        second_runner, second_handle = _open_runner(
            tmp_path=tmp_path,
            db_path=db_path,
            run_root=run_root,
        )
        assert second_handle.checkpointer is not first_saver
        assert db_path.exists()

        # 4. get_state on the same thread_id.
        view = second_runner.get_state(thread_id)
        assert view.run_id == thread_id
        assert view.status == WorkflowStatus.COMPLETED.value
        assert view.retrieved_count == 2
        assert view.filtered_count == 2
        assert view.returned_count == 2
        assert view.validation_passed is True
        assert view.finished_at is not None
        assert view.exports

        # 5. get_history.
        history = second_runner.get_history(thread_id)
        assert history

        # 6. Verify status, next and checkpoint IDs on the newest checkpoint.
        newest = history[0]
        assert newest.status == WorkflowStatus.COMPLETED.value
        assert newest.next_nodes == ()
        assert _CHECKPOINT_ID_PATTERN.fullmatch(newest.checkpoint_id)
        assert all(
            _CHECKPOINT_ID_PATTERN.fullmatch(item.checkpoint_id) for item in history
        )

        # 7. Artifacts remain verifiable with a fresh artifact store.
        fresh_store = FileRunArtifactStore(run_root)
        manifest_path = run_root / thread_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert set(manifest["artifacts"]) == {
            "retrieval",
            "filtered",
            "filter_trace",
            "ranked",
            "validation",
            "screening_result",
        }
        for name, ref_dict in manifest["artifacts"].items():
            ref = ArtifactRef.model_validate(ref_dict)
            assert fresh_store.verify(ref) is True, name
            assert fresh_store.get_json(ref) is not None, name

        # Export manifest is also still present and readable.
        export_manifest = run_root / thread_id / "exports" / "export_manifest.json"
        assert export_manifest.is_file()

        second_runner.close()

    def test_restart_without_global_memory(self, tmp_path: Path) -> None:
        """The second runner must not reuse the first runner's in-memory state."""
        db_path = tmp_path / "checkpoints.sqlite"
        run_root = tmp_path / "runs"
        first_runner, first_handle = _open_runner(
            tmp_path=tmp_path,
            db_path=db_path,
            run_root=run_root,
        )
        first_saver = first_handle.checkpointer
        first_output = first_runner.run(_request_input(run_root))
        thread_id = first_output.thread_id
        first_runner.close()

        second_runner, second_handle = _open_runner(
            tmp_path=tmp_path,
            db_path=db_path,
            run_root=run_root,
        )
        view = second_runner.get_state(thread_id)
        assert view.status == WorkflowStatus.COMPLETED.value
        assert second_handle.checkpointer is not first_saver
        second_runner.close()
