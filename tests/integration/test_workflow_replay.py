"""Controlled workflow replay tests (S3-M7)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langgraph.errors import GraphRecursionError  # noqa: E402

from materials_screening.models import (  # noqa: E402
    MaterialRecord,
    PropertyProvenance,
    PropertyValueType,
    ScreeningResult,
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
from materials_screening.workflow.state import WorkflowStatus  # noqa: E402


def _fixed_clock() -> datetime:
    return datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


class _RecordsRepository:
    def __init__(self, records: tuple[MaterialRecord, ...]) -> None:
        self._records = records
        self.search_calls = 0

    def search(self, request: object) -> RetrievalResult:
        self.search_calls += 1
        return RetrievalResult(
            source="mock",
            database_version="fixture-v1",
            retrieved_at=_fixed_clock(),
            records=self._records,
        )

    def healthcheck(self) -> bool:
        return True


class _CountingFilterService(FilterService):
    def __init__(self) -> None:
        super().__init__()
        self.apply_calls = 0

    def apply(
        self,
        records: Any,
        request: Any,
    ) -> tuple[tuple[MaterialRecord, ...], Any]:
        self.apply_calls += 1
        return super().apply(records, request)


class _CountingRankingService(RankingService):
    def __init__(self) -> None:
        super().__init__()
        self.rank_calls = 0

    def rank(self, records: Any, request: Any) -> Any:
        self.rank_calls += 1
        return super().rank(records, request)


class _CountingValidationService(ValidationService):
    def __init__(self) -> None:
        super().__init__()
        self.validate_calls = 0

    def validate(self, result: ScreeningResult) -> Any:
        self.validate_calls += 1
        return super().validate(result)


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


class _RunContext:
    def __init__(self, run_root: Path) -> None:
        self.repository = _RecordsRepository(
            (_record("mp-1", 1.5), _record("mp-2", 1.6))
        )
        self.filter_service = _CountingFilterService()
        self.ranking_service = _CountingRankingService()
        self.validation_service = _CountingValidationService()
        self.run_root = run_root

    def build(self) -> WorkflowContext:
        return WorkflowContext(
            planner_service=_FakePlanner(),
            materials_repository=self.repository,
            filter_service=self.filter_service,
            ranking_service=self.ranking_service,
            validation_service=self.validation_service,
            export_service=WorkflowExportAdapter(self.run_root),
            artifact_store=FileRunArtifactStore(self.run_root),
            clock=_fixed_clock,
            id_generator=_IdGenerator(),
        )


def _open_runner(run_context: _RunContext) -> WorkflowRunner:
    return WorkflowRunner(
        settings=WorkflowSettings(_env_file=None),
        context=run_context.build(),
        checkpointer=create_checkpointer_handle(backend="memory"),
    )


def _request_input(run_root: Path) -> WorkflowInput:
    return WorkflowInput(
        request={
            "limit": 10,
            "band_gap_ev": {"min": 1.0, "max": 2.0},
        },
        output_root=str(run_root),
    )


def _filter_checkpoint_id(runner: WorkflowRunner, thread_id: str) -> str:
    for view in runner.get_history(thread_id):
        if view.next_nodes == ("rank_materials",):
            return view.checkpoint_id
    raise AssertionError("filter-after checkpoint not found")


def _manifest(run_root: Path, thread_id: str) -> dict[str, Any]:
    path = run_root / thread_id / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _export_file_names(run_root: Path, thread_id: str) -> set[str]:
    exports = run_root / thread_id / "exports"
    return {
        str(path.relative_to(exports).as_posix())
        for path in exports.rglob("*")
        if path.is_file()
    }


class TestControlledReplay:
    def test_replay_from_filter_checkpoint(self, tmp_path: Path) -> None:
        run_root = tmp_path / "runs"
        run_context = _RunContext(run_root)
        runner = _open_runner(run_context)
        output = runner.run(_request_input(run_root))
        assert output.status is WorkflowStatus.COMPLETED
        thread_id = output.thread_id
        assert run_context.repository.search_calls == 1
        assert run_context.filter_service.apply_calls == 1
        assert run_context.ranking_service.rank_calls == 1
        assert run_context.validation_service.validate_calls == 1

        checkpoint_id = _filter_checkpoint_id(runner, thread_id)
        manifest_before = _manifest(run_root, thread_id)
        exports_before = _export_file_names(run_root, thread_id)

        replay_output = runner.replay(
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            confirm_remote_calls=True,
        )
        assert replay_output.status is WorkflowStatus.COMPLETED
        assert run_context.repository.search_calls == 1
        assert run_context.filter_service.apply_calls == 1
        assert run_context.ranking_service.rank_calls == 2
        assert run_context.validation_service.validate_calls == 2

        assert _manifest(run_root, thread_id) == manifest_before
        assert _export_file_names(run_root, thread_id) == exports_before
        run_dirs = [path.name for path in (run_root / thread_id).iterdir()]
        assert "exports" in run_dirs
        assert not any(name.startswith("run_") for name in run_dirs)
        runner.close()

    def test_replay_unconfirmed_rejected(self, tmp_path: Path) -> None:
        run_root = tmp_path / "runs"
        run_context = _RunContext(run_root)
        runner = _open_runner(run_context)
        output = runner.run(_request_input(run_root))
        thread_id = output.thread_id
        checkpoint_id = _filter_checkpoint_id(runner, thread_id)
        calls_before = run_context.repository.search_calls
        with pytest.raises(WorkflowCheckpointError, match="confirm_remote_calls"):
            runner.replay(thread_id=thread_id, checkpoint_id=checkpoint_id)
        assert run_context.repository.search_calls == calls_before
        runner.close()

    def test_replay_invalid_checkpoint_rejected(self, tmp_path: Path) -> None:
        run_root = tmp_path / "runs"
        run_context = _RunContext(run_root)
        runner = _open_runner(run_context)
        output = runner.run(_request_input(run_root))
        with pytest.raises(WorkflowCheckpointError, match="not found"):
            runner.replay(
                thread_id=output.thread_id,
                checkpoint_id="not-a-real-checkpoint",
                confirm_remote_calls=True,
            )
        runner.close()

    def test_replay_checkpoint_from_other_thread_rejected(self, tmp_path: Path) -> None:
        run_root = tmp_path / "runs"
        run_context = _RunContext(run_root)
        runner = _open_runner(run_context)
        first = runner.run(_request_input(run_root))
        second = runner.run(_request_input(run_root))
        other_checkpoint = _filter_checkpoint_id(runner, second.thread_id)
        with pytest.raises(WorkflowCheckpointError, match="not found"):
            runner.replay(
                thread_id=first.thread_id,
                checkpoint_id=other_checkpoint,
                confirm_remote_calls=True,
            )
        runner.close()

    def test_replay_corrupted_artifact_rejected(self, tmp_path: Path) -> None:
        run_root = tmp_path / "runs"
        run_context = _RunContext(run_root)
        runner = _open_runner(run_context)
        output = runner.run(_request_input(run_root))
        thread_id = output.thread_id
        checkpoint_id = _filter_checkpoint_id(runner, thread_id)
        filtered_path = run_root / thread_id / "artifacts" / "filtered.json"
        filtered_path.write_text('[{"tampered": true}]', encoding="utf-8")
        with pytest.raises(WorkflowCheckpointError, match="missing or corrupted"):
            runner.replay(
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                confirm_remote_calls=True,
            )
        runner.close()

    def test_replay_does_not_expose_update_state(self, tmp_path: Path) -> None:
        run_root = tmp_path / "runs"
        runner = _open_runner(_RunContext(run_root))
        assert not hasattr(runner, "update_state")
        runner.close()

    def test_replay_recursion_error_mapped(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        run_root = tmp_path / "runs"
        run_context = _RunContext(run_root)
        runner = _open_runner(run_context)
        output = runner.run(_request_input(run_root))
        thread_id = output.thread_id
        checkpoint_id = _filter_checkpoint_id(runner, thread_id)

        def _raise_recursion(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise GraphRecursionError("recursion limit reached")

        monkeypatch.setattr(runner._graph, "invoke", _raise_recursion)
        replay_output = runner.replay(
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            confirm_remote_calls=True,
        )
        assert replay_output.status is WorkflowStatus.FAILED
        assert replay_output.error["code"] == "WORKFLOW_RECURSION_LIMIT"
        runner.close()

    def test_replay_checkpoint_error_mapped(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        run_root = tmp_path / "runs"
        run_context = _RunContext(run_root)
        runner = _open_runner(run_context)
        output = runner.run(_request_input(run_root))
        thread_id = output.thread_id
        checkpoint_id = _filter_checkpoint_id(runner, thread_id)

        def _raise_checkpoint(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise WorkflowCheckpointError("checkpoint boom")

        monkeypatch.setattr(runner._graph, "invoke", _raise_checkpoint)
        replay_output = runner.replay(
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            confirm_remote_calls=True,
        )
        assert replay_output.status is WorkflowStatus.FAILED
        assert replay_output.error["code"] == "CHECKPOINT_FAILED"
        runner.close()

    def test_verify_replay_artifacts_rejects_invalid_values(
        self, tmp_path: Path
    ) -> None:
        runner = _open_runner(_RunContext(tmp_path / "runs"))
        with pytest.raises(WorkflowCheckpointError, match="invalid"):
            runner._verify_replay_artifacts(None)
        with pytest.raises(WorkflowCheckpointError, match="missing or corrupted"):
            runner._verify_replay_artifacts({"ranked_ref": {"bad": 1}})
        runner.close()
