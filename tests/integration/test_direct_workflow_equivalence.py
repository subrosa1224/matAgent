"""Direct (stage 1) vs workflow equivalence test (S3-M8)."""

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pymatgen.core.structure import Structure

pytest.importorskip("langgraph")

from materials_screening.models import (  # noqa: E402
    FloatRange,
    MaterialRecord,
    PropertyProvenance,
    PropertyValueType,
    ScreeningRequest,
)
from materials_screening.repositories.materials_project import (  # noqa: E402
    PROVENANCE_PROPERTIES,
)
from materials_screening.repositories.mock import MockMaterialsRepository  # noqa: E402
from materials_screening.services.export_service import ExportService  # noqa: E402
from materials_screening.services.filter_service import FilterService  # noqa: E402
from materials_screening.services.ranking_service import RankingService  # noqa: E402
from materials_screening.services.screening_service import (
    ScreeningService,  # noqa: E402
)
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
from materials_screening.workflow.export_adapter import (  # noqa: E402
    WorkflowExportAdapter,
)
from materials_screening.workflow.input_output import WorkflowInput  # noqa: E402
from materials_screening.workflow.runner import WorkflowRunner  # noqa: E402
from materials_screening.workflow.settings import WorkflowSettings  # noqa: E402
from materials_screening.workflow.state import ArtifactRef  # noqa: E402

FIXED_CLOCK = datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


def _request() -> ScreeningRequest:
    return ScreeningRequest(
        band_gap_ev=FloatRange(min=1.0, max=2.0),
        energy_above_hull_ev_atom=FloatRange(min=0.0, max=0.5),
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
    structure: bool,
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
        "structure_dict": _structure_dict() if structure else None,
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
            structure=True,
        ),
        _record(
            "mp-2",
            band_gap=1.6,
            hull=0.1,
            formation=-1.8,
            density=3.2,
            direct_gap=False,
            stable=True,
            structure=True,
        ),
        _record(
            "mp-3",
            band_gap=1.4,
            hull=0.2,
            formation=-1.5,
            density=2.9,
            direct_gap=False,
            stable=False,
            structure=False,
        ),
        _record(
            "mp-4",
            band_gap=1.7,
            hull=0.05,
            formation=-1.9,
            density=3.0,
            direct_gap=True,
            stable=True,
            structure=True,
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


class TestDirectWorkflowEquivalence:
    def test_ranked_candidates_match(self, tmp_path: Path) -> None:
        request = _request()
        records = _records()
        direct, direct_csv = _direct_result(tmp_path, request, records)
        workflow, workflow_csv = _workflow_result(tmp_path, request, records)

        direct_ranked = direct["ranked_materials"]
        workflow_ranked = workflow["ranked_materials"]

        assert [item["record"]["material_id"] for item in workflow_ranked] == [
            item["record"]["material_id"] for item in direct_ranked
        ]
        assert [item["rank"] for item in workflow_ranked] == [
            item["rank"] for item in direct_ranked
        ]
        assert [item["total_score"] for item in workflow_ranked] == [
            item["total_score"] for item in direct_ranked
        ]
        assert [item["score_breakdown"] for item in workflow_ranked] == [
            item["score_breakdown"] for item in direct_ranked
        ]

    def test_validation_matches(self, tmp_path: Path) -> None:
        request = _request()
        records = _records()
        direct, _ = _direct_result(tmp_path, request, records)
        workflow, _ = _workflow_result(tmp_path, request, records)

        assert workflow["validation"] == direct["validation"]
        assert workflow["validation"]["passed"] is True

    def test_csv_core_data_matches(self, tmp_path: Path) -> None:
        request = _request()
        records = _records()
        direct, direct_csv = _direct_result(tmp_path, request, records)
        workflow, workflow_csv = _workflow_result(tmp_path, request, records)
        assert direct["metadata"]["run_id"].startswith("run_")
        assert workflow["metadata"]["run_id"] != direct["metadata"]["run_id"]
        assert direct_csv == workflow_csv
        assert direct_csv[0] == workflow_csv[0]

    def test_counts_and_limits_match(self, tmp_path: Path) -> None:
        request = _request()
        records = _records()
        direct, _ = _direct_result(tmp_path, request, records)
        workflow, _ = _workflow_result(tmp_path, request, records)
        assert workflow["retrieved_count"] == direct["retrieved_count"] == 4
        assert workflow["passed_filter_count"] == direct["passed_filter_count"] == 4
        assert len(workflow["ranked_materials"]) == len(direct["ranked_materials"])
