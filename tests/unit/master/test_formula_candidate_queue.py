"""Formula selection must deduplicate the complete immutable snapshot first."""

from pathlib import Path

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.master.data_analysis_handoffs import (
    DataAnalysisCrossAgentCoordinator,
)
from materials_screening.models import MaterialRecord
from materials_screening.services.query_result_store import QueryResultStore


def test_formula_queue_uses_full_snapshot_and_lowest_hull_representative(
    tmp_path: Path,
):
    records = tuple(
        MaterialRecord(
            source="materials_project",
            material_id=f"mp-{i}",
            formula_pretty=formula,
            elements=("Li", "Fe", "O"),
            energy_above_hull_ev_atom=hull,
        )
        for i, (formula, hull) in enumerate(
            [("Li2FeO3", 0.09)] * 21
            + [("LiFeO2", 0.02), ("Li5FeO4", 0.0), ("Li2FeO3", 0.01)]
        )
    )
    store = QueryResultStore(tmp_path / "queries")
    store.save_query("query-test", {"source": "materials_project"}, records)
    coordinator = DataAnalysisCrossAgentCoordinator(
        dataset_store=DatasetStore(tmp_path / "datasets"), query_store=store
    )
    preview = coordinator.material_query_candidates("query-test", limit=2)
    assert [x["material_id"] for x in preview] == ["mp-0", "mp-1"]
    queue = coordinator.material_query_candidates(
        "query-test", limit=3, unique_formulas=True
    )
    assert [x["formula_pretty"] for x in queue] == ["Li5FeO4", "Li2FeO3", "LiFeO2"]
    assert queue[1]["material_id"] == "mp-23"
    assert (
        len(
            coordinator.material_query_candidates(
                "query-test", limit=2, unique_formulas=True
            )
        )
        == 2
    )
    assert store.load_records("query-test") == records


def test_formula_queue_sorts_missing_hull_last_with_deterministic_ties(tmp_path: Path):
    records = tuple(
        MaterialRecord(
            source="materials_project",
            material_id=f"mp-{i}",
            formula_pretty=f,
            elements=("Li", "Fe", "O"),
            energy_above_hull_ev_atom=h,
        )
        for i, (f, h) in enumerate(
            [("LiFeO2", None), ("Li2FeO3", 0.0), ("Li5FeO4", 0.0)]
        )
    )
    store = QueryResultStore(tmp_path / "queries")
    store.save_query("query-test", {"source": "materials_project"}, records)
    coordinator = DataAnalysisCrossAgentCoordinator(
        dataset_store=DatasetStore(tmp_path / "datasets"), query_store=store
    )
    queue = coordinator.material_query_candidates(
        "query-test", limit=3, unique_formulas=True
    )
    assert [x["formula_pretty"] for x in queue] == ["Li2FeO3", "Li5FeO4", "LiFeO2"]
