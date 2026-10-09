"""DA-7 end-to-end tests for the three bounded cross-agent paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.statistics import DataStatisticsService
from materials_screening.master import (
    DataAnalysisCrossAgentCoordinator,
    SourceSynthesisSection,
    UnifiedResultEnvelope,
    default_renderer_registry,
)
from materials_screening.models import MaterialRecord
from materials_screening.repositories.mock import MockMaterialsRepository
from materials_screening.services.material_database_service import (
    MaterialDatabaseService,
)
from materials_screening.services.query_result_store import QueryResultStore
from materials_screening.sub_agents.literature.models import (
    ExperimentalGroup,
    ExperimentalMeasurement,
    ExperimentMatrixDataTable,
)


def _records() -> tuple[MaterialRecord, ...]:
    return (
        MaterialRecord(
            source="materials_project",
            material_id="mp-1",
            formula_pretty="Li2O",
            elements=("Li", "O"),
            band_gap_ev=2.0,
            formation_energy_ev_atom=-2.1,
        ),
        MaterialRecord(
            source="materials_project",
            material_id="mp-2",
            formula_pretty="MgO",
            elements=("Mg", "O"),
            band_gap_ev=4.0,
            formation_energy_ev_atom=-3.2,
        ),
    )


def test_material_query_snapshot_to_data_analysis_end_to_end(tmp_path: Path) -> None:
    query_store = QueryResultStore(tmp_path / "queries")
    query_store.save_query("query-cross-1", {"source": "materials_project"}, _records())
    datasets = DatasetStore(tmp_path / "analysis")
    coordinator = DataAnalysisCrossAgentCoordinator(
        dataset_store=datasets, query_store=query_store
    )

    handoff = coordinator.material_query_to_analysis(
        "query-cross-1", task="分析带隙与形成能"
    )
    analysis = DataStatisticsService(datasets).analyze_correlations(
        handoff.partition.dataset_id,
        columns=("band_gap_ev", "formation_energy_ev_atom"),
        evidence_id=handoff.evidence_id,
    )

    assert handoff.partition.source_kind == "materials_database"
    assert handoff.partition.evidence_ids == ("query-cross-1",)
    assert analysis.dataset_id == handoff.partition.dataset_id
    assert analysis.evidence_id == handoff.evidence_id
    candidates = coordinator.material_query_candidates("query-cross-1", limit=1)
    assert candidates[0]["material_id"] == "mp-1"
    assert candidates[0]["formula_pretty"] == "Li2O"
    prioritized = coordinator.prioritized_material_query_candidates(
        "query-cross-1", limit=1
    )
    assert prioritized[0].material_id == "mp-1"
    assert prioritized[0].priority == 1


def test_approved_literature_matrix_to_data_analysis_end_to_end(
    tmp_path: Path,
) -> None:
    digest = "a" * 64
    groups = tuple(
        ExperimentalGroup(
            group_id=f"group-{index}",
            document_id="doc-1234567890abcdef12345678",
            label=label,
            role="control" if index == 1 else "treatment",
            material="TiO2 scaffold",
            variables={"temperature": temperature},
            conditions={},
            source_quote=f"{label} {temperature}",
            chunk_id=f"chunk-{index}",
            page_from=2,
            page_to=2,
            source_text_sha256=digest,
            extraction_method="manual",
            review_status="approved",
        )
        for index, (label, temperature) in enumerate(
            (("A", "800 C"), ("B", "900 C")), 1
        )
    )
    measurements = tuple(
        ExperimentalMeasurement(
            measurement_id=f"measurement-{index}",
            group_id=f"group-{index}",
            document_id="doc-1234567890abcdef12345678",
            metric="compressive_strength",
            value_text=str(value),
            numeric_value=value,
            unit="MPa",
            sample_size=3,
            source_quote=f"strength {value} MPa",
            chunk_id=f"chunk-{index}",
            page_from=2,
            page_to=2,
            source_text_sha256=digest,
            extraction_method="manual",
            review_status="approved",
        )
        for index, value in enumerate((10.0, 14.0), 1)
    )
    table = ExperimentMatrixDataTable(
        title="Approved matrix",
        document_id="doc-1234567890abcdef12345678",
        groups=groups,
        measurements=measurements,
        comparisons=(),
        claims=(),
        claim_evidence_links=(),
    )
    datasets = DatasetStore(tmp_path / "analysis")
    coordinator = DataAnalysisCrossAgentCoordinator(dataset_store=datasets)

    handoff = coordinator.literature_matrix_to_analysis(
        table, task="比较不同温度下的抗压强度"
    )
    result = DataStatisticsService(datasets).describe_dataset(
        handoff.partition.dataset_id,
        columns=("numeric_value",),
        group_by="group_label",
        evidence_id=handoff.evidence_id,
    )

    assert handoff.partition.source_kind == "literature"
    assert handoff.partition.evidence_ids == ("chunk-1", "chunk-2")
    assert result.summary["statistics"][0]["count"] == 1


def test_data_analysis_material_ids_to_database_and_partitioned_synthesis(
    tmp_path: Path,
) -> None:
    datasets = DatasetStore(tmp_path / "analysis")
    dataset = datasets.register_records(
        ({"material_id": "mp-1", "score": 0.9}, {"material_id": "mp-2", "score": 0.8}),
        source_artifact_id="artifact-data-user-selection",
        display_name="selected-materials.json",
    )
    database = MaterialDatabaseService(
        MockMaterialsRepository(_records()), QueryResultStore(tmp_path / "queries")
    )
    coordinator = DataAnalysisCrossAgentCoordinator(
        dataset_store=datasets, database_service=database
    )

    handoff = coordinator.analysis_to_material_lookup(
        dataset.dataset_id, analysis_ids=("analysis-selection",)
    )
    database_section = coordinator.execute_material_lookup(
        handoff, fields=("band_gap_ev",)
    )
    analysis_section = SourceSynthesisSection(
        source_kind="data_analysis",
        source_id="analysis-selection",
        title="用户数据分析结果",
        markdown="按用户数据中的 score 选择了 2 个材料。",
        evidence_ids=("analysis-selection",),
    )
    synthesis = coordinator.synthesize((analysis_section, database_section))
    envelope = UnifiedResultEnvelope(
        agent_name="master",
        status="completed",
        result_type="cross_agent_analysis_answer",
        result=synthesis.model_dump(mode="json"),
    )
    rendered = default_renderer_registry().render(envelope)

    assert handoff.material_ids == ("mp-1", "mp-2")
    assert "数据分析" in rendered
    assert "材料数据库" in rendered
    assert "mp-1" in rendered
    assert rendered.index("用户数据分析结果") < rendered.index(
        "Materials Project 回查结果"
    )


def test_cross_agent_failures_are_closed_and_partial_results_are_retained(
    tmp_path: Path,
) -> None:
    datasets = DatasetStore(tmp_path / "analysis")
    invalid = datasets.register_records(
        ({"material_id": "not-an-mp-id"},),
        source_artifact_id="artifact-data-invalid-material",
        display_name="invalid.json",
    )
    coordinator = DataAnalysisCrossAgentCoordinator(dataset_store=datasets)
    with pytest.raises(ValueError, match="invalid Materials Project IDs"):
        coordinator.analysis_to_material_lookup(invalid.dataset_id)

    retained = SourceSynthesisSection(
        source_kind="data_analysis",
        source_id="analysis-retained",
        title="已完成的数据分析",
        markdown="描述统计已完成。",
    )
    synthesis = coordinator.synthesize(
        (retained,), warnings=("materials database lookup failed",)
    )
    assert synthesis.status == "partial"
    assert synthesis.sections == (retained,)
