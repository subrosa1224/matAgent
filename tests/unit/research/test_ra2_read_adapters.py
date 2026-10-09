from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from materials_screening.data_analysis.models import AnalysisResult
from materials_screening.models import MaterialRecord
from materials_screening.repositories.mock import MockMaterialsRepository
from materials_screening.research import (
    CriterionOperator,
    CriterionRole,
    EvidenceReviewStatus,
    MaterialScope,
    MissingValuePolicy,
    ProjectStatus,
    ScreeningCriterion,
    ScreeningProject,
)
from materials_screening.research.adapters import (
    DataAnalysisReadAdapter,
    LiteratureEvidenceReadAdapter,
    MaterialsDatabaseCandidateSource,
    MaterialsDatabaseReadAdapter,
)
from materials_screening.research.property_registry import default_property_registry
from materials_screening.services.material_database_service import (
    MaterialDatabaseService,
)
from materials_screening.services.query_result_store import QueryResultStore
from materials_screening.sub_agents.literature.models import (
    ExperimentalGroup,
    ExperimentalMeasurement,
)
from materials_screening.sub_agents.materials_database.models import (
    QueryResultReference,
)


def test_database_adapter_preserves_ids_sources_units_and_boundaries() -> None:
    result = QueryResultReference(
        query_id="query-1",
        source="materials_project",
        matched_count=1,
        returned_count=1,
        fields=("material_id", "formula_pretty", "band_gap_ev"),
        materials=(
            {
                "material_id": "mp-1",
                "formula_pretty": "TiO2",
                "chemsys": "Ti-O",
                "band_gap_ev": 3.1,
                "is_stable": True,
                "unknown_application_score": 99.0,
            },
        ),
        created_at=datetime(2026, 9, 2, tzinfo=UTC),
    )

    batch = MaterialsDatabaseReadAdapter(default_property_registry()).normalize(
        "project-1", result, database_version="2026.04.13"
    )

    assert len(batch.candidates) == 1
    assert {item.property_id for item in batch.evidence} == {
        "band_gap_ev",
        "is_stable",
    }
    band_gap = next(
        item for item in batch.evidence if item.property_id == "band_gap_ev"
    )
    assert band_gap.unit == "eV"
    assert band_gap.locator.query_id == "query-1"
    assert "不能直接替代实验光学带隙" in band_gap.limitations[0]
    assert any("unknown_application_score" in warning for warning in batch.warnings)


def test_literature_adapter_keeps_pending_review_and_sample_conditions() -> None:
    group = ExperimentalGroup(
        group_id="group-1",
        document_id="doc-1",
        label="bulk sample",
        role="treatment",
        material="TiO2",
        conditions={"temperature": "300 K"},
        source_quote="The bulk sample was measured at 300 K.",
        chunk_id="chunk-1",
        page_from=3,
        page_to=3,
        source_text_sha256="a" * 64,
        review_status="pending",
    )
    measurement = ExperimentalMeasurement(
        measurement_id="measurement-1",
        group_id="group-1",
        document_id="doc-1",
        metric="band gap",
        value_text="3.2 eV",
        numeric_value=3.2,
        unit="eV",
        source_quote="The band gap was 3.2 eV.",
        chunk_id="chunk-2",
        page_from=4,
        page_to=4,
        source_text_sha256="b" * 64,
        review_status="pending",
    )

    batch = LiteratureEvidenceReadAdapter(
        default_property_registry()
    ).normalize_measurements(
        "project-1",
        "candidate-1",
        groups=(group,),
        measurements=(measurement,),
    )

    assert len(batch.evidence) == 1
    evidence = batch.evidence[0]
    assert evidence.review_status is EvidenceReviewStatus.PENDING
    assert evidence.locator.page_from == 4
    assert evidence.conditions["conditions"] == {"temperature": "300 K"}


def test_literature_adapter_does_not_guess_unregistered_property() -> None:
    group = ExperimentalGroup(
        group_id="group-1",
        document_id="doc-1",
        label="sample",
        role="unknown",
        material="TiO2",
        source_quote="sample",
        chunk_id="chunk-1",
        page_from=1,
        page_to=1,
        source_text_sha256="a" * 64,
    )
    measurement = ExperimentalMeasurement(
        measurement_id="measurement-1",
        group_id="group-1",
        document_id="doc-1",
        metric="device excellence score",
        value_text="99",
        numeric_value=99,
        unit=None,
        source_quote="score 99",
        chunk_id="chunk-2",
        page_from=2,
        page_to=2,
        source_text_sha256="b" * 64,
    )

    batch = LiteratureEvidenceReadAdapter(
        default_property_registry()
    ).normalize_measurements(
        "project-1",
        "candidate-1",
        groups=(group,),
        measurements=(measurement,),
    )

    assert not batch.evidence
    assert "unregistered property" in batch.warnings[0]


def test_data_analysis_adapter_keeps_result_as_context_not_property_value() -> None:
    result = AnalysisResult(
        analysis_id="analysis-1",
        dataset_id="dataset-1",
        analysis_type="descriptive",
        method="describe",
        parameters={"columns": ["band_gap_ev"]},
        summary={"mean": 3.2},
        evidence_id="evidence-source-1",
    )

    batch = DataAnalysisReadAdapter().normalize("project-1", result)

    evidence = batch.evidence[0]
    assert evidence.property_id is None
    assert evidence.value is None
    assert evidence.locator.analysis_id == "analysis-1"
    assert evidence.conditions["summary"] == {"mean": 3.2}


def test_database_candidate_source_reuses_persisted_idempotency_cache(
    tmp_path: Path,
) -> None:
    record = MaterialRecord(
        source="materials_project",
        material_id="mp-1",
        formula_pretty="TiO2",
        elements=("O", "Ti"),
        chemsys="O-Ti",
        band_gap_ev=3.1,
        is_metal=False,
    )
    service = MaterialDatabaseService(
        MockMaterialsRepository((record,)),
        QueryResultStore(tmp_path / "queries"),
    )
    source = MaterialsDatabaseCandidateSource(
        service,
        default_property_registry(),
        tmp_path / "cache",
    )
    project = ScreeningProject(
        project_id="project-1",
        title="test",
        research_question="find candidate",
        material_scope=MaterialScope(required_elements=("O",)),
        criteria=(
            ScreeningCriterion(
                property_id="band_gap_ev",
                role=CriterionRole.HARD_FILTER,
                operator=CriterionOperator.GTE,
                values=(3.0,),
                unit="eV",
                missing_policy=MissingValuePolicy.EXCLUDE,
            ),
        ),
        status=ProjectStatus.CONFIRMED,
        revision=2,
        confirmed_revision=2,
    )

    first = source.fetch_candidates(project, idempotency_key="work-query-1")
    second = source.fetch_candidates(project, idempotency_key="work-query-1")

    assert first == second
    assert len(first.candidates) == 1
    assert len(tuple((tmp_path / "queries").iterdir())) == 1
