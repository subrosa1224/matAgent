from pathlib import Path

import pytest

from materials_screening.models import MaterialRecord
from materials_screening.repositories.mock import MockMaterialsRepository
from materials_screening.services.material_database_service import (
    MaterialDatabaseService,
)
from materials_screening.services.query_result_store import QueryResultStore
from materials_screening.sub_agents.materials_database.models import (
    PropertyFilter,
    SearchMaterialsInput,
    SortRule,
)


def _record(mid: str, formula: str, density: float, gap: float) -> MaterialRecord:
    return MaterialRecord(
        source="materials_project",
        material_id=mid,
        formula_pretty=formula,
        elements=("O", "Ti"),
        chemsys="O-Ti",
        density_g_cm3=density,
        band_gap_ev=gap,
        energy_above_hull_ev_atom=0.0,
        is_stable=True,
    )


@pytest.fixture
def service(tmp_path: Path) -> MaterialDatabaseService:
    records = (
        _record("mp-1", "TiO2", 4.0, 3.0),
        _record("mp-2", "Ti2O3", 5.0, 1.0),
        _record("mp-3", "TiO", 3.0, 0.2),
    )
    return MaterialDatabaseService(
        MockMaterialsRepository(records), QueryResultStore(tmp_path / "queries")
    )


def test_search_filters_sorts_and_persists_snapshot(
    service: MaterialDatabaseService,
) -> None:
    result = service.search(
        SearchMaterialsInput(
            filters=(PropertyFilter(field="density_g_cm3", operator="gte", value=3.5),),
            sort=(SortRule(field="density_g_cm3", direction="desc"),),
            fields=("material_id", "formula_pretty", "density_g_cm3"),
            limit=2,
        )
    )
    assert result["matched_count"] == 2
    assert [row["material_id"] for row in result["materials"]] == ["mp-2", "mp-1"]
    page = service.page(result["query_id"], 0, 20, ("density_g_cm3",))
    assert page["total"] == 2


def test_exact_chemsys_rejects_records_with_additional_elements() -> None:
    exact_request = SearchMaterialsInput(chemsys="Li-Fe-O")
    exact = MaterialRecord(
        source="materials_project",
        material_id="mp-exact",
        formula_pretty="LiFeO2",
        elements=("Fe", "Li", "O"),
        chemsys="Fe-Li-O",
    )
    with_extra = exact.model_copy(
        update={
            "material_id": "mp-extra",
            "formula_pretty": "LiFePO4",
            "elements": ("Fe", "Li", "O", "P"),
            "chemsys": "Fe-Li-O-P",
        }
    )

    assert MaterialDatabaseService._matches(exact, exact_request)
    assert not MaterialDatabaseService._matches(with_extra, exact_request)


def test_required_and_excluded_elements_are_enforced_after_retrieval() -> None:
    safe = _record("mp-safe", "TiO2", 4.0, 3.0)
    contains_lead = safe.model_copy(
        update={
            "material_id": "mp-lead",
            "formula_pretty": "PbTiO3",
            "elements": ("O", "Pb", "Ti"),
            "chemsys": "O-Pb-Ti",
        }
    )
    no_oxygen = safe.model_copy(
        update={
            "material_id": "mp-no-oxygen",
            "formula_pretty": "TiC",
            "elements": ("C", "Ti"),
            "chemsys": "C-Ti",
        }
    )
    request = SearchMaterialsInput(
        required_elements=("O",), excluded_elements=("Pb", "Cd", "Hg")
    )

    assert MaterialDatabaseService._matches(safe, request)
    assert not MaterialDatabaseService._matches(contains_lead, request)
    assert not MaterialDatabaseService._matches(no_oxygen, request)


def test_describe_and_export_are_based_on_query_snapshot(
    service: MaterialDatabaseService, tmp_path: Path
) -> None:
    query = service.search(
        SearchMaterialsInput(
            fields=("material_id", "formula_pretty", "density_g_cm3"),
            limit=3,
        )
    )
    analysis = service.describe(query["query_id"], ("density_g_cm3",), False)
    assert analysis["statistics"]["density_g_cm3"]["mean"] == 4.0
    exported = service.export(query["query_id"], None, "csv", ("density_g_cm3",))
    assert Path(exported["path"]).is_file()


def test_unsupported_analysis_property_is_rejected(
    service: MaterialDatabaseService,
) -> None:
    query = service.search(SearchMaterialsInput(limit=3))
    with pytest.raises(ValueError, match="not numeric"):
        service.describe(query["query_id"], ("formula_pretty",), False)
