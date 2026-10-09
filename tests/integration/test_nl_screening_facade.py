"""NaturalLanguageScreeningFacade integration tests (D2)."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pymatgen.core.structure import Structure

from materials_screening.llm.errors import LLMTimeoutError
from materials_screening.llm.mock_provider import (
    MockStructuredProvider,
    MockErrorKind,
)
from materials_screening.models import (
    CrystalSystem,
    MaterialRecord,
    PropertyProvenance,
    PropertyValueType,
    ScreeningRequest,
    SymmetryInfo,
)
from materials_screening.planner.facade import NaturalLanguageScreeningFacade
from materials_screening.planner.models import (
    DraftStatus,
    EnergyUnit,
    PlannerDraft,
    PlannerStatus,
)
from materials_screening.planner.service import PlannerService
from materials_screening.planner.settings import Settings
from materials_screening.repositories.base import RetrievalResult
from materials_screening.repositories.mock import MockMaterialsRepository
from materials_screening.services.export_service import ExportService
from materials_screening.services.filter_service import FilterService
from materials_screening.services.ranking_service import RankingService
from materials_screening.services.screening_service import ScreeningService
from materials_screening.services.validation_service import ValidationService

FIXED_TIME = datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC)
DATABASE_VERSION = "v2026.08"


class CountingRepository:
    """Repository spy that records how often search is called."""

    def __init__(self, inner: MockMaterialsRepository) -> None:
        self._inner = inner
        self.search_calls = 0

    def search(self, request: ScreeningRequest) -> RetrievalResult:
        self.search_calls += 1
        return self._inner.search(request)

    def healthcheck(self) -> bool:
        return self._inner.healthcheck()


def _structure_dict() -> dict[str, object]:
    structure = Structure(
        lattice=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Fe", "O"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    return structure.as_dict()


def _record(**overrides: object) -> MaterialRecord:
    values: dict[str, object] = {
        "source": "mock",
        "material_id": "mp-1",
        "formula_pretty": "Fe2O3",
        "elements": ("Fe", "O"),
        "chemsys": "Fe-O",
        "band_gap_ev": 2.0,
        "energy_above_hull_ev_atom": 0.01,
        "formation_energy_ev_atom": -1.5,
        "density_g_cm3": 5.2,
        "is_metal": False,
        "is_gap_direct": True,
        "is_stable": True,
        "theoretical": False,
        "deprecated": False,
        "symmetry": SymmetryInfo(
            crystal_system=CrystalSystem.TRIGONAL,
            symbol="R-3c",
            number=167,
        ),
        "structure_dict": _structure_dict(),
        "provenance": (),
    }
    values.update(overrides)
    return MaterialRecord.model_validate(values)


def _with_provenance(record: MaterialRecord) -> MaterialRecord:
    provenance = tuple(
        PropertyProvenance(
            property_name=name,
            source="mock",
            source_material_id=record.material_id,
            value_type=PropertyValueType.DFT_CALCULATED,
            database_version=DATABASE_VERSION,
            retrieved_at=FIXED_TIME,
        )
        for name in (
            "band_gap_ev",
            "energy_above_hull_ev_atom",
            "formation_energy_ev_atom",
            "density_g_cm3",
            "is_metal",
            "is_gap_direct",
            "is_stable",
            "theoretical",
            "symmetry",
            "structure_dict",
        )
        if getattr(record, name) is not None
    )
    return record.model_copy(update={"provenance": provenance})


def _draft(**overrides: object) -> PlannerDraft:
    values: dict[str, object] = {
        "status": DraftStatus.EXTRACTED,
        "required_elements": [],
        "excluded_elements": [],
        "chemsys": None,
        "formula": None,
        "band_gap_min": None,
        "band_gap_max": None,
        "band_gap_unit": EnergyUnit.UNSPECIFIED,
        "hull_min": None,
        "hull_max": None,
        "hull_unit": "unspecified",
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
        "limit": None,
        "ambiguities": [],
        "unsupported_requirements": [],
        "conflicts": [],
        "assumptions": [],
        "clarification_question": "",
        "evidence": [],
    }
    values.update(overrides)
    return PlannerDraft.model_validate(values)


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "llm_provider": "mock",
        "planner_prompt_version": "planner-v2",
        "planner_schema_version": "planner-draft-v1",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _planner_service(provider: MockStructuredProvider) -> PlannerService:
    return PlannerService(
        settings=_settings(),
        provider=provider,
        clock=lambda: FIXED_TIME,
    )


def _screening_service(
    repository: CountingRepository,
) -> ScreeningService:
    return ScreeningService(
        repository=repository,
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=ExportService(),
        clock=lambda: FIXED_TIME,
    )


def _facade(
    provider: MockStructuredProvider,
) -> tuple[NaturalLanguageScreeningFacade, CountingRepository]:
    repository = CountingRepository(
        MockMaterialsRepository(records=(_with_provenance(_record()),))
    )
    facade = NaturalLanguageScreeningFacade(
        planner_service=_planner_service(provider),
        screening_service=_screening_service(repository),
    )
    return facade, repository


class TestFacadeCallCounting:
    def test_ready_runs_screening_once(self, tmp_path: Path) -> None:
        facade, repository = _facade(MockStructuredProvider(fixtures={"q": _draft()}))

        result = facade.screen("q", tmp_path)

        assert result.planner.status is PlannerStatus.READY
        assert result.screening is not None
        assert result.screening.result.request == result.planner.request
        assert result.screening.result.validation.passed is True
        assert repository.search_calls == 1
        assert (result.screening.exports.run_dir / "report.md").exists()

    def test_screen_ready_runs_without_reparsing(self, tmp_path: Path) -> None:
        facade, repository = _facade(MockStructuredProvider(fixtures={"q": _draft()}))
        planner = _planner_service(
            MockStructuredProvider(fixtures={"q": _draft()})
        ).parse("q")

        result = facade.screen_ready(planner, tmp_path)

        assert result.planner.status is PlannerStatus.READY
        assert result.screening is not None
        assert repository.search_calls == 1

    def test_screen_ready_rejects_non_ready(self, tmp_path: Path) -> None:
        facade, repository = _facade(
            MockStructuredProvider(
                fixtures={
                    "q": _draft(band_gap_min=5000, band_gap_unit=EnergyUnit.UNSPECIFIED)
                }
            )
        )
        planner = _planner_service(
            MockStructuredProvider(
                fixtures={
                    "q": _draft(band_gap_min=5000, band_gap_unit=EnergyUnit.UNSPECIFIED)
                }
            )
        ).parse("q")

        with pytest.raises(Exception, match="screen_ready requires a READY"):
            facade.screen_ready(planner, tmp_path)
        assert repository.search_calls == 0

    def test_needs_clarification_skips_screening(self, tmp_path: Path) -> None:
        facade, repository = _facade(
            MockStructuredProvider(
                fixtures={
                    "q": _draft(band_gap_min=5000, band_gap_unit=EnergyUnit.UNSPECIFIED)
                }
            )
        )

        result = facade.screen("q", tmp_path)

        assert result.planner.status is PlannerStatus.NEEDS_CLARIFICATION
        assert result.screening is None
        assert repository.search_calls == 0

    def test_invalid_skips_screening(self, tmp_path: Path) -> None:
        facade, repository = _facade(
            MockStructuredProvider(fixtures={"q": _draft(required_elements=["Xx"])})
        )

        result = facade.screen("q", tmp_path)

        assert result.planner.status is PlannerStatus.INVALID
        assert result.screening is None
        assert repository.search_calls == 0

    def test_unsupported_skips_screening(self, tmp_path: Path) -> None:
        facade, repository = _facade(
            MockStructuredProvider(
                fixtures={"q": _draft(unsupported_requirements=["predict"])}
            )
        )

        result = facade.screen("q", tmp_path)

        assert result.planner.status is PlannerStatus.UNSUPPORTED
        assert result.screening is None
        assert repository.search_calls == 0

    def test_provider_error_never_calls_repository(self, tmp_path: Path) -> None:
        facade, repository = _facade(
            MockStructuredProvider(error_fixtures={"q": MockErrorKind.TIMEOUT})
        )

        with pytest.raises(LLMTimeoutError):
            facade.screen("q", tmp_path)

        assert repository.search_calls == 0
