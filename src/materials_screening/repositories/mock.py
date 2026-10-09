"""Mock repository for offline testing (M3)."""

from datetime import UTC, datetime

from materials_screening.models import MaterialRecord, ScreeningRequest
from materials_screening.repositories.base import RetrievalResult

FIXED_TEST_TIME = datetime(2026, 8, 5, 0, 0, tzinfo=UTC)

# Extra records used by the documented offline outlier-detection example.
# They are formula-lookup fixtures only and deliberately do not participate in
# ``search`` so existing screening demos keep their original candidate set.
_FORMULA_LOOKUP_FIXTURES: tuple[MaterialRecord, ...] = (
    MaterialRecord(
        source="mock", material_id="mock-tio2", formula_pretty="TiO2",
        elements=("Ti", "O"), band_gap_ev=3.2, density_g_cm3=4.23,
        energy_above_hull_ev_atom=0.0,
    ),
    MaterialRecord(
        source="mock", material_id="mock-batio3", formula_pretty="BaTiO3",
        elements=("Ba", "Ti", "O"), band_gap_ev=3.0, density_g_cm3=6.02,
        energy_above_hull_ev_atom=0.04,
    ),
    MaterialRecord(
        source="mock", material_id="mock-srtio3", formula_pretty="SrTiO3",
        elements=("Sr", "Ti", "O"), band_gap_ev=3.3, density_g_cm3=5.12,
        energy_above_hull_ev_atom=0.02,
    ),
    MaterialRecord(
        source="mock", material_id="mock-zno", formula_pretty="ZnO",
        elements=("Zn", "O"), band_gap_ev=3.4, density_g_cm3=5.61,
        energy_above_hull_ev_atom=0.01,
    ),
    MaterialRecord(
        source="mock", material_id="mock-sio2", formula_pretty="SiO2",
        elements=("Si", "O"), band_gap_ev=9.0, density_g_cm3=2.20,
        energy_above_hull_ev_atom=0.10,
    ),
)


class MockMaterialsRepository:
    """Repository that returns pre-built records without any network access."""

    def __init__(self, records: tuple[MaterialRecord, ...]) -> None:
        self._records = records

    def search(self, request: ScreeningRequest) -> RetrievalResult:
        return RetrievalResult(
            source="mock",
            database_version="fixture-v1",
            retrieved_at=FIXED_TEST_TIME,
            records=self._records,
        )

    def query_by_formula(self, formula: str) -> MaterialRecord | None:
        """Look up a single material by its reduced formula."""
        primary: MaterialRecord | None = None
        for record in self._records:
            if record.formula_pretty == formula:
                primary = record
                break
        fallback = next(
            (r for r in _FORMULA_LOOKUP_FIXTURES if r.formula_pretty == formula),
            None,
        )
        if primary is None:
            return fallback
        if fallback is None:
            return primary
        # Preserve caller-provided values and identity while filling properties
        # omitted by the compact bundled screening fixture.
        fields = (
            "band_gap_ev", "energy_above_hull_ev_atom",
            "formation_energy_ev_atom", "density_g_cm3",
        )
        updates = {
            field: getattr(fallback, field)
            for field in fields
            if getattr(primary, field) is None and getattr(fallback, field) is not None
        }
        return primary.model_copy(update=updates) if updates else primary

    def query_all_by_formula(self, formula: str) -> tuple[MaterialRecord, ...]:
        """Return every mock entry for a formula (one per bundled formula)."""
        record = self.query_by_formula(formula)
        return (record,) if record is not None else ()

    def query_by_material_id(self, material_id: str) -> MaterialRecord | None:
        """Return an exact bundled or formula-lookup fixture by id."""
        for record in self._records:
            if record.material_id == material_id:
                # Apply the same missing-property enrichment as formula lookup.
                return self.query_by_formula(record.formula_pretty)
        return next(
            (
                record
                for record in _FORMULA_LOOKUP_FIXTURES
                if record.material_id == material_id
            ),
            None,
        )

    def healthcheck(self) -> bool:
        return True
