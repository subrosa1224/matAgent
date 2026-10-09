"""Unit tests for MP query building and summary document mapping (M3)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from materials_screening.errors import RepositoryMappingError
from materials_screening.models import (
    CrystalSystem,
    FloatRange,
    PropertyValueType,
    ScreeningRequest,
    SymmetryInfo,
)
from materials_screening.repositories.materials_project import (
    build_mp_query,
    map_summary_document,
    range_to_mp_tuple,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
RETRIEVED_AT = datetime(2026, 8, 5, 0, 0, tzinfo=UTC)
DATABASE_VERSION = "v2026.08"


def _load_document(material_id: str) -> dict[str, object]:
    raw = (FIXTURES_DIR / "mp_documents.json").read_text(encoding="utf-8")
    documents = json.loads(raw)
    return next(
        document for document in documents if document["material_id"] == material_id
    )


class FakeSummaryDoc:
    """Attribute-style SummaryDoc stand-in backed by a plain dict."""

    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> object:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None


class TestRangeToMpTuple:
    def test_open_min_uses_default(self) -> None:
        assert range_to_mp_tuple(FloatRange(max=5.0), 0.0, 100.0) == (0.0, 5.0)

    def test_open_max_uses_default(self) -> None:
        assert range_to_mp_tuple(FloatRange(min=1.0), 0.0, 100.0) == (1.0, 100.0)

    def test_closed_range_passthrough(self) -> None:
        assert range_to_mp_tuple(FloatRange(min=1.0, max=2.0), 0.0, 100.0) == (
            1.0,
            2.0,
        )


class TestBuildMpQuery:
    def test_default_query(self) -> None:
        query = build_mp_query(ScreeningRequest())
        assert query["deprecated"] is False
        assert query["all_fields"] is False
        assert query["fields"] == [
            "material_id",
            "formula_pretty",
            "elements",
            "chemsys",
            "band_gap",
            "energy_above_hull",
            "formation_energy_per_atom",
            "density",
            "is_metal",
            "is_gap_direct",
            "is_stable",
            "theoretical",
            "deprecated",
            "symmetry",
            "structure",
        ]
        assert query["chunk_size"] == 500

    def test_full_request(self) -> None:
        request = ScreeningRequest(
            required_elements=("Fe", "O"),
            excluded_elements=("Pb",),
            chemsys="Fe-O",
            formula="Fe2O3",
            band_gap_ev=FloatRange(min=1.2, max=2.0),
            energy_above_hull_ev_atom=FloatRange(min=0.0, max=0.05),
            density_g_cm3=FloatRange(min=2.0, max=5.0),
            crystal_system="Cubic",
            spacegroup_numbers=(225, 221),
            is_metal=False,
            is_stable=True,
            theoretical=False,
        )
        query = build_mp_query(request)
        assert query["band_gap"] == (1.2, 2.0)
        assert query["energy_above_hull"] == (0.0, 0.05)
        assert query["density"] == (2.0, 5.0)
        assert query["elements"] == ["Fe", "O"]
        assert query["exclude_elements"] == ["Pb"]
        assert query["chemsys"] == "Fe-O"
        assert query["formula"] == "Fe2O3"
        assert query["crystal_system"] == "Cubic"
        assert query["spacegroup_number"] == [221, 225]
        assert query["is_metal"] is False
        assert query["is_stable"] is True
        assert query["theoretical"] is False

    def test_open_ranges_use_defaults(self) -> None:
        query = build_mp_query(
            ScreeningRequest(
                band_gap_ev=FloatRange(min=1.2),
                density_g_cm3=FloatRange(max=5.0),
            )
        )
        assert query["band_gap"] == (1.2, 100.0)
        assert query["density"] == (0.0, 5.0)

    def test_limit_is_not_passed_to_api(self) -> None:
        query = build_mp_query(ScreeningRequest(limit=5))
        assert "limit" not in query

    def test_element_count_is_passed_as_supported_mp_constraint(self) -> None:
        query = build_mp_query(
            ScreeningRequest(required_elements=("O",), num_elements=2)
        )
        assert query["num_elements"] == 2

    def test_query_contains_no_none_values(self) -> None:
        query = build_mp_query(ScreeningRequest())
        assert all(value is not None for value in query.values())


class TestMapSummaryDocument:
    def test_maps_dict_document(self) -> None:
        record = map_summary_document(
            _load_document("mp-1"), DATABASE_VERSION, RETRIEVED_AT
        )
        assert record.source == "materials_project"
        assert record.material_id == "mp-1"
        assert record.formula_pretty == "LiFeO2"
        assert record.elements == ("Fe", "Li", "O")
        assert record.chemsys == "Fe-Li-O"
        assert record.band_gap_ev == 2.1
        assert record.energy_above_hull_ev_atom == 0.01
        assert record.formation_energy_ev_atom == -1.5
        assert record.density_g_cm3 == 5.2
        assert record.is_metal is False
        assert record.is_gap_direct is True
        assert record.is_stable is True
        assert record.theoretical is False
        assert record.deprecated is False
        assert record.symmetry == SymmetryInfo(
            crystal_system=CrystalSystem.TRIGONAL,
            symbol="R-3c",
            number=167,
        )
        assert record.structure_dict is not None
        assert record.structure_dict == _load_document("mp-1")["structure"]
        assert len(record.provenance) == 10

    def test_maps_attribute_object_document(self) -> None:
        record = map_summary_document(
            FakeSummaryDoc(_load_document("mp-1")), DATABASE_VERSION, RETRIEVED_AT
        )
        assert record.material_id == "mp-1"
        assert record.band_gap_ev == 2.1
        assert record.symmetry == SymmetryInfo(
            crystal_system=CrystalSystem.TRIGONAL,
            symbol="R-3c",
            number=167,
        )

    def test_missing_structure_maps_to_none(self) -> None:
        record = map_summary_document(
            _load_document("mp-2"), DATABASE_VERSION, RETRIEVED_AT
        )
        assert record.structure_dict is None
        property_names = {entry.property_name for entry in record.provenance}
        assert "structure_dict" not in property_names
        assert len(record.provenance) == 9

    def test_minimal_document_maps_optional_fields_to_none(self) -> None:
        data = {
            "material_id": "mp-99",
            "formula_pretty": "Fe",
            "elements": ["Fe"],
        }
        record = map_summary_document(data, DATABASE_VERSION, RETRIEVED_AT)
        assert record.material_id == "mp-99"
        assert record.band_gap_ev is None
        assert record.energy_above_hull_ev_atom is None
        assert record.formation_energy_ev_atom is None
        assert record.density_g_cm3 is None
        assert record.is_metal is None
        assert record.is_gap_direct is None
        assert record.is_stable is None
        assert record.theoretical is None
        assert record.deprecated is None
        assert record.symmetry is None
        assert record.structure_dict is None
        assert record.provenance == ()

    def test_missing_material_id_raises(self) -> None:
        data = dict(_load_document("mp-2"))
        del data["material_id"]
        with pytest.raises(RepositoryMappingError, match="material_id is missing"):
            map_summary_document(data, DATABASE_VERSION, RETRIEVED_AT)

    def test_mapping_error_includes_material_id(self) -> None:
        data = dict(_load_document("mp-2"))
        data["material_id"] = "mp-7"
        data["structure"] = "not-a-structure"
        with pytest.raises(RepositoryMappingError, match="mp-7"):
            map_summary_document(data, DATABASE_VERSION, RETRIEVED_AT)

    def test_missing_numeric_fields_map_to_none(self) -> None:
        record = map_summary_document(
            _load_document("mp-3"), DATABASE_VERSION, RETRIEVED_AT
        )
        assert record.band_gap_ev is None
        assert record.density_g_cm3 is None
        assert record.formation_energy_ev_atom is None
        assert record.energy_above_hull_ev_atom == 0.0

    def test_structure_object_with_as_dict(self) -> None:
        class FakeStructure:
            def as_dict(self) -> dict[str, object]:
                return {"lattice": {"a": 1.0}, "sites": []}

        data = dict(_load_document("mp-2"))
        data["structure"] = FakeStructure()
        record = map_summary_document(data, DATABASE_VERSION, RETRIEVED_AT)
        assert record.structure_dict == {"lattice": {"a": 1.0}, "sites": []}

    def test_element_objects_convert_to_symbols(self) -> None:
        class FakeElement:
            def __init__(self, symbol: str) -> None:
                self._symbol = symbol

            def __str__(self) -> str:
                return self._symbol

        data = dict(_load_document("mp-2"))
        data["elements"] = [FakeElement("O"), FakeElement("Fe")]
        record = map_summary_document(data, DATABASE_VERSION, RETRIEVED_AT)
        assert record.elements == ("Fe", "O")

    def test_invalid_structure_type_raises(self) -> None:
        data = dict(_load_document("mp-2"))
        data["material_id"] = "mp-7"
        data["structure"] = "not-a-structure"
        with pytest.raises(RepositoryMappingError):
            map_summary_document(data, DATABASE_VERSION, RETRIEVED_AT)

    def test_invalid_as_dict_return_raises(self) -> None:
        class BadStructure:
            def as_dict(self) -> str:
                return "nope"

        data = dict(_load_document("mp-2"))
        data["structure"] = BadStructure()
        with pytest.raises(RepositoryMappingError):
            map_summary_document(data, DATABASE_VERSION, RETRIEVED_AT)

    def test_symmetry_object_mapping(self) -> None:
        class FakeSymmetry:
            crystal_system = "Cubic"
            symbol = "Fm-3m"
            number = 225

        data = dict(_load_document("mp-2"))
        data["symmetry"] = FakeSymmetry()
        record = map_summary_document(data, DATABASE_VERSION, RETRIEVED_AT)
        assert record.symmetry == SymmetryInfo(
            crystal_system=CrystalSystem.CUBIC,
            symbol="Fm-3m",
            number=225,
        )

    def test_symmetry_enum_value_mapping(self) -> None:
        class FakeEmmetCrystalSystem:
            value = "Monoclinic"

        class FakeSymmetry:
            crystal_system = FakeEmmetCrystalSystem()
            symbol = "P2/c"
            number = 13

        data = dict(_load_document("mp-2"))
        data["symmetry"] = FakeSymmetry()
        record = map_summary_document(data, DATABASE_VERSION, RETRIEVED_AT)
        assert record.symmetry == SymmetryInfo(
            crystal_system=CrystalSystem.MONOCLINIC,
            symbol="P2/c",
            number=13,
        )

    def test_invalid_symmetry_number_raises(self) -> None:
        data = dict(_load_document("mp-2"))
        data["symmetry"] = {
            "crystal_system": "Cubic",
            "symbol": "Fm-3m",
            "number": 231,
        }
        with pytest.raises(RepositoryMappingError):
            map_summary_document(data, DATABASE_VERSION, RETRIEVED_AT)

    def test_elements_not_a_list_raises(self) -> None:
        data = dict(_load_document("mp-2"))
        data["elements"] = "FeO"
        with pytest.raises(RepositoryMappingError):
            map_summary_document(data, DATABASE_VERSION, RETRIEVED_AT)

    def test_unknown_extra_fields_are_ignored(self) -> None:
        data = dict(_load_document("mp-2"))
        data["junk"] = {"anything": 1}
        record = map_summary_document(data, DATABASE_VERSION, RETRIEVED_AT)
        assert record.material_id == "mp-2"

    def test_provenance_content(self) -> None:
        record = map_summary_document(
            _load_document("mp-1"), DATABASE_VERSION, RETRIEVED_AT
        )
        band_gap_provenance = next(
            entry for entry in record.provenance if entry.property_name == "band_gap_ev"
        )
        assert band_gap_provenance.source == "materials_project"
        assert band_gap_provenance.source_material_id == "mp-1"
        assert band_gap_provenance.value_type is PropertyValueType.DFT_CALCULATED
        assert band_gap_provenance.database_version == DATABASE_VERSION
        assert band_gap_provenance.retrieved_at == RETRIEVED_AT
        assert band_gap_provenance.method is None

    def test_provenance_property_names(self) -> None:
        record = map_summary_document(
            _load_document("mp-1"), DATABASE_VERSION, RETRIEVED_AT
        )
        property_names = {entry.property_name for entry in record.provenance}
        assert property_names == {
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
        }
