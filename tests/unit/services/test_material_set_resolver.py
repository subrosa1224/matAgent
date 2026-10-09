"""Unit tests for MaterialSetResolver."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import mkstemp
from os import close as _os_close
from unittest.mock import Mock

import pytest

from materials_screening.models import MaterialRecord
from materials_screening.parsers.data_file_parser import DataFileParser
from materials_screening.services.material_set_resolver import MaterialSetResolver
from materials_screening.sub_agents.outlier_detection.models import MaterialSetReference


def _rec(material_id: str = "mp-1", formula: str = "TiO2") -> MaterialRecord:
    return MaterialRecord(
        source="mock",
        material_id=material_id,
        formula_pretty=formula,
        elements=("Ti", "O"),
        band_gap_ev=2.0,
    )


def _write_temp(content: str, suffix: str = ".csv") -> Path:
    fd, path = mkstemp(suffix=suffix)
    _os_close(fd)
    p = Path(path)
    p.write_text(content, encoding="utf-8")
    return p


# ── workflow_thread_ids path ────────────────────────────────────────────────


class TestResolveFromIds:
    def test_resolves_single_thread(self) -> None:
        fake_reader = Mock()
        fake_reader.read.return_value = {
            "ranked_materials": [
                {
                    "record": {
                        "source": "mock",
                        "material_id": "mp-1234",
                        "formula_pretty": "TiO2",
                        "elements": ["Ti", "O"],
                    },
                    "rank": 1,
                }
            ]
        }
        repo = Mock()
        parser = Mock()
        resolver = MaterialSetResolver(fake_reader, repo, parser)

        ref = MaterialSetReference(workflow_thread_ids=("thread-1",))
        records = resolver.resolve(ref)
        assert len(records) == 1
        assert records[0].material_id == "mp-1234"
        assert records[0].formula_pretty == "TiO2"

    def test_no_records_in_artifact_raises(self) -> None:
        fake_reader = Mock()
        fake_reader.read.return_value = {"ranked_materials": []}
        resolver = MaterialSetResolver(fake_reader, Mock(), Mock())

        ref = MaterialSetReference(workflow_thread_ids=("thread-empty",))
        with pytest.raises(ValueError, match="No material records"):
            resolver.resolve(ref)

    def test_resolves_multiple_threads(self) -> None:
        fake_reader = Mock()
        fake_reader.read.side_effect = [
            {
                "ranked_materials": [
                    {
                        "record": {
                            "source": "mock",
                            "material_id": "mp-1",
                            "formula_pretty": "TiO2",
                            "elements": ["Ti", "O"],
                        },
                        "rank": 1,
                    }
                ]
            },
            {
                "ranked_materials": [
                    {
                        "record": {
                            "source": "mock",
                            "material_id": "mp-2",
                            "formula_pretty": "BaTiO3",
                            "elements": ["Ba", "Ti", "O"],
                        },
                        "rank": 1,
                    }
                ]
            },
        ]
        resolver = MaterialSetResolver(fake_reader, Mock(), Mock())

        ref = MaterialSetReference(workflow_thread_ids=("t1", "t2"))
        records = resolver.resolve(ref)
        assert len(records) == 2
        ids = {r.material_id for r in records}
        assert ids == {"mp-1", "mp-2"}

    def test_handles_records_field(self) -> None:
        fake_reader = Mock()
        fake_reader.read.return_value = {
            "records": [
                {
                    "source": "mock",
                    "material_id": "mp-1",
                    "formula_pretty": "Si",
                    "elements": ["Si"],
                }
            ]
        }
        resolver = MaterialSetResolver(fake_reader, Mock(), Mock())
        ref = MaterialSetReference(workflow_thread_ids=("t1",))
        records = resolver.resolve(ref)
        assert len(records) == 1
        assert records[0].material_id == "mp-1"


# ── material_formulas path ──────────────────────────────────────────────────


class TestResolveFromFormulas:
    def test_resolves_formulas(self) -> None:
        fake_reader = Mock()
        repo = Mock()
        repo.query_all_by_formula.side_effect = lambda f: (
            (_rec("mp-1", "TiO2"),) if f == "TiO2" else ()
        )
        parser = Mock()
        resolver = MaterialSetResolver(fake_reader, repo, parser)

        ref = MaterialSetReference(material_formulas=("TiO2",))
        records = resolver.resolve(ref)
        assert len(records) == 1
        assert records[0].formula_pretty == "TiO2"

    def test_no_formulas_found_raises(self) -> None:
        repo = Mock()
        repo.query_all_by_formula.return_value = ()
        resolver = MaterialSetResolver(Mock(), repo, Mock())

        ref = MaterialSetReference(material_formulas=("UnknownXyz",))
        with pytest.raises(ValueError, match="No materials found"):
            resolver.resolve(ref)

    def test_partial_match_returns_found_with_warning(self) -> None:
        repo = Mock()
        repo.query_all_by_formula.side_effect = lambda f: (
            (_rec("mp-1", f),) if f == "TiO2" else ()
        )
        resolver = MaterialSetResolver(Mock(), repo, Mock())

        ref = MaterialSetReference(material_formulas=("TiO2", "Unknown"))
        records, warnings = resolver.resolve_with_warnings(ref)
        assert len(records) == 1
        assert records[0].formula_pretty == "TiO2"
        assert warnings == (
            "No material found for formula 'Unknown'; excluded from analysis",
        )

    def test_resolves_all_polymorphs_for_formula(self) -> None:
        repo = Mock()
        repo.query_all_by_formula.return_value = (
            _rec("mp-1", "TiO2"),
            _rec("mp-2", "TiO2"),
        )
        resolver = MaterialSetResolver(Mock(), repo, Mock())

        records = resolver.resolve(MaterialSetReference(material_formulas=("TiO2",)))

        assert [record.material_id for record in records] == ["mp-1", "mp-2"]


class TestResolveFromMaterialIds:
    def test_resolves_only_exact_ids(self) -> None:
        repo = Mock()
        repo.query_by_material_id.side_effect = lambda material_id: (
            _rec(material_id, "TiO2") if material_id == "mp-2" else None
        )
        resolver = MaterialSetResolver(Mock(), repo, Mock())

        records = resolver.resolve(MaterialSetReference(material_ids=("mp-2",)))

        assert [record.material_id for record in records] == ["mp-2"]
        repo.query_all_by_formula.assert_not_called()


# ── data_file path ──────────────────────────────────────────────────────────


class TestResolveFromFile:
    def test_resolves_csv_file(self, tmp_path: Path) -> None:
        csv = "formula,band_gap_ev\nTiO2,3.2"
        path = tmp_path / "materials.csv"
        path.write_text(csv, encoding="utf-8")
        parser = DataFileParser()
        resolver = MaterialSetResolver(
            Mock(), Mock(), parser, allowed_data_dir=tmp_path
        )
        ref = MaterialSetReference(data_file=str(path))
        records = resolver.resolve(ref)
        assert len(records) == 1
        assert records[0].formula_pretty == "TiO2"
        assert records[0].band_gap_ev == 3.2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        resolver = MaterialSetResolver(
            Mock(), Mock(), DataFileParser(), allowed_data_dir=tmp_path
        )
        ref = MaterialSetReference(data_file=str(tmp_path / "missing.csv"))
        with pytest.raises(ValueError, match="not found"):
            resolver.resolve(ref)

    def test_file_outside_allowed_directory_raises(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside.csv"
        outside.write_text("formula,band_gap_ev\nTiO2,3.2", encoding="utf-8")
        resolver = MaterialSetResolver(
            Mock(), Mock(), DataFileParser(), allowed_data_dir=tmp_path
        )
        with pytest.raises(ValueError, match="allowed directory"):
            resolver.resolve(MaterialSetReference(data_file=str(outside)))


# ── Input validation ────────────────────────────────────────────────────────


class TestMaterialSetReferenceValidation:
    def test_exactly_one_field_required(self) -> None:
        with pytest.raises(ValueError, match="必须且只能"):
            MaterialSetReference()

    def test_too_many_fields_raises(self) -> None:
        with pytest.raises(ValueError, match="必须且只能"):
            MaterialSetReference(
                workflow_thread_ids=("run-1",), material_formulas=("TiO2",)
            )

    def test_frozen_after_construction(self) -> None:
        ref = MaterialSetReference(workflow_thread_ids=("run-1",))
        with pytest.raises(Exception):
            ref.workflow_thread_ids = ("run-2",)  # type: ignore[misc]
