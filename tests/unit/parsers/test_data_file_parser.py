"""Unit tests for DataFileParser."""

from __future__ import annotations

import json
from os import close as _os_close
from pathlib import Path
from tempfile import mkstemp

import pytest

from materials_screening.parsers.data_file_parser import DataFileParser


def _write_temp(content: str, suffix: str = ".csv") -> Path:
    fd, path = mkstemp(suffix=suffix)
    _os_close(fd)
    p = Path(path)
    p.write_text(content, encoding="utf-8")
    return p


# ── CSV ────────────────────────────────────────────────────────────────────


class TestCsvParsing:
    def test_utf8_bom_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "powershell.csv"
        path.write_text(
            "formula,band_gap_ev\nTiO2,3.2", encoding="utf-8-sig"
        )

        records = DataFileParser().parse(path)

        assert records[0].formula_pretty == "TiO2"
        assert records[0].band_gap_ev == 3.2

    def test_basic_csv(self) -> None:
        csv = (
            "formula,band_gap_ev,formation_energy_ev_atom\n"
            "TiO2,3.2,-3.5\nBaTiO3,3.0,-3.8"
        )
        path = _write_temp(csv)
        try:
            records = DataFileParser().parse(path)
            assert len(records) == 2
            assert records[0].formula_pretty == "TiO2"
            assert records[0].band_gap_ev == 3.2
            assert records[0].formation_energy_ev_atom == -3.5
            assert records[1].formula_pretty == "BaTiO3"
            assert records[1].band_gap_ev == 3.0
        finally:
            path.unlink(missing_ok=True)

    def test_csv_with_material_id_column(self) -> None:
        csv = "formula,material_id,band_gap_ev\nTiO2,mp-1234,3.2"
        path = _write_temp(csv)
        try:
            records = DataFileParser().parse(path)
            assert records[0].material_id == "mp-1234"
            assert records[0].formula_pretty == "TiO2"
        finally:
            path.unlink(missing_ok=True)

    def test_csv_missing_formula_raises(self) -> None:
        csv = "band_gap_ev,density\n3.2,4.0"
        path = _write_temp(csv)
        try:
            with pytest.raises(ValueError, match="formula"):
                DataFileParser().parse(path)
        finally:
            path.unlink(missing_ok=True)

    def test_csv_empty_value_treated_as_none(self) -> None:
        csv = "formula,band_gap_ev\nTiO2,\nBaTiO3,3.0"
        path = _write_temp(csv)
        try:
            records = DataFileParser().parse(path)
            assert records[0].band_gap_ev is None
            assert records[1].band_gap_ev == 3.0
        finally:
            path.unlink(missing_ok=True)

    def test_csv_non_numeric_float_column(self) -> None:
        csv = "formula,band_gap_ev\nTiO2,hello"
        path = _write_temp(csv)
        try:
            records = DataFileParser().parse(path)
            assert records[0].band_gap_ev is None
        finally:
            path.unlink(missing_ok=True)

    def test_csv_column_name_aliases(self) -> None:
        csv = "formula,band_gap,density,formation_energy\nTiO2,3.2,4.0,-3.5"
        path = _write_temp(csv)
        try:
            records = DataFileParser().parse(path)
            r = records[0]
            assert r.band_gap_ev == 3.2
            assert r.density_g_cm3 == 4.0
            assert r.formation_energy_ev_atom == -3.5
        finally:
            path.unlink(missing_ok=True)

    def test_csv_empty_file_raises(self) -> None:
        path = _write_temp("")
        try:
            with pytest.raises(ValueError, match="empty"):
                DataFileParser().parse(path)
        finally:
            path.unlink(missing_ok=True)

    def test_csv_unknown_columns_ignored(self) -> None:
        csv = "formula,band_gap_ev,color,weight\nTiO2,3.2,white,79.9"
        path = _write_temp(csv)
        try:
            records = DataFileParser().parse(path)
            assert records[0].band_gap_ev == 3.2
        finally:
            path.unlink(missing_ok=True)


# ── JSON ───────────────────────────────────────────────────────────────────


class TestJsonParsing:
    def test_basic_json(self) -> None:
        data = [
            {"formula": "TiO2", "band_gap_ev": 3.2},
            {"formula": "BaTiO3", "band_gap_ev": 3.0},
        ]
        path = _write_temp(json.dumps(data), suffix=".json")
        try:
            records = DataFileParser().parse(path)
            assert len(records) == 2
            assert records[0].formula_pretty == "TiO2"
            assert records[1].formula_pretty == "BaTiO3"
        finally:
            path.unlink(missing_ok=True)

    def test_json_missing_formula_raises(self) -> None:
        path = _write_temp('[{"band_gap_ev": 3.2}]', suffix=".json")
        try:
            with pytest.raises(ValueError, match="formula"):
                DataFileParser().parse(path)
        finally:
            path.unlink(missing_ok=True)

    def test_json_not_array_raises(self) -> None:
        path = _write_temp('{"formula": "TiO2"}', suffix=".json")
        try:
            with pytest.raises(ValueError, match="array"):
                DataFileParser().parse(path)
        finally:
            path.unlink(missing_ok=True)

    def test_json_invalid_syntax_raises(self) -> None:
        path = _write_temp("not json", suffix=".json")
        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                DataFileParser().parse(path)
        finally:
            path.unlink(missing_ok=True)

    def test_json_empty_file_raises(self) -> None:
        path = _write_temp("", suffix=".json")
        try:
            with pytest.raises(ValueError, match="empty"):
                DataFileParser().parse(path)
        finally:
            path.unlink(missing_ok=True)


# ── Unsupported format ─────────────────────────────────────────────────────


class TestUnsupportedFormat:
    def test_unsupported_extension_raises(self) -> None:
        path = _write_temp("data", suffix=".xlsx")
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                DataFileParser().parse(path)
        finally:
            path.unlink(missing_ok=True)
