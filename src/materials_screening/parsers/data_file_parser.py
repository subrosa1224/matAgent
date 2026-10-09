"""Parser for user-provided CSV / JSON material property data files.

Pure I/O — reads files, returns MaterialRecord sequences.
No network access, no database access, no outlier computation.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from materials_screening.models import MaterialRecord

# Columns that map directly to MaterialRecord fields
_CSV_COLUMN_MAP: dict[str, str] = {
    "formula": "formula_pretty",
    "formula_pretty": "formula_pretty",
    "material_id": "material_id",
    "band_gap_ev": "band_gap_ev",
    "band_gap": "band_gap_ev",
    "formation_energy_ev_atom": "formation_energy_ev_atom",
    "formation_energy": "formation_energy_ev_atom",
    "energy_above_hull_ev_atom": "energy_above_hull_ev_atom",
    "energy_above_hull": "energy_above_hull_ev_atom",
    "density_g_cm3": "density_g_cm3",
    "density": "density_g_cm3",
}

_FLOAT_FIELDS = frozenset({
    "band_gap_ev",
    "formation_energy_ev_atom",
    "energy_above_hull_ev_atom",
    "density_g_cm3",
})


class DataFileParser:
    """Parse user-provided CSV / JSON property data files.

    Expected CSV format::

        formula,band_gap_ev,formation_energy_ev_atom,...
        TiO2,3.2,-3.5,...
        BaTiO3,3.0,-3.8,...

    Expected JSON format::

        [
            {"formula": "TiO2", "band_gap_ev": 3.2, ...},
            {"formula": "BaTiO3", "band_gap_ev": 3.0, ...}
        ]

    Pure I/O: reads file, returns MaterialRecord sequence.
    - No network access
    - No database access
    - No outlier computation (that is OutlierDetectionService's role)
    """

    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".json"})

    def parse(self, path: Path) -> Sequence[MaterialRecord]:
        """Parse a CSV or JSON file into MaterialRecord instances.

        Args:
            path: Path to a .csv or .json file.

        Returns:
            Tuple of MaterialRecord instances.

        Raises:
            ValueError: If the file format is unsupported or parsing fails.
        """
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported data file format: {suffix}. "
                f"Supported: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )
        if suffix == ".csv":
            return self._parse_csv(path)
        if suffix == ".json":
            return self._parse_json(path)
        raise AssertionError("unreachable")

    def _parse_csv(self, path: Path) -> Sequence[MaterialRecord]:
        """Parse a CSV file with header row."""
        # ``utf-8-sig`` transparently accepts both plain UTF-8 and the BOM
        # emitted by Windows PowerShell / Excel exports.
        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError(f"CSV file is empty: {path}")

        lines = text.splitlines()
        reader = csv.DictReader(lines)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path}")

        records: list[MaterialRecord] = []
        for row_num, row in enumerate(reader, start=2):
            mapped = self._map_row(row, path, row_num)
            records.append(mapped)
        return tuple(records)

    def _parse_json(self, path: Path) -> Sequence[MaterialRecord]:
        """Parse a JSON array of material objects."""
        raw = path.read_text(encoding="utf-8-sig").strip()
        if not raw:
            raise ValueError(f"JSON file is empty: {path}")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

        if not isinstance(data, list):
            raise ValueError(
                f"JSON file must contain a top-level array, got "
                f"{type(data).__name__}: {path}"
            )

        records: list[MaterialRecord] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Row {idx + 1}: expected object, got {type(item).__name__}"
                )
            mapped = self._map_row(item, path, idx + 1)
            records.append(mapped)
        return tuple(records)

    def _map_row(
        self,
        row: dict[str, str],
        path: Path,
        row_num: int,
    ) -> MaterialRecord:
        """Map a single row/dict to a MaterialRecord."""
        values: dict[str, object] = {
            "source": f"user_file:{path.name}",
            "material_id": f"user:{path.stem}:{row_num}",
            "formula_pretty": "",
            "elements": (),
        }

        for csv_col, value in row.items():
            field = _CSV_COLUMN_MAP.get(csv_col.lower(), csv_col.lower())
            if field == "formula_pretty":
                values["formula_pretty"] = str(value).strip()
            elif field == "material_id":
                values["material_id"] = str(value).strip()
            elif field in _FLOAT_FIELDS:
                try:
                    values[field] = float(value) if str(value).strip() else None
                except (ValueError, TypeError):
                    values[field] = None
            else:
                # Unknown column — ignore
                pass

        formula = str(values.get("formula_pretty", "")).strip()
        if not formula:
            raise ValueError(
                f"Row {row_num} in {path.name}: 'formula' column is required"
            )

        return MaterialRecord.model_validate(values)
