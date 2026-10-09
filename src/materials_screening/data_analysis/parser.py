"""Safe parser for general-purpose CSV, JSON, and XLSX tabular datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]


class TabularDataParser:
    """Parse bounded CSV/JSON/XLSX files without domain-specific mapping."""

    SUPPORTED_EXTENSIONS = frozenset({".csv", ".json", ".xlsx"})

    def __init__(
        self,
        *,
        max_file_bytes: int = 50 * 1024 * 1024,
        max_rows: int = 100_000,
        max_columns: int = 500,
        max_cells: int = 5_000_000,
        max_cell_characters: int = 10_000,
    ) -> None:
        self.max_file_bytes = max_file_bytes
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.max_cells = max_cells
        self.max_cell_characters = max_cell_characters

    def parse(self, path: Path) -> pd.DataFrame:
        """Return a validated DataFrame for one local CSV or JSON file."""

        resolved = path.resolve()
        if not resolved.is_file():
            raise ValueError("dataset file does not exist")
        suffix = resolved.suffix.casefold()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"unsupported dataset format: {suffix or '<none>'}; "
                "supported formats are .csv, .json and .xlsx"
            )
        size = resolved.stat().st_size
        if size == 0:
            raise ValueError("dataset file is empty")
        if size > self.max_file_bytes:
            raise ValueError(
                f"dataset file exceeds {self.max_file_bytes} byte limit"
            )

        if suffix == ".csv":
            frame = self._parse_csv(resolved)
        elif suffix == ".json":
            frame = self._parse_json(resolved)
        else:
            frame = self._parse_xlsx(resolved)
        self._validate_shape(frame)
        self._validate_cells(frame)
        return frame

    def _parse_csv(self, path: Path) -> pd.DataFrame:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle), None)
        except UnicodeDecodeError as exc:
            raise ValueError("CSV must use UTF-8 encoding") from exc
        if not header or not any(name.strip() for name in header):
            raise ValueError("CSV must contain a non-blank header row")
        normalized = [name.strip() for name in header]
        if any(not name for name in normalized):
            raise ValueError("CSV column names must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("CSV column names must be unique")

        try:
            frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError as exc:
            raise ValueError("CSV must use UTF-8 encoding") from exc
        except (pd.errors.ParserError, ValueError) as exc:
            raise ValueError(f"invalid CSV dataset: {exc}") from exc
        frame.columns = normalized
        return frame

    def _parse_json(self, path: Path) -> pd.DataFrame:
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("JSON must use UTF-8 encoding") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON dataset: {exc.msg}") from exc
        if not isinstance(payload, list):
            raise ValueError("JSON dataset must be a top-level array of objects")
        if not payload:
            raise ValueError("JSON dataset must contain at least one row")
        for row_number, row in enumerate(payload, 1):
            if not isinstance(row, dict):
                raise ValueError(f"JSON row {row_number} must be an object")
            for key, value in row.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(
                        f"JSON row {row_number} contains an invalid column name"
                    )
                if isinstance(value, (dict, list)):
                    raise ValueError(
                        f"JSON row {row_number} column {key!r} contains nested data"
                    )
        return pd.DataFrame(payload)

    def _parse_xlsx(self, path: Path) -> pd.DataFrame:
        try:
            header_frame = pd.read_excel(
                path,
                sheet_name=0,
                header=None,
                nrows=1,
                engine="openpyxl",
            )
        except ImportError as exc:
            raise ValueError(
                "XLSX support requires the analysis extra with openpyxl"
            ) from exc
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid XLSX dataset: {exc}") from exc
        if header_frame.empty:
            raise ValueError("XLSX must contain a non-blank header row")
        header = [
            "" if pd.isna(value) else str(value).strip()
            for value in header_frame.iloc[0].tolist()
        ]
        if not header or any(not name for name in header):
            raise ValueError("XLSX column names must not be blank")
        if len(header) != len(set(header)):
            raise ValueError("XLSX column names must be unique")
        try:
            frame = pd.read_excel(path, sheet_name=0, engine="openpyxl")
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid XLSX dataset: {exc}") from exc
        frame.columns = header
        return frame

    def _validate_shape(self, frame: pd.DataFrame) -> None:
        rows, columns = frame.shape
        if rows == 0:
            raise ValueError("dataset must contain at least one data row")
        if columns == 0:
            raise ValueError("dataset must contain at least one column")
        if rows > self.max_rows:
            raise ValueError(f"dataset exceeds {self.max_rows} row limit")
        if columns > self.max_columns:
            raise ValueError(f"dataset exceeds {self.max_columns} column limit")
        if rows * columns > self.max_cells:
            raise ValueError(f"dataset exceeds {self.max_cells} cell limit")
        names = [str(name) for name in frame.columns]
        if any(not name.strip() or len(name) > 256 for name in names):
            raise ValueError("dataset column names must be non-blank and bounded")
        if len(names) != len(set(names)):
            raise ValueError("dataset column names must be unique")

    def _validate_cells(self, frame: pd.DataFrame) -> None:
        for name in frame.select_dtypes(include=["object", "string"]).columns:
            values = frame[name].dropna()
            if values.empty:
                continue
            lengths = values.astype(str).str.len()
            if bool((lengths > self.max_cell_characters).any()):
                raise ValueError(
                    f"column {str(name)!r} contains a cell longer than "
                    f"{self.max_cell_characters} characters"
                )


def schema_fingerprint(frame: pd.DataFrame) -> str:
    """Return a stable SHA-256 fingerprint for names and inferred dtypes."""

    import hashlib

    schema: list[dict[str, Any]] = [
        {"name": str(name), "dtype": str(frame[name].dtype)}
        for name in frame.columns
    ]
    encoded = json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
