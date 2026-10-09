"""Durable query/analysis snapshots: JSON metadata plus JSONL records."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from materials_screening.models import MaterialRecord


class QueryResultStore:
    def __init__(self, root: Path = Path("data/material_queries")) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_query(
        self, query_id: str, metadata: dict[str, Any], records: Sequence[MaterialRecord]
    ) -> None:
        folder = self.root / query_id
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str), "utf-8"
        )
        with (folder / "records.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(record.model_dump_json() + "\n")

    def load_records(self, query_id: str) -> tuple[MaterialRecord, ...]:
        path = self._safe_folder(query_id) / "records.jsonl"
        if not path.is_file():
            raise ValueError(f"unknown query_id: {query_id}")
        return tuple(
            MaterialRecord.model_validate_json(line)
            for line in path.read_text("utf-8").splitlines()
            if line.strip()
        )

    def save_analysis(self, analysis_id: str, payload: dict[str, Any]) -> None:
        folder = self.root / analysis_id
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), "utf-8"
        )

    def load_analysis(self, analysis_id: str) -> dict[str, Any]:
        path = self._safe_folder(analysis_id) / "result.json"
        if not path.is_file():
            raise ValueError(f"unknown analysis_id: {analysis_id}")
        value = json.loads(path.read_text("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("analysis result is invalid")
        return value

    def _safe_folder(self, identifier: str) -> Path:
        if not identifier or any(
            ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for ch in identifier
        ):
            raise ValueError("invalid result identifier")
        return self.root / identifier
