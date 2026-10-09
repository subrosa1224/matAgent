"""Atomic local snapshots for literature query results."""

from __future__ import annotations

import json
from pathlib import Path

from .models import (
    CandidateLiteratureScreenOutput,
    ExpandedQuery,
    LiteratureSearchOutput,
)


class LiteratureQueryStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def save(self, result: LiteratureSearchOutput) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = (self.root / result.query_id / "result.json").resolve()
        if self.root not in target.parents:
            raise ValueError("literature query path escapes configured root")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = result.model_dump_json(indent=2).encode("utf-8")
        if target.exists():
            existing = target.read_bytes()
            if existing != payload:
                raise ValueError(
                    "literature query snapshot already exists with different content"
                )
            return target
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
        return target

    def exists(self, query_id: str) -> bool:
        if not query_id.startswith("lit-") or not query_id[4:].isalnum():
            return False
        target = (self.root / query_id / "result.json").resolve()
        return self.root in target.parents and target.is_file()

    def replace_degraded(self, result: LiteratureSearchOutput) -> Path:
        """Replace a recovered snapshot while retaining the degraded response."""
        target = (self.root / result.query_id / "result.json").resolve()
        if self.root not in target.parents or not target.is_file():
            raise ValueError("degraded literature query snapshot does not exist")
        existing = LiteratureSearchOutput.model_validate_json(target.read_bytes())
        if not any(row.status == "degraded" for row in existing.provider_statuses):
            raise ValueError("only a degraded query snapshot can be replaced")
        if any(row.status == "degraded" for row in result.provider_statuses):
            raise ValueError("replacement query snapshot is still degraded")
        archive = target.with_name("result.degraded.json")
        if not archive.exists():
            archive.write_bytes(target.read_bytes())
        temporary = target.with_suffix(".tmp")
        temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(target)
        return target

    def load(self, query_id: str) -> LiteratureSearchOutput:
        if not query_id.startswith("lit-") or not query_id[4:].isalnum():
            raise ValueError("invalid literature query id")
        target = (self.root / query_id / "result.json").resolve()
        if self.root not in target.parents:
            raise ValueError("literature query path escapes configured root")
        return LiteratureSearchOutput.model_validate(
            json.loads(target.read_text("utf-8"))
        )

    def save_candidate_screen(
        self, result: CandidateLiteratureScreenOutput
    ) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = (self.root / result.screening_id / "result.json").resolve()
        if self.root not in target.parents:
            raise ValueError("candidate screening path escapes configured root")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = result.model_dump_json(indent=2).encode("utf-8")
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError(
                    "candidate screening snapshot exists with different content"
                )
            return target
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
        return target

    def candidate_screen_exists(self, screening_id: str) -> bool:
        if not screening_id.startswith("lit-screen-"):
            return False
        suffix = screening_id.removeprefix("lit-screen-")
        if len(suffix) != 24 or any(char not in "0123456789abcdef" for char in suffix):
            return False
        target = (self.root / screening_id / "result.json").resolve()
        return self.root in target.parents and target.is_file()

    def load_candidate_screen(
        self, screening_id: str
    ) -> CandidateLiteratureScreenOutput:
        if not self.candidate_screen_exists(screening_id):
            raise ValueError("unknown candidate screening id")
        target = (self.root / screening_id / "result.json").resolve()
        return CandidateLiteratureScreenOutput.model_validate_json(
            target.read_bytes()
        )

    def latest_compatible_success(
        self, expanded: ExpandedQuery, *, exclude_query_id: str
    ) -> LiteratureSearchOutput | None:
        """Find the newest successful snapshot for the same expanded concepts."""
        if not self.root.is_dir():
            return None
        matches: list[LiteratureSearchOutput] = []
        for target in self.root.glob("lit-*/result.json"):
            try:
                result = LiteratureSearchOutput.model_validate_json(
                    target.read_bytes()
                )
            except (OSError, ValueError):
                continue
            if (
                result.query_id == exclude_query_id
                or result.expanded_query is None
                or not result.papers
                or any(row.status == "degraded" for row in result.provider_statuses)
                or not _compatible_expansion(result.expanded_query, expanded)
            ):
                continue
            matches.append(result)
        return max(matches, key=lambda item: item.created_at, default=None)


def _compatible_expansion(left: ExpandedQuery, right: ExpandedQuery) -> bool:
    return (
        set(left.normalized_materials) == set(right.normalized_materials)
        and set(left.performance_terms) == set(right.performance_terms)
        and set(left.process_terms) == set(right.process_terms)
    )
