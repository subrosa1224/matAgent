"""Repository contracts and shared result models (M3)."""

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from materials_screening.models import MaterialRecord, ScreeningRequest


class RetrievalResult(BaseModel):
    """Result of one repository query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    database_version: str | None
    retrieved_at: datetime
    records: tuple[MaterialRecord, ...]
    warnings: tuple[str, ...] = ()


class MaterialsRepository(Protocol):
    """Protocol implemented by all material repositories."""

    def search(self, request: ScreeningRequest) -> RetrievalResult: ...

    def query_by_formula(self, formula: str) -> MaterialRecord | None:
        """Look up one material by its reduced formula (legacy convenience API).

        Returns None when the formula is not found.
        """
        ...

    def query_all_by_formula(self, formula: str) -> tuple[MaterialRecord, ...]:
        """Return every available polymorph/entry for a reduced formula."""
        ...

    def query_by_material_id(self, material_id: str) -> MaterialRecord | None:
        """Look up exactly one material entry by its database identifier."""
        ...

    def healthcheck(self) -> bool: ...
