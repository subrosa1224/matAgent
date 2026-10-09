"""Manual real Materials Project API smoke test (M7).

Runs only when RUN_REAL_MP_TESTS=1 and MP_API_KEY is set.
Uses a single fixed material ID so the download stays minimal.
"""

import os
from datetime import UTC, datetime

import pytest
from mp_api.client import MPRester

from materials_screening.repositories.materials_project import (
    REQUESTED_FIELDS,
    map_summary_document,
)

SMOKE_MATERIAL_ID = "mp-149"


@pytest.mark.real_api
def test_summary_search_mapping_and_provenance() -> None:
    """Query one fixed material, map it, and verify provenance metadata."""
    if os.getenv("RUN_REAL_MP_TESTS") != "1":
        pytest.skip("set RUN_REAL_MP_TESTS=1 to run real API tests")
    api_key = os.getenv("MP_API_KEY", "").strip()
    if not api_key:
        pytest.skip("MP_API_KEY is not set")

    with MPRester(api_key=api_key, mute_progress_bars=True) as mpr:
        database_version = mpr.db_version or None
        documents = mpr.materials.summary.search(
            material_ids=[SMOKE_MATERIAL_ID],
            all_fields=False,
            fields=list(REQUESTED_FIELDS),
        )

    assert len(documents) >= 1, "narrow smoke query returned no documents"
    record = map_summary_document(
        documents[0],
        database_version=database_version,
        retrieved_at=datetime.now(UTC),
    )
    assert record.material_id == SMOKE_MATERIAL_ID
    assert record.source == "materials_project"
    assert record.provenance, "expected provenance entries"
    for entry in record.provenance:
        assert entry.source == "materials_project"
        assert entry.database_version == database_version
