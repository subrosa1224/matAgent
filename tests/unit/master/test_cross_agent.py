"""MA-4 typed literature-to-database coordination tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from materials_screening.master.cross_agent import (
    CrossAgentResult,
    LiteratureDatabaseCoordinator,
    LiteratureMaterialClueExtractor,
)
from materials_screening.sub_agents.literature.dossier import (
    DossierItem,
    PaperDossier,
    PaperDossierStore,
)
from materials_screening.sub_agents.materials_database.models import (
    SearchMaterialsInput,
)

DOCUMENT_ID = "doc-1234567890abcdef12345678"


def _save_dossier(root: Path) -> None:
    PaperDossierStore(root).save_pending(
        PaperDossier(
            document_id=DOCUMENT_ID,
            title="58S bioactive glass scaffold",
            extraction_method="llm",
            review_status="pending",
            items=(
                DossierItem(
                    category="materials",
                    summary="58S BG scaffold containing SiO2, CaO and P2O5.",
                    source_quote="The glass contained 58% SiO2, 33% CaO and 9% P2O5.",
                    chunk_id="chunk-evidence-1",
                    page=3,
                    confidence=0.9,
                    risk_level="low",
                ),
            ),
        )
    )


class _Database:
    def __init__(self) -> None:
        self.request: SearchMaterialsInput | None = None

    def search(self, request: SearchMaterialsInput) -> dict[str, Any]:
        self.request = request
        return {
            "query_id": "query-1",
            "materials": [
                {
                    "material_id": "mp-1",
                    "formula_pretty": "Ca2P2SiO9",
                    "elements": ["Ca", "P", "Si", "O"],
                    "is_stable": True,
                    "energy_above_hull_ev_atom": 0.0,
                    "band_gap_ev": 3.2,
                }
            ],
        }


def test_extractor_creates_evidence_bound_material_clue(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    _save_dossier(pending)
    extractor = LiteratureMaterialClueExtractor(
        approved_root=tmp_path / "approved", pending_root=pending
    )
    clue = extractor.extract(DOCUMENT_ID)

    assert clue is not None
    assert clue.required_elements == ("Ca", "O", "P", "Si")
    assert set(clue.formula_candidates) == {"SiO2", "CaO", "P2O5"}
    assert clue.evidence_page == 3
    assert clue.review_status == "pending"


def test_coordinator_passes_typed_elements_not_exact_formula(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    _save_dossier(pending)
    database = _Database()
    coordinator = LiteratureDatabaseCoordinator(
        database,  # type: ignore[arg-type]
        LiteratureMaterialClueExtractor(
            approved_root=tmp_path / "approved", pending_root=pending
        ),
    )
    result = coordinator.run([DOCUMENT_ID])

    assert result.status == "partial"
    assert len(result.candidates) == 1
    assert database.request is not None
    assert database.request.required_elements == ("Ca", "O", "P", "Si")
    assert database.request.formula is None
    assert database.request.filters[0].field == "is_stable"


def test_coordinator_fails_closed_without_validated_clue(tmp_path: Path) -> None:
    result = LiteratureDatabaseCoordinator(
        _Database(),  # type: ignore[arg-type]
        LiteratureMaterialClueExtractor(
            approved_root=tmp_path / "approved", pending_root=tmp_path / "pending"
        ),
    ).run([DOCUMENT_ID])
    assert result == CrossAgentResult(
        status="failed",
        warnings=(f"{DOCUMENT_ID}: no validated material clue",),
    )
