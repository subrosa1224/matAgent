"""Common immutable output from read-only research adapters."""

from pydantic import BaseModel, ConfigDict

from materials_screening.research.candidate_ledger import (
    ClaimEvidenceLink,
    MaterialCandidate,
    ScreeningClaim,
    ScreeningEvidence,
)


class AdapterBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: tuple[MaterialCandidate, ...] = ()
    evidence: tuple[ScreeningEvidence, ...] = ()
    claims: tuple[ScreeningClaim, ...] = ()
    links: tuple[ClaimEvidenceLink, ...] = ()
    warnings: tuple[str, ...] = ()
