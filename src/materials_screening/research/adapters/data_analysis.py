"""Read-only normalization of deterministic DataAnalysisAgent results."""

from __future__ import annotations

import hashlib

from materials_screening.data_analysis.models import AnalysisResult
from materials_screening.research.adapters.models import AdapterBatch
from materials_screening.research.candidate_ledger import (
    EvidenceLocator,
    EvidenceReviewStatus,
    EvidenceSourceKind,
    ScreeningEvidence,
)


class DataAnalysisReadAdapter:
    """Register an existing result as analysis evidence without reinterpreting it."""

    def normalize(
        self,
        project_id: str,
        result: AnalysisResult,
        *,
        candidate_id: str | None = None,
    ) -> AdapterBatch:
        digest = hashlib.sha256(
            f"{project_id}|{result.analysis_id}|{candidate_id or ''}".encode()
        ).hexdigest()[:24]
        evidence = ScreeningEvidence(
            evidence_id=f"evidence-{digest}",
            project_id=project_id,
            candidate_id=candidate_id,
            source_kind=EvidenceSourceKind.DATA_ANALYSIS,
            source_id=result.analysis_id,
            locator=EvidenceLocator(
                dataset_id=result.dataset_id,
                analysis_id=result.analysis_id,
            ),
            method=result.method,
            conditions={
                "analysis_type": result.analysis_type,
                "parameters": result.parameters,
                "summary": result.summary,
            },
            review_status=EvidenceReviewStatus.VALIDATED,
            limitations=result.warnings,
        )
        return AdapterBatch(evidence=(evidence,))
