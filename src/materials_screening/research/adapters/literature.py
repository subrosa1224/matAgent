"""Read-only normalization of reviewed LiteratureAgent measurements."""

from __future__ import annotations

import hashlib

from materials_screening.research.adapters.models import AdapterBatch
from materials_screening.research.candidate_ledger import (
    EvidenceLocator,
    EvidenceReviewStatus,
    EvidenceSourceKind,
    ScreeningEvidence,
)
from materials_screening.research.property_registry import PropertyRegistry
from materials_screening.sub_agents.literature.models import (
    ExperimentalGroup,
    ExperimentalMeasurement,
)


class LiteratureEvidenceReadAdapter:
    """Normalize measurements only after an explicit candidate/group mapping."""

    def __init__(self, registry: PropertyRegistry) -> None:
        self._registry = registry

    def normalize_measurements(
        self,
        project_id: str,
        candidate_id: str,
        *,
        groups: tuple[ExperimentalGroup, ...],
        measurements: tuple[ExperimentalMeasurement, ...],
    ) -> AdapterBatch:
        group_index = {group.group_id: group for group in groups}
        evidence: list[ScreeningEvidence] = []
        warnings: list[str] = []
        for measurement in measurements:
            if measurement.review_status == "rejected":
                warnings.append(
                    f"{measurement.measurement_id}: rejected measurement skipped"
                )
                continue
            group = group_index.get(measurement.group_id)
            if group is None or group.document_id != measurement.document_id:
                warnings.append(
                    f"{measurement.measurement_id}: group mapping is missing"
                )
                continue
            definition = self._registry.resolve(measurement.metric)
            if definition is None:
                warnings.append(
                    f"{measurement.measurement_id}: unregistered property "
                    f"{measurement.metric!r} skipped"
                )
                continue
            if measurement.numeric_value is None:
                warnings.append(
                    f"{measurement.measurement_id}: no structured numeric value"
                )
                continue
            if measurement.unit != definition.canonical_unit:
                warnings.append(
                    f"{measurement.measurement_id}: unit {measurement.unit!r} is not "
                    f"canonical {definition.canonical_unit!r}; retained as evidence "
                    "only"
                )
            evidence_id = _id(
                project_id,
                candidate_id,
                measurement.document_id,
                measurement.measurement_id,
            )
            evidence.append(
                ScreeningEvidence(
                    evidence_id=evidence_id,
                    project_id=project_id,
                    candidate_id=candidate_id,
                    source_kind=EvidenceSourceKind.LITERATURE_FULL_TEXT,
                    source_id=measurement.document_id,
                    locator=EvidenceLocator(
                        document_id=measurement.document_id,
                        chunk_id=measurement.chunk_id,
                        page_from=measurement.page_from,
                        page_to=measurement.page_to,
                    ),
                    property_id=definition.property_id,
                    value=measurement.numeric_value,
                    unit=measurement.unit,
                    method=measurement.extraction_method,
                    conditions={
                        "group_label": group.label,
                        "material": group.material,
                        "variables": group.variables,
                        "conditions": group.conditions,
                        "sample_size": measurement.sample_size,
                        "uncertainty": measurement.uncertainty_text,
                    },
                    source_excerpt=measurement.source_quote,
                    source_text_sha256=measurement.source_text_sha256,
                    review_status=(
                        EvidenceReviewStatus.HUMAN_REVIEWED
                        if measurement.review_status == "approved"
                        and group.review_status == "approved"
                        else EvidenceReviewStatus.PENDING
                    ),
                    limitations=(
                        "实验样品到数据库候选的映射必须由独立身份评估记录支持。",
                    ),
                )
            )
        return AdapterBatch(evidence=tuple(evidence), warnings=tuple(warnings))


def _id(*parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]
    return f"evidence-{digest}"
