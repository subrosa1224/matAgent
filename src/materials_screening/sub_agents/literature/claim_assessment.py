"""Conservative rule-based links between author claims and body evidence."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from .models import (
    ClaimEvidenceLink,
    ExperimentalComparison,
    ExperimentalMeasurement,
    PaperClaim,
)

_INCREASE_WORDS = frozenset(
    {
        "increase",
        "increased",
        "improve",
        "improved",
        "enhance",
        "enhanced",
        "enhancement",
        "higher",
    }
)
_DECREASE_WORDS = frozenset(
    {"decrease", "decreased", "reduce", "reduced", "lower", "decline", "declined"}
)
_METRIC_ALIASES = {
    "pce": ("power conversion efficiency", "power-conversion efficiency"),
    "photoconversion efficiency": (
        "power conversion efficiency",
        "power-conversion efficiency",
    ),
    "jsc": ("short circuit current", "short-circuit current"),
    "voc": ("open circuit voltage", "open-circuit voltage"),
    "ff": ("fill factor",),
}


def assess_claim_against_comparison(
    claim: PaperClaim,
    comparison: ExperimentalComparison,
) -> ClaimEvidenceLink:
    if claim.document_id != comparison.document_id:
        raise ValueError("claim and comparison belong to different documents")
    claimed_direction = _claimed_direction(claim.claim_text)
    metric_present = _metric_present(claim.claim_text, comparison.metric)
    if not metric_present:
        assessment = "not_verifiable"
        explanation = "Claim does not identify the comparison metric."
    elif claimed_direction is None or comparison.direction == "not_computable":
        assessment = "not_verifiable"
        explanation = "Claim direction could not be matched to a computable comparison."
    elif claimed_direction == comparison.direction:
        assessment = "supported"
        explanation = (
            f"Claim direction ({claimed_direction}) matches the deterministic "
            f"comparison for {comparison.metric}."
        )
    else:
        assessment = "contradicted"
        explanation = (
            f"Claim direction ({claimed_direction}) conflicts with the deterministic "
            f"comparison direction ({comparison.direction}) for {comparison.metric}."
        )
    identity = f"{claim.claim_id}|comparison|{comparison.comparison_id}|rule"
    return ClaimEvidenceLink.model_validate(
        {
            "link_id": f"link-{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
            "claim_id": claim.claim_id,
            "document_id": claim.document_id,
            "evidence_type": "comparison",
            "evidence_id": comparison.comparison_id,
            "assessment": assessment,
            "explanation": explanation,
            "assessment_method": "rule",
        }
    )


def assess_claim_against_measurement(
    claim: PaperClaim,
    measurement: ExperimentalMeasurement,
) -> ClaimEvidenceLink:
    if claim.document_id != measurement.document_id:
        raise ValueError("claim and measurement belong to different documents")
    claim_normalized = _normalize(claim.claim_text)
    value_present = _normalize(measurement.value_text) in claim_normalized
    metric_present = _metric_present(claim.claim_text, measurement.metric)
    if value_present and metric_present:
        assessment = "supported"
        explanation = (
            f"Claim contains the reported {measurement.metric} value "
            f"{measurement.value_text} {measurement.unit or ''}.".strip()
        )
    else:
        assessment = "not_verifiable"
        explanation = "Claim does not contain both the metric and its measured value."
    identity = f"{claim.claim_id}|measurement|{measurement.measurement_id}|rule"
    return ClaimEvidenceLink.model_validate(
        {
            "link_id": f"link-{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
            "claim_id": claim.claim_id,
            "document_id": claim.document_id,
            "evidence_type": "measurement",
            "evidence_id": measurement.measurement_id,
            "assessment": assessment,
            "explanation": explanation,
            "assessment_method": "rule",
        }
    )


def assess_claim_against_body(
    claim: PaperClaim,
    *,
    measurements: Sequence[ExperimentalMeasurement],
    comparisons: Sequence[ExperimentalComparison],
) -> ClaimEvidenceLink:
    """Choose the strongest deterministic body-evidence assessment."""
    related_comparisons = [
        item
        for item in comparisons
        if item.document_id == claim.document_id
        and _metric_present(claim.claim_text, item.metric)
    ]
    links = [
        assess_claim_against_comparison(claim, item)
        for item in related_comparisons
    ]
    related_measurements = [
        item
        for item in measurements
        if item.document_id == claim.document_id
        and _metric_present(claim.claim_text, item.metric)
    ]
    links.extend(
        assess_claim_against_measurement(claim, item)
        for item in related_measurements
    )
    priority = {"contradicted": 4, "supported": 3, "partially_supported": 2}
    decisive = [item for item in links if item.assessment in priority]
    if decisive:
        return max(decisive, key=lambda item: priority[item.assessment])
    if related_measurements:
        evidence = related_measurements[0]
        identity = (
            f"{claim.claim_id}|measurement|{evidence.measurement_id}|rule-partial"
        )
        return ClaimEvidenceLink(
            link_id=f"link-{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
            claim_id=claim.claim_id,
            document_id=claim.document_id,
            evidence_type="measurement",
            evidence_id=evidence.measurement_id,
            assessment="partially_supported",
            explanation=(
                f"Body evidence reports {evidence.metric}, but the abstract claim "
                "does not contain a matching explicit value or direction."
            ),
            assessment_method="rule",
            review_status="pending",
        )
    identity = f"{claim.claim_id}|text_chunk|{claim.chunk_id}|rule-unverifiable"
    return ClaimEvidenceLink(
        link_id=f"link-{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
        claim_id=claim.claim_id,
        document_id=claim.document_id,
        evidence_type="text_chunk",
        evidence_id=claim.chunk_id,
        assessment="not_verifiable",
        explanation="No matching body measurement or comparison was extracted.",
        assessment_method="rule",
        review_status="pending",
    )


def _claimed_direction(text: str) -> str | None:
    words = set(re.findall(r"[a-z]+", text.casefold()))
    has_increase = bool(words & _INCREASE_WORDS)
    has_decrease = bool(words & _DECREASE_WORDS)
    if has_increase == has_decrease:
        return None
    return "increase" if has_increase else "decrease"


def _metric_present(claim_text: str, metric: str) -> bool:
    claim_normalized = _normalize(claim_text)
    metric_name = _normalize(metric)
    aliases = _METRIC_ALIASES.get(metric_name, ())
    if metric_name in claim_normalized or any(
        alias in claim_normalized for alias in aliases
    ):
        return True
    metric_terms = set(re.findall(r"[a-z]{3,}", metric_name))
    claim_terms = set(re.findall(r"[a-z]{3,}", claim_normalized))
    informative = metric_terms - {"the", "and", "value", "measured"}
    return bool(informative) and informative <= claim_terms


def _normalize(value: str) -> str:
    value = value.replace("−", "-").replace("–", "-").replace("×", "x")
    return re.sub(r"\s+", " ", value).strip().casefold()
