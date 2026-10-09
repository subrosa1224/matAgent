"""Independent P3 metrics for automatic matrices versus approved gold data."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .matrix_automation import PendingMatrixExtraction
from .models import ExperimentalGroup, ExperimentalMeasurement


class MatrixEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    predicted_groups: int
    gold_groups: int
    matched_groups: int
    key_group_recall: float = Field(ge=0, le=1)
    predicted_measurements: int
    scored_measurements: int
    unadjudicated_measurements: int
    gold_measurements: int
    matched_measurements: int
    numeric_precision: float = Field(ge=0, le=1)
    unit_fidelity: float = Field(ge=0, le=1)
    approved_automatic_records: int = Field(ge=0)
    warnings: tuple[str, ...] = ()


class MatrixEvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    documents: tuple[MatrixEvaluation, ...]
    macro_numeric_precision: float = Field(ge=0, le=1)
    macro_key_group_recall: float = Field(ge=0, le=1)
    macro_unit_fidelity: float = Field(ge=0, le=1)
    approved_automatic_records: int = Field(ge=0)
    unadjudicated_measurements: int = Field(ge=0)
    passed: bool


def evaluate_matrix(
    automatic: PendingMatrixExtraction,
    *,
    document_id: str,
    gold_groups: Sequence[ExperimentalGroup],
    gold_measurements: Sequence[ExperimentalMeasurement],
    adjudications: Mapping[str, str] | None = None,
) -> MatrixEvaluation:
    decisions = adjudications or {}
    group_matches = _group_matches(automatic.groups, gold_groups)
    matched_gold_group_ids = set(group_matches.values())
    value_matches = 0
    unit_matches = 0
    scored_count = 0
    unadjudicated_count = 0
    gold_metrics = {_metric(item.metric) for item in gold_measurements}
    used_gold: set[str] = set()
    for predicted in automatic.measurements:
        if _metric(predicted.metric) not in gold_metrics:
            decision = decisions.get(predicted.measurement_id)
            if decision == "rejected":
                continue
            if decision == "approved":
                scored_count += 1
                value_matches += 1
                unit_matches += 1
                continue
            unadjudicated_count += 1
            continue
        scored_count += 1
        gold_group_id = group_matches.get(predicted.group_id)
        if gold_group_id is None:
            continue
        candidates = [
            gold
            for gold in gold_measurements
            if gold.measurement_id not in used_gold
            and gold.group_id == gold_group_id
            and _metric(gold.metric) == _metric(predicted.metric)
            and _value_matches(predicted, gold)
        ]
        if not candidates:
            continue
        gold = candidates[0]
        value_matches += 1
        if _unit(predicted.unit) == _unit(gold.unit):
            unit_matches += 1
        used_gold.add(gold.measurement_id)
    predicted_count = len(automatic.measurements)
    approved_count = sum(
        row.review_status == "approved"
        for rows in (
            automatic.groups,
            automatic.measurements,
            automatic.comparisons,
            automatic.claims,
            automatic.claim_evidence_links,
        )
        for row in rows
    )
    return MatrixEvaluation(
        document_id=document_id,
        predicted_groups=len(automatic.groups),
        gold_groups=len(gold_groups),
        matched_groups=len(matched_gold_group_ids),
        key_group_recall=(
            len(matched_gold_group_ids) / len(gold_groups) if gold_groups else 1.0
        ),
        predicted_measurements=predicted_count,
        scored_measurements=scored_count,
        unadjudicated_measurements=unadjudicated_count,
        gold_measurements=len(gold_measurements),
        matched_measurements=value_matches,
        numeric_precision=value_matches / scored_count if scored_count else 0.0,
        unit_fidelity=unit_matches / value_matches if value_matches else 0.0,
        approved_automatic_records=approved_count,
        warnings=automatic.warnings,
    )


def summarize_matrix_evaluations(
    documents: Sequence[MatrixEvaluation],
) -> MatrixEvaluationSummary:
    if not documents:
        raise ValueError("at least one matrix evaluation is required")
    numeric = sum(item.numeric_precision for item in documents) / len(documents)
    groups = sum(item.key_group_recall for item in documents) / len(documents)
    units = sum(item.unit_fidelity for item in documents) / len(documents)
    approved = sum(item.approved_automatic_records for item in documents)
    unadjudicated = sum(item.unadjudicated_measurements for item in documents)
    return MatrixEvaluationSummary(
        documents=tuple(documents),
        macro_numeric_precision=numeric,
        macro_key_group_recall=groups,
        macro_unit_fidelity=units,
        approved_automatic_records=approved,
        unadjudicated_measurements=unadjudicated,
        passed=(
            numeric >= 0.95
            and groups >= 0.80
            and units >= 0.95
            and approved == 0
            and unadjudicated == 0
        ),
    )


def _group_matches(
    predicted: Sequence[ExperimentalGroup],
    gold: Sequence[ExperimentalGroup],
) -> dict[str, str]:
    matches: dict[str, str] = {}
    used: set[str] = set()
    for candidate in predicted:
        ranked = sorted(
            (
                (_group_score(candidate, reference), reference)
                for reference in gold
                if reference.group_id not in used
            ),
            key=lambda item: (-item[0], item[1].group_id),
        )
        if ranked and ranked[0][0] >= 2:
            reference = ranked[0][1]
            matches[candidate.group_id] = reference.group_id
            used.add(reference.group_id)
    return matches


def _group_score(candidate: ExperimentalGroup, gold: ExperimentalGroup) -> int:
    score = 0
    if _text(candidate.label) == _text(gold.label):
        score += 3
    if candidate.role == gold.role:
        score += 1
    candidate_values = {
        _variable_value(value) for value in candidate.variables.values()
    }
    gold_values = {_variable_value(value) for value in gold.variables.values()}
    score += 2 * len(candidate_values & gold_values)
    if _text(candidate.material) == _text(gold.material):
        score += 1
    return score


def _value_matches(
    predicted: ExperimentalMeasurement, gold: ExperimentalMeasurement
) -> bool:
    if predicted.numeric_value is not None and gold.numeric_value is not None:
        tolerance = max(1e-9, abs(gold.numeric_value) * 1e-6)
        return abs(predicted.numeric_value - gold.numeric_value) <= tolerance
    predicted_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", predicted.value_text)
    gold_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", gold.value_text)
    if len(predicted_numbers) >= 2 and predicted_numbers == gold_numbers:
        return True
    return _text(predicted.value_text) == _text(gold.value_text)


def _variable_value(value: str) -> str:
    normalized = re.sub(r"(?<=\d):(?=\d)", ".", value)
    numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", normalized)
    if len(numbers) == 1:
        return f"{float(numbers[0]):g}"
    return _text(normalized)


def _metric(value: str) -> str:
    normalized = _text(value)
    compact = normalized.replace("-", " ")
    rules = (
        (("jsc", "short circuit current"), "jsc"),
        (("voc", "open circuit voltage"), "voc"),
        (("fill factor",), "ff"),
        (("ecbm", "conduction band minimum"), "ecbm-ef"),
        (("optical bandgap", "bandgap eg"), "optical bandgap"),
        (
            (
                "pce",
                "photoconversion efficiency",
                "photovoltaic conversion efficiency",
                "power conversion efficiency",
            ),
            "photoconversion efficiency",
        ),
    )
    for aliases, canonical in rules:
        if normalized == canonical or any(alias in compact for alias in aliases):
            return canonical
    if normalized == "ff":
        return "ff"
    return normalized


def _unit(value: str | None) -> str:
    normalized = (value or "").replace("−", "-").replace("²", "2")
    return re.sub(r"\s+", "", normalized).casefold()


def _text(value: str) -> str:
    normalized = re.sub(r"(?<=\d):(?=\d)", ".", value.casefold())
    normalized = re.sub(r"\s*%", "%", normalized)
    return re.sub(r"[^a-z0-9.%+-]+", " ", normalized).strip()
