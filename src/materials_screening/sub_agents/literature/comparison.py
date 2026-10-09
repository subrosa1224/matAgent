"""Deterministic experimental-group comparison calculations."""

from __future__ import annotations

import hashlib
import math
from typing import Literal

from .models import ExperimentalComparison, ExperimentalMeasurement


def calculate_comparison(
    baseline: ExperimentalMeasurement,
    target: ExperimentalMeasurement,
) -> ExperimentalComparison:
    if baseline.document_id != target.document_id:
        raise ValueError("measurements belong to different documents")
    if baseline.metric != target.metric:
        raise ValueError("measurement metrics differ")
    if _unit(baseline.unit) != _unit(target.unit):
        raise ValueError("measurement units differ")
    if baseline.numeric_value is None or target.numeric_value is None:
        direction: Literal["increase", "decrease", "unchanged", "not_computable"] = (
            "not_computable"
        )
        absolute_change = None
        relative_change = None
    else:
        absolute_change = target.numeric_value - baseline.numeric_value
        if math.isclose(absolute_change, 0.0, abs_tol=1e-12):
            direction = "unchanged"
        elif absolute_change > 0:
            direction = "increase"
        else:
            direction = "decrease"
        relative_change = (
            None
            if math.isclose(baseline.numeric_value, 0.0, abs_tol=1e-12)
            else absolute_change / baseline.numeric_value * 100.0
        )
    identity = "|".join(
        (
            baseline.measurement_id,
            target.measurement_id,
            "calculated",
        )
    )
    return ExperimentalComparison(
        comparison_id=(
            f"comparison-{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
        ),
        document_id=baseline.document_id,
        baseline_group_id=baseline.group_id,
        target_group_id=target.group_id,
        metric=baseline.metric,
        baseline_measurement_id=baseline.measurement_id,
        target_measurement_id=target.measurement_id,
        baseline_value=baseline.numeric_value,
        target_value=target.numeric_value,
        absolute_change=absolute_change,
        relative_change_percent=relative_change,
        direction=direction,
        provenance_type="calculated",
        unit=baseline.unit,
    )


def _unit(value: str | None) -> str:
    return " ".join((value or "").split()).casefold()
