"""Planner evaluator with full stage-2 metrics (D2-M7).

Metric semantics:

- ``schema_success_rate``: fraction of cases where the provider returned a
  parseable PlannerDraft (provider errors count as failures).
- ``status_accuracy``: fraction of cases where the resolved status matches the
  expected status.
- ``ready_precision`` / ``ready_recall``: READY treated as the positive class.
- ``request_field_exact_match``: for cases with a non-empty expected_request,
  fraction where every expected field matches the resolved request. Expected
  request is a *subset*; extra fields in the resolved request are ignored.
  Elements and space groups compare as sets; floats compare with tolerance;
  other fields compare exactly.
- ``numeric_constraint_accuracy``: matched numeric bounds (band gap, hull,
  density ranges and band-gap target) over all expected numeric instances.
- ``element_constraint_accuracy``: matched required/excluded element fields
  (set comparison) over all expected element fields.
- ``ambiguity_recall``: mean fraction of expected ambiguity codes found in
  ``result.ambiguities | result.assumptions``.
- ``conflict_recall``: mean fraction of expected conflict codes found in
  ``result.conflicts``.
- ``unsupported_accuracy``: fraction of cases tagged ``unsupported`` that
  resolve to UNSUPPORTED.
- ``injection_resilience``: fraction of cases tagged ``injection`` that keep
  schema success, expected status and the expected request subset.
- latency and token metrics: averages over schema-successful cases.

The evaluator never calls an LLM for scoring; it only runs the deterministic
planner pipeline.
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from materials_screening.errors import InvalidRequestError
from materials_screening.models import ScreeningRequest
from materials_screening.planner.models import PlannerResult, PlannerStatus
from materials_screening.planner.service import PlannerService

_NUMERIC_REQUEST_FIELDS = (
    "band_gap_ev",
    "energy_above_hull_ev_atom",
    "density_g_cm3",
    "target_band_gap_ev",
)
_ELEMENT_REQUEST_FIELDS = ("required_elements", "excluded_elements")
_SET_FIELDS = (*_ELEMENT_REQUEST_FIELDS, "spacegroup_numbers")

_REL_TOL = 1e-6
_ABS_TOL = 1e-9


class PlannerEvalCase(BaseModel):
    """One evaluation case with reference labels for the planner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    query: str
    expected_status: PlannerStatus
    expected_request: dict[str, object] = Field(
        description="Expected ScreeningRequest field subset; empty when non-READY."
    )
    expected_ambiguity_codes: tuple[str, ...] = ()
    expected_conflict_codes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


class PlannerEvalMetrics(BaseModel):
    """Aggregated planner evaluation metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total: int
    schema_success: int
    schema_success_rate: float
    status_accurate: int
    status_accuracy: float
    ready_precision: float
    ready_recall: float
    request_exact_match: int
    request_exact_instances: int
    request_exact_match_rate: float
    numeric_matches: int
    numeric_instances: int
    numeric_constraint_accuracy: float
    element_matches: int
    element_instances: int
    element_constraint_accuracy: float
    ambiguity_recall: float
    conflict_recall: float
    unsupported_accuracy: float
    injection_resilience: float
    average_latency_ms: float
    average_input_tokens: float
    average_output_tokens: float
    average_reasoning_tokens: float


class PlannerCaseResult(BaseModel):
    """Per-case outcome with redacted details for evaluation reports."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    tags: tuple[str, ...] = ()
    expected_status: PlannerStatus
    schema_success: bool
    actual_status: PlannerStatus | None = None
    error_type: str | None = None
    request_id: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None


def load_cases(path: Path) -> list[PlannerEvalCase]:
    """Load JSONL evaluation cases; empty or malformed files are rejected."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InvalidRequestError(f"cannot read eval file {path}: {exc}") from exc
    cases: list[PlannerEvalCase] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            cases.append(PlannerEvalCase.model_validate(payload))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise InvalidRequestError(
                f"invalid eval case at line {line_number}: {exc}"
            ) from exc
    if not cases:
        raise InvalidRequestError("eval file contains no cases")
    return cases


def _float_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _value_matches(expected: object, actual: object) -> bool:
    """Compare one expected value against a resolved request value."""
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and _float_close(expected, float(actual))
        )
    if isinstance(expected, int):
        return (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected
        )
    if isinstance(expected, str):
        return str(actual) == expected
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and set(expected) == set(actual)
    return False


def _field_matches(field_name: str, expected: object, actual: object) -> bool:
    """Compare one expected_request field against the resolved request."""
    if field_name in _SET_FIELDS:
        return (
            isinstance(expected, (list, tuple))
            and isinstance(actual, (list, tuple))
            and set(expected) == set(actual)
        )
    if isinstance(expected, dict):
        if actual is None:
            return False
        return all(
            _value_matches(value, getattr(actual, key, None))
            for key, value in expected.items()
        )
    return _value_matches(expected, actual)


def _request_subset_matches(
    request: ScreeningRequest | None,
    expected_request: Mapping[str, object],
) -> bool:
    """Return True when every expected field matches the resolved request."""
    if not expected_request:
        return True
    if request is None:
        return False
    return all(
        _field_matches(name, value, getattr(request, name, None))
        for name, value in expected_request.items()
    )


def _rate(matches: int, instances: int) -> float:
    return matches / instances if instances else 0.0


class PlannerEvaluator:
    """Run the planner over eval cases and aggregate all metrics."""

    def __init__(self, planner_service: PlannerService) -> None:
        self._planner_service = planner_service

    def evaluate(self, cases: list[PlannerEvalCase]) -> PlannerEvalMetrics:
        """Evaluate cases deterministically; never uses an LLM for scoring."""
        metrics, _ = self.evaluate_with_details(cases)
        return metrics

    def evaluate_with_details(
        self,
        cases: list[PlannerEvalCase],
    ) -> tuple[PlannerEvalMetrics, tuple[PlannerCaseResult, ...]]:
        """Evaluate cases and also return redacted per-case outcomes."""
        total = len(cases)
        schema_success = 0
        status_accurate = 0
        ready_tp = 0
        ready_fp = 0
        ready_fn = 0
        request_exact_match = 0
        request_exact_instances = 0
        numeric_matches = 0
        numeric_instances = 0
        element_matches = 0
        element_instances = 0
        ambiguity_numerator = 0.0
        ambiguity_denominator = 0
        conflict_numerator = 0.0
        conflict_denominator = 0
        unsupported_cases = 0
        unsupported_correct = 0
        injection_cases = 0
        injection_resilient = 0
        latency_total = 0
        input_tokens_total = 0
        output_tokens_total = 0
        reasoning_tokens_total = 0
        outcomes: list[PlannerCaseResult] = []

        for case in cases:
            try:
                result = self._planner_service.parse(case.query)
            except Exception as exc:
                if case.expected_status is PlannerStatus.READY:
                    ready_fn += 1
                outcomes.append(
                    PlannerCaseResult(
                        id=case.id,
                        tags=case.tags,
                        expected_status=case.expected_status,
                        schema_success=False,
                        error_type=type(exc).__name__,
                    )
                )
                continue

            schema_success += 1
            metadata = result.provider_metadata
            if metadata is not None:
                latency_total += metadata.latency_ms
                input_tokens_total += metadata.input_tokens or 0
                output_tokens_total += metadata.output_tokens or 0
                reasoning_tokens_total += metadata.reasoning_tokens or 0

            if result.status is case.expected_status:
                status_accurate += 1

            expected_ready = case.expected_status is PlannerStatus.READY
            actual_ready = result.status is PlannerStatus.READY
            if expected_ready and actual_ready:
                ready_tp += 1
            elif expected_ready and not actual_ready:
                ready_fn += 1
            elif not expected_ready and actual_ready:
                ready_fp += 1

            if case.expected_request:
                request_exact_instances += 1
                if _request_subset_matches(result.request, case.expected_request):
                    request_exact_match += 1
                numeric_matches, numeric_instances = _accumulate_numeric(
                    result, case.expected_request, numeric_matches, numeric_instances
                )
                element_matches, element_instances = _accumulate_elements(
                    result, case.expected_request, element_matches, element_instances
                )

            if case.expected_ambiguity_codes:
                ambiguity_denominator += 1
                actual_codes = set(result.ambiguities) | set(result.assumptions)
                matched = len(set(case.expected_ambiguity_codes) & actual_codes)
                ambiguity_numerator += matched / len(case.expected_ambiguity_codes)

            if case.expected_conflict_codes:
                conflict_denominator += 1
                actual_codes = set(result.conflicts)
                matched = len(set(case.expected_conflict_codes) & actual_codes)
                conflict_numerator += matched / len(case.expected_conflict_codes)

            if "unsupported" in case.tags:
                unsupported_cases += 1
                if result.status is PlannerStatus.UNSUPPORTED:
                    unsupported_correct += 1

            if "injection" in case.tags:
                injection_cases += 1
                if result.status is case.expected_status and _request_subset_matches(
                    result.request, case.expected_request
                ):
                    injection_resilient += 1

            outcomes.append(
                PlannerCaseResult(
                    id=case.id,
                    tags=case.tags,
                    expected_status=case.expected_status,
                    schema_success=True,
                    actual_status=result.status,
                    request_id=metadata.request_id if metadata else None,
                    latency_ms=metadata.latency_ms if metadata else None,
                    input_tokens=metadata.input_tokens if metadata else None,
                    output_tokens=metadata.output_tokens if metadata else None,
                    reasoning_tokens=(metadata.reasoning_tokens if metadata else None),
                )
            )

        metrics = PlannerEvalMetrics(
            total=total,
            schema_success=schema_success,
            schema_success_rate=_rate(schema_success, total),
            status_accurate=status_accurate,
            status_accuracy=_rate(status_accurate, total),
            ready_precision=_rate(ready_tp, ready_tp + ready_fp),
            ready_recall=_rate(ready_tp, ready_tp + ready_fn),
            request_exact_match=request_exact_match,
            request_exact_instances=request_exact_instances,
            request_exact_match_rate=_rate(
                request_exact_match, request_exact_instances
            ),
            numeric_matches=numeric_matches,
            numeric_instances=numeric_instances,
            numeric_constraint_accuracy=_rate(numeric_matches, numeric_instances),
            element_matches=element_matches,
            element_instances=element_instances,
            element_constraint_accuracy=_rate(element_matches, element_instances),
            ambiguity_recall=(
                ambiguity_numerator / ambiguity_denominator
                if ambiguity_denominator
                else 0.0
            ),
            conflict_recall=(
                conflict_numerator / conflict_denominator
                if conflict_denominator
                else 0.0
            ),
            unsupported_accuracy=_rate(unsupported_correct, unsupported_cases),
            injection_resilience=_rate(injection_resilient, injection_cases),
            average_latency_ms=(
                latency_total / schema_success if schema_success else 0.0
            ),
            average_input_tokens=(
                input_tokens_total / schema_success if schema_success else 0.0
            ),
            average_output_tokens=(
                output_tokens_total / schema_success if schema_success else 0.0
            ),
            average_reasoning_tokens=(
                reasoning_tokens_total / schema_success if schema_success else 0.0
            ),
        )
        return metrics, tuple(outcomes)


def _accumulate_numeric(
    result: PlannerResult,
    expected_request: Mapping[str, object],
    matches: int,
    instances: int,
) -> tuple[int, int]:
    """Count matched numeric bounds across expected numeric fields."""
    request = result.request
    for name, expected in expected_request.items():
        if name not in _NUMERIC_REQUEST_FIELDS:
            continue
        actual = getattr(request, name, None) if request is not None else None
        if isinstance(expected, dict):
            for bound in ("min", "max"):
                value = expected.get(bound)
                if value is None:
                    continue
                instances += 1
                actual_bound = (
                    getattr(actual, bound, None) if actual is not None else None
                )
                if actual_bound is not None and _value_matches(value, actual_bound):
                    matches += 1
        else:
            instances += 1
            if _value_matches(expected, actual):
                matches += 1
    return matches, instances


def _accumulate_elements(
    result: PlannerResult,
    expected_request: Mapping[str, object],
    matches: int,
    instances: int,
) -> tuple[int, int]:
    """Count matched element fields (set comparison) across expected fields."""
    request = result.request
    for name, expected in expected_request.items():
        if name not in _ELEMENT_REQUEST_FIELDS:
            continue
        instances += 1
        actual = getattr(request, name, None) if request is not None else None
        if _field_matches(name, expected, actual):
            matches += 1
    return matches, instances
