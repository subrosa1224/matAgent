"""Generic, evidence-gated extraction of pending experiment matrices."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal, TypeVar

from openai import APIConnectionError, APITimeoutError, OpenAIError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

from materials_screening.llm.base import StructuredLLM
from materials_screening.llm.errors import (
    LLMConnectionError,
    LLMError,
    LLMStructuredOutputError,
    LLMTimeoutError,
)

from .claim_assessment import assess_claim_against_body
from .comparison import calculate_comparison
from .evidence_scope import (
    extract_percent_sample_tables,
    is_prior_work,
    is_reference_evidence,
    measurement_binding_context,
    measurement_sample_bound,
    sample_table_rows,
)
from .matrix import (
    measurement_value_present,
    unit_present_in_evidence,
    validate_claim_evidence,
    validate_matrix_evidence,
)
from .metric_coverage import (
    RequiredMetric,
    RequiredMetricCoverage,
    check_required_metric_coverage,
)
from .models import (
    ClaimEvidenceLink,
    ExperimentalComparison,
    ExperimentalGroup,
    ExperimentalMeasurement,
    PaperClaim,
)
from .rag import ChunkRecord, VectorStore

_SYSTEM_PROMPT = """Extract an experimental matrix from the supplied scientific
paper evidence. Copy every source_quote verbatim from its named chunk. Create a
stable local group_key shared by each group and its measurements.
Every measurement must include the group_key of a group in the same response;
reuse that key for the same device/sample rather than emitting null group keys.
For multi-sample tables create one group per EXACT sample label, including loading
or doping level. Quote the matching row and its metric header, never transfer a
number from one row to another. Do not extract cited previous studies as this work.
Extract only explicitly reported groups, variables, preparation/test conditions,
metrics, values, units, uncertainty, and sample size. value_text must contain only the
reported numeric value or range, without its unit. Do not calculate, normalize,
convert, or infer missing values. Claims must be verbatim statements from the
abstract only. Empty lists are valid. Figure-only values must not be estimated.
"""

_EVIDENCE_TERMS = (
    "table",
    "sample",
    "group",
    "control",
    "wt%",
    "mol%",
    "porosity",
    "efficiency",
    "strength",
    "current density",
    "voltage",
    "experimental",
    "results",
)

_RowT = TypeVar("_RowT")


class GroupCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    group_key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    role: Literal["control", "treatment", "reference", "unknown"] = "unknown"
    material: str = Field(min_length=1)
    variables: dict[str, str] = Field(default_factory=dict)
    conditions: dict[str, str] = Field(default_factory=dict)
    source_quote: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)

    @field_validator("variables", "conditions", mode="before")
    @classmethod
    def _flatten_mapping_values(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        flattened: dict[str, str] = {}
        for key, item_value in value.items():
            if isinstance(item_value, list):
                flattened.update(
                    {
                        f"{key}[{index}]": str(item)
                        for index, item in enumerate(item_value, 1)
                    }
                )
            else:
                flattened[str(key)] = str(item_value)
        return flattened


class MeasurementCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    group_key: str | None = None
    metric: str = Field(min_length=1)
    value_text: str = Field(min_length=1)
    numeric_value: float | None = None
    unit: str | None = None
    uncertainty_text: str | None = None
    sample_size: int | None = Field(default=None, ge=1)
    source_quote: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)


class ClaimCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    claim_text: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)


class MatrixExtractionBatch(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    groups: tuple[GroupCandidate, ...] = Field(default=(), max_length=50)
    measurements: tuple[MeasurementCandidate, ...] = Field(default=(), max_length=200)
    abstract_claims: tuple[ClaimCandidate, ...] = Field(default=(), max_length=30)
    # Server-only diagnostic metadata; never alter the existing provider schema.
    source_selection_rejections: SkipJsonSchema[
        tuple[
            Literal[
                "nonliteral sample label", "nonnumeric value", "unbound sample/value"
            ],
            ...,
        ]
    ] = Field(default=(), max_length=250)


@dataclass(frozen=True)
class PendingMatrixExtraction:
    groups: tuple[ExperimentalGroup, ...]
    measurements: tuple[ExperimentalMeasurement, ...]
    comparisons: tuple[ExperimentalComparison, ...]
    claims: tuple[PaperClaim, ...]
    claim_evidence_links: tuple[ClaimEvidenceLink, ...]
    warnings: tuple[str, ...]
    diagnostics: MatrixExtractionDiagnostics = field(
        default_factory=lambda: MatrixExtractionDiagnostics()
    )


class MatrixBatchAttemptDiagnostic(BaseModel):
    """Safe provider-call metadata; deliberately excludes raw error/output text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_index: int
    attempt: int
    status: Literal["ok", "error", "reused"]
    chunk_ids: tuple[str, ...]
    page_ranges: tuple[tuple[int, int], ...]
    evidence_chars: int
    output_token_budget: int
    error_type: str | None = None
    error_category: str | None = None
    schema_location: str | None = None


_ConstructionRejection = Literal[
    "missing_or_ambiguous_group_key",
    "unknown_group_key",
    "unlocated_measurement_evidence",
    "numeric_value_mismatch",
]


class MatrixMeasurementCandidateDiagnostic(BaseModel):
    """Parsed-candidate construction summary, not final evidence approval.

    Text fields are bounded; quotes and complete model output are never retained.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_index: int
    candidate_index: int
    metric: str
    value_text: str
    numeric_value: float | None
    unit: str | None
    group_key: str | None
    chunk_id: str
    source_quote_sha256: str
    source_quote_chars: int
    resolved_group_id: str | None
    resolved_chunk_id: str | None
    resolved_quote_sha256: str | None
    status: Literal["constructed", "rejected"]
    rejection_reasons: tuple[_ConstructionRejection, ...] = ()
    truncated_fields: tuple[str, ...] = ()


def _measurement_candidate_diagnostic(
    candidate: MeasurementCandidate,
    *,
    origin: tuple[int, int],
    resolved_group_id: str | None,
    chunk: ChunkRecord | None,
    quote: str | None,
) -> MatrixMeasurementCandidateDiagnostic:
    """Observe existing preconstruction decisions without changing their gates."""
    reasons: list[_ConstructionRejection] = []
    if resolved_group_id is None:
        reasons.append(
            "unknown_group_key"
            if candidate.group_key
            else "missing_or_ambiguous_group_key"
        )
    if chunk is None or quote is None:
        reasons.append("unlocated_measurement_evidence")
    if not reasons and not _numeric_value_consistent(
        candidate.value_text, candidate.numeric_value
    ):
        reasons.append("numeric_value_mismatch")
    fields = {
        "metric": candidate.metric,
        "value_text": candidate.value_text,
        "unit": candidate.unit,
        "group_key": candidate.group_key,
        "chunk_id": candidate.chunk_id,
    }
    truncated = tuple(
        name for name, value in fields.items() if value is not None and len(value) > 120
    )
    return MatrixMeasurementCandidateDiagnostic(
        batch_index=origin[0],
        candidate_index=origin[1],
        **{
            name: value[:120] if value is not None else None
            for name, value in fields.items()
        },
        numeric_value=candidate.numeric_value,
        source_quote_sha256=hashlib.sha256(
            candidate.source_quote.encode(errors="surrogatepass")
        ).hexdigest(),
        source_quote_chars=len(candidate.source_quote),
        resolved_group_id=resolved_group_id,
        resolved_chunk_id=chunk.chunk_id if chunk is not None else None,
        resolved_quote_sha256=(
            hashlib.sha256(quote.encode(errors="surrogatepass")).hexdigest()
            if quote is not None
            else None
        ),
        status="rejected" if reasons else "constructed",
        rejection_reasons=tuple(reasons),
        truncated_fields=truncated,
    )


def _safe_batch_error(error: Exception) -> tuple[str, str | None]:
    if isinstance(error, (LLMTimeoutError, APITimeoutError)):
        return "timeout", None
    if isinstance(error, (LLMConnectionError, APIConnectionError)):
        return "connection_error", None
    if not isinstance(error, LLMStructuredOutputError):
        return "provider_error", None
    message = str(error)
    if message.startswith("Intern response exceeded max_tokens"):
        return "output_truncated", None
    if message.startswith("Intern returned empty structured output"):
        return "empty_output", None
    if message.startswith("Intern output is not valid JSON"):
        category = (
            "prompt_processing_error"
            if message
            == (
                "Intern output is not valid JSON "
                "(provider reported prompt processing error)"
            )
            else "invalid_json"
        )
        return category, None
    prefix = "Intern output does not match schema at "
    if message.startswith(prefix):
        location, separator, _ = message[len(prefix) :].partition(":")
        fields = {
            "groups",
            "measurements",
            "abstract_claims",
            "group_key",
            "label",
            "role",
            "material",
            "variables",
            "conditions",
            "source_quote",
            "chunk_id",
            "metric",
            "value_text",
            "numeric_value",
            "unit",
            "uncertainty_text",
            "sample_size",
            "claim_text",
        }
        safe_location = (
            separator
            and 0 < len(location) <= 160
            and all(
                part in fields or re.fullmatch(r"[0-9]{1,6}", part)
                for part in location.split(".")
            )
        )
        return "schema_mismatch", location if safe_location else None
    return "structured_output_error", None


class MatrixExtractionDiagnostics(BaseModel):
    """Counts at extraction boundaries, not inferred absence of experimental data."""

    input_chunks: int = 0
    selected_chunks: int = 0
    attempted_batches: int = 0
    successful_batches: int = 0
    llm_groups: int = 0
    llm_measurements: int = 0
    table_measurements: int = 0
    before_validation_measurements: int = 0
    after_validation_measurements: int = 0
    before_sanitize_measurements: int = 0
    accepted_measurements: int = 0
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    batch_attempts: tuple[MatrixBatchAttemptDiagnostic, ...] = ()
    measurement_candidates: tuple[MatrixMeasurementCandidateDiagnostic, ...] = ()
    required_metric_coverage: RequiredMetricCoverage = Field(
        default_factory=RequiredMetricCoverage
    )


class AutomatedMatrixExtractor:
    """Extract matrix candidates while treating the LLM as untrusted input."""

    def __init__(self, llm: StructuredLLM, store: VectorStore) -> None:
        self.llm = llm
        self.store = store

    def extract(
        self,
        *,
        document_id: str,
        chunks: Sequence[ChunkRecord],
        max_output_tokens: int = 8192,
        max_evidence_chars: int = 24000,
        batch_chars: int = 8000,
        required_metrics: Sequence[RequiredMetric] = (),
        cached_batches: Mapping[int, MatrixExtractionBatch] | None = None,
        on_batch: Callable[[int, MatrixExtractionBatch], None] | None = None,
        request_missing: bool = True,
    ) -> PendingMatrixExtraction:
        selected = _select_evidence(chunks, max_evidence_chars=max_evidence_chars)
        warnings = list(detect_matrix_warnings(chunks))
        batches: list[MatrixExtractionBatch] = []
        batch_attempts: list[MatrixBatchAttemptDiagnostic] = []
        measurement_origins: list[tuple[int, int]] = []
        candidate_diagnostics: list[MatrixMeasurementCandidateDiagnostic] = []
        evidence_batches = tuple(_chunk_batches(selected, batch_chars=batch_chars))
        cached_batches = cached_batches or {}
        if any(
            index not in range(1, len(evidence_batches) + 1) for index in cached_batches
        ):
            raise ValueError("Cached batch is outside current evidence plan")
        for batch_index, evidence_batch in enumerate(evidence_batches, 1):
            evidence_text = _evidence_text(evidence_batch)
            metadata = {
                "batch_index": batch_index,
                "chunk_ids": tuple(chunk.chunk_id for chunk in evidence_batch),
                "page_ranges": tuple(
                    dict.fromkeys(
                        (chunk.page_from, chunk.page_to) for chunk in evidence_batch
                    )
                ),
                "evidence_chars": len(evidence_text),
                "output_token_budget": max_output_tokens,
            }
            response = None
            cached = cached_batches.get(batch_index)
            if cached is not None:
                batch_attempts.append(
                    MatrixBatchAttemptDiagnostic(
                        **metadata,
                        attempt=0,
                        status="reused",
                    )
                )
                batches.append(cached)
                measurement_origins.extend(
                    (batch_index, index)
                    for index in range(1, len(cached.measurements) + 1)
                )
                continue
            if not request_missing:
                warnings.append(
                    f"evidence batch {batch_index} not requested: partial checkpoint"
                )
                continue
            last_error: Exception | None = None
            for _attempt in (1, 2):
                try:
                    response = self.llm.generate_structured(
                        system_prompt=_SYSTEM_PROMPT,
                        user_text=evidence_text,
                        output_model=MatrixExtractionBatch,
                        schema_name="literature_matrix_candidates_v1",
                        max_output_tokens=max_output_tokens,
                    )
                    batch_attempts.append(
                        MatrixBatchAttemptDiagnostic(
                            **metadata, attempt=_attempt, status="ok"
                        )
                    )
                    break
                except (OpenAIError, LLMError) as exc:
                    last_error = exc
                    category, location = _safe_batch_error(exc)
                    batch_attempts.append(
                        MatrixBatchAttemptDiagnostic(
                            **metadata,
                            attempt=_attempt,
                            status="error",
                            error_type=type(exc).__name__,
                            error_category=category,
                            schema_location=location,
                        )
                    )
            if response is None:
                warnings.append(
                    f"evidence batch {batch_index} rejected after retry: "
                    f"{type(last_error).__name__} ({batch_attempts[-1].error_category})"
                )
                continue
            batches.append(response.parsed)
            if on_batch is not None:
                on_batch(batch_index, response.parsed)
            measurement_origins.extend(
                (batch_index, index)
                for index in range(1, len(response.parsed.measurements) + 1)
            )
        group_candidates = tuple(item for batch in batches for item in batch.groups)
        warnings.extend(
            f"source-selection candidate rejected: {reason}"
            for batch in batches
            for reason in batch.source_selection_rejections
        )
        measurement_candidates = tuple(
            item for batch in batches for item in batch.measurements
        )
        claim_candidates = tuple(
            item for batch in batches for item in batch.abstract_claims
        )
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        groups: list[ExperimentalGroup] = []
        group_ids: dict[str, str] = {}
        original_variables_by_id: dict[str, dict[str, str] | None] = {}
        for group_candidate in group_candidates:
            group_chunk, group_quote = _locate_group_evidence(
                group_candidate, chunks_by_id, chunks
            )
            if group_chunk is None or group_quote is None:
                warnings.append(
                    f"group {group_candidate.group_key} rejected: invalid quote"
                )
                continue
            if is_reference_evidence(group_quote, group_chunk, chunks):
                warnings.append(
                    f"group {group_candidate.group_key} rejected: reference section"
                )
                continue
            grounded_variables = _ground_mapping(
                group_candidate.variables,
                group_quote,
                numeric_fallback=True,
            )
            grounded_conditions = _ground_mapping(
                group_candidate.conditions,
                group_quote,
                numeric_fallback=False,
            )
            dropped = (
                len(group_candidate.variables)
                + len(group_candidate.conditions)
                - len(grounded_variables)
                - len(grounded_conditions)
            )
            if dropped:
                warnings.append(
                    f"group {group_candidate.group_key}: dropped {dropped} "
                    "ungrounded variable/condition fields"
                )
            if group_candidate.variables and not grounded_variables:
                warnings.append(
                    f"group {group_candidate.group_key} rejected: no grounded variable"
                )
                continue
            generated_group_id = _id("group", document_id, group_candidate.group_key)
            original_variables = dict(group_candidate.variables)
            if (
                generated_group_id in original_variables_by_id
                and original_variables_by_id[generated_group_id] != original_variables
            ):
                original_variables_by_id[generated_group_id] = None
            else:
                original_variables_by_id.setdefault(
                    generated_group_id, original_variables
                )
            group_ids[group_candidate.group_key] = generated_group_id
            groups.append(
                ExperimentalGroup(
                    group_id=generated_group_id,
                    document_id=document_id,
                    label=group_candidate.label,
                    role=group_candidate.role,
                    material=group_candidate.material,
                    variables=grounded_variables,
                    conditions=grounded_conditions,
                    source_quote=group_quote,
                    chunk_id=group_chunk.chunk_id,
                    page_from=group_chunk.page_from,
                    page_to=group_chunk.page_to,
                    source_text_sha256=group_chunk.text_sha256,
                    llm_extracted=True,
                    extraction_method="llm",
                    review_status="pending",
                )
            )
        measurements: list[ExperimentalMeasurement] = []
        for origin, measurement_candidate in zip(
            measurement_origins, measurement_candidates, strict=True
        ):
            measurement_chunk, measurement_quote = _locate_measurement_evidence(
                measurement_candidate, chunks_by_id, chunks
            )
            resolved_group_id = (
                group_ids.get(measurement_candidate.group_key)
                if measurement_candidate.group_key
                else _infer_group_id(measurement_quote, groups)
            )
            if not measurement_candidate.group_key and resolved_group_id is None:
                adjacent = _adjacent_capacity_group_evidence(
                    measurement_candidate,
                    measurement_chunk,
                    groups,
                    chunks_by_id,
                    original_variables_by_id,
                )
                if adjacent is not None:
                    resolved_group_id, measurement_quote = adjacent
                    warnings.append(
                        f"measurement {measurement_candidate.metric}: adjacent sample "
                        f"context resolved group {resolved_group_id}; quote sha256 "
                        f"{hashlib.sha256(measurement_quote.encode()).hexdigest()}; "
                        f"source chunk {measurement_candidate.chunk_id} unchanged"
                    )
            candidate_diagnostics.append(
                _measurement_candidate_diagnostic(
                    measurement_candidate,
                    origin=origin,
                    resolved_group_id=resolved_group_id,
                    chunk=measurement_chunk,
                    quote=measurement_quote,
                )
            )
            if not measurement_candidate.group_key and resolved_group_id is None:
                warnings.append(
                    f"measurement {measurement_candidate.metric} rejected: "
                    "missing or ambiguous group_key"
                )
                continue
            if (
                resolved_group_id is None
                or measurement_chunk is None
                or measurement_quote is None
            ):
                warnings.append(
                    f"measurement {measurement_candidate.metric} rejected: "
                    "unknown group or quote"
                )
                continue
            if not _numeric_value_consistent(
                measurement_candidate.value_text,
                measurement_candidate.numeric_value,
            ):
                warnings.append(
                    f"measurement {measurement_candidate.metric} rejected: "
                    "numeric value mismatch"
                )
                continue
            measurements.append(
                ExperimentalMeasurement(
                    measurement_id=_id(
                        "measurement",
                        document_id,
                        resolved_group_id,
                        measurement_candidate.metric,
                        measurement_candidate.value_text,
                        measurement_candidate.unit or "",
                    ),
                    group_id=resolved_group_id,
                    document_id=document_id,
                    metric=measurement_candidate.metric,
                    value_text=measurement_candidate.value_text,
                    numeric_value=measurement_candidate.numeric_value,
                    unit=measurement_candidate.unit,
                    uncertainty_text=measurement_candidate.uncertainty_text,
                    sample_size=measurement_candidate.sample_size,
                    source_quote=measurement_quote,
                    chunk_id=measurement_chunk.chunk_id,
                    page_from=measurement_chunk.page_from,
                    page_to=measurement_chunk.page_to,
                    source_text_sha256=measurement_chunk.text_sha256,
                    llm_extracted=True,
                    extraction_method="llm",
                    review_status="pending",
                )
            )
        measurements = list(
            expand_measurement_group_evidence(
                measurements,
                groups=groups,
                chunks_by_id=chunks_by_id,
                warnings=warnings,
            )
        )
        table_groups, table_measurements = extract_percent_sample_tables(
            document_id, chunks
        )
        groups.extend(table_groups)
        measurements.extend(table_measurements)
        before_validation = len(measurements)
        groups, measurements = _validate_individually(
            self.store,
            document_id=document_id,
            groups=groups,
            measurements=measurements,
            warnings=warnings,
        )
        after_validation = len(measurements)
        groups.extend(_extract_series_groups(document_id, chunks, groups))
        groups.extend(_extract_factorial_groups(document_id, chunks, groups))
        groups = list(_unique_by_id(groups, "group_id"))
        for generated_measurement in _extract_factorial_measurements(
            document_id, chunks, groups
        ):
            parent = next(
                group
                for group in groups
                if group.group_id == generated_measurement.group_id
            )
            try:
                validate_matrix_evidence(
                    self.store,
                    document_id=document_id,
                    groups=(parent,),
                    measurements=(generated_measurement,),
                )
            except ValueError as exc:
                warnings.append(f"factorial measurement rejected: {exc}")
            else:
                measurements.append(generated_measurement)
        groups, measurements = _consolidate_semantic_groups(groups, measurements)
        before_sanitize = len(measurements)
        measurements = list(
            sanitize_measurements(
                _unique_by_id(measurements, "measurement_id"),
                groups=groups,
                warnings=warnings,
                chunks_by_id=chunks_by_id,
            )
        )
        coverage = check_required_metric_coverage(measurements, required_metrics)
        if coverage.status == "incomplete":
            missing = sum(check.status == "missing" for check in coverage.checks)
            warnings.append(
                "REQUIRED_METRIC_COVERAGE_INCOMPLETE: "
                f"{missing} of {len(coverage.checks)} specified requirements "
                "have no validated measurement; this does not imply paper absence"
            )
        claims: list[PaperClaim] = []
        for claim_candidate in claim_candidates:
            claim_chunk = chunks_by_id.get(claim_candidate.chunk_id)
            claim_quote = _verbatim_quote(claim_candidate.source_quote, claim_chunk)
            if (
                claim_chunk is None
                or claim_quote is None
                or not _contains(claim_quote, claim_candidate.claim_text)
            ):
                warnings.append("abstract claim rejected: invalid verbatim evidence")
                continue
            claim = PaperClaim(
                claim_id=_id("claim", document_id, claim_candidate.claim_text),
                document_id=document_id,
                claim_text=claim_candidate.claim_text,
                source_section="abstract",
                source_quote=claim_quote,
                chunk_id=claim_chunk.chunk_id,
                page_from=claim_chunk.page_from,
                page_to=claim_chunk.page_to,
                source_text_sha256=claim_chunk.text_sha256,
                llm_extracted=True,
                review_status="pending",
            )
            try:
                validate_claim_evidence(self.store, claim)
            except ValueError as exc:
                warnings.append(f"abstract claim rejected: {exc}")
            else:
                claims.append(claim)
        comparisons = _build_deterministic_comparisons(groups, measurements)
        claim_links = tuple(
            assess_claim_against_body(
                claim,
                measurements=measurements,
                comparisons=comparisons,
            )
            for claim in claims
        )
        return PendingMatrixExtraction(
            groups=tuple(groups),
            measurements=tuple(measurements),
            comparisons=comparisons,
            claims=tuple(claims),
            claim_evidence_links=claim_links,
            warnings=tuple(warnings),
            diagnostics=MatrixExtractionDiagnostics(
                input_chunks=len(chunks),
                selected_chunks=len(selected),
                attempted_batches=len(evidence_batches),
                successful_batches=len(batches),
                llm_groups=len(group_candidates),
                llm_measurements=len(measurement_candidates),
                table_measurements=len(table_measurements),
                before_validation_measurements=before_validation,
                after_validation_measurements=after_validation,
                before_sanitize_measurements=before_sanitize,
                accepted_measurements=len(measurements),
                rejection_reasons=_rejection_counts(warnings),
                batch_attempts=tuple(batch_attempts),
                measurement_candidates=tuple(candidate_diagnostics),
                required_metric_coverage=coverage,
            ),
        )


def _rejection_counts(warnings: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for warning in warnings:
        if " rejected:" in warning:
            reason = warning.split(" rejected:", 1)[1].strip()
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def detect_matrix_warnings(chunks: Sequence[ChunkRecord]) -> tuple[str, ...]:
    warnings: set[str] = set()
    for chunk in chunks:
        text = chunk.text.casefold()
        if "table" in text and chunk.page_from != chunk.page_to:
            warnings.add("CROSS_PAGE_TABLE: table evidence spans multiple pages")
        if re.search(r"(?:shown|presented|plotted) in (?:fig\.|figure)", text):
            warnings.add("FIGURE_ONLY_VALUES: plotted values require manual review")
        if re.search(r"table\s+[s]?\d+", text) and not re.search(
            r"\d+(?:\.\d+)?\s*(?:%|mpa|gpa|ma|mv|v|ev)", text
        ):
            warnings.add(
                "MISSING_TABLE_DATA: table is referenced without parseable data"
            )
    return tuple(sorted(warnings))


def _build_deterministic_comparisons(
    groups: Sequence[ExperimentalGroup],
    measurements: Sequence[ExperimentalMeasurement],
) -> tuple[ExperimentalComparison, ...]:
    groups_by_id = {group.group_id: group for group in groups}
    baselines = [
        item
        for item in measurements
        if groups_by_id[item.group_id].role in {"control", "reference"}
    ]
    targets = [
        item for item in measurements if groups_by_id[item.group_id].role == "treatment"
    ]
    comparisons: list[ExperimentalComparison] = []
    for baseline in baselines:
        for target in targets:
            if _normalized_metric(baseline.metric) != _normalized_metric(target.metric):
                continue
            if _normalized_unit(baseline.unit) != _normalized_unit(target.unit):
                continue
            comparisons.append(calculate_comparison(baseline, target))
    return tuple(comparisons)


def measurement_binding_rejection_reason(
    row: ExperimentalMeasurement,
    parent: ExperimentalGroup | None,
    *,
    chunks_by_id: Mapping[str, ChunkRecord] | None = None,
) -> str | None:
    """Shared attribution/sample gate; do not normalize or mutate source records."""
    chunk = (chunks_by_id or {}).get(row.chunk_id)
    context = chunk.text if chunk is not None else row.source_quote
    if chunk is not None and is_reference_evidence(
        row.source_quote, chunk, tuple((chunks_by_id or {}).values())
    ):
        return "reference section"
    if is_prior_work(row.source_quote, context):
        return "prior work"
    binding_context = measurement_binding_context(row.source_quote, context)
    if (
        parent is not None
        and binding_context != context
        and not _contains(binding_context, parent.label)
        and not _has_group_binding(parent, binding_context)
    ):
        return "scoped prose does not bind sample"
    if parent is not None and not measurement_sample_bound(
        label=parent.label,
        value=row.value_text,
        unit=row.unit,
        metric=row.metric,
        quote=binding_context,
    ):
        return "ambiguous or incorrect sample/value binding"
    if (
        parent is not None
        and parent.variables
        and not _has_group_binding(parent, row.source_quote)
    ):
        return "evidence does not bind the assigned group variable"
    return None


def sanitize_measurements(
    rows: Sequence[ExperimentalMeasurement],
    *,
    groups: Sequence[ExperimentalGroup] = (),
    warnings: list[str],
    chunks_by_id: Mapping[str, ChunkRecord] | None = None,
) -> tuple[ExperimentalMeasurement, ...]:
    """Normalize table-header artifacts and remove duplicate/pseudo measurements."""
    accepted: list[ExperimentalMeasurement] = []
    seen: set[tuple[str, str, str, str]] = set()
    groups_by_id = {group.group_id: group for group in groups}
    for row in rows:
        parent = groups_by_id.get(row.group_id)
        reason = measurement_binding_rejection_reason(
            row, parent, chunks_by_id=chunks_by_id
        )
        if reason is not None:
            warnings.append(f"measurement {row.metric} rejected: {reason}")
            continue
        metric = _canonical_measurement_metric(row.metric, row.source_quote, row.unit)
        if _is_comparative_pseudomeasurement(metric, row.value_text):
            warnings.append(
                f"measurement {row.metric} rejected: comparative claim is not "
                "a single-group measurement"
            )
            continue
        normalized = row.model_copy(update={"metric": metric})
        key = (
            normalized.group_id,
            _normalized_metric(normalized.metric),
            _evidence_normalize(normalized.value_text),
            _normalized_unit(normalized.unit),
        )
        if key in seen:
            warnings.append(
                f"measurement {row.metric} removed: duplicate normalized measurement"
            )
            continue
        seen.add(key)
        accepted.append(normalized)
    return tuple(accepted)


def expand_measurement_group_evidence(
    rows: Sequence[ExperimentalMeasurement],
    *,
    groups: Sequence[ExperimentalGroup],
    chunks_by_id: dict[str, ChunkRecord],
    warnings: list[str] | None = None,
) -> tuple[ExperimentalMeasurement, ...]:
    """Expand quote context without inventing additional sample/value pairs.

    Co-occurring group variables do not establish that a measured value belongs
    to every group. Even shared outcomes must be supplied as explicit records;
    this step preserves the original measurement and group IDs.
    """
    groups_by_id = {group.group_id: group for group in groups}
    expanded: list[ExperimentalMeasurement] = []
    for row in rows:
        group = groups_by_id.get(row.group_id)
        chunk = chunks_by_id.get(row.chunk_id)
        scoped_quote = _explicit_capacity_quote(row, group, chunk, groups, chunks_by_id)
        if scoped_quote is not None:
            expanded.append(row.model_copy(update={"source_quote": scoped_quote}))
            if warnings is not None:
                warnings.append(
                    f"measurement {row.measurement_id} quote scope narrowed: "
                    f"{hashlib.sha256(row.source_quote.encode()).hexdigest()} -> "
                    f"{hashlib.sha256(scoped_quote.encode()).hexdigest()}; "
                    f"source chunk {row.chunk_id} unchanged"
                )
            continue
        if (
            group is not None
            and chunk is not None
            and not _has_group_binding(group, row.source_quote)
            and _has_group_binding(group, chunk.text)
        ):
            expanded.append(row.model_copy(update={"source_quote": chunk.text}))
        else:
            expanded.append(row)
    return tuple(expanded)


def _explicit_capacity_quote(
    row: ExperimentalMeasurement,
    group: ExperimentalGroup | None,
    chunk: ChunkRecord | None,
    groups: Sequence[ExperimentalGroup],
    chunks_by_id: dict[str, ChunkRecord],
) -> str | None:
    """Recover only a unique, explicit capacity sentence outside a known table.

    Never select a convenient fragment of a list, normalize source text, or
    resolve conflicting conditions. Missing measurements are not synthesized.
    """
    if (
        group is None
        or chunk is None
        or not group.variables
        or row.document_id != chunk.document_id
        or group.document_id != chunk.document_id
        or row.source_text_sha256 != chunk.text_sha256
        or (row.page_from, row.page_to) != (chunk.page_from, chunk.page_to)
        or chunk.text.count(row.source_quote) != 1
        or not re.search(r"\bcapacity\b", row.metric, re.IGNORECASE)
        or re.search(r"\bretention\b", row.metric, re.IGNORECASE)
        or row.unit not in {"mAh/g", "Ah/g"}
    ):
        return None
    if (
        measurement_binding_rejection_reason(row, group, chunks_by_id=chunks_by_id)
        != "ambiguous or incorrect sample/value binding"
    ):
        return None

    def binds_all(parent: ExperimentalGroup, sentence: str) -> bool:
        return bool(parent.variables) and all(
            _field_value_present(key, value, sentence)
            if _evidence_normalize(value) in _BOOLEAN_VALUES
            else _has_group_binding(
                parent.model_copy(update={"variables": {key: value}}), sentence
            )
            for key, value in parent.variables.items()
        )

    candidates: list[tuple[str, str, str]] = []
    for match in re.finditer(r".+?(?:[.!?](?=\s|$)|$)", row.source_quote, re.DOTALL):
        # A sentence match can include a preceding line-separated table. Keep
        # the earliest prose suffix outside its established boundary; do not
        # choose the suffix based on which value or group it happens to contain.
        raw = match.group()
        starts = [0, *(line.end() for line in re.finditer(r"\n", raw))]
        sentence = next(
            (
                raw[start:].strip()
                for start in starts
                if raw[start:].strip()
                and measurement_binding_context(raw[start:].strip(), chunk.text)
                == raw[start:].strip()
            ),
            None,
        )
        if sentence is None or not binds_all(group, sentence):
            continue
        if not re.search(r"\bcapacit(?:y|ies)\b", sentence, re.IGNORECASE):
            continue
        if re.search(r"\brespectively\b", sentence, re.IGNORECASE):
            return None
        if any(
            other.group_id != group.group_id and binds_all(other, sentence)
            for other in groups
        ):
            return None
        if len(set(re.findall(r"[+−-]?\d+(?:\.\d+)?\s*[°◦]\s*[CF]", sentence))) > 1:
            return None
        quantities = list(
            re.finditer(
                r"(?<![A-Za-z\d.])(?P<value>[+-]?\d+(?:\.\d+)?)\s*"
                r"(?P<prefix>m?Ah)\s*/\s*g(?![A-Za-z])",
                sentence,
            )
        )
        if len(quantities) != 1:
            return None
        quantity = quantities[0]
        candidates.append((sentence, quantity["value"], quantity["prefix"] + "/g"))
    if len(candidates) != 1:
        return None
    sentence, value, unit = candidates[0]
    if value != row.value_text.strip() or unit != row.unit:
        return None
    if row.uncertainty_text and not _value_present(row.uncertainty_text, sentence):
        return None
    if re.search(r"\binitial\b", row.metric, re.IGNORECASE) and not re.search(
        r"\b(?:initial|first)\b", sentence, re.IGNORECASE
    ):
        return None
    narrowed = row.model_copy(update={"source_quote": sentence})
    return (
        sentence
        if measurement_binding_rejection_reason(
            narrowed, group, chunks_by_id=chunks_by_id
        )
        is None
        else None
    )


def _has_group_binding(group: ExperimentalGroup, evidence: str) -> bool:
    normalized = _evidence_normalize(evidence)
    normalized_ratios = re.sub(r"(?<=\d):(?=\d)", ".", normalized)
    for key, value in group.variables.items():
        normalized_value = _evidence_normalize(value)
        if re.search(r"[a-z]", normalized_value) and _field_value_present(
            key, value, evidence
        ):
            return True
        if normalized_value in _BOOLEAN_VALUES:
            continue
        ratio_value = _ratio_normalize(value)
        anchors = _binding_anchors(key)
        if anchors:
            if not all(anchor in normalized_ratios for anchor in anchors):
                continue
            first_anchor = re.escape(anchors[0])
            if re.search(
                rf"{first_anchor}.{{0,24}}(?<![\d.])"
                rf"{re.escape(ratio_value)}(?![\d.])",
                normalized_ratios,
            ) or re.search(
                rf"(?<![\d.]){re.escape(ratio_value)}(?![\d.])"
                rf".{{0,24}}{first_anchor}",
                normalized_ratios,
            ):
                return True
        elif _value_present(value, evidence):
            return True
    return False


def _binding_anchors(variable_name: str) -> tuple[str, ...]:
    stopwords = {"ratio", "molar", "content", "concentration", "doping", "amount"}
    tokens = tuple(
        token
        for token in re.findall(r"[a-z]+\d*", variable_name.casefold())
        if token not in stopwords and len(token) >= 2
    )
    return tokens[:2]


def _consolidate_semantic_groups(
    groups: Sequence[ExperimentalGroup],
    measurements: Sequence[ExperimentalMeasurement],
) -> tuple[list[ExperimentalGroup], list[ExperimentalMeasurement]]:
    canonical_by_key: dict[tuple[str, ...], ExperimentalGroup] = {}
    remap: dict[str, str] = {}
    for group in groups:
        # Equal variable values alone do not establish sample identity. Preserve
        # material, experimental role, field names, units, and test conditions.
        key = (
            group.document_id,
            _evidence_normalize(group.material),
            group.role,
            repr(
                sorted(
                    (_evidence_normalize(name), _evidence_normalize(value))
                    for name, value in group.variables.items()
                )
            ),
            repr(
                sorted(
                    (_evidence_normalize(name), _evidence_normalize(value))
                    for name, value in group.conditions.items()
                )
            ),
            "" if group.variables else _evidence_normalize(group.label),
        )
        canonical = canonical_by_key.setdefault(key, group)
        remap[group.group_id] = canonical.group_id
    remapped = [
        item.model_copy(update={"group_id": remap.get(item.group_id, item.group_id)})
        for item in measurements
    ]
    return list(canonical_by_key.values()), remapped


def _extract_series_groups(
    document_id: str,
    chunks: Sequence[ChunkRecord],
    existing: Sequence[ExperimentalGroup],
) -> tuple[ExperimentalGroup, ...]:
    existing_values = {
        _ratio_normalize(value)
        for group in existing
        for value in group.variables.values()
    }
    generated: list[ExperimentalGroup] = []
    pattern = re.compile(
        r"(?:ratios?|contents?|concentrations?)\s+(?:for|of)\s+"
        r"(?P<variable>[A-Za-z0-9/=]+).*?varying\s+from\s+"
        r"(?P<series>.{5,180}?)\.(?:\s|$)",
        re.IGNORECASE | re.DOTALL,
    )
    for chunk in chunks:
        for match in pattern.finditer(chunk.text):
            variable = re.sub(r"\s+", " ", match.group("variable")).strip()
            for value in _number_tokens(match.group("series")):
                normalized = _ratio_normalize(value)
                if normalized in existing_values:
                    continue
                existing_values.add(normalized)
                generated.append(
                    ExperimentalGroup(
                        group_id=_id("group", document_id, variable, value),
                        document_id=document_id,
                        label=f"{variable} {value}",
                        role="unknown",
                        material=variable,
                        variables={f"{variable} ratio": value},
                        conditions={},
                        source_quote=chunk.text,
                        chunk_id=chunk.chunk_id,
                        page_from=chunk.page_from,
                        page_to=chunk.page_to,
                        source_text_sha256=chunk.text_sha256,
                        llm_extracted=True,
                        extraction_method="table_parser",
                        review_status="pending",
                    )
                )
    return tuple(generated)


def _extract_factorial_groups(
    document_id: str,
    chunks: Sequence[ChunkRecord],
    existing: Sequence[ExperimentalGroup],
) -> tuple[ExperimentalGroup, ...]:
    existing_labels = {_evidence_normalize(group.label) for group in existing}
    generated: list[ExperimentalGroup] = []
    pattern = re.compile(
        r"included\s+(?P<factors>.{5,160}?)\s+structures?,\s*each\s+"
        r"featuring\s+(?P<count>\w+)\s+porosity\s+levels?\s+"
        r"ranging\s+from\s+(?P<start>\d+(?:\.\d+)?)\s*%\s+to\s+"
        r"(?P<end>\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE | re.DOTALL,
    )
    number_words = {
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    for chunk in chunks:
        for match in pattern.finditer(chunk.text):
            count_text = match.group("count").casefold()
            count = number_words.get(count_text)
            if count is None and count_text.isdigit():
                count = int(count_text)
            if count is None or count < 2:
                continue
            start = float(match.group("start"))
            end = float(match.group("end"))
            step = (end - start) / (count - 1)
            levels = [start + index * step for index in range(count)]
            factor_text = re.sub(r"\([^)]*\)", "", match.group("factors"))
            factors = tuple(
                token.casefold()
                for token in re.findall(r"[A-Za-z][A-Za-z-]+", factor_text)
                if token.casefold() not in {"and", "the"}
            )
            series_text = f"{start:g} % to {end:g} %"
            reference_label = f"all {len(factors) * count} scaffold configurations"
            if _evidence_normalize(reference_label) not in existing_labels:
                existing_labels.add(_evidence_normalize(reference_label))
                generated.append(
                    ExperimentalGroup(
                        group_id=_id("group", document_id, reference_label),
                        document_id=document_id,
                        label=reference_label,
                        role="reference",
                        material="porous scaffold",
                        variables={
                            "analysis scope": f"{len(factors) * count} structures"
                        },
                        conditions={},
                        source_quote=chunk.text,
                        chunk_id=chunk.chunk_id,
                        page_from=chunk.page_from,
                        page_to=chunk.page_to,
                        source_text_sha256=chunk.text_sha256,
                        llm_extracted=True,
                        extraction_method="table_parser",
                        review_status="pending",
                    )
                )
            for factor in factors:
                for level in levels:
                    level_text = f"{level:g} %"
                    label = f"{factor} {level_text}"
                    if _evidence_normalize(label) in existing_labels:
                        continue
                    existing_labels.add(_evidence_normalize(label))
                    generated.append(
                        ExperimentalGroup(
                            group_id=_id("group", document_id, factor, level_text),
                            document_id=document_id,
                            label=label,
                            role="treatment",
                            material="porous scaffold",
                            variables={
                                "pore geometry": factor,
                                "porosity": level_text,
                                "porosity series": series_text,
                            },
                            conditions={},
                            source_quote=chunk.text,
                            chunk_id=chunk.chunk_id,
                            page_from=chunk.page_from,
                            page_to=chunk.page_to,
                            source_text_sha256=chunk.text_sha256,
                            llm_extracted=True,
                            extraction_method="table_parser",
                            review_status="pending",
                        )
                    )
    return tuple(generated)


def _extract_factorial_measurements(
    document_id: str,
    chunks: Sequence[ChunkRecord],
    groups: Sequence[ExperimentalGroup],
) -> tuple[ExperimentalMeasurement, ...]:
    pattern = re.compile(
        r"(?P<count>\d+)\s+structures?.{0,240}?porosity\s+levels?\s+"
        r"ranging\s+from\s+(?P<start>\d+(?:\.\d+)?)\s*%\s+to\s+"
        r"(?P<end>\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE | re.DOTALL,
    )
    generated: list[ExperimentalMeasurement] = []
    for chunk in chunks:
        for match in pattern.finditer(chunk.text):
            count = int(match.group("count"))
            group = next(
                (
                    item
                    for item in groups
                    if item.label == f"all {count} scaffold configurations"
                ),
                None,
            )
            if group is None:
                continue
            value_text = f"{match.group('start')} % to {match.group('end')} %"
            generated.append(
                ExperimentalMeasurement(
                    measurement_id=_id(
                        "measurement", document_id, group.group_id, value_text
                    ),
                    group_id=group.group_id,
                    document_id=document_id,
                    metric="screened porosity range",
                    value_text=value_text,
                    numeric_value=None,
                    unit=None,
                    source_quote=chunk.text,
                    chunk_id=chunk.chunk_id,
                    page_from=chunk.page_from,
                    page_to=chunk.page_to,
                    source_text_sha256=chunk.text_sha256,
                    llm_extracted=True,
                    extraction_method="table_parser",
                    review_status="pending",
                )
            )
    return tuple(generated)


def _canonical_measurement_metric(
    metric: str, source_quote: str, unit: str | None
) -> str:
    if (
        _normalized_metric(metric) == "h"
        and _normalized_unit(unit) == "%"
        and "photoconversion" in source_quote.casefold()
    ):
        return "photoconversion efficiency"
    return metric


def _is_comparative_pseudomeasurement(metric: str, value_text: str) -> bool:
    metric_text = metric.casefold()
    is_comparative_metric = any(
        term in metric_text
        for term in ("improvement", "change", "increase", "decrease")
    )
    has_range = bool(
        re.search(
            r"\d[^\n]{0,12}\b(?:to|through|-)\b[^\n]{0,12}\d",
            value_text.casefold(),
        )
    )
    return is_comparative_metric and has_range


def _validate_individually(
    store: VectorStore,
    *,
    document_id: str,
    groups: list[ExperimentalGroup],
    measurements: list[ExperimentalMeasurement],
    warnings: list[str],
) -> tuple[list[ExperimentalGroup], list[ExperimentalMeasurement]]:
    valid_groups: list[ExperimentalGroup] = []
    for group in groups:
        try:
            validate_matrix_evidence(
                store, document_id=document_id, groups=(group,), measurements=()
            )
        except ValueError as exc:
            warnings.append(f"group {group.label} rejected: {exc}")
        else:
            valid_groups.append(group)
    valid_ids = {group.group_id for group in valid_groups}
    valid_measurements: list[ExperimentalMeasurement] = []
    groups_by_id = {group.group_id: group for group in valid_groups}
    for measurement in measurements:
        parent_group = groups_by_id.get(measurement.group_id)
        if parent_group is None or measurement.group_id not in valid_ids:
            continue
        try:
            validate_matrix_evidence(
                store,
                document_id=document_id,
                groups=(parent_group,),
                measurements=(measurement,),
            )
        except ValueError as exc:
            warnings.append(f"measurement {measurement.metric} rejected: {exc}")
        else:
            valid_measurements.append(measurement)
    return valid_groups, valid_measurements


def _select_evidence(
    chunks: Sequence[ChunkRecord], *, max_evidence_chars: int
) -> tuple[ChunkRecord, ...]:
    ranked = sorted(
        chunks,
        key=lambda chunk: (
            -sum(term in chunk.text.casefold() for term in _EVIDENCE_TERMS),
            chunk.page_from,
            chunk.chunk_id,
        ),
    )
    selected: list[ChunkRecord] = []
    used = 0
    for chunk in ranked:
        if selected and used + len(chunk.text) > max_evidence_chars:
            continue
        selected.append(chunk)
        used += len(chunk.text)
    return tuple(selected)


def _chunk_batches(
    chunks: Sequence[ChunkRecord], *, batch_chars: int
) -> tuple[tuple[ChunkRecord, ...], ...]:
    batches: list[list[ChunkRecord]] = []
    current: list[ChunkRecord] = []
    used = 0
    for chunk in chunks:
        if current and used + len(chunk.text) > batch_chars:
            batches.append(current)
            current = []
            used = 0
        current.append(chunk)
        used += len(chunk.text)
    if current:
        batches.append(current)
    return tuple(tuple(batch) for batch in batches)


def _evidence_text(chunks: Sequence[ChunkRecord]) -> str:
    return "\n\n".join(
        f"[chunk_id={chunk.chunk_id}; pages={chunk.page_from}-{chunk.page_to}]\n"
        f"{chunk.text}"
        for chunk in chunks
    )


def _verbatim_quote(value: str, chunk: ChunkRecord | None) -> str | None:
    if chunk is None:
        return None
    start = chunk.text.find(value)
    if start >= 0:
        return chunk.text[start : start + len(value)]
    normalized_value = _evidence_normalize(value)
    if normalized_value and normalized_value in _evidence_normalize(chunk.text):
        return value
    sentences = tuple(
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n{2,}", chunk.text)
        if part.strip()
    )
    best_text: str | None = None
    best_score = 0.0
    for width in (1, 2):
        for index in range(0, len(sentences) - width + 1):
            candidate = " ".join(sentences[index : index + width])
            score = SequenceMatcher(
                None,
                normalized_value,
                _evidence_normalize(candidate),
            ).ratio()
            if score > best_score:
                best_score = score
                best_text = candidate
    return best_text if best_score >= 0.9 else None


def _group_evidence(candidate: GroupCandidate, chunk: ChunkRecord | None) -> str | None:
    if chunk is None:
        return None
    quote = _verbatim_quote(candidate.source_quote, chunk)
    variable_values = tuple(candidate.variables.values())
    if quote is not None and (
        _all_present(variable_values, quote)
        or _has_numeric_anchor(variable_values, quote)
    ):
        return quote
    if variable_values and (
        _all_present(variable_values, chunk.text)
        or _has_numeric_anchor(variable_values, chunk.text)
    ):
        return chunk.text
    if not variable_values and (
        _evidence_normalize(candidate.label) in _evidence_normalize(chunk.text)
        or _evidence_normalize(candidate.material) in _evidence_normalize(chunk.text)
    ):
        return chunk.text
    return None


def _ground_mapping(
    values: dict[str, str],
    evidence: str,
    *,
    numeric_fallback: bool,
) -> dict[str, str]:
    grounded: dict[str, str] = {}
    for key, value in values.items():
        temperature = re.fullmatch(
            r"(?P<number>[+\-−]?\d+(?:\.\d+)?)\s*[°◦]\s*(?P<scale>[CF])",
            value.strip(),
        )
        if temperature is not None:
            # Typography matching must retain the source's full, literal unit.
            # Never reduce an explicit temperature to a bare number or convert
            # scales, signs, or numeric precision to make it match.
            number = re.escape(temperature.group("number")).replace("−", "-")
            if number.startswith(r"\-"):
                number = "[-−]" + number[2:]
            elif number.startswith("-"):
                number = "[-−]" + number[1:]
            match = re.search(
                rf"(?<![\w.+\-−–]){number}\s*[°◦]\s*"
                rf"{temperature.group('scale')}(?![A-Za-z0-9])"
                r"(?!\s*/|\s+(?i:per)\b)"
                r"(?!\s*(?:[·⋅]\s*)?(?:min|s|h)\s*"
                r"(?:\^?\s*[-−]\s*1|⁻¹)\b)",
                evidence,
            )
            if match is not None:
                grounded[key] = match.group(0)
            continue
        if _field_value_present(key, value, evidence):
            grounded[key] = value
            continue
        numbers = _number_tokens(value)
        if numeric_fallback and len(numbers) == 1:
            match = re.search(
                rf"(?<![\d.]){re.escape(numbers[0])}(?![\d.])",
                evidence.replace("−", "-"),
            )
            if match is not None:
                grounded[key] = match.group(0)
    return grounded


def _has_numeric_anchor(values: Sequence[str], evidence: str) -> bool:
    evidence_numbers = set(_number_tokens(evidence))
    return any(
        numbers and set(numbers).issubset(evidence_numbers)
        for numbers in (_number_tokens(value) for value in values)
    )


def _number_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", value))


def _locate_group_evidence(
    candidate: GroupCandidate,
    chunks_by_id: dict[str, ChunkRecord],
    chunks: Sequence[ChunkRecord],
) -> tuple[ChunkRecord | None, str | None]:
    named = chunks_by_id.get(candidate.chunk_id)
    quote = _group_evidence(candidate, named)
    if named is not None and quote is not None:
        return named, quote
    for chunk in chunks:
        quote = _group_evidence(candidate, chunk)
        if quote is not None:
            return chunk, quote
    return None, None


def _measurement_evidence(
    candidate: MeasurementCandidate, chunk: ChunkRecord | None
) -> str | None:
    if chunk is None:
        return None
    required = tuple(value for value in (candidate.uncertainty_text,) if value)

    def grounded(text: str) -> bool:
        return (
            measurement_value_present(candidate.value_text, candidate.unit, text)
            and _all_present(required, text)
            and (
                candidate.unit is None or unit_present_in_evidence(candidate.unit, text)
            )
        )

    quote = _verbatim_quote(candidate.source_quote, chunk)
    if quote is not None and grounded(quote):
        return quote
    return chunk.text if grounded(chunk.text) else None


def _locate_measurement_evidence(
    candidate: MeasurementCandidate,
    chunks_by_id: dict[str, ChunkRecord],
    chunks: Sequence[ChunkRecord],
) -> tuple[ChunkRecord | None, str | None]:
    named = chunks_by_id.get(candidate.chunk_id)
    quote = _measurement_evidence(candidate, named)
    if named is not None and quote is not None:
        return named, quote
    for chunk in chunks:
        quote = _measurement_evidence(candidate, chunk)
        if quote is not None:
            return chunk, quote
    return None, None


def _adjacent_capacity_group_evidence(
    candidate: MeasurementCandidate,
    chunk: ChunkRecord | None,
    groups: Sequence[ExperimentalGroup],
    chunks_by_id: Mapping[str, ChunkRecord],
    original_variables_by_id: Mapping[str, Mapping[str, str] | None],
) -> tuple[str, str] | None:
    """Bind a missing key to one local parent using only two adjacent sentences.

    This is not a replacement for the existing inference/fallback paths. Never
    invent measurements, use a page-wide number match, or select by score.
    """
    cycles = re.search(r"\bafter\s+(\d+)\s+cycles?\b", candidate.metric, re.I)
    if (
        candidate.group_key
        or chunk is None
        or candidate.chunk_id != chunk.chunk_id
        or chunk.page_from != chunk.page_to
        or hashlib.sha256(chunk.text.encode()).hexdigest() != chunk.text_sha256
        or cycles is None
        or not re.search(r"\bdischarge capacity\b", candidate.metric, re.I)
        or re.search(r"\b(?:retention|initial)\b", candidate.metric, re.I)
        or candidate.unit not in {"mAh/g", "Ah/g"}
        or not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", candidate.value_text.strip())
        or not _numeric_value_consistent(candidate.value_text, candidate.numeric_value)
    ):
        return None
    # Only whitespace may differ. Recover the literal source span, not a fuzzy
    # sentence or a conveniently matching number from another page.
    pattern = r"\s+".join(re.escape(token) for token in candidate.source_quote.split())
    locations = list(re.finditer(pattern, chunk.text))
    if len(locations) != 1:
        return None
    located = locations[0]
    quote = located.group()
    sentences = list(re.finditer(r"\S.*?(?:[.!?](?=\s|$)|$)", chunk.text, re.S))
    index = next(
        (
            i
            for i, sentence in enumerate(sentences)
            if sentence.span() == located.span()
        ),
        None,
    )
    if index is None or index == 0:
        return None
    anchor = sentences[index - 1].group()
    evidence = chunk.text[sentences[index - 1].start() : located.end()]
    if (
        re.search(r"\n\s*\n", evidence)
        or not re.search(r"\b(?:sample|electrode|device)\b", anchor, re.I)
        or not re.search(
            r"\b(?:obtained|prepared|synthesized|calcined|sintered)\b", anchor, re.I
        )
        or len(re.findall(r"\b(?:sample|electrode|device)s?\b", anchor, re.I)) != 1
        or re.search(r"\b(?:samples|electrodes|devices|respectively)\b", evidence, re.I)
        or re.search(
            r"\b(?:sample|electrode|device|another|other|different)\b", quote, re.I
        )
        or re.search(r"\d\s*[°◦]\s*[CF]\b", quote)
        or len(re.findall(r"[+−-]?\d+(?:\.\d+)?\s*[°◦]\s*[CF]\b", anchor)) > 1
        or len(
            re.findall(
                r"\b\d+(?:\.\d+)?\s*(?:h|hours?|min|minutes?|s|seconds?)\b", anchor
            )
        )
        > 1
        or len(set(re.findall(r"(?<![\d.])\d+(?:\.\d+)?\s+C\b", evidence))) > 1
        or not re.search(rf"\bafter\s+{re.escape(cycles[1])}\s+cycles?\b", quote, re.I)
        or len(re.findall(r"\b\d+\s+cycles?\b", quote, re.I)) != 1
        or not re.search(r"\bdischarge capacity\b", quote, re.I)
        or is_prior_work(evidence, chunk.text)
        or is_reference_evidence(evidence, chunk, tuple(chunks_by_id.values()))
        or sample_table_rows(chunk.text)
        and measurement_binding_context(evidence, chunk.text) != evidence
    ):
        return None
    # A row-major table must not become antecedent prose even when its sample
    # header happens to share the target temperature.
    if any(
        re.fullmatch(r"\s*(?:Sample|Sample name|Table\b.*|Figure\b.*)\s*", line, re.I)
        for line in evidence.splitlines()
    ):
        return None
    quantities = list(
        re.finditer(
            r"(?<![A-Za-z\d.])([+-]?\d+(?:\.\d+)?)\s*(m?Ah)\s*/\s*g(?![A-Za-z])", quote
        )
    )
    if (
        len(quantities) != 1
        or quantities[0][1] != candidate.value_text.strip()
        or quantities[0][2] + "/g" != candidate.unit
        or candidate.uncertainty_text
        and not _value_present(candidate.uncertainty_text, quote)
    ):
        return None

    def variable_bound(key: str, value: str) -> bool:
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value.strip()):
            # Legacy grounding may have retained only a duration's numeric part.
            # Require a literal, unit-bearing 'for N h/min/s', never page/header N.
            if key.casefold().endswith(("_time", "_duration")):
                return bool(
                    re.search(
                        rf"\bfor\s+{re.escape(value.strip())}\s+"
                        r"(?:h|hours?|min|minutes?|s|seconds?)\b",
                        anchor,
                        re.I,
                    )
                )
            return False
        return _field_value_present(key, value, anchor)

    local = [g for g in groups if g.chunk_id == chunk.chunk_id]
    matching: dict[str, ExperimentalGroup] = {}
    for group in local:
        original_variables = original_variables_by_id.get(group.group_id)
        if (
            group.document_id != chunk.document_id
            or group.source_text_sha256 != chunk.text_sha256
            or (group.page_from, group.page_to) != (chunk.page_from, chunk.page_to)
            or not group.variables
            or original_variables is None
            or set(original_variables) != set(group.variables)
            or not _contains(group.source_quote, anchor)
            or not all(variable_bound(k, v) for k, v in group.variables.items())
            or any(
                re.fullmatch(
                    r"[+-]?\d+(?:\.\d+)?\s*(?:h|hours?|min|minutes?|s|seconds?)",
                    v,
                    re.I,
                )
                and not unit_present_in_evidence(v, anchor)
                for v in original_variables.values()
            )
        ):
            continue
        if group.group_id in matching and matching[group.group_id] != group:
            return None
        matching[group.group_id] = group
    if len(matching) != 1:
        return None
    group = next(iter(matching.values()))
    if not all(
        _field_value_present(k, v, evidence) for k, v in group.conditions.items()
    ):
        return None
    return group.group_id, evidence


def _infer_group_id(
    evidence: str | None, groups: Sequence[ExperimentalGroup]
) -> str | None:
    if not evidence:
        return None
    scores_by_id: dict[str, int] = {}
    for group in groups:
        score = int(_value_present(group.label, evidence)) + sum(
            _field_value_present(key, value, evidence)
            for key, value in group.variables.items()
        )
        if score:
            scores_by_id[group.group_id] = max(
                score, scores_by_id.get(group.group_id, 0)
            )
    scores = sorted(
        ((score, group_id) for group_id, score in scores_by_id.items()), reverse=True
    )
    if not scores or (len(scores) > 1 and scores[0][0] == scores[1][0]):
        return None
    return scores[0][1]


def _all_present(values: Sequence[str], text: str) -> bool:
    return all(_value_present(value, text) for value in values)


_BOOLEAN_VALUES = frozenset({"yes", "no", "true", "false"})


def _value_present(value: str, evidence: str) -> bool:
    normalized_value = _evidence_normalize(value)
    if not normalized_value:
        return False
    return (
        re.search(
            rf"(?<![a-z0-9])(?<!\d\.){re.escape(normalized_value)}(?![a-z0-9]|\.\d)",
            _evidence_normalize(evidence),
        )
        is not None
    )


def _field_value_present(key: str, value: str, evidence: str) -> bool:
    if not _value_present(value, evidence):
        return False
    if _evidence_normalize(value) not in _BOOLEAN_VALUES:
        return True
    # A bare "No" elsewhere on the page cannot prove a negative condition.
    anchors = _binding_anchors(key)
    if not anchors:
        return False
    normalized = _evidence_normalize(evidence)
    field = r"[^.;\n]{0,24}".join(re.escape(anchor) for anchor in anchors)
    literal = re.escape(_evidence_normalize(value))
    return (
        re.search(
            rf"(?<![a-z0-9])(?:{field}[^.;\n]{{0,24}}\b{literal}\b|"
            rf"{literal}\b[^.;\n]{{0,24}}{field})(?![a-z0-9])",
            normalized,
        )
        is not None
    )


def group_has_ungrounded_boolean_fields(group: ExperimentalGroup) -> bool:
    """Detect stale pending groups accepted by the former substring validator."""
    return any(
        _evidence_normalize(value) in _BOOLEAN_VALUES
        and not _field_value_present(key, value, group.source_quote)
        for key, value in (*group.variables.items(), *group.conditions.items())
    )


def _unique_by_id(rows: Sequence[_RowT], attribute: str) -> tuple[_RowT, ...]:
    unique: dict[str, _RowT] = {}
    for row in rows:
        unique.setdefault(str(getattr(row, attribute)), row)
    return tuple(unique.values())


def _contains(container: str, value: str) -> bool:
    return _evidence_normalize(value) in _evidence_normalize(container)


def _numeric_value_consistent(value_text: str, numeric_value: float | None) -> bool:
    if numeric_value is None:
        return True
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", value_text.replace("−", "-"))
    return any(abs(float(value) - numeric_value) <= 1e-9 for value in numbers)


def _normalized_metric(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _normalized_unit(value: str | None) -> str:
    return re.sub(r"\s+", "", (value or "").casefold())


def _ratio_normalize(value: str) -> str:
    normalized = _evidence_normalize(value)
    normalized = re.sub(r"(?<=\d):(?=\d)", ".", normalized)
    numbers = _number_tokens(normalized)
    return numbers[-1] if numbers else normalized


def _evidence_normalize(value: str) -> str:
    value = value.replace("−", "-").replace("–", "-").replace("×", "x")
    value = value.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"\s+", " ", value).strip().casefold()


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"
