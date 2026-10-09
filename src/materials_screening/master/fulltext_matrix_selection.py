"""Compact Master-only proposals resolved to unchanged, supplied PDF chunks.

These are candidates, never approved facts. The generic matrix extractor retains
all numeric, unit, sample/table binding, prior-work and provenance validation.
"""

import hashlib
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.llm.errors import LLMStructuredOutputError
from materials_screening.sub_agents.literature.evidence_scope import (
    measurement_sample_bound,
    sample_table_rows,
)
from materials_screening.sub_agents.literature.matrix import measurement_value_present
from materials_screening.sub_agents.literature.matrix_automation import (
    GroupCandidate,
    MatrixExtractionBatch,
    MeasurementCandidate,
    _evidence_text,
)


class SelectedGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str = Field(min_length=1, max_length=160)
    material: str = Field(min_length=1, max_length=160)
    source_id: str = Field(pattern=r"^c[1-9][0-9]*$")
    role: Literal["control", "treatment", "reference", "unknown"] = "unknown"
    variables: dict[str, str] = Field(default_factory=dict)
    conditions: dict[str, str] = Field(default_factory=dict)


class SelectedMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_label: str = Field(min_length=1, max_length=160)
    metric: str = Field(min_length=1, max_length=160)
    value_text: str = Field(min_length=1, max_length=160)
    unit: str | None = Field(default=None, max_length=60)
    source_id: str = Field(pattern=r"^c[1-9][0-9]*$")
    uncertainty_text: str | None = Field(default=None, max_length=80)
    sample_size: int | None = Field(default=None, ge=1)


class SelectedMatrixBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    groups: tuple[SelectedGroup, ...] = Field(default=(), max_length=50)
    measurements: tuple[SelectedMeasurement, ...] = Field(default=(), max_length=200)


SELECTION_PROMPT = """Extract experimental measurements needed by the user from
the supplied paper evidence. Return compact JSON matching the supplied schema.
Do not copy or rewrite source quotes, chunk IDs, page numbers, or numeric_value.
Select source_id (c1, c2, ...) from THIS request; the server retrieves unchanged
source text. A citation is not proof: every number/unit/sample is checked later.
For each measurement create its group in the same response. sample_label must
EXACTLY equal that group's label AND be a literal sample/device label in the
selected group source. Reuse the exact label across batches, not arbitrary keys.
Never use a generic family such as NTO for measurements of NTO3, invent a sample,
or map a list of sample labels to values without explicit table-column evidence.
Extract only metrics explicitly requested for experimental analysis, and their
reported doping/preparation/test conditions. Do not expand to unrelated metrics.
Do not include missing values (not specified, unknown, N/A) as measurements; omit
such rows. Empty lists are valid, including methods-only or references-only text.
Keep value_text as the reported numeric value/range/bound, without its unit.
Do not calculate, convert, infer signs/exponents damaged by PDF text extraction,
estimate a plot, or treat cited previous studies as this work. Do not turn a
figure of merit into sheet resistance/resistivity. Keep ambiguous data unfilled.
Claims and summaries are handled separately; do not add abstract_claims here.
Example of attribution (illustration, NOT paper evidence): for "S-2 film has
transmittance 91%", label and sample_label are "S-2", value_text is "91", unit
is "%". They are NOT "films", "91%", "increased" or "not specified". When only
the sentence names one best-performing sample, emit only that sample's measured
value; do not invent values for the other films. A numeric preparation parameter
belongs in conditions/variables, not as a performance measurement.
"""


def supplied_sources(text, chunks):
    """Bind only the exact current batch, not all chunks known to the task."""
    ids = re.findall(r"^\[chunk_id=([^;\n]+); pages=\d+-\d+\]$", text, re.M)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(by_id) != len(chunks) or not ids or len(set(ids)) != len(ids):
        raise ValueError("Invalid extraction batch source identities")
    if any(key not in by_id for key in ids):
        raise ValueError("Unknown extraction batch source identity")
    selected = tuple(by_id[key] for key in ids)
    if text != _evidence_text(selected) or any(
        hashlib.sha256(chunk.text.encode()).hexdigest() != chunk.text_sha256
        for chunk in selected
    ):
        raise ValueError("Extraction batch differs from checked source")
    sources = {f"c{index}": chunk for index, chunk in enumerate(selected, 1)}
    evidence = "\n\n".join(
        f"[source_id={key}; page={chunk.page_from}-{chunk.page_to}]\n{chunk.text}"
        for key, chunk in sources.items()
    )
    return sources, evidence


def _numeric(value):
    # Only an explicit plain scalar yields a scalar; ranges, bounds, and damaged
    # scientific notation remain non-scalars for existing isolation checks.
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value.strip()):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def resolve_selected_matrix(proposal, sources):
    def source(key):
        if key not in sources:
            raise LLMStructuredOutputError("Unknown selected extraction source")
        return sources[key]

    groups, labels, rejected = [], set(), []
    for row in proposal.groups:
        chunk = source(row.source_id)
        if row.label in labels:
            raise LLMStructuredOutputError("Duplicate sample label in response")
        if re.search(rf"(?<!\w){re.escape(row.label)}(?!\w)", chunk.text) is None:
            # Invalid scientific proposals are not failed network requests.
            rejected.append("nonliteral sample label")
            continue
        labels.add(row.label)
        groups.append(
            GroupCandidate(
                group_key=row.label,
                **row.model_dump(exclude={"source_id"}),
                chunk_id=chunk.chunk_id,
                source_quote=chunk.text,
            )
        )
    measurements = []
    for row in proposal.measurements:
        chunk = source(row.source_id)
        value = row.value_text
        # Separate a literal repeated unit suffix into its existing unit field;
        # never convert units or repair signs/exponents damaged in the PDF.
        if row.unit and value.endswith(row.unit):
            value = value[: -len(row.unit)].rstrip()
        value = value or row.value_text
        if not re.fullmatch(
            r"(?:[<>≤≥~≈]\s*)?[+−-]?(?:\d+(?:\.\d*)?|\.\d+)"
            r"[\d.eExX×*^+−–\-/()\s]*",
            value,
        ):
            rejected.append("nonnumeric value")
            continue
        # Recover a short continuous literal clause where possible, never rewrite
        # PDF symbols or stitch distant samples/values. Remaining candidates are
        # still rejected by the generic evidence/unit/sample gates if ambiguous.
        quote = None
        if (
            sample_table_rows(chunk.text)
            and measurement_sample_bound(
                label=row.sample_label,
                value=value,
                unit=row.unit,
                metric=row.metric,
                quote=chunk.text,
            )
            and measurement_value_present(value, row.unit, chunk.text)
        ):
            quote = chunk.text
        for clause in re.split(r";|(?<=[.!?])\s+", chunk.text):
            if re.search(
                rf"(?<!\w){re.escape(row.sample_label)}(?!\w)", clause
            ) and measurement_value_present(value, row.unit, clause):
                # A column heading is not a binding for an arbitrary value in a
                # flattened, unsupported table.
                if re.search(rf"^\s*{re.escape(row.sample_label)}\s*$", clause, re.M):
                    continue
                quote = clause
                break
        if quote is None:
            rejected.append("unbound sample/value")
            continue
        if row.sample_label not in labels:
            # The same literal measured-sample anchor establishes an opaque
            # parent. Never infer composition, doping, role or conditions from
            # a missing/mismatched model group.
            labels.add(row.sample_label)
            groups.append(
                GroupCandidate(
                    group_key=row.sample_label,
                    label=row.sample_label,
                    material=row.sample_label,
                    role="unknown",
                    variables={},
                    conditions={},
                    source_quote=quote,
                    chunk_id=chunk.chunk_id,
                )
            )
        measurements.append(
            MeasurementCandidate(
                group_key=row.sample_label,
                **row.model_dump(exclude={"sample_label", "source_id", "value_text"}),
                value_text=value,
                numeric_value=_numeric(value),
                chunk_id=chunk.chunk_id,
                source_quote=quote,
            )
        )
    return MatrixExtractionBatch(
        groups=tuple(groups),
        measurements=tuple(measurements),
        source_selection_rejections=tuple(rejected),
    )
