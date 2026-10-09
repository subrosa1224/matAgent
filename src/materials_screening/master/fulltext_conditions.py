"""Supplementary source-bound attributes, never automatic expert approval.

These checks establish literal citation, identity and numeric binding. They do
not prove semantic applicability or a scientifically complete comparable set.
The caller must keep the result pending and preserve the original snapshot.
"""

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.sub_agents.literature.evidence_scope import (
    is_reference_evidence,
    normalized,
)


class SourceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    chunk_id: str = Field(min_length=1, max_length=300)
    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=1800)


class AttributeBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    measurement_ids: tuple[str, ...] = Field(min_length=1, max_length=200)
    kind: Literal["condition", "preparation"]
    key: str = Field(min_length=1, max_length=100)
    value_text: str = Field(min_length=1, max_length=160)
    numeric_value: float | None = Field(default=None, allow_inf_nan=False)
    unit: str | None = Field(default=None, max_length=32)
    value_source: SourceSpan
    applicability: Literal["per_sample", "reported_common_protocol"]
    applicability_source: SourceSpan


class IsolatedAttribute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    key: str = Field(min_length=1, max_length=100)
    proposed_value: str = Field(min_length=1, max_length=160)
    reason: Literal[
        "unlocated_value",
        "nonliteral_value",
        "numeric_or_phase_mismatch",
        "source_evidence_failed",
    ]


class ConditionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    bindings: tuple[AttributeBinding, ...] = Field(default=(), max_length=40)
    unresolved: tuple[str, ...] = Field(default=(), max_length=20)
    review_status: Literal["pending"] = "pending"
    isolated_attributes: tuple[IsolatedAttribute, ...] = Field(
        default=(), max_length=40
    )


class CitationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    key: str = Field(pattern=r"^s[0-9]{1,3}$")
    chunk_key: str = Field(pattern=r"^c[0-9]{1,3}$")
    quote: str = Field(min_length=1, max_length=1800)


class AttributeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    measurement_keys: tuple[str, ...] = Field(min_length=1, max_length=200)
    kind: Literal["condition", "preparation"]
    key: str = Field(min_length=1, max_length=100)
    value_text: str = Field(min_length=1, max_length=160)
    numeric_value: float | None = Field(default=None, allow_inf_nan=False)
    unit: str | None = Field(default=None, max_length=32)
    value_source: str = Field(pattern=r"^s[0-9]{1,3}$")
    applicability: Literal["per_sample", "reported_common_protocol"]
    applicability_source: str = Field(pattern=r"^s[0-9]{1,3}$")


class ConditionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    citations: tuple[CitationProposal, ...] = Field(default=(), max_length=40)
    bindings: tuple[AttributeProposal, ...] = Field(default=(), max_length=40)
    unresolved: tuple[str, ...] = Field(default=(), max_length=20)


def _resolve_proposal(proposal, measurements, chunks):
    proposal = ConditionProposal.model_validate(proposal.model_dump())
    spans = {}
    for citation in proposal.citations:
        if citation.key in spans or citation.chunk_key not in chunks:
            raise ValueError("Invalid or duplicate compact citation reference")
        chunk = chunks[citation.chunk_key]
        spans[citation.key] = SourceSpan(
            chunk_id=chunk.chunk_id,
            text_sha256=chunk.text_sha256,
            page_from=chunk.page_from,
            page_to=chunk.page_to,
            quote=citation.quote,
        )
    bindings = []
    for row in proposal.bindings:
        if (
            not set(row.measurement_keys).issubset(measurements)
            or row.value_source not in spans
            or row.applicability_source not in spans
        ):
            raise ValueError("Unknown compact attribute reference")
        bindings.append(
            AttributeBinding(
                measurement_ids=tuple(
                    measurements[key] for key in row.measurement_keys
                ),
                **row.model_dump(
                    exclude={"measurement_keys", "value_source", "applicability_source"}
                ),
                value_source=spans[row.value_source],
                applicability_source=spans[row.applicability_source],
            )
        )
    return ConditionPlan(bindings=tuple(bindings), unresolved=proposal.unresolved)


def validate_condition_plan(plan, candidates, chunks):
    """Fail closed for wrong sources, values, targets and duplicate attributes."""
    plan = ConditionPlan.model_validate(plan.model_dump())
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise ValueError("Duplicate condition source identity")

    def checked(span, document):
        chunk = by_id.get(span.chunk_id)
        if (
            chunk is None
            or chunk.document_id != document
            or span.text_sha256 != chunk.text_sha256
            or hashlib.sha256(chunk.text.encode()).hexdigest() != chunk.text_sha256
            or (span.page_from, span.page_to) != (chunk.page_from, chunk.page_to)
            or normalized(span.quote) not in normalized(chunk.text)
            or is_reference_evidence(span.quote, chunk, chunks)
        ):
            raise ValueError("Supplementary attribute source is not verified")

    seen = set()
    for binding in plan.bindings:
        if (
            not binding.key.strip()
            or not binding.value_text.strip()
            or len(set(binding.measurement_ids)) != len(binding.measurement_ids)
            or not set(binding.measurement_ids).issubset(candidates)
        ):
            raise ValueError("Invalid supplementary attribute target")
        documents = {candidates[mid]["document_id"] for mid in binding.measurement_ids}
        if len(documents) != 1:
            raise ValueError("Common protocol cannot cross documents")
        document = next(iter(documents))
        checked(binding.value_source, document)
        checked(binding.applicability_source, document)
        if normalized(binding.value_text) not in normalized(binding.value_source.quote):
            raise ValueError("Attribute value not present in its literal source")
        if binding.numeric_value is not None:
            # No conversions, midpoint estimates, label-only numeric inference,
            # or scaling. Keep the exact displayed number and unit as reported.
            value = normalized(binding.value_text)
            unit = normalized(binding.unit or "")
            pattern = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)"
            match = re.fullmatch(pattern + r"\s*" + re.escape(unit), value)
            if match is None or float(match.group(1)) != binding.numeric_value:
                raise ValueError("Unsupported attribute numeric derivation")
        scope = normalized(binding.applicability_source.quote)
        if binding.applicability == "reported_common_protocol":
            if not re.search(
                r"\ball\s+(?:the\s+)?samples\b|所有样品|全部样品|各样品", scope
            ):
                raise ValueError("Missing explicit common-sample protocol statement")
        else:
            for mid in binding.measurement_ids:
                if normalized(candidates[mid]["group_label"]).replace(
                    " ", ""
                ) not in scope.replace(" ", ""):
                    raise ValueError("Missing explicit sample applicability quote")
        for mid in binding.measurement_ids:
            identity = (mid, binding.kind, binding.key)
            if identity in seen:
                raise ValueError("Duplicate or conflicting supplementary attribute")
            seen.add(identity)
    return plan


_PROMPT = """Read only the supplied paper evidence. Propose supplementary test
conditions and preparation variables for the explicit selected measurement keys.
Do not rewrite measured values, infer facts from the user request, approve records,
or declare a complete comparable set. Each value needs its shortest sufficient
contiguous literal quote. Put each reusable quote ONCE in citations with a local
s0/s1/... key and its provided c0/c1/... chunk_key. Bindings reference citation keys
and provided m0/m1/... measurement_keys; NEVER invent unknown keys. Server assigns
the original IDs/hashes/pages. For common attributes use ONE binding with all
applicable measurement keys, never repeat it once per sample. Numeric values
must equal the displayed number/unit (e.g. 10% -> 10, %, never 0.10). Do not infer
preparation numbers merely from the user or a label without paper evidence.
Per-sample applicability needs the sample's explicit label in a scope quote.
reported_common_protocol needs an explicit 'all samples' statement in the scope
quote, from the same paper. Citation presence alone is not semantic approval.
Each measurement/kind/key must occur only once. Unclear scope, abbreviations,
contradictory or unsupported conditions go in unresolved, not invented bindings.
Use compact JSON, unique reusable citations and short sufficient quotes."""

_PROMPT += """
CRITICAL extraction contract: ONE binding = ONE atomic test condition or
preparation variable. value_text is an EXACT SUBSTRING of value_source's quote,
not a paraphrase, explanation, sentence assembled from several places, or a
general paper summary. For example, from 'All the samples were measured in water
at 25 C.', produce two separate bindings: key=test_medium,value_text=water;
key=temperature,value_text=25 C,numeric_value=25,unit=C. Both can reference the
same citation and use reported_common_protocol with the explicit all-samples
quote. Do not choose per_sample for a shared protocol: it requires every named
sample label in its applicability quote. Leave unresolved if no such evidence.
Include only attributes relevant to the SELECTED measurement's test or changed
preparation variable; omit unrelated characterization and other measured
outcomes (such as bandgap when reviewing degradation). Do not assign composite
preparation or loading to pure controls. Cite only quotes that support a
binding or an unresolved item, rather than one summary from every chunk.
Keep each key short and stable (e.g. pollutant, light_source, cutoff_wavelength,
catalyst_mass, solution_volume, initial_concentration, irradiation_time,
loading_fraction). A loading range is not an individual sample's loading.
"""


def request_condition_plan(
    *, calls, requirements, candidates, chunks, max_evidence_chars=24000
):
    """One shared-budget proposal; bounded selected evidence, never fake output."""
    if not candidates:
        return ConditionPlan(unresolved=("没有任务选定的候选；未请求条件模型。",))
    selected, remaining = [], min(max_evidence_chars, 24000)
    # Prefer full protocol/result headings, not table-only sample/value snippets.
    pattern = re.compile(
        r"methods|experimental|measurements|performance|方法|测试|实验", re.I
    )
    ordered = sorted(
        chunks,
        key=lambda chunk: (
            not bool(pattern.search(chunk.text)),
            chunk.page_from,
            chunk.chunk_id,
        ),
    )
    for chunk in ordered:
        if len(selected) >= 6:
            break
        if len(chunk.text) <= remaining:
            selected.append(chunk)
            remaining -= len(chunk.text)
    if not selected:
        return ConditionPlan(
            unresolved=("没有完整源片段适合有界条件窗口；未补推条件。",)
        )
    measurements = {f"m{index}": mid for index, mid in enumerate(sorted(candidates))}
    chunk_keys = {f"c{index}": chunk for index, chunk in enumerate(selected)}
    metadata = [
        dict(
            measurement_key=alias,
            **{key: candidates[mid][key] for key in ("group_label", "metric", "unit")},
        )
        for alias, mid in measurements.items()
    ]
    payload = json.dumps(
        {
            "requirements_only": requirements,
            "selected_measurements": metadata,
            "evidence": [
                {
                    "chunk_key": alias,
                    "text": c.text,
                }
                for alias, c in chunk_keys.items()
            ],
        },
        ensure_ascii=False,
    )
    if len(payload.encode()) > 65536:
        raise ValueError("Condition request exceeds bounded input window")
    response = calls.generate_structured(
        system_prompt=_PROMPT,
        user_text=payload,
        output_model=ConditionProposal,
        schema_name="fulltext_supplementary_conditions_v3_atomic",
        max_output_tokens=4096,
    )
    plan = _resolve_proposal(response.parsed, measurements, chunk_keys)
    return validate_condition_plan(plan, candidates, selected)
