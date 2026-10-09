"""Literal composite evidence for the new Master-only extraction contract.

No legacy matrix validators are weakened. Models select server source IDs;
the server restores independent citations and refuses ambiguous flattened tables.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.sub_agents.literature.evidence_scope import (
    is_prior_work,
    is_reference_evidence,
)

from .fulltext_conditions import SourceSpan


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SampleProposal(_Record):
    label: str = Field(min_length=1, max_length=100)
    source_id: str = Field(pattern=r"^s[0-9]+$")


class InventoryProposal(_Record):
    samples: tuple[SampleProposal, ...] = Field(default=(), max_length=12)


class LiteralSourceSpan(SourceSpan):
    # Catalogue-only oversized sources stay inspectable/explicitly blocked;
    # no model request may include a block over 3000 characters.
    quote: str = Field(min_length=1, max_length=65536)


class VerifiedSample(_Record):
    sample_id: str = Field(pattern=r"^sample-[a-f0-9]{24}$")
    document_id: str
    label: str
    definition: LiteralSourceSpan
    identity_status: Literal["literal_anchor", "defined", "ambiguous"] = (
        "literal_anchor"
    )
    material_status: Literal["unknown"] = "unknown"
    role: Literal["unknown"] = "unknown"
    # Composition/role are never inferred from title, question, or label.


class MetricProposal(_Record):
    sample_id: str = Field(pattern=r"^sample-[a-f0-9]{24}$")
    metric: str = Field(min_length=1, max_length=80)
    value_text: str = Field(min_length=1, max_length=100)
    unit: str = Field(min_length=1, max_length=32)
    source_id: str = Field(pattern=r"^s[0-9]+$")


class MetricsProposal(_Record):
    measurements: tuple[MetricProposal, ...] = Field(default=(), max_length=12)


class VerifiedMetric(_Record):
    measurement_id: str = Field(pattern=r"^measurement-[a-f0-9]{24}$")
    document_id: str
    sample_id: str
    metric: str
    value_text: str
    unit: str
    qualifier: Literal["exact", "approximate", "range", "bound"]
    source: LiteralSourceSpan
    sample_binding_source: LiteralSourceSpan
    review_status: Literal["pending"] = "pending"


_METRICS = {
    "transmittance": r"transmittance|transmission|透过率|透射率",
    "resistivity": r"resistivity|电阻率|ρ",
    "responsivity": r"responsivity|响应度",
    "strength": r"strength|强度",
    "degradation": r"degradation|降解率|降解效率",
}
_NUMBER = r"[+−-]?(?:\d+(?:\.\d*)?|\.\d+)"
# A caret or an explicit signed exponent is required. Never reinterpret an
# unsigned concatenated '104' or a damaged glyph as '10^4' / '10^-4'.
_SCALAR = _NUMBER + (r"(?:[eE][+−-]?\d+|\s*[×x]\s*10\s*(?:\^\s*[+−-]?\d+|[+−-]\d+))?")
_APPROX = r"(?:about|around(?:\s+of)?|approximately|约|~|≈)\s*"
_BOUND = (
    r"(?:(?:<=|>=|[<>≤≥])\s*|"
    r"(?:more\s+than|less\s+than|at\s+least|at\s+most|above|below|up\s+to)\s+)"
)


def _id(prefix, *parts):
    return prefix + hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]


def _contains_label(text, label):
    pattern = r"\s+".join(re.escape(word) for word in label.split())
    return (
        bool(pattern)
        and re.search(r"(?<![\w])" + pattern + r"(?![\w])", text) is not None
    )


def own_result_text(quote):
    """Keep an explicitly preceding own result, not its prior-report comparison.

    This is a literal prefix, never an inferred alias or a rewritten citation.
    Unsupported prior-report prose remains ineligible rather than guessed.
    """
    prior = re.search(r"\b(?:previous|prior|earlier)\s+reports?\b", quote, re.I)
    if prior is None:
        return quote
    comparison = re.search(
        r"\bwhich\s+(?:are|is|were|was)\s+compara\s*ble\s+(?:with|to)\b",
        quote[: prior.start()],
        re.I,
    )
    if comparison is None:
        raise ValueError("Prior report is not a current experimental sample source")
    prefix = quote[: comparison.start()]
    if re.search(r"\b(?:obtained|measured|observed)\s+for\b", prefix, re.I) is None:
        raise ValueError("Prior comparison lacks an explicit current sample result")
    return prefix


@lru_cache(maxsize=256)
def _chemical_formula(label):
    if re.fullmatch(r"[A-Za-z0-9().]+", label) is None:
        return False
    from pymatgen.core import Composition, Element

    # Composition also accepts isotope shortcuts such as T (tritium).
    # Do not reinterpret an experimental code like NTO3 as a chemical family.
    if any(
        not Element.is_valid_symbol(s) or Element(s).symbol != s
        for s in re.findall(r"[A-Z][a-z]?", label)
    ):
        return False

    try:
        return bool(Composition(label, strict=True).elements)
    except ValueError:
        return False


def _specific_description(label):
    # A source-literal single control or one explicit concentration is not a
    # generic doped family. Do not derive labels from a concentration list.
    control = re.fullmatch(r"(?:pure|undoped|un-doped)\s+(\S+)", label)
    if control:
        return _chemical_formula(control.group(1))
    doped = re.fullmatch(
        r"\d+(?:\.\d+)?\s+(?:at|wt|mol)\.?\s*%\s+"
        r"([A-Z][a-z]?)\s+doped\s+(\S+)",
        label,
    )
    return bool(doped) and all(_chemical_formula(v) for v in doped.groups())


def _named_definition(quote, label):
    label_pattern = r"\s+".join(map(re.escape, label.split()))
    sample_pattern = (
        label_pattern if label.startswith("Sample ") else r"Sample\s+" + label_pattern
    )
    if re.search(r"\b" + sample_pattern + r"\s+(?:is|was|contains|with)\b", quote):
        return True
    return (
        re.search(
            r"(?:referred\s+to\s+as|denoted\s+(?:as|by)|designated\s+as|"
            r"labelled|labeled|named|命名(?:为)?|记为)\s*"
            r"(?:(?:Sample\s+)?[A-Za-z][A-Za-z0-9_.:%+-]*\s*(?:,\s*|and\s+))*"
            + label_pattern
            + r"(?!\w)",
            quote,
        )
        is not None
    )


def _span(chunk, quote):
    return LiteralSourceSpan(
        chunk_id=chunk.chunk_id,
        text_sha256=chunk.text_sha256,
        page_from=chunk.page_from,
        page_to=chunk.page_to,
        quote=quote,
    )


def build_catalogue(chunks):
    """Cover all parsed chunks with literal sentence-sized sources, no top-k cap.

    Oversized single sentences/tables remain in the catalogue (and are explicitly
    blocked by the request planner); they are not silently cropped or submitted.
    Reference spans are retained for coverage but refused as experimental facts.
    """
    result = {}
    seen = set()
    documents = {c.document_id for c in chunks}
    if len(documents) != 1:
        raise ValueError("Catalogue requires one document")
    for chunk in chunks:
        if (
            chunk.chunk_id in seen
            or hashlib.sha256(chunk.text.encode()).hexdigest() != chunk.text_sha256
        ):
            raise ValueError("Invalid or duplicate source identity")
        seen.add(chunk.chunk_id)
        for quote in re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[。！？])", chunk.text):
            quote = quote.strip()
            if not quote:
                continue
            # Very long prose is split ONLY at original paragraph/line boundaries.
            pieces = [quote]
            if len(quote) > 3000:
                pieces, current = [], ""
                for line in quote.splitlines(keepends=True):
                    if current and len(current) + len(line) > 3000:
                        pieces.append(current.strip())
                        current = ""
                    current += line
                if current.strip():
                    pieces.append(current.strip())
            for piece in pieces:
                result[f"s{len(result)}"] = _span(chunk, piece)
    return result


def checked_span(span, chunks, *, document_id):
    by_id = {c.chunk_id: c for c in chunks}
    c = by_id.get(span.chunk_id)
    if (
        c is None
        or len(by_id) != len(chunks)
        or c.document_id != document_id
        or hashlib.sha256(c.text.encode()).hexdigest() != c.text_sha256
        or span.text_sha256 != c.text_sha256
        or (span.page_from, span.page_to) != (c.page_from, c.page_to)
        or span.quote not in c.text
        or is_reference_evidence(span.quote, c, chunks)
        or is_prior_work(span.quote, c.text)
    ):
        raise ValueError("Composite source is not verified own-work evidence")
    own_result_text(span.quote)
    return c


def checked_sample(proposal, catalogue, *, document_id="doc-1", chunks=None):
    proposal = SampleProposal.model_validate(proposal.model_dump())
    span = catalogue.get(proposal.source_id)
    label = " ".join(proposal.label.split())
    if span is None or not _contains_label(own_result_text(span.quote), label):
        raise ValueError("Unknown or nonliteral sample source")
    defined = _named_definition(own_result_text(span.quote), label)
    if re.search(r"[,，]|\b(?:films?|samples?|doped|pure)\b", label, re.I) and not (
        re.fullmatch(r"Sample\s+[A-Za-z0-9_.-]+", label, re.I)
        or _specific_description(label)
    ):
        raise ValueError("A family/series description is not one specific sample label")
    if _chemical_formula(label) and not defined:
        raise ValueError(
            "A material family formula is not a specific sample definition"
        )
    if chunks is not None:
        checked_span(span, chunks, document_id=document_id)
    return VerifiedSample(
        sample_id=_id("sample-", document_id, label),
        document_id=document_id,
        label=label,
        definition=span,
        identity_status="defined" if defined else "literal_anchor",
    )


def looks_like_flattened_table(text):
    # A text table's reading order alone cannot establish header/column/cell binding.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(lines) >= 5 and bool(
        re.search(r"(?im)^\s*(?:Table\s*\d|Parameter\b|表\s*\d)", text)
    )


def _literal_value(proposal, quote):
    value = proposal.value_text.strip()
    unit = proposal.unit.strip()
    # A duplicated exact declared unit is formatting, not a scientific
    # transformation. Number/unit must still occur together in the source.
    if value.endswith(unit):
        value = value[: -len(unit)].strip()
    # Accept only a raw numeric expression, not 'estimated', missing values, etc.
    raw = re.sub(r"^" + _BOUND, "", value, flags=re.I)
    raw = re.sub(r"^" + _APPROX, "", raw, flags=re.I)
    if re.fullmatch(rf"{_SCALAR}(?:\s*(?:-|–|to|至)\s*{_SCALAR})?", raw) is None:
        raise ValueError("Not a reported numeric expression")
    # Unit must immediately follow this value, not merely occur on the same page.
    pattern = re.compile(
        r"(?<![\w.])(?P<bound>"
        + _BOUND
        + r")?"
        + r"(?P<approx>"
        + _APPROX
        + r")?"
        + r"(?P<value>"
        + re.escape(raw)
        + r")\s*"
        + r"(?-i:"
        + re.escape(unit)
        + r")"
        + r"(?!\w)",
        re.I,
    )
    matches = list(pattern.finditer(quote))
    if len(matches) != 1:
        raise ValueError("Value/unit are not uniquely literal in the sample clause")
    match = matches[0]
    anchors = list(re.finditer(_METRICS[proposal.metric], quote, re.I))
    bound = False
    for anchor in anchors:
        if anchor.end() <= match.start():
            gap = quote[anchor.end() : match.start()]
            other_metrics = [p for key, p in _METRICS.items() if key != proposal.metric]
            if len(gap) <= 180 and not any(
                re.search(p, gap, re.I) for p in other_metrics
            ):
                bound = True
        elif match.end() <= anchor.start():
            gap = quote[match.end() : anchor.start()]
            if len(gap) <= 60 and re.fullmatch(
                r"\s*(?:(?:average|mean|optical|visible|light|UV[-–]visible|of|the)\s*)*",
                gap,
                re.I,
            ):
                bound = True
    if not bound:
        raise ValueError("This numeric value is not locally bound to the named metric")
    if match.group("bound"):
        return match.group("bound") + raw, "bound"
    if re.fullmatch(_SCALAR, raw) is None:
        return raw, "range"
    if match.group("approx"):
        return match.group("approx").strip() + " " + raw, "approximate"
    return raw, "exact"


def _metric_clause(quote, sample, samples, metric):
    """Check explicit contrast clauses independently, keeping the full citation.

    No inherited metric, pronoun resolution, list-order binding or table layout
    is inferred. Every side must name one different verified sample AND the
    requested metric. An incomplete inventory cannot remove the competition.
    """
    clauses = re.split(r",\s*(?:whereas|while)\b\s*", quote, flags=re.I)
    if len(clauses) > 1:
        bound, seen = None, set()
        for clause in clauses:
            matched = [s for s in samples.values() if _contains_label(clause, s.label)]
            if (
                len(matched) != 1
                or matched[0].identity_status == "ambiguous"
                or matched[0].sample_id in seen
                or re.search(_METRICS[metric], clause, re.I) is None
            ):
                raise ValueError(
                    "Contrast requires independent explicit sample/metric clauses"
                )
            seen.add(matched[0].sample_id)
            if matched[0].sample_id == sample.sample_id:
                bound = clause
        if bound is None:
            raise ValueError("Contrast does not bind the requested sample")
        return bound
    if any(
        s.sample_id != sample.sample_id and _contains_label(quote, s.label)
        for s in samples.values()
    ):
        raise ValueError("Multi-sample clause requires explicit cell/relation binding")
    return quote


def checked_metric(proposal, samples, catalogue, *, chunks=None):
    proposal = MetricProposal.model_validate(proposal.model_dump())
    sample, span = samples.get(proposal.sample_id), catalogue.get(proposal.source_id)
    if sample is None or sample.identity_status == "ambiguous" or span is None:
        raise ValueError("Unknown source or unsafe sample")
    own_quote = own_result_text(span.quote)
    if not _contains_label(own_quote, sample.label):
        raise ValueError("Metric clause does not explicitly bind its sample")
    if looks_like_flattened_table(span.quote):
        raise ValueError("Flattened table requires independently checked layout")
    pattern = _METRICS.get(proposal.metric)
    if pattern is None or re.search(pattern, own_quote, re.I) is None:
        raise ValueError("Metric not explicitly supported by this clause")
    metric_clause = _metric_clause(own_quote, sample, samples, proposal.metric)
    compact_unit = re.sub(r"\s+", "", proposal.unit)
    accepted_units = {
        "transmittance": {"%"},
        "degradation": {"%"},
        "strength": {"Pa", "kPa", "MPa", "GPa"},
        "responsivity": {"A/W", "mA/W", "μA/W", "µA/W", "uA/W"},
    }
    if (
        proposal.metric in accepted_units
        and compact_unit not in accepted_units[proposal.metric]
    ):
        raise ValueError("Unit does not represent the requested physical quantity")
    if (
        proposal.metric == "resistivity"
        and re.fullmatch(
            r"(?:[munμµ]?(?:Ω|Ω)|(?:micro|milli)?ohm)(?:[·.*-]?)(?:cm|m)", compact_unit
        )
        is None
    ):
        raise ValueError(
            "Resistance, sheet resistance or unknown unit is not resistivity"
        )
    if chunks is not None:
        checked_span(sample.definition, chunks, document_id=sample.document_id)
        checked_span(span, chunks, document_id=sample.document_id)
    value, qualifier = _literal_value(proposal, metric_clause)
    return VerifiedMetric(
        measurement_id=_id(
            "measurement-",
            sample.sample_id,
            proposal.metric,
            value,
            proposal.unit,
            span.chunk_id,
            span.quote,
        ),
        document_id=sample.document_id,
        sample_id=sample.sample_id,
        metric=proposal.metric,
        value_text=value,
        unit=proposal.unit,
        qualifier=qualifier,
        source=span,
        sample_binding_source=span,
    )


def reported_scalar(text):
    raw = re.sub(r"^" + _APPROX, "", text.strip(), flags=re.I)
    if re.fullmatch(_SCALAR, raw) is None:
        raise ValueError("Range, bound or damaged exponent is not a scalar")
    # Normalize a *present* Unicode minus only after literal evidence checking;
    # do not insert a missing sign, multiplication mark, or exponent.
    normalized = raw.replace("−", "-")
    match = re.fullmatch(
        rf"({_NUMBER})\s*[×x]\s*10\s*(?:\^\s*([+-]?\d+)|([+-]\d+))",
        normalized,
    )
    value = (
        float(match.group(1) + "e" + (match.group(2) or match.group(3)))
        if match
        else float(normalized)
    )
    if not math.isfinite(value):
        raise ValueError("Non-finite scalar")
    return value


def validate_composite(sample, metric, chunks, *, samples=None):
    sample = VerifiedSample.model_validate(sample.model_dump())
    metric = VerifiedMetric.model_validate(metric.model_dump())
    if sample.sample_id != _id("sample-", sample.document_id, sample.label):
        raise ValueError("Server sample identity changed")
    checked_span(sample.definition, chunks, document_id=sample.document_id)
    if not _contains_label(sample.definition.quote, sample.label):
        raise ValueError("Sample definition no longer supports its identity")
    if metric.document_id != sample.document_id or metric.sample_id != sample.sample_id:
        raise ValueError("Cross-document or cross-sample metric")
    if metric.sample_binding_source != metric.source:
        raise ValueError("Unsupported alternate sample binding")
    rebuilt = checked_metric(
        MetricProposal(
            sample_id=sample.sample_id,
            metric=metric.metric,
            value_text=metric.value_text,
            unit=metric.unit,
            source_id="s0",
        ),
        samples or {sample.sample_id: sample},
        {"s0": metric.source},
        chunks=chunks,
    )
    if rebuilt != metric:
        raise ValueError("Composite record differs from source replay")
    return metric
