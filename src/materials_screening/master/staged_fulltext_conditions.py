"""Adapters for existing condition proposals under composite sample evidence."""

import re

from .fulltext_conditions import ConditionPlan, validate_condition_plan
from .staged_fulltext_evidence import (
    _METRICS,
    _contains_label,
    checked_span,
    own_result_text,
)

_WAVELENGTH_KEYS = {
    "wavelength",
    "measurement_wavelength",
    "optical_wavelength",
    "transmittance_wavelength",
    "wavelength_range",
    "measurement_wavelength_range",
    "optical_wavelength_range",
    "transmittance_wavelength_range",
}


def _validate_wavelength(binding, candidates):
    if binding.kind != "condition":
        raise ValueError("A test wavelength is not a preparation variable")
    number = r"\+?(?:\d+(?:\.\d*)?|\.\d+)"
    units = r"(?:nm|µm|μm|um|mm|cm|m|Å|angstrom)"
    if binding.key.endswith("_range"):
        pattern = (
            r"(?P<lo>" + number + r")\s*(?P<unit>" + units + r")\s*"
            r"(?:to|[-–−])\s*(?P<hi>" + number + r")\s*(?P=unit)"
        )
        match = re.fullmatch(pattern, binding.value_text)
        if match is None:
            pattern = (
                r"(?P<lo>" + number + r")\s*(?:to|[-–−])\s*"
                r"(?P<hi>" + number + r")\s*(?P<unit>" + units + r")"
            )
            match = re.fullmatch(pattern, binding.value_text)
        if (
            match is None
            or binding.numeric_value is not None
            or not 0 < float(match["lo"]) <= float(match["hi"])
        ):
            raise ValueError("A wavelength range needs literal positive endpoints")
    else:
        match = re.fullmatch(
            r"(?P<value>" + number + r")\s*(?P<unit>" + units + r")", binding.value_text
        )
        if match is None or float(match["value"]) <= 0:
            raise ValueError("A wavelength condition needs a literal length value")
    if binding.unit is not None and binding.unit != match["unit"]:
        raise ValueError("Wavelength unit does not match its literal length value")
    if binding.key.startswith("transmittance_") and any(
        candidates[mid]["metric"] != "transmittance" for mid in binding.measurement_ids
    ):
        raise ValueError("Transmittance wavelength cannot apply to another metric")


def restore_literal(proposed, source):
    """Recover original whitespace only; no fuzzy spelling or exponent repairs."""
    if proposed in source:
        return proposed
    tokens = re.findall(r"\d+(?:\.\d+)?|[^\W\d_]+|[^\w\s]", proposed)
    if not tokens:
        return None
    matches = list(re.finditer(r"\s*".join(map(re.escape, tokens)), source))
    return matches[0].group() if len(matches) == 1 else None


def _ordered_value(quote, label, all_labels):
    relation = re.search(
        r"referred\s+to\s+as|denoted\s+(?:as|by)|designated\s+as", quote, re.I
    )
    if relation is None:
        return None
    prefix, tail = quote[: relation.start()], quote[relation.end() :]
    values = re.findall(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:wt\s*%|mol\s*%|at\s*%)", prefix, re.I
    )
    labels = sorted(
        (s for s in all_labels if _contains_label(tail, s)), key=lambda s: tail.index(s)
    )
    # Literal pure-and-doped definition: the pure label has no listed dopant
    # value. Never invent its concentration; only bind the explicitly ordered
    # doped series when all remaining counts match.
    if (
        len(labels) == len(values) + 1
        and re.search(r"\bpure\b", prefix, re.I)
        and labels
        and not re.search(r"\d", labels[0])
        and all(re.search(r"\d", s) for s in labels[1:])
    ):
        labels = labels[1:]
    if len(labels) != len(values) or label not in labels:
        return None
    return values[labels.index(label)]


def validate_staged_attribute(binding, candidates, chunks, *, samples):
    identity_key = re.sub(r"[\s-]+", "_", binding.key.casefold())
    if identity_key in {
        "sample_label",
        "sample_name",
        "sample_code",
        "sample_id",
        "sample_type",
        "group_label",
        "group_id",
        "group_type",
        "measurement_id",
        "specimen_label",
        "specimen_type",
    }:
        raise ValueError(
            "Sample identity is not an experimental condition or preparation"
        )
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", binding.value_text):
        raise ValueError(
            "Damaged literal attribute needs verification, not normalization"
        )
    wavelength = binding.key in _WAVELENGTH_KEYS
    if not wavelength and any(
        re.search(pattern, binding.key, re.I) for pattern in _METRICS.values()
    ):
        raise ValueError(
            "A performance is not a test condition or preparation variable"
        )
    if binding.kind == "condition" and re.search(
        r"dop|loading|deposition|anneal", binding.key, re.I
    ):
        raise ValueError("Preparation variable was labelled as test condition")
    validate_condition_plan(ConditionPlan(bindings=(binding,)), candidates, chunks)
    if wavelength:
        _validate_wavelength(binding, candidates)
    document = candidates[binding.measurement_ids[0]]["document_id"]
    checked_span(binding.value_source, chunks, document_id=document)
    checked_span(binding.applicability_source, chunks, document_id=document)
    if any(
        own_result_text(s.quote) != s.quote
        for s in (binding.value_source, binding.applicability_source)
    ):
        raise ValueError("Mixed prior-report comparison is not a condition scope")
    labels = [s.label for s in samples.values()]
    if binding.kind == "condition":
        scope = binding.applicability_source.quote
        for mid in binding.measurement_ids:
            pattern = _METRICS.get(candidates[mid]["metric"])
            if pattern is None or not re.search(pattern, scope, re.I):
                raise ValueError(
                    "Condition does not explicitly apply to this measurement type"
                )
    if binding.applicability == "reported_common_protocol":
        # Explicit 'all samples' plus local protocol, not unrelated same-paper
        # characterization paragraphs or earlier experiments.
        if binding.value_source.chunk_id != binding.applicability_source.chunk_id:
            raise ValueError("Shared attribute crossed a local protocol")
        return binding
    for mid in binding.measurement_ids:
        label = candidates[mid]["group_label"]
        scope = binding.applicability_source.quote
        if sum(_contains_label(scope, s) for s in labels) > 1:
            expected = _ordered_value(scope, label, labels)
            if expected is None or re.sub(r"\s+", "", expected) != re.sub(
                r"\s+", "", binding.value_text
            ):
                raise ValueError(
                    "Multi-sample condition lacks the exact ordered value binding"
                )
        elif not (
            _contains_label(binding.value_source.quote, label)
            or (binding.value_text in scope and _contains_label(scope, label))
        ):
            raise ValueError("Value and identity citations do not prove applicability")
    return binding


def ordered_preparation_plan(candidates, chunks, *, samples):
    """Only explicit, ordered dopant lists. No title/label inference or unit repair."""
    from .fulltext_conditions import AttributeBinding

    bindings = []
    labels = [s.label for s in samples.values() if s.identity_status != "ambiguous"]
    for sample in samples.values():
        if sample.identity_status != "defined":
            continue
        mids = tuple(
            mid for mid, c in candidates.items() if c["group_label"] == sample.label
        )
        if not mids:
            continue
        value = _ordered_value(sample.definition.quote, sample.label, labels)
        if value is None:
            continue
        binding = AttributeBinding(
            measurement_ids=mids,
            kind="preparation",
            key="dopant_concentration",
            value_text=value,
            value_source=sample.definition,
            applicability="per_sample",
            applicability_source=sample.definition,
        )
        validate_staged_attribute(binding, candidates, chunks, samples=samples)
        bindings.append(binding)
    return ConditionPlan(bindings=tuple(bindings))
