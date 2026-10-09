import pytest

from materials_screening.master.fulltext_conditions import AttributeBinding
from materials_screening.master.staged_fulltext_conditions import (
    restore_literal,
    validate_staged_attribute,
)
from materials_screening.master.staged_fulltext_evidence import (
    SampleProposal,
    build_catalogue,
    checked_sample,
)
from tests.unit.master.test_staged_fulltext_evidence import source


def fixture():
    chunk = source(
        "The pure and Nb-doped SnO2 films of 0.5 wt %, 1 wt %, 1.5 wt %, "
        "2 wt % and 2.5 wt % are referred to as TO, NTO1, NTO2, NTO3, NTO4, and NTO5 "
        "with respect of Nb concentration."
    )
    cat = build_catalogue((chunk,))
    samples = {
        s.sample_id: s
        for s in (
            checked_sample(SampleProposal(label=label, source_id="s0"), cat)
            for label in ("TO", "NTO1", "NTO2", "NTO3", "NTO4", "NTO5")
        )
    }
    candidates = {
        "m": dict(
            document_id="doc-1", group_label="NTO3", metric="transmittance", unit="%"
        )
    }
    return (chunk,), cat["s0"], samples, candidates


def test_ordered_definition_binds_nto3_not_an_arbitrary_list_value():
    chunks, span, samples, candidates = fixture()
    binding = AttributeBinding(
        measurement_ids=("m",),
        kind="preparation",
        key="doping_concentration",
        value_text="1.5 wt %",
        value_source=span,
        applicability="per_sample",
        applicability_source=span,
    )
    validate_staged_attribute(binding, candidates, chunks, samples=samples)
    with pytest.raises(ValueError):
        validate_staged_attribute(
            binding.model_copy(update={"value_text": "0.5 wt %"}),
            candidates,
            chunks,
            samples=samples,
        )


def test_unrelated_value_citation_cannot_be_linked_to_an_identity_only_quote():
    chunks, span, samples, candidates = fixture()
    other = source("NTO4 was annealed at 450 C.", "other")
    value_span = build_catalogue((other,))["s0"]
    binding = AttributeBinding(
        measurement_ids=("m",),
        kind="preparation",
        key="annealing_temperature",
        value_text="450 C",
        value_source=value_span,
        applicability="per_sample",
        applicability_source=span,
    )
    with pytest.raises(ValueError):
        validate_staged_attribute(
            binding, candidates, (*chunks, other), samples=samples
        )


def test_whitespace_restoration_is_literal_not_fuzzy_or_exponent_repair():
    assert restore_literal("1.5 wt%", "Nb concentration 1.5 wt %.") == "1.5 wt %"
    assert restore_literal("around 82%", "around\nof 82%") is None
    assert restore_literal("6.45e-4", "6.45 \x02 10 4") is None


def test_common_conditions_need_explicit_scope_not_same_document():
    chunks, span, samples, candidates = fixture()
    other = source("The measurements were at 25 C.", "other")
    value_span = build_catalogue((other,))["s0"]
    binding = AttributeBinding(
        measurement_ids=("m",),
        kind="condition",
        key="temperature",
        value_text="25 C",
        value_source=value_span,
        applicability="reported_common_protocol",
        applicability_source=span,
    )
    with pytest.raises(ValueError):
        validate_staged_attribute(
            binding, candidates, (*chunks, other), samples=samples
        )


def test_performance_cannot_be_added_as_a_test_condition():
    chunks, span, samples, candidates = fixture()
    c = source("NTO3 transmittance was 82%.", "optics")
    s = build_catalogue((c,))["s0"]
    b = AttributeBinding(
        measurement_ids=("m",),
        kind="condition",
        key="transmittance",
        value_text="82%",
        value_source=s,
        applicability="per_sample",
        applicability_source=s,
    )
    with pytest.raises(ValueError):
        validate_staged_attribute(b, candidates, (*chunks, c), samples=samples)


def test_prior_comparison_condition_cannot_attach_to_current_sample():
    c = source(
        "A transmittance of 82% was obtained for NTO3 which is comparable with "
        "previous reports of NTO3 measured at 550 nm.",
        "comparison",
    )
    span = build_catalogue((c,))["s0"]
    sample = checked_sample(SampleProposal(label="NTO3", source_id="s0"), {"s0": span})
    binding = AttributeBinding(
        measurement_ids=("m",),
        kind="condition",
        key="wavelength",
        value_text="550 nm",
        numeric_value=550,
        unit="nm",
        value_source=span,
        applicability="per_sample",
        applicability_source=span,
    )
    candidates = {
        "m": dict(
            document_id="doc-1", group_label="NTO3", metric="transmittance", unit="%"
        )
    }
    with pytest.raises(ValueError, match="comparison"):
        validate_staged_attribute(
            binding, candidates, (c,), samples={sample.sample_id: sample}
        )


@pytest.mark.parametrize(
    "key",
    [
        "sample_label",
        "sample label",
        "sample_code",
        "group_label",
        "sample_type",
        "Sample Type",
        "sample-type",
        "specimen_type",
        "group_type",
    ],
)
@pytest.mark.parametrize("kind", ["condition", "preparation"])
def test_sample_identity_is_not_an_experimental_test_condition(key, kind):
    c = source("NTO3 transmittance was 82%.", "identity-as-condition")
    span = build_catalogue((c,))["s0"]
    sample = checked_sample(SampleProposal(label="NTO3", source_id="s0"), {"s0": span})
    binding = AttributeBinding(
        measurement_ids=("m",),
        kind=kind,
        key=key,
        value_text="NTO3",
        value_source=span,
        applicability="per_sample",
        applicability_source=span,
    )
    candidates = {
        "m": dict(
            document_id="doc-1", group_label="NTO3", metric="transmittance", unit="%"
        )
    }
    with pytest.raises(ValueError, match="identity"):
        validate_staged_attribute(
            binding, candidates, (c,), samples={sample.sample_id: sample}
        )


@pytest.mark.parametrize(
    "key,value,numeric,unit",
    [
        ("sample_thickness", "150 nm", 150, "nm"),
        ("sample_temperature", "25 C", 25, "C"),
    ],
)
def test_physical_sample_conditions_are_not_discarded_as_identity(
    key, value, numeric, unit
):
    physical_name = key.replace("_", " ")
    c = source(
        f"NTO3 transmittance was measured with {physical_name} of {value}.",
        "physical-condition",
    )
    span = build_catalogue((c,))["s0"]
    sample = checked_sample(SampleProposal(label="NTO3", source_id="s0"), {"s0": span})
    binding = AttributeBinding(
        measurement_ids=("m",),
        kind="condition",
        key=key,
        value_text=value,
        numeric_value=numeric,
        unit=unit,
        value_source=span,
        applicability="per_sample",
        applicability_source=span,
    )
    candidates = {
        "m": dict(
            document_id="doc-1", group_label="NTO3", metric="transmittance", unit="%"
        )
    }
    validate_staged_attribute(
        binding, candidates, (c,), samples={sample.sample_id: sample}
    )


def test_thermal_resistance_temperature_cannot_attach_to_optical_measurement():
    chunks, span, samples, candidates = fixture()
    c = source("NTO3 sheet resistance was measured at 450 C.", "thermal")
    s = build_catalogue((c,))["s0"]
    b = AttributeBinding(
        measurement_ids=("m",),
        kind="condition",
        key="temperature",
        value_text="450 C",
        value_source=s,
        applicability="per_sample",
        applicability_source=s,
    )
    with pytest.raises(ValueError):
        validate_staged_attribute(b, candidates, (*chunks, c), samples=samples)


@pytest.mark.parametrize(
    "key",
    [
        "transmittance_wavelength",
        "measurement_wavelength",
        "optical_wavelength",
        "wavelength",
    ],
)
def test_wavelength_is_a_condition_not_a_transmittance_performance(key):
    c = source("The SFTO film showed a transmittance of 74.3% at 550 nm.", "sf-optics")
    span = build_catalogue((c,))["s0"]
    sample = checked_sample(SampleProposal(label="SFTO", source_id="s0"), {"s0": span})
    binding = AttributeBinding(
        measurement_ids=("m",),
        kind="condition",
        key=key,
        value_text="550 nm",
        numeric_value=550,
        unit="nm",
        value_source=span,
        applicability="per_sample",
        applicability_source=span,
    )
    candidates = {
        "m": dict(
            document_id="doc-1", group_label="SFTO", metric="transmittance", unit="%"
        )
    }
    validate_staged_attribute(
        binding, candidates, (c,), samples={sample.sample_id: sample}
    )


@pytest.mark.parametrize(
    "damage",
    [
        "performance_value",
        "wrong_sample",
        "wrong_metric",
        "wrong_unit",
        "preparation_kind",
        "invented_value",
        "unknown_target",
    ],
)
def test_wavelength_name_cannot_bypass_value_source_or_measurement_scope(damage):
    c = source("The SFTO film showed a transmittance of 74.3% at 550 nm.", "sf-optics")
    span = build_catalogue((c,))["s0"]
    sample = checked_sample(SampleProposal(label="SFTO", source_id="s0"), {"s0": span})
    binding = AttributeBinding(
        measurement_ids=("m",),
        kind="condition",
        key="transmittance_wavelength",
        value_text="550 nm",
        numeric_value=550,
        unit="nm",
        value_source=span,
        applicability="per_sample",
        applicability_source=span,
    )
    candidates = {
        "m": dict(
            document_id="doc-1", group_label="SFTO", metric="transmittance", unit="%"
        )
    }
    if damage == "performance_value":
        binding = binding.model_copy(
            update=dict(value_text="74.3%", numeric_value=74.3, unit="%")
        )
    elif damage == "wrong_sample":
        candidates["m"]["group_label"] = "OTHER"
    elif damage == "wrong_metric":
        candidates["m"]["metric"] = "resistivity"
    elif damage == "wrong_unit":
        binding = binding.model_copy(update=dict(unit="%"))
    elif damage == "preparation_kind":
        binding = binding.model_copy(update=dict(kind="preparation"))
    elif damage == "unknown_target":
        binding = binding.model_copy(update={"measurement_ids": ("unknown",)})
    else:
        binding = binding.model_copy(
            update=dict(value_text="600 nm", numeric_value=600)
        )
    with pytest.raises(ValueError):
        validate_staged_attribute(
            binding, candidates, (c,), samples={sample.sample_id: sample}
        )


def test_wavelength_range_preserves_range_and_cannot_be_scalarized():
    c = source("The SFTO transmittance was measured from 190 nm to 1100 nm.", "range")
    span = build_catalogue((c,))["s0"]
    sample = checked_sample(SampleProposal(label="SFTO", source_id="s0"), {"s0": span})
    binding = AttributeBinding(
        measurement_ids=("m",),
        kind="condition",
        key="transmittance_wavelength_range",
        value_text="190 nm to 1100 nm",
        unit="nm",
        value_source=span,
        applicability="per_sample",
        applicability_source=span,
    )
    candidates = {
        "m": dict(
            document_id="doc-1", group_label="SFTO", metric="transmittance", unit="%"
        )
    }
    validate_staged_attribute(
        binding, candidates, (c,), samples={sample.sample_id: sample}
    )
    with pytest.raises(ValueError):
        validate_staged_attribute(
            binding.model_copy(update={"numeric_value": 190}),
            candidates,
            (c,),
            samples={sample.sample_id: sample},
        )
