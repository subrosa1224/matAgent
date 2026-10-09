"""New composite-source path, intentionally independent of legacy matrices."""

import hashlib
from dataclasses import replace

import pytest

from materials_screening.master.staged_fulltext_evidence import (
    MetricProposal,
    SampleProposal,
    build_catalogue,
    checked_metric,
    checked_sample,
    reported_scalar,
    validate_composite,
)
from tests.unit.sub_agents.test_literature_matrix_automation import _chunk


def source(text, key="definition"):
    return replace(
        _chunk(),
        chunk_id=key,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def example():
    chunks = (
        source("Sample NTO3 is the Nb-doped SnO2 film with 1.5 wt% Nb."),
        source(
            "The maximum average transmittance of around 82% was observed "
            "in NTO3 film at UV-visible region.",
            "optics",
        ),
    )
    cat = build_catalogue(chunks)
    sample = checked_sample(SampleProposal(label="NTO3", source_id="s0"), cat)
    metric = checked_metric(
        MetricProposal(
            sample_id=sample.sample_id,
            metric="transmittance",
            value_text="82",
            unit="%",
            source_id="s1",
        ),
        {sample.sample_id: sample},
        cat,
    )
    return chunks, cat, sample, metric


def test_cross_paragraph_sample_definition_retains_approximate_metric():
    chunks, _, sample, metric = example()
    assert sample.sample_id == metric.sample_id
    assert sample.definition.chunk_id != metric.source.chunk_id
    assert metric.value_text == "around 82"
    assert metric.qualifier == "approximate"
    assert "UV-visible" in metric.source.quote
    validate_composite(sample, metric, chunks)


def test_literal_unit_suffix_is_removed_without_losing_source_approximation():
    chunks, cat, sample, expected = example()
    actual = checked_metric(
        MetricProposal(
            sample_id=sample.sample_id,
            metric="transmittance",
            value_text="82%",
            unit="%",
            source_id="s1",
        ),
        {sample.sample_id: sample},
        cat,
    )
    assert actual == expected
    validate_composite(sample, actual, chunks)


def test_unknown_source_or_invented_sample_rejected():
    _, cat, sample, _ = example()
    for label, key in (("NTO4", "s0"), ("NTO3", "s99")):
        with pytest.raises(ValueError):
            checked_sample(SampleProposal(label=label, source_id=key), cat)
    with pytest.raises(ValueError):
        checked_metric(
            MetricProposal(
                sample_id="sample-" + "0" * 24,
                metric="transmittance",
                value_text="82",
                unit="%",
                source_id="s1",
            ),
            {sample.sample_id: sample},
            cat,
        )


@pytest.mark.parametrize(
    "label", ["NTO thin films", "NTO2, NTO3", "Nb-doped SnO2 film"]
)
def test_generic_family_and_joined_labels_are_not_specific_sample_ids(label):
    cat = build_catalogue((source(label + " were investigated."),))
    with pytest.raises(ValueError):
        checked_sample(SampleProposal(label=label, source_id="s0"), cat)


@pytest.mark.parametrize("label", ["pure SnO2", "5 at % Ga doped SnO2"])
def test_single_literal_specimen_description_survives_pdf_whitespace(label):
    quote = label.replace(" ", " \n") + " film showed transmittance of 74%."
    chunks = (source(quote),)
    cat = build_catalogue(chunks)
    sample = checked_sample(
        SampleProposal(label=label, source_id="s0"),
        cat,
        chunks=chunks,
        document_id=chunks[0].document_id,
    )
    assert sample.label == label and sample.definition.quote == quote
    measurement = checked_metric(
        MetricProposal(
            sample_id=sample.sample_id,
            metric="transmittance",
            value_text="74",
            unit="%",
            source_id="s0",
        ),
        {sample.sample_id: sample},
        cat,
        chunks=chunks,
    )
    validate_composite(sample, measurement, chunks)


def test_pdf_whitespace_does_not_create_different_sample_ids():
    cat = build_catalogue((source("Sample A is a specimen. Sample\nA was tested."),))
    a = checked_sample(SampleProposal(label="Sample A", source_id="s0"), cat)
    b = checked_sample(SampleProposal(label="Sample\nA", source_id="s0"), cat)
    assert a.sample_id == b.sample_id and b.label == "Sample A"


def test_material_formula_alone_is_not_a_specific_experimental_sample():
    cat = build_catalogue((source("SnO2 was studied with multiple dopants."),))
    with pytest.raises(ValueError, match="family"):
        checked_sample(SampleProposal(label="SnO2", source_id="s0"), cat)
    named = build_catalogue(
        (source("Sample SnO2 is the control, explicitly named SnO2."),)
    )
    assert checked_sample(SampleProposal(label="SnO2", source_id="s0"), named)


def test_concentration_list_cannot_create_unwritten_single_sample_label():
    cat = build_catalogue(
        (
            source(
                "Different concentrations (1, 3, and 5 at %) of Ga doped "
                "SnO2 films were prepared."
            ),
        )
    )
    with pytest.raises(ValueError):
        checked_sample(
            SampleProposal(label="5 at % Ga doped SnO2", source_id="s0"), cat
        )


def test_metric_cannot_take_another_samples_value():
    _, cat, _, _ = example()
    wrong = source("Sample NTO4 is another film.", "wrong")
    wrong_cat = build_catalogue((wrong,))
    sample = checked_sample(SampleProposal(label="NTO4", source_id="s0"), wrong_cat)
    with pytest.raises(ValueError):
        checked_metric(
            MetricProposal(
                sample_id=sample.sample_id,
                metric="transmittance",
                value_text="82",
                unit="%",
                source_id="s1",
            ),
            {sample.sample_id: sample},
            cat,
        )


@pytest.mark.parametrize("value", ["estimated values", "not reported", "6.45e-4"])
def test_missing_text_and_damaged_exponent_not_recovered_by_guessing(value):
    c = source("NTO3 resistivity was 6.45 \x02 10 4 Ω cm.")
    cat = build_catalogue((c,))
    sample = checked_sample(SampleProposal(label="NTO3", source_id="s0"), cat)
    with pytest.raises(ValueError):
        checked_metric(
            MetricProposal(
                sample_id=sample.sample_id,
                metric="resistivity",
                value_text=value,
                unit="Ω cm",
                source_id="s0",
            ),
            {sample.sample_id: sample},
            cat,
        )


def test_same_page_number_does_not_validate_flattened_table():
    c = source("Table 1\nParameter\nNTO3\nNTO4\nResistivity (Ω cm)\n6.45e-4\n9.0e-4")
    cat = build_catalogue((c,))
    sample = checked_sample(SampleProposal(label="NTO3", source_id="s0"), cat)
    with pytest.raises(ValueError):
        checked_metric(
            MetricProposal(
                sample_id=sample.sample_id,
                metric="resistivity",
                value_text="6.45e-4",
                unit="Ω cm",
                source_id="s0",
            ),
            {sample.sample_id: sample},
            cat,
        )


def test_range_is_retained_but_never_midpoint_scalar():
    c = source("NTO3 transmittance was 80-82% in the visible region.")
    cat = build_catalogue((c,))
    sample = checked_sample(SampleProposal(label="NTO3", source_id="s0"), cat)
    metric = checked_metric(
        MetricProposal(
            sample_id=sample.sample_id,
            metric="transmittance",
            value_text="80-82",
            unit="%",
            source_id="s0",
        ),
        {sample.sample_id: sample},
        cat,
    )
    assert metric.qualifier == "range"
    with pytest.raises(ValueError):
        reported_scalar(metric.value_text)


@pytest.mark.parametrize(
    "value,expected",
    [("6.45e-4", 0.000645), ("6.45 × 10^-4", 0.000645), ("around 82", 82)],
)
def test_literal_scalar_scientific_notation(value, expected):
    assert reported_scalar(value) == expected


def test_body_is_not_silently_reduced_to_a_24k_window():
    chunks = tuple(
        source("NTO3 transmittance was 82%. " * 150, f"c{i}") for i in range(9)
    )
    cat = build_catalogue(chunks)
    assert {span.chunk_id for span in cat.values()} == {c.chunk_id for c in chunks}
    assert all(len(span.quote) <= 3000 for span in cat.values())


def test_oversized_single_sentence_is_retained_for_explicit_blocking():
    cat = build_catalogue((source("NTO3 " + "x" * 3100),))
    assert len(cat["s0"].quote) == 3105


def test_dopant_percentage_is_not_the_transmittance_in_the_same_sentence():
    cat = build_catalogue((source("NTO3 with 1.5% Nb showed transmittance of 82%."),))
    s = checked_sample(SampleProposal(label="NTO3", source_id="s0"), cat)
    with pytest.raises(ValueError):
        checked_metric(
            MetricProposal(
                sample_id=s.sample_id,
                metric="transmittance",
                value_text="1.5",
                unit="%",
                source_id="s0",
            ),
            {s.sample_id: s},
            cat,
        )


def test_responsivity_peak_wavelength_is_not_responsivity():
    cat = build_catalogue((source("NTO3 responsivity peaked at wavelength 370 nm."),))
    s = checked_sample(SampleProposal(label="NTO3", source_id="s0"), cat)
    with pytest.raises(ValueError):
        checked_metric(
            MetricProposal(
                sample_id=s.sample_id,
                metric="responsivity",
                value_text="370",
                unit="nm",
                source_id="s0",
            ),
            {s.sample_id: s},
            cat,
        )


def test_unit_prefix_case_is_not_silently_changed():
    cat = build_catalogue((source("NTO3 responsivity was 12 MA/W."),))
    s = checked_sample(SampleProposal(label="NTO3", source_id="s0"), cat)
    with pytest.raises(ValueError):
        checked_metric(
            MetricProposal(
                sample_id=s.sample_id,
                metric="responsivity",
                value_text="12",
                unit="mA/W",
                source_id="s0",
            ),
            {s.sample_id: s},
            cat,
        )


@pytest.mark.parametrize(
    "value,expected",
    [("16.130 × 10−4", 0.001613), ("4.78 x 10-3", 0.00478)],
)
def test_explicit_signed_exponent_without_caret_is_literal_not_a_repair(
    value, expected
):
    text = f"NTO3 resistivity was {value} Ω cm."
    cat = build_catalogue((source(text),))
    sample = checked_sample(SampleProposal(label="NTO3", source_id="s0"), cat)
    measurement = checked_metric(
        MetricProposal(
            sample_id=sample.sample_id,
            metric="resistivity",
            value_text=value,
            unit="Ω cm",
            source_id="s0",
        ),
        {sample.sample_id: sample},
        cat,
    )
    assert measurement.value_text == value
    assert reported_scalar(measurement.value_text) == pytest.approx(expected)


def test_word_bound_preserved_and_not_sent_to_scalar_statistics():
    cat = build_catalogue((source("NTO3 transmittance was more than 85%."),))
    sample = checked_sample(SampleProposal(label="NTO3", source_id="s0"), cat)
    measurement = checked_metric(
        MetricProposal(
            sample_id=sample.sample_id,
            metric="transmittance",
            value_text="more than 85%",
            unit="%",
            source_id="s0",
        ),
        {sample.sample_id: sample},
        cat,
    )
    assert measurement.qualifier == "bound"
    assert measurement.value_text == "more than 85"
    with pytest.raises(ValueError):
        reported_scalar(measurement.value_text)


def test_unsigned_concatenated_104_is_not_reinterpreted_as_exponent():
    with pytest.raises(ValueError):
        reported_scalar("6.45 x 104")


_MIXED_CONTROL_RESULT = (
    "The lowest conductivity and highest resistivity (ρ) of 619.960 (Ω cm)−1 "
    "and 16.130 × 10−4 Ω cm, respectively were obtained for undoped SnO2 film "
    "which are comparable with the values found in previous reports [13,65] "
    "for spray deposited pure SnO2 film."
)


def test_prior_comparison_label_cannot_receive_current_paper_value():
    chunks = (
        source("The pure SnO2 film was investigated."),
        source(_MIXED_CONTROL_RESULT, "result"),
    )
    cat = build_catalogue(chunks)
    sample = checked_sample(SampleProposal(label="pure SnO2", source_id="s0"), cat)
    with pytest.raises(ValueError, match="sample"):
        checked_metric(
            MetricProposal(
                sample_id=sample.sample_id,
                metric="resistivity",
                value_text="16.130 × 10−4",
                unit="Ω cm",
                source_id="s1",
            ),
            {sample.sample_id: sample},
            cat,
            chunks=chunks,
        )


@pytest.mark.parametrize("comparison", ["comparable", "compara\nble"])
def test_prior_comparison_cannot_define_current_sample_or_infer_alias(comparison):
    quote = _MIXED_CONTROL_RESULT.replace("comparable", comparison)
    cat = build_catalogue((source(quote),))
    with pytest.raises(ValueError, match="sample"):
        checked_sample(SampleProposal(label="pure SnO2", source_id="s0"), cat)
    sample = checked_sample(SampleProposal(label="undoped SnO2", source_id="s0"), cat)
    measurement = checked_metric(
        MetricProposal(
            sample_id=sample.sample_id,
            metric="resistivity",
            value_text="16.130 × 10−4",
            unit="Ω cm",
            source_id="s0",
        ),
        {sample.sample_id: sample},
        cat,
    )
    assert measurement.source.quote == quote
    assert reported_scalar(measurement.value_text) == pytest.approx(0.001613)


def test_previous_report_alone_is_not_a_current_sample():
    chunks = (
        source(
            "Previous reports [13] found pure SnO2 resistivity of 16.130 × 10−4 Ω cm."
        ),
    )
    with pytest.raises(ValueError):
        checked_sample(
            SampleProposal(label="pure SnO2", source_id="s0"),
            build_catalogue(chunks),
            chunks=chunks,
            document_id=chunks[0].document_id,
        )


def test_tampered_or_foreign_source_invalidates_cached_composite():
    chunks, _, sample, metric = example()
    damaged = replace(chunks[1], text=chunks[1].text.replace("82", "99"))
    with pytest.raises(ValueError):
        validate_composite(sample, metric, (chunks[0], damaged))
    with pytest.raises(ValueError):
        validate_composite(
            sample,
            metric,
            (chunks[0], replace(chunks[1], document_id="doc-" + "f" * 24)),
        )


_GA_CONTRAST = (
    "In the visible \nlight region, the average transmittance of pure SnO2 film "
    "found to be above 85%, whereas Ga \ndoped SnO2 films were found to be a "
    "decrease of transmittance up to 74 % in 5 at % Ga doped \nSnO2 film."
)


def contrast_example(quote=_GA_CONTRAST, labels=("pure SnO2", "5 at % Ga doped SnO2")):
    chunks = (source(quote),)
    catalogue = build_catalogue(chunks)
    samples = tuple(
        checked_sample(
            SampleProposal(label=label, source_id="s0"),
            catalogue,
            document_id=chunks[0].document_id,
            chunks=chunks,
        )
        for label in labels
    )
    return chunks, catalogue, {s.sample_id: s for s in samples}, samples


@pytest.mark.parametrize("connector", ["whereas", "while"])
@pytest.mark.parametrize(
    "index,value,expected,qualifier",
    [
        (0, "85", "above 85", "bound"),
        (1, "74", "up to 74", "bound"),
    ],
)
def test_explicit_contrast_clauses_bind_each_samples_own_value(
    connector, index, value, expected, qualifier
):
    quote = _GA_CONTRAST.replace("whereas", connector)
    chunks, cat, samples, ordered = contrast_example(quote)
    sample = ordered[index]
    measurement = checked_metric(
        MetricProposal(
            sample_id=sample.sample_id,
            metric="transmittance",
            value_text=value,
            unit="%",
            source_id="s0",
        ),
        samples,
        cat,
        chunks=chunks,
    )
    assert measurement.value_text == expected
    assert measurement.qualifier == qualifier
    assert measurement.source.quote == quote  # No cropped or invented citation.
    validate_composite(sample, measurement, chunks, samples=samples)
    if qualifier == "bound":
        with pytest.raises(ValueError):
            reported_scalar(measurement.value_text)


@pytest.mark.parametrize("index,value", [(0, "74"), (1, "85")])
def test_explicit_contrast_rejects_swapped_values(index, value):
    chunks, cat, samples, ordered = contrast_example()
    with pytest.raises(ValueError):
        checked_metric(
            MetricProposal(
                sample_id=ordered[index].sample_id,
                metric="transmittance",
                value_text=value,
                unit="%",
                source_id="s0",
            ),
            samples,
            cat,
            chunks=chunks,
        )


@pytest.mark.parametrize("value", ["74", "85"])
def test_contrast_cannot_pass_by_removing_competing_sample_from_inventory(value):
    chunks, cat, _, ordered = contrast_example()
    sample = ordered[0]
    with pytest.raises(ValueError):
        checked_metric(
            MetricProposal(
                sample_id=sample.sample_id,
                metric="transmittance",
                value_text=value,
                unit="%",
                source_id="s0",
            ),
            {sample.sample_id: sample},
            cat,
            chunks=chunks,
        )


@pytest.mark.parametrize(
    "quote",
    [
        "Sample A and Sample B showed transmittance of 85% and 74%, respectively.",
        "Sample A transmittance was 85%, whereas Sample B was 74%.",
        "Sample A transmittance was 85%, whereas Sample B and Sample A "
        "had transmittance of 74%.",
        "Sample A transmittance was 85%, whereas that of Sample B was 74%.",
    ],
)
def test_ambiguous_or_implicit_parallel_metrics_remain_blocked(quote):
    chunks, cat, samples, ordered = contrast_example(quote, ("Sample A", "Sample B"))
    with pytest.raises(ValueError):
        checked_metric(
            MetricProposal(
                sample_id=ordered[1].sample_id,
                metric="transmittance",
                value_text="74",
                unit="%",
                source_id="s0",
            ),
            samples,
            cat,
            chunks=chunks,
        )


@pytest.mark.parametrize("word", ["above", "below", "up to"])
def test_literal_word_bounds_are_not_exact_scalars(word):
    chunks, cat, samples, ordered = contrast_example(
        f"Sample A transmittance was {word} 85%.", ("Sample A",)
    )
    measurement = checked_metric(
        MetricProposal(
            sample_id=ordered[0].sample_id,
            metric="transmittance",
            value_text="85",
            unit="%",
            source_id="s0",
        ),
        samples,
        cat,
        chunks=chunks,
    )
    assert measurement.value_text == word + " 85" and measurement.qualifier == "bound"
    with pytest.raises(ValueError):
        reported_scalar(measurement.value_text)


@pytest.mark.parametrize(
    "quote,value,unit",
    [
        (
            "Sample A resistivity was 6.45 × 104 Ω cm, whereas Sample B "
            "resistivity was 9.0e-4 Ω cm.",
            "6.45e-4",
            "Ω cm",
        ),
        (
            "Sample A resistivity was 6.45 × 10−4 a·cm, whereas Sample B "
            "resistivity was 9.0e-4 a·cm.",
            "6.45 × 10−4",
            "Ω cm",
        ),
        (
            "Sample A resistivity was 6.45 × 10−4 a·cm, whereas Sample B "
            "resistivity was 9.0e-4 a·cm.",
            "6.45 × 10−4",
            "a·cm",
        ),
    ],
)
def test_contrast_does_not_repair_missing_exponent_or_unit(quote, value, unit):
    chunks, cat, samples, ordered = contrast_example(quote, ("Sample A", "Sample B"))
    with pytest.raises(ValueError):
        checked_metric(
            MetricProposal(
                sample_id=ordered[0].sample_id,
                metric="resistivity",
                value_text=value,
                unit=unit,
                source_id="s0",
            ),
            samples,
            cat,
            chunks=chunks,
        )


def test_source_replay_rejects_lost_contrast_inventory():
    chunks, cat, samples, ordered = contrast_example()
    sample = ordered[0]
    measurement = checked_metric(
        MetricProposal(
            sample_id=sample.sample_id,
            metric="transmittance",
            value_text="85",
            unit="%",
            source_id="s0",
        ),
        samples,
        cat,
        chunks=chunks,
    )
    with pytest.raises(ValueError):
        validate_composite(
            sample, measurement, chunks, samples={sample.sample_id: sample}
        )
