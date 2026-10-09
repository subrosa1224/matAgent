"""Evidence completion must not infer common times or physical mass fractions."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from materials_screening.master.fulltext_targeted_evidence import (
    SampleSources,
    TargetedProposal,
    request_targeted_evidence,
    validate_targeted_evidence,
)


def environment():
    text = (
        "2. Preparation of MVO4/g-C3N4 (M = Bi, Ce) composites\n"
        "Lastly, 7%, 14% MVO4/g-C3N4 composites were prepared by adjusting "
        "the mass ratios of g-C3N4 and MVO4.\n"
        "Results and Discussion\n"
        "The degradation of BiVO4 under visible light for 4 h reached 45%. "
        "The degradation of 7% BiVO4/g-C3N4 after irradiation for 2 h reached 60%. "
        "The samples were stirred for 1 h for adsorption equilibrium.\n"
        "Table 1. Results\nSample Name\n7% BiVO4/g-C3N4\n60%\n"
        "14% BiVO4/g-C3N4\n70%\n"
    )
    chunk = SimpleNamespace(
        chunk_id="chunk-targeted",
        document_id="doc-targeted",
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        page_from=4,
        page_to=4,
    )
    rows = {
        "mid-control": dict(
            document_id=chunk.document_id,
            group_label="BiVO4",
            metric="Degradation",
            unit="%",
        ),
        "mid-seven": dict(
            document_id=chunk.document_id,
            group_label="7% BiVO4/g-C3N4",
            metric="Degradation",
            unit="%",
        ),
        "mid-fourteen": dict(
            document_id=chunk.document_id,
            group_label="14% BiVO4/g-C3N4",
            metric="Degradation",
            unit="%",
        ),
    }
    return rows, (chunk,)


class Calls:
    def __init__(self, *, bad_time=False, missing_basis=False):
        self.calls = 0
        self.bad_time, self.missing_basis = bad_time, missing_basis

    def generate_structured(self, **kwargs):
        self.calls += 1
        assert kwargs["max_output_tokens"] == 4096
        data = json.loads(kwargs["user_text"])
        evidence = data["evidence"]

        def source(needle, kind):
            return next(
                key
                for key, row in evidence.items()
                if needle in row["quote"] and row["kind"] == kind
            )

        control = source("BiVO4 under visible light", "time")
        seven = source("after irradiation for 2 h", "time")
        definition = source("mass ratios", "preparation")
        context = source("M = Bi, Ce", "preparation_context")
        return SimpleNamespace(
            parsed=TargetedProposal(
                samples={
                    "m0": SampleSources(time_source=control),
                    "m1": SampleSources(
                        sample_source=source("14% BiVO4", "sample"),
                        preparation_source=definition,
                        preparation_context=context,
                        time_source=control if self.bad_time else None,
                    ),
                    "m2": SampleSources(
                        sample_source=source("7% BiVO4", "sample"),
                        preparation_source=None if self.missing_basis else definition,
                        preparation_context=context,
                        time_source=seven,
                    ),
                }
            )
        )


def test_targeted_sources_bind_distinct_times_and_reported_percentages():
    candidates, chunks = environment()
    calls = Calls()
    plan = request_targeted_evidence(calls=calls, candidates=candidates, chunks=chunks)
    assert calls.calls == 1 and plan.review_status == "pending"
    values = {
        (b.measurement_ids[0], b.key): b.numeric_value for b in plan.attributes.bindings
    }
    assert values == {
        ("mid-control", "irradiation_time"): 4,
        ("mid-seven", "irradiation_time"): 2,
        ("mid-seven", "reported_loading_percent"): 7,
        ("mid-fourteen", "reported_loading_percent"): 14,
    }
    assert "mid-fourteen" in plan.preparation_sources
    assert all(b.applicability == "per_sample" for b in plan.attributes.bindings)
    assert any("质量分数" in note for note in plan.attributes.unresolved)


def test_control_time_cannot_spread_to_composite_and_bad_field_is_isolated():
    candidates, chunks = environment()
    plan = request_targeted_evidence(
        calls=Calls(bad_time=True), candidates=candidates, chunks=chunks
    )
    assert len(plan.attributes.bindings) == 4
    assert any(b.key == "irradiation_time" for b in plan.attributes.isolated_attributes)
    assert not any(
        b.measurement_ids == ("mid-fourteen",) and b.key == "irradiation_time"
        for b in plan.attributes.bindings
    )


def test_percentage_label_without_preparation_definition_does_not_bind():
    candidates, chunks = environment()
    plan = request_targeted_evidence(
        calls=Calls(missing_basis=True), candidates=candidates, chunks=chunks
    )
    assert not any(
        b.measurement_ids == ("mid-seven",) and b.kind == "preparation"
        for b in plan.attributes.bindings
    )
    assert any(
        b.key == "reported_loading_percent" for b in plan.attributes.isolated_attributes
    )


@pytest.mark.parametrize("change", ["hash", "time", "definition"])
def test_replay_revalidates_sources_and_semantics(change):
    candidates, chunks = environment()
    plan = request_targeted_evidence(
        calls=Calls(), candidates=candidates, chunks=chunks
    )
    if change == "hash":
        chunks[0].text += " changed"
    elif change == "time":
        binding = plan.attributes.bindings[0].model_copy(update={"numeric_value": 999})
        plan = plan.model_copy(
            update={
                "attributes": plan.attributes.model_copy(
                    update={"bindings": (binding, *plan.attributes.bindings[1:])}
                )
            }
        )
    else:
        plan = plan.model_copy(update={"preparation_sources": {}})
    with pytest.raises(ValueError):
        validate_targeted_evidence(plan, candidates, chunks)


def test_empty_or_irrelevant_sources_do_not_call_model():
    candidates, chunks = environment()
    chunks[0].text = "XRD characterization found a peak at 20 degrees."
    chunks[0].text_sha256 = hashlib.sha256(chunks[0].text.encode()).hexdigest()
    calls = Calls()
    plan = request_targeted_evidence(calls=calls, candidates=candidates, chunks=chunks)
    assert calls.calls == 0 and not plan.attributes.bindings


def test_schema_rejects_generated_values_as_source_ids():
    with pytest.raises(ValueError):
        SampleSources(time_source="3 h")
    with pytest.raises(ValueError):
        TargetedProposal(samples={"invented": SampleSources()})


def test_pure_material_name_inside_composite_is_not_explicit_control_scope():
    from materials_screening.master.fulltext_targeted_evidence import _explicit_label

    assert not _explicit_label("BiVO4", "7% BiVO4/g-C3N4 after irradiation for 2 h")
    assert _explicit_label("BiVO4", "BiVO4 under visible light for 4 h")


def test_preparation_range_is_not_a_list_of_prepared_sample_percentages():
    candidates, chunks = environment()
    chunks[0].text = chunks[0].text.replace("7%, 14%", "7%–14%")
    chunks[0].text_sha256 = hashlib.sha256(chunks[0].text.encode()).hexdigest()
    plan = request_targeted_evidence(
        calls=Calls(), candidates=candidates, chunks=chunks
    )
    assert not any(
        binding.kind == "preparation" for binding in plan.attributes.bindings
    )


def test_preparation_material_name_cannot_match_inside_another_formula():
    from materials_screening.master.fulltext_targeted_evidence import _loading

    candidates, chunks = environment()
    span = SimpleNamespace(quote="7% BiVO4/g-C3N4", chunk_id="same")
    definition = SimpleNamespace(
        quote="7% composites prepared by adjusting mass ratios.", chunk_id="same"
    )
    context = SimpleNamespace(quote="Preparation of ABiVO4/g-C3N4", chunk_id="same")
    with pytest.raises(ValueError):
        _loading(candidates["mid-seven"], span, definition, context)


@pytest.mark.parametrize("change", ["cross_document", "duplicate_identity"])
def test_targeted_catalogue_rejects_foreign_or_ambiguous_chunks(change):
    candidates, chunks = environment()
    if change == "cross_document":
        chunks[0].document_id = "doc-foreign"
    else:
        chunks = (*chunks, chunks[0])
    with pytest.raises(ValueError):
        request_targeted_evidence(calls=Calls(), candidates=candidates, chunks=chunks)
