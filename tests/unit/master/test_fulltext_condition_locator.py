"""Fixed two-stage fixtures; literal checks stay fail-closed per attribute."""

import hashlib
from types import SimpleNamespace

import pytest

from materials_screening.master.fulltext_condition_locator import (
    LiteralValues,
    SentenceLocations,
    locate_shared_conditions,
    read_located_values,
    validate_locations,
)


def environment():
    text = (
        "Methods\nAll the samples were tested in water. "
        "The samples were stirred for 1 h for adsorption equilibration. "
        "A 400 nm cutoff filter was used for illumination. "
        "The solution volume was 120 mL.\nResults and Discussion\n"
        "A different experiment used 20 mL."
    )
    chunk = SimpleNamespace(
        chunk_id="chunk-unit",
        document_id="doc-unit",
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        page_from=2,
        page_to=2,
    )
    candidates = {
        "measurement-unit": dict(
            document_id="doc-unit",
            group_label="Sample A",
            metric="degradation",
            unit="%",
        )
    }
    return candidates, (chunk,)


class Calls:
    def __init__(self):
        self.calls = 0

    def generate_structured(self, **kwargs):
        import json

        self.calls += 1
        assert kwargs["max_output_tokens"] == 4096
        if kwargs["output_model"] is SentenceLocations:
            evidence = json.loads(kwargs["user_text"])["evidence"]
            assert len(evidence) == 4
            assert not any(
                "different experiment" in value for value in evidence.values()
            )
            return SimpleNamespace(
                parsed=SentenceLocations(
                    shared_scope="s0",
                    adsorption_equilibration_time="s1",
                    irradiation_time="s1",
                    cutoff_wavelength="s2",
                    solution_volume="s3",
                )
            )
        assert kwargs["output_model"] is LiteralValues
        return SimpleNamespace(
            parsed=LiteralValues(
                adsorption_equilibration_time="1 h",
                irradiation_time="1 h",
                cutoff_wavelength="400 nm",
                solution_volume="120 mL",
                pollutant="invented pollutant",
            )
        )


def test_valid_attributes_preserved_wrong_phase_and_unlocated_values_isolated():
    candidates, chunks = environment()
    calls = Calls()
    locations = locate_shared_conditions(
        calls=calls, candidates=candidates, chunks=chunks
    )
    plan = read_located_values(
        calls=calls, locations=locations, candidates=candidates, chunks=chunks
    )
    assert calls.calls == 2 and plan.review_status == "pending"
    assert {row.key for row in plan.bindings} == {
        "adsorption_equilibration_time",
        "cutoff_wavelength",
        "solution_volume",
    }
    assert {row.key for row in plan.isolated_attributes} == {
        "irradiation_time",
        "pollutant",
    }
    assert all(row.numeric_value in (1, 400, 120) for row in plan.bindings)


def test_quoted_null_is_missing_not_a_source_or_fact():
    assert (
        SentenceLocations(shared_scope="null", temperature="null").temperature is None
    )
    assert LiteralValues(temperature="null").temperature is None


def test_locator_schema_cannot_accept_condition_values_as_sentence_ids():
    with pytest.raises(ValueError):
        SentenceLocations(shared_scope="ALL SAMPLES", pollutant="tetracycline")
    schema = SentenceLocations.model_json_schema()
    assert next(iter(schema["properties"])) == "shared_scope"
    assert schema["properties"]["pollutant"]["anyOf"][0]["pattern"] == r"^s[0-9]{1,3}$"


def test_no_explicit_shared_test_protocol_starts_no_model():
    candidates, chunks = environment()
    chunks[0].text = "All samples were characterized by XRD. Preparation used 25 C."
    chunks[0].text_sha256 = hashlib.sha256(chunks[0].text.encode()).hexdigest()
    calls = Calls()
    locations = locate_shared_conditions(
        calls=calls, candidates=candidates, chunks=chunks
    )
    assert locations is None and calls.calls == 0


@pytest.mark.parametrize("change", ["source", "scope", "hash"])
def test_locations_reject_unknown_scope_or_source_change(change):
    candidates, chunks = environment()
    locations = locate_shared_conditions(
        calls=Calls(), candidates=candidates, chunks=chunks
    )
    if change == "source":
        locations = locations.model_copy(
            update={
                "attributes": {
                    "temperature": locations.shared_scope.model_copy(
                        update={"chunk_id": "unknown"}
                    )
                }
            }
        )
    elif change == "scope":
        locations = locations.model_copy(
            update={
                "shared_scope": locations.shared_scope.model_copy(
                    update={"quote": "400 nm"}
                )
            }
        )
    else:
        chunks[0].text += " replaced"
    with pytest.raises(ValueError):
        validate_locations(locations, candidates, chunks)


def test_range_is_not_coerced_into_one_numeric_condition():
    candidates, chunks = environment()
    calls = Calls()
    locations = locate_shared_conditions(
        calls=calls, candidates=candidates, chunks=chunks
    )
    locations = locations.model_copy(
        update={"attributes": {"temperature": locations.shared_scope}}
    )
    calls.generate_structured = lambda **kwargs: SimpleNamespace(
        parsed=LiteralValues(temperature="25–30 C")
    )
    plan = read_located_values(
        calls=calls, locations=locations, candidates=candidates, chunks=chunks
    )
    assert not plan.bindings and plan.isolated_attributes[0].key == "temperature"
