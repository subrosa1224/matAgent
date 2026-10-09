import pytest

from materials_screening.sub_agents.literature.matrix import measurement_value_present


@pytest.mark.parametrize(
    "evidence",
    [
        "R = 113 AW − 1 and G = 385",
        "R = 113 AW⁻¹ and G = 385",
        "R = 113 AW^-1 and G = 385",
    ],
)
def test_contiguous_numeric_unit_typography(evidence):
    assert measurement_value_present("113 AW−1", "AW−1", evidence)


@pytest.mark.parametrize(
    "evidence",
    [
        "R = 113 mAW − 1",
        "R = 113 AW − 2",
        "R = 2113 AW − 1",
        "R = 113.5 AW − 1",
        "R = 113; unit is AW − 1",
        "R = 113 in table one. Elsewhere: AW − 1",
        "R = 113 aw − 1",
    ],
)
def test_wrong_or_disconnected_numeric_units(evidence):
    assert not measurement_value_present("113 AW−1", "AW−1", evidence)


def test_value_suffix_must_match_declared_unit():
    assert not measurement_value_present("113 mAW−1", "AW−1", "113 mAW − 1")
    assert not measurement_value_present("113 mAW−1", "AW−1", "113 mAW−1")
    assert not measurement_value_present("113 AW−1", "AW−1", "2113 AW−1")
    assert not measurement_value_present("113 AW−1", "AW−1", "113 aw−1")


def test_candidate_location_keeps_actual_source_text():
    import hashlib

    from materials_screening.sub_agents.literature.matrix_automation import (
        MeasurementCandidate,
        _measurement_evidence,
    )
    from materials_screening.sub_agents.literature.rag import ChunkRecord

    text = "The responsivity is 113 AW − 1 and gain is 385."
    chunk = ChunkRecord(
        "chunk-test",
        "doc-test",
        None,
        4,
        4,
        text,
        hashlib.sha256(text.encode()).hexdigest(),
    )
    candidate = MeasurementCandidate(
        group_key="device",
        metric="Responsivity",
        value_text="113 AW−1",
        numeric_value=113,
        unit="AW−1",
        source_quote=text,
        chunk_id=chunk.chunk_id,
    )
    assert _measurement_evidence(candidate, chunk) == text
    assert (
        _measurement_evidence(
            candidate.model_copy(update={"value_text": "114 AW−1"}), chunk
        )
        is None
    )
