"""Temperature typography must not erase units or relax sample attribution."""

import hashlib

import pytest
from tests.unit.master.test_shared_measurement_binding import (
    test_extraction_and_handoff_share_binding_rules as check_shared_binding,
)
from tests.unit.sub_agents.test_literature_matrix_automation import FakeLlm

from materials_screening.sub_agents.literature.matrix_automation import (
    AutomatedMatrixExtractor,
    GroupCandidate,
    MatrixExtractionBatch,
    MeasurementCandidate,
    _ground_mapping,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord


@pytest.mark.parametrize("numeric_fallback", [True, False])
@pytest.mark.parametrize(
    "candidate,reported",
    [
        ("150°C", "150 ◦C"),
        ("150 ◦C", "150°C"),
        ("150 °C", "150\n◦ C"),
        ("37.5°C", "37.5 ◦C"),
        ("-20°C", "−20 ◦C"),
        ("−20°C", "-20 ◦C"),
        ("+20°C", "+20 ◦C"),
        ("150°F", "150 ◦F"),
        ("150°C", "150°C"),
    ],
)
def test_temperature_grounding_preserves_complete_verbatim_span(
    candidate, reported, numeric_fallback
):
    quote = f"The sample was prepared at {reported} for 24 h."
    original = {"temperature": candidate}
    result = _ground_mapping(original, quote, numeric_fallback=numeric_fallback)
    assert result == {"temperature": reported}
    assert result["temperature"] in quote
    assert original == {"temperature": candidate}


@pytest.mark.parametrize("numeric_fallback", [True, False])
@pytest.mark.parametrize(
    "candidate,reported",
    [
        ("150°C", "150 ◦F"),
        ("150°F", "150 ◦C"),
        ("150°C", "150 K"),
        ("150°C", "1150 ◦C"),
        ("150°C", "150.0 ◦C"),
        ("150.0°C", "150 ◦C"),
        ("150°C", "-150 ◦C"),
        ("150°C", "+150 ◦C"),
        ("150°C", "150 °C/min"),
        ("150°C", "150 °C per minute"),
        ("150°C", "150 °C min−1"),
        ("150°C", "150 °C·s^-1"),
        ("150°C", "150 °C h⁻¹"),
        ("150°C", "150 °C2"),
    ],
)
def test_explicit_temperature_never_falls_back_to_a_bare_or_different_value(
    candidate, reported, numeric_fallback
):
    assert not _ground_mapping(
        {"temperature": candidate},
        f"The reported condition was {reported}.",
        numeric_fallback=numeric_fallback,
    )


def test_non_temperature_grounding_is_unchanged():
    assert _ground_mapping(
        {"pH": "9", "Nb": "1.5 mol%"},
        "pH 9 and 1.5 mol% Nb were used.",
        numeric_fallback=True,
    ) == {"pH": "9", "Nb": "1.5 mol%"}
    assert not _ground_mapping({"pH": "9.0"}, "pH 9", numeric_fallback=True)


@pytest.mark.parametrize("quote_style", ["150°C", "150 ◦C", "150\n◦ C"])
def test_extractor_and_handoff_keep_temperature_units_and_source(quote_style):
    quote = (
        f"Among these samples, the sample prepared at {quote_style} delivered a "
        "discharge capacity of 194.5 mAh/g after 50 cycles."
    )
    table = "Table 1. Surface area.\nSample\nSBET\n100 ◦C\n10\n150 ◦C\n20\n"
    text = table + quote
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        text=text,
        page_from=6,
        page_to=6,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )

    class MemoryStore:
        def get_chunks(self, ids):
            return [chunk] if chunk.chunk_id in ids else []

    batch = MatrixExtractionBatch(
        groups=(
            GroupCandidate(
                group_key="150C",
                label="150°C",
                role="unknown",
                material="LiFeO2",
                variables={"hydrothermal temperature": "150°C"},
                conditions={"temperature": "150°C"},
                source_quote=quote,
                chunk_id=chunk.chunk_id,
            ),
        ),
        measurements=(
            MeasurementCandidate(
                group_key="150C",
                metric="discharge capacity",
                value_text="194.5",
                numeric_value=194.5,
                unit="mAh/g",
                source_quote=quote,
                chunk_id=chunk.chunk_id,
            ),
        ),
    )
    result = AutomatedMatrixExtractor(FakeLlm(batch), MemoryStore()).extract(
        document_id="doc-1",
        chunks=(chunk,),
    )
    assert len(result.groups) == len(result.measurements) == 1
    assert result.groups[0].variables == {"hydrothermal temperature": quote_style}
    assert result.groups[0].conditions == {"temperature": quote_style}
    assert result.groups[0].source_quote == quote
    row = result.measurements[0]
    assert row.source_quote == quote and row.source_text_sha256 == chunk.text_sha256
    assert row.page_from == row.page_to == 6
    assert row.review_status == "pending"
    assert batch.groups[0].variables == {"hydrothermal temperature": "150°C"}
    check_shared_binding(
        text,
        quote,
        "150°C",
        "194.5",
        "discharge capacity",
        "mAh/g",
        1,
        False,
        variables=result.groups[0].variables,
    )


def test_preserved_temperature_does_not_allow_a_wrong_table_row():
    quote = (
        "Table 1. Photocatalytic results.\nSample Name\nDegradation (%)\n"
        "100 ◦C\n25.3%\n150 ◦C\n63.2%\n"
    )
    variables = _ground_mapping(
        {"temperature": "150°C"},
        quote,
        numeric_fallback=True,
    )
    assert variables == {"temperature": "150 ◦C"}
    check_shared_binding(
        quote,
        quote,
        "150 ◦C",
        "25.3",
        "photocatalytic degradation efficiency",
        "%",
        0,
        False,
        variables=variables,
    )
