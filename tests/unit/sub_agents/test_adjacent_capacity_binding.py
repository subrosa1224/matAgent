"""Recover missing sample keys only from literal, adjacent cyclic-capacity prose."""

import hashlib
from dataclasses import replace

import pytest
from tests.unit.sub_agents.test_literature_matrix_automation import FakeLlm

from materials_screening.sub_agents.literature import matrix_automation as matrix
from materials_screening.sub_agents.literature.matrix_automation import (
    AutomatedMatrixExtractor,
    GroupCandidate,
    MatrixExtractionBatch,
    MeasurementCandidate,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord

ANCHOR = (
    "The sample obtained at 150 ◦C for 6 h exhibited an initial discharge "
    "capacity of 223.2 mAh/g."
)
QUOTE = "After 50 cycles, the discharge capacity remained 194.5 mAh/g."


def run(
    text=None,
    *,
    anchor=ANCHOR,
    quote=QUOTE,
    updates=None,
    extra_groups=(),
    group_updates=None,
    extra_chunks=(),
):
    text = text if text is not None else anchor + " " + quote
    chunk = ChunkRecord(
        chunk_id="chunk-local",
        document_id="doc-local",
        paper_id=None,
        page_from=1,
        page_to=1,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    group = GroupCandidate(
        group_key="sample-local",
        label="Optimal oxide sample",
        material="oxide",
        variables={"calcination_temperature": "150 ◦C", "calcination_time": "6 h"},
        source_quote=anchor,
        chunk_id=chunk.chunk_id,
    ).model_copy(update=group_updates or {})
    candidate = MeasurementCandidate(
        metric="discharge capacity after 50 cycles",
        value_text="194.5",
        numeric_value=194.5,
        unit="mAh/g",
        source_quote=quote,
        chunk_id=chunk.chunk_id,
    ).model_copy(update=updates or {})
    batch = MatrixExtractionBatch(
        groups=(group, *extra_groups), measurements=(candidate,)
    )
    original = batch.model_dump_json()
    chunk_map = {c.chunk_id: c for c in (chunk, *extra_chunks)}

    class SnapshotStore:
        def get_chunks(self, ids):
            return [chunk_map[key] for key in ids if key in chunk_map]

    result = AutomatedMatrixExtractor(FakeLlm(batch), SnapshotStore()).extract(
        document_id=chunk.document_id, chunks=tuple(chunk_map.values())
    )
    assert original == batch.model_dump_json()
    return result, chunk


def test_literal_adjacent_sample_capacity_is_retained_as_pending():
    result, chunk = run()
    assert len(result.measurements) == 1
    row = result.measurements[0]
    assert row.source_quote == ANCHOR + " " + QUOTE
    assert row.source_quote in chunk.text
    assert row.value_text == "194.5" and row.numeric_value == 194.5
    assert row.unit == "mAh/g" and row.review_status == "pending"
    assert (
        row.chunk_id == chunk.chunk_id and row.source_text_sha256 == chunk.text_sha256
    )
    diagnostic = result.diagnostics.measurement_candidates[0]
    assert diagnostic.group_key is None and diagnostic.status == "constructed"
    assert diagnostic.source_quote_sha256 == hashlib.sha256(QUOTE.encode()).hexdigest()
    assert (
        diagnostic.resolved_quote_sha256
        == hashlib.sha256(row.source_quote.encode()).hexdigest()
    )
    assert any("adjacent sample context" in w for w in result.warnings)


def test_line_wraps_are_preserved_not_rewritten():
    text = (ANCHOR + " " + QUOTE).replace("capacity remained", "capacity\nremained")
    result, chunk = run(text)
    assert len(result.measurements) == 1
    assert result.measurements[0].source_quote == chunk.text


def test_duplicate_identical_parent_id_is_not_a_second_sample():
    duplicate = GroupCandidate(
        group_key="sample-local",
        label="Optimal oxide sample",
        material="oxide",
        variables={"calcination_temperature": "150 ◦C", "calcination_time": "6 h"},
        source_quote=ANCHOR,
        chunk_id="chunk-local",
    )
    result, _ = run(extra_groups=(duplicate,))
    assert len(result.measurements) == 1


@pytest.mark.parametrize(
    "anchor,quote,updates",
    [
        (ANCHOR, QUOTE, {"group_key": "unknown"}),
        (ANCHOR, QUOTE, {"chunk_id": "unknown"}),
        (ANCHOR, QUOTE, {"unit": "Ah/g"}),
        (ANCHOR, QUOTE, {"unit": "MAh/g"}),
        (ANCHOR, QUOTE, {"metric": "capacity retention after 50 cycles"}),
        (ANCHOR, QUOTE, {"metric": "initial discharge capacity"}),
        (ANCHOR, QUOTE, {"metric": "discharge capacity after 100 cycles"}),
        (ANCHOR, QUOTE, {"numeric_value": 195.5}),
        (ANCHOR, QUOTE, {"uncertainty_text": "± 0.4"}),
        (ANCHOR, QUOTE, {"value_text": "194.5 mAh/g"}),
        ("The samples obtained at 100 ◦C and 150 ◦C for 6 h were tested.", QUOTE, {}),
        (
            ANCHOR,
            "After 50 cycles, capacities were 194.5 and 180 mAh/g, respectively.",
            {},
        ),
        (
            ANCHOR,
            "After 50 cycles, it had 194.5 mAh/g at 0.1 C and 180 mAh/g at 0.2 C.",
            {},
        ),
        ("Previous studies reported: " + ANCHOR, QUOTE, {}),
        ("References\n" + ANCHOR, QUOTE, {}),
        ("The sample obtained at 150 ◦C for 6 h and 12 h was tested.", QUOTE, {}),
        ("Sample A at 150 ◦C for 6 h and Sample B were tested.", QUOTE, {}),
    ],
    ids=[
        "unknown-key",
        "unknown-chunk",
        "scale",
        "unit-case",
        "retention",
        "initial",
        "wrong-cycle",
        "wrong-number",
        "uncertainty",
        "unit-in-value",
        "multi-temperature",
        "respectively",
        "multi-rate",
        "prior-work",
        "references",
        "multi-time",
        "multi-sample",
    ],
)
def test_unsafe_missing_key_is_not_rescued(anchor, quote, updates):
    result, _ = run(anchor=anchor, quote=quote, updates=updates)
    assert not result.measurements
    assert not any("adjacent sample context" in w for w in result.warnings)


def test_unlocated_quote_is_not_rescued_when_old_inference_is_unresolved(monkeypatch):
    monkeypatch.setattr(matrix, "_infer_group_id", lambda *_: None)
    result, _ = run(
        updates={"source_quote": "After 50 cycles, capacity was 194.5 mAh/g."}
    )
    assert not result.measurements
    assert not any("adjacent sample context" in w for w in result.warnings)


def test_preexisting_whole_page_fallback_is_not_silently_changed():
    # This limitation predates the fix: the old inference resolves from the page
    # before the new strict adjacent-context path can run. It is NOT new rescue.
    result, chunk = run(
        updates={"source_quote": "After 50 cycles, capacity was 194.5 mAh/g."}
    )
    assert len(result.measurements) == 1
    assert result.measurements[0].source_quote == chunk.text
    assert not any("adjacent sample context" in w for w in result.warnings)


@pytest.mark.parametrize("separator", ["\n\n", "\n2. Results\n", "\nTable 1.\n"])
def test_paragraph_section_and_table_boundaries_are_not_crossed(separator):
    result, _ = run(ANCHOR + separator + QUOTE)
    assert not result.measurements


def test_only_the_immediate_previous_sentence_can_bind():
    result, _ = run(ANCHOR + " Samples were tested at another rate. " + QUOTE)
    assert not result.measurements


def test_repeated_quote_is_not_resolved_by_convenient_occurrence():
    result, _ = run(ANCHOR + " " + QUOTE + " " + QUOTE)
    assert not result.measurements


@pytest.mark.parametrize(
    "variables,conditions",
    [
        ({"calcination_temperature": "150 ◦C", "calcination_time": "6 h"}, {}),
        ({"calcination_temperature": "150 ◦C"}, {}),
        (
            {"calcination_temperature": "150 ◦C", "calcination_time": "6 h"},
            {"rate": "0.2 C"},
        ),
    ],
)
def test_competing_local_parent_is_not_resolved_by_score(variables, conditions):
    other = GroupCandidate(
        group_key="other",
        label="Other oxide sample",
        material="oxide",
        variables=variables,
        conditions=conditions,
        source_quote=ANCHOR + " Test conditions included 0.2 C.",
        chunk_id="chunk-local",
    )
    result, _ = run(
        ANCHOR + " Test conditions included 0.2 C. " + ANCHOR + " " + QUOTE,
        extra_groups=(other,),
    )
    assert not result.measurements


def test_local_parent_cannot_be_rescued_with_unstated_conditions():
    result, _ = run(
        group_updates={
            "conditions": {"rate": "0.2 C"},
            "source_quote": ANCHOR + "\nTesting used 0.2 C.",
        },
        text=ANCHOR + "\nTesting used 0.2 C.\n\n" + ANCHOR + " " + QUOTE,
    )
    assert not result.measurements


def test_reference_continuation_is_not_own_work():
    text = "References\n"
    header = ChunkRecord(
        chunk_id="references",
        document_id="doc-local",
        paper_id=None,
        page_from=0,
        page_to=0,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    result, _ = run(extra_chunks=(header,))
    assert not result.measurements


def test_same_number_from_different_document_does_not_bind():
    result, chunk = run()
    foreign = replace(chunk, chunk_id="foreign", document_id="other-doc")
    result, _ = run(group_updates={"chunk_id": "foreign"}, extra_chunks=(foreign,))
    assert not result.measurements


def test_no_missing_measurement_is_synthesized():
    # The candidate asks for a different quantity; finding 194.5 in prose is not enough.
    result, _ = run(updates={"value_text": "999", "numeric_value": 999})
    assert not result.measurements


@pytest.mark.parametrize(
    "quote",
    [
        "After 50 cycles, another electrode's discharge capacity was 194.5 mAh/g.",
        "After 50 cycles, the discharge capacity of the 100 ◦C material "
        "was 194.5 mAh/g.",
        "After 50 cycles, a different material's discharge capacity was 194.5 mAh/g.",
    ],
)
def test_changed_subject_does_not_inherit_previous_sample(quote):
    result, _ = run(quote=quote)
    assert not result.measurements


def test_adjacent_context_never_spans_chunks():
    text = ANCHOR
    previous = ChunkRecord(
        chunk_id="previous",
        document_id="doc-local",
        paper_id=None,
        page_from=1,
        page_to=1,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    result, _ = run(
        text=QUOTE, group_updates={"chunk_id": "previous"}, extra_chunks=(previous,)
    )
    assert not result.measurements


def test_local_anchor_keeps_explicit_test_conditions_when_available():
    anchor = ANCHOR.replace("exhibited", "tested at 0.1 C exhibited")
    result, _ = run(anchor=anchor, group_updates={"conditions": {"rate": "0.1 C"}})
    assert len(result.measurements) == 1
    assert "0.1 C" in result.measurements[0].source_quote


def test_changing_rate_in_target_sentence_is_not_rescued():
    anchor = ANCHOR.replace("exhibited", "tested at 0.1 C exhibited")
    quote = QUOTE.replace("194.5 mAh/g", "194.5 mAh/g at 0.2 C")
    result, _ = run(anchor=anchor, quote=quote)
    assert not result.measurements


def test_dropped_original_sample_variable_must_not_be_ignored_by_rescue():
    result, _ = run(
        group_updates={
            "variables": {
                "calcination_temperature": "150 ◦C",
                "calcination_time": "12 h",
            }
        }
    )
    assert not result.measurements


def test_duration_numeric_fallback_cannot_match_a_different_time_unit():
    result, _ = run(anchor=ANCHOR.replace("6 h", "6 min"))
    assert not result.measurements


def test_multiple_cycle_counts_do_not_share_one_quantity_implicitly():
    result, _ = run(
        quote=(
            "After 50 cycles and after 100 cycles, the discharge capacity "
            "remained 194.5 mAh/g."
        )
    )
    assert not result.measurements
