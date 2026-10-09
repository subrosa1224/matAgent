"""Source-bound supplementary attributes, not professional comparability approval."""

import hashlib
from types import SimpleNamespace

import pytest

from materials_screening.master.fulltext_conditions import (
    AttributeBinding,
    ConditionPlan,
    SourceSpan,
    validate_condition_plan,
)


def environment():
    text = "All the samples were measured in water at 25 C. Sample A has 10% loading."
    chunk = SimpleNamespace(
        chunk_id="chunk-a",
        document_id="doc-unit",
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        page_from=2,
        page_to=2,
    )
    rows = {"measurement-a": {"document_id": "doc-unit", "group_label": "Sample A"}}
    span = SourceSpan(
        chunk_id=chunk.chunk_id,
        text_sha256=chunk.text_sha256,
        page_from=2,
        page_to=2,
        quote="All the samples were measured in water at 25 C.",
    )
    binding = AttributeBinding(
        measurement_ids=("measurement-a",),
        kind="condition",
        key="temperature",
        value_text="25 C",
        numeric_value=25,
        unit="C",
        value_source=span,
        applicability_source=span,
        applicability="reported_common_protocol",
    )
    return rows, (chunk,), binding


def test_common_protocol_preserves_quotes_and_pending_not_approval():
    rows, chunks, binding = environment()
    plan = ConditionPlan(bindings=(binding,))
    checked = validate_condition_plan(plan, rows, chunks)
    assert checked == plan and checked.review_status == "pending"
    assert checked.bindings[0].value_source.page_from == 2


@pytest.mark.parametrize(
    "change", ["value", "hash", "page", "scope", "target", "scale"]
)
def test_invalid_condition_or_numeric_derivation_fails_closed(change):
    rows, chunks, binding = environment()
    if change == "value":
        binding = binding.model_copy(update={"value_text": "30 C", "numeric_value": 30})
    elif change == "hash":
        binding = binding.model_copy(
            update={
                "value_source": binding.value_source.model_copy(
                    update={"text_sha256": "a" * 64}
                )
            }
        )
    elif change == "page":
        binding = binding.model_copy(
            update={
                "value_source": binding.value_source.model_copy(
                    update={"page_from": 3, "page_to": 3}
                )
            }
        )
    elif change == "scope":
        binding = binding.model_copy(
            update={
                "applicability_source": binding.value_source.model_copy(
                    update={"quote": "25 C"}
                )
            }
        )
    elif change == "target":
        binding = binding.model_copy(update={"measurement_ids": ("measurement-other",)})
    else:
        binding = binding.model_copy(update={"numeric_value": 0.25})
    with pytest.raises(ValueError):
        validate_condition_plan(ConditionPlan(bindings=(binding,)), rows, chunks)


def test_preparation_value_must_be_literal_not_inferred_from_label_only():
    rows, chunks, binding = environment()
    specific = binding.model_copy(
        update={
            "kind": "preparation",
            "key": "nominal_loading",
            "value_text": "10%",
            "numeric_value": 10,
            "unit": "%",
            "applicability": "per_sample",
            "value_source": binding.value_source.model_copy(
                update={"quote": "Sample A has 10% loading."}
            ),
            "applicability_source": binding.value_source.model_copy(
                update={"quote": "Sample A has 10% loading."}
            ),
        }
    )
    checked = validate_condition_plan(ConditionPlan(bindings=(specific,)), rows, chunks)
    assert checked.bindings[0].numeric_value == 10
    with pytest.raises(ValueError):
        validate_condition_plan(
            ConditionPlan(bindings=(specific, specific)), rows, chunks
        )


def test_other_document_or_reference_section_cannot_bind_condition():
    rows, chunks, binding = environment()
    chunks[0].document_id = "doc-other"
    with pytest.raises(ValueError):
        validate_condition_plan(ConditionPlan(bindings=(binding,)), rows, chunks)
    chunks[0].document_id = "doc-unit"
    chunks[0].text = "References\n" + chunks[0].text
    chunks[0].text_sha256 = hashlib.sha256(chunks[0].text.encode()).hexdigest()
    span = binding.value_source.model_copy(
        update={"text_sha256": chunks[0].text_sha256}
    )
    binding = binding.model_copy(
        update={"value_source": span, "applicability_source": span}
    )
    with pytest.raises(ValueError):
        validate_condition_plan(ConditionPlan(bindings=(binding,)), rows, chunks)


def test_empty_condition_plan_preserves_missing_not_equal_conditions():
    rows, chunks, _ = environment()
    plan = ConditionPlan(unresolved=("No applicable protocol found",))
    assert validate_condition_plan(plan, rows, chunks).bindings == ()


def test_generated_summary_cannot_replace_literal_atomic_condition():
    # Real compact diagnostic returned prose summaries despite literal schema.
    rows, chunks, binding = environment()
    summary = binding.model_copy(
        update={
            "value_text": "Measured in water at a temperature of 25 C",
            "numeric_value": None,
        }
    )
    with pytest.raises(ValueError, match="Attribute value not present"):
        validate_condition_plan(ConditionPlan(bindings=(summary,)), rows, chunks)


def test_prompt_teaches_atomic_literal_values_not_general_paper_summaries():
    from materials_screening.master.fulltext_conditions import _PROMPT

    assert "value_text is an EXACT SUBSTRING" in _PROMPT
    assert "25 C" in _PROMPT and "reported_common_protocol" in _PROMPT
    assert "unrelated characterization" in _PROMPT


def test_compact_response_reuses_citations_and_server_owned_identity():
    from materials_screening.master.fulltext_conditions import (
        AttributeProposal,
        CitationProposal,
        ConditionProposal,
        request_condition_plan,
    )

    rows, chunks, binding = environment()
    rows["measurement-a"].update(
        measurement_id="measurement-a", metric="strength", unit="MPa"
    )

    class Calls:
        def generate_structured(self, **kwargs):
            assert kwargs["output_model"] is ConditionProposal
            assert kwargs["max_output_tokens"] == 4096
            return SimpleNamespace(
                parsed=ConditionProposal(
                    citations=(
                        CitationProposal(
                            key="s0", chunk_key="c0", quote=binding.value_source.quote
                        ),
                    ),
                    bindings=(
                        AttributeProposal(
                            measurement_keys=("m0",),
                            kind="condition",
                            key="temperature",
                            value_text="25 C",
                            numeric_value=25,
                            unit="C",
                            value_source="s0",
                            applicability_source="s0",
                            applicability="reported_common_protocol",
                        ),
                    ),
                )
            )

    plan = request_condition_plan(
        calls=Calls(), requirements="Review strength", candidates=rows, chunks=chunks
    )
    assert plan.bindings == (binding,)
    assert "text_sha256" not in CitationProposal.model_fields
    assert "page_from" not in CitationProposal.model_fields
