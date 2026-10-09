"""Source identifiers replace model-rewritten quotes, not evidence checks."""

import hashlib
from dataclasses import replace

import pytest

from materials_screening.llm.base import StructuredProviderResponse
from materials_screening.llm.errors import LLMStructuredOutputError
from materials_screening.master.fulltext_extraction import _TaskCalls
from materials_screening.master.fulltext_snapshots import CheckedChunkStore
from materials_screening.sub_agents.literature.matrix_automation import (
    AutomatedMatrixExtractor,
    MatrixExtractionBatch,
    _evidence_text,
)
from tests.unit.sub_agents.test_literature_matrix_automation import _chunk as base_chunk


def _chunk(text):
    return replace(
        base_chunk(), text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest()
    )


class Calls:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def generate_structured(self, **kwargs):
        self.requests.append(kwargs)
        return StructuredProviderResponse(
            parsed=kwargs["output_model"].model_validate(self.payload),
            provider="offline",
            model="unit",
            request_id="unit",
            latency_ms=0,
            input_tokens=None,
            output_tokens=None,
            reasoning_tokens=None,
            raw_output_sha256=None,
        )


def task():
    from types import SimpleNamespace

    return SimpleNamespace(
        original_question="提取可见光透过率、电阻率和掺杂及制备条件；不可比时不排名。",
        user_instructions=("确认详细分析全部内容",),
    )


def payload(label="NTO3", value="82", source="c1", unit="%"):
    return {
        "groups": [{"label": label, "material": "SnO2", "source_id": source}],
        "measurements": [
            {
                "sample_label": label,
                "metric": "average transmittance",
                "value_text": value,
                "unit": unit,
                "source_id": source,
            }
        ],
    }


def adapter(calls, chunks):
    return _TaskCalls(calls, task(), chunks=chunks, source_selection=True)


def test_model_selects_source_without_retyping_pdf_control_glyphs():
    chunk = _chunk("NTO3 SnO2 film at 300 \x0eC had average transmittance 82%.")
    calls = Calls(payload())
    reply = adapter(calls, (chunk,)).generate_structured(
        system_prompt="Extract",
        user_text=_evidence_text((chunk,)),
        output_model=MatrixExtractionBatch,
        schema_name="matrix",
        max_output_tokens=4096,
    )
    assert reply.parsed.measurements[0].source_quote == chunk.text
    assert reply.parsed.groups[0].source_quote == chunk.text
    assert reply.parsed.measurements[0].group_key == "NTO3"
    assert reply.parsed.measurements[0].numeric_value == 82.0
    request = calls.requests[0]
    assert request["output_model"] is not MatrixExtractionBatch
    assert "source_quote" not in request["output_model"].model_fields
    assert "Do not copy or rewrite" in request["system_prompt"]
    assert "For source_quote copy" not in request["system_prompt"]


@pytest.mark.parametrize(
    "bad",
    [
        payload(label="NTO"),
        payload(value="not specified"),
        {
            **payload(),
            "measurements": [{**payload()["measurements"][0], "sample_label": "TO"}],
        },
    ],
)
def test_invalid_scientific_candidates_are_rejected_not_transport_failures(bad):
    chunk = _chunk("NTO3 SnO2 film had average transmittance 82%.")
    calls = Calls(bad)
    result = AutomatedMatrixExtractor(
        adapter(calls, (chunk,)), CheckedChunkStore((chunk,))
    ).extract(document_id=chunk.document_id, chunks=(chunk,))
    assert not result.measurements
    assert len(calls.requests) == 1
    assert result.diagnostics.successful_batches == 1
    assert result.diagnostics.rejection_reasons


def test_source_not_supplied_in_this_batch_cannot_be_selected():
    one = _chunk("NTO3 SnO2 film had average transmittance 82%.")
    two = replace(one, chunk_id="chunk-other")
    calls = Calls(payload(source="c2"))
    with pytest.raises(LLMStructuredOutputError):
        adapter(calls, (one, two)).generate_structured(
            system_prompt="Extract",
            user_text=_evidence_text((one,)),
            output_model=MatrixExtractionBatch,
            schema_name="matrix",
            max_output_tokens=4096,
        )


def test_original_numeric_unit_and_sample_validation_still_run():
    chunk = _chunk("NTO3 SnO2 film had average transmittance 82%.")
    for proposal, expected in (
        (payload(), 1),
        (payload(value="99"), 0),
        (payload(unit="MPa"), 0),
    ):
        calls = Calls(proposal)
        result = AutomatedMatrixExtractor(
            adapter(calls, (chunk,)), CheckedChunkStore((chunk,))
        ).extract(
            document_id=chunk.document_id,
            chunks=(chunk,),
        )
        assert len(result.measurements) == expected
        if expected:
            assert result.measurements[0].source_quote == chunk.text
            assert result.measurements[0].review_status == "pending"


def test_second_attempt_changes_feedback_but_never_adds_calls():
    chunk = _chunk("NTO3 SnO2 film had average transmittance 82%.")
    calls = Calls(payload(source="c99"))
    result = AutomatedMatrixExtractor(
        adapter(calls, (chunk,)), CheckedChunkStore((chunk,))
    ).extract(
        document_id=chunk.document_id,
        chunks=(chunk,),
    )
    assert len(calls.requests) == 2 and not result.measurements
    assert "Retry" in calls.requests[1]["system_prompt"]
    assert result.diagnostics.successful_batches == 0


def test_wrong_sample_number_in_same_source_is_not_approved():
    chunk = _chunk(
        "NTO1 SnO2 film had average transmittance 70%. "
        "NTO3 SnO2 film had average transmittance 82%."
    )
    calls = Calls(payload(label="NTO1"))
    result = AutomatedMatrixExtractor(
        adapter(calls, (chunk,)), CheckedChunkStore((chunk,))
    ).extract(
        document_id=chunk.document_id,
        chunks=(chunk,),
    )
    assert not result.measurements


def test_damaged_pdf_exponent_is_not_guessed():
    chunk = _chunk("NTO3 SnO2 film had resistivity 9.02 \x02 10 4 Ω cm.")
    proposal = payload(value="9.02 × 10^-4", unit="Ω cm")
    proposal["measurements"][0]["metric"] = "resistivity"
    calls = Calls(proposal)
    result = AutomatedMatrixExtractor(
        adapter(calls, (chunk,)), CheckedChunkStore((chunk,))
    ).extract(
        document_id=chunk.document_id,
        chunks=(chunk,),
    )
    assert not result.measurements


def test_one_missing_proposal_does_not_lose_another_valid_numeric_row():
    chunk = _chunk("NTO3 SnO2 film had average transmittance 82%.")
    good = payload(value="82%")
    good["measurements"].extend(payload(value="not specified")["measurements"])
    calls = Calls(good)
    result = AutomatedMatrixExtractor(
        adapter(calls, (chunk,)), CheckedChunkStore((chunk,))
    ).extract(document_id=chunk.document_id, chunks=(chunk,))
    assert len(calls.requests) == 1
    assert len(result.measurements) == 1
    assert result.measurements[0].value_text == "82"
    assert result.measurements[0].unit == "%"
    assert result.diagnostics.llm_measurements == 1
    assert result.diagnostics.rejection_reasons


def test_literal_measured_sample_restores_parent_without_guessing_material():
    chunk = _chunk("NTO3 film had average transmittance 82%.")
    proposal = payload()
    proposal["groups"] = [
        {"label": "Nb-doped In2O3 films", "material": "In2O3", "source_id": "c1"}
    ]
    calls = Calls(proposal)
    result = AutomatedMatrixExtractor(
        adapter(calls, (chunk,)), CheckedChunkStore((chunk,))
    ).extract(document_id=chunk.document_id, chunks=(chunk,))
    assert len(result.measurements) == 1
    assert result.groups[0].label == result.groups[0].material == "NTO3"
    assert result.groups[0].variables == result.groups[0].conditions == {}
    assert result.groups[0].role == "unknown"


def test_flattened_column_table_never_binds_arbitrary_column_value():
    chunk = _chunk("Table 1\nParameter\nNTO1\nNTO3\ntransmittance (%)\n70\n82\n")
    calls = Calls(payload(label="NTO1"))
    result = AutomatedMatrixExtractor(
        adapter(calls, (chunk,)), CheckedChunkStore((chunk,))
    ).extract(document_id=chunk.document_id, chunks=(chunk,))
    assert not result.measurements
    assert result.diagnostics.rejection_reasons
