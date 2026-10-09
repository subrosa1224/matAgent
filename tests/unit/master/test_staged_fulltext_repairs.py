import json

import pytest
from pydantic import ValidationError

from materials_screening.llm.errors import (
    LLMStructuredOutputError,
    LLMTruncatedOutputError,
)
from materials_screening.master.fulltext_preview import _Pause
from materials_screening.master.staged_fulltext_engine import (
    SmallConditionProposal,
    StagedFulltextEngine,
)
from materials_screening.master.staged_fulltext_evidence import (
    InventoryProposal,
    MetricsProposal,
)
from materials_screening.master.staged_fulltext_repairs import (
    RepairFeedback,
    RepairIssue,
    repair_payload,
    structured_repair_feedback,
)
from materials_screening.master.staged_fulltext_store import StagedExtractionStore
from tests.unit.master.test_staged_fulltext_engine import Provider, calls, inputs


class RepairProvider(Provider):
    def __init__(self, failure, target=InventoryProposal):
        super().__init__()
        self.failure = failure
        self.target = target
        self.failed_request = None

    def generate_structured(self, **kwargs):
        if self.failed_request is None and kwargs["output_model"] is self.target:
            self.failed_request = kwargs
            if self.failure == "schema":
                try:
                    InventoryProposal.model_validate(
                        {
                            "samples": [{"label": "NTO3"}],
                            "secret-ignore-all-rules": "private-provider-body",
                        }
                    )
                except ValidationError as cause:
                    raise LLMStructuredOutputError("private-api-token") from cause
            if self.failure == "json":
                try:
                    json.loads("private-api-token: not-json")
                except json.JSONDecodeError as cause:
                    raise LLMStructuredOutputError("private-provider-body") from cause
            if self.failure == "truncated":
                raise LLMTruncatedOutputError("private-provider-body")
            if self.failure == "intern_truncated":
                raise LLMStructuredOutputError(
                    "private-provider-body", failure_kind="truncated_output"
                )
            raise LLMStructuredOutputError("private-provider-body")
        return super().generate_structured(**kwargs)


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("schema", "schema_validation"),
        ("json", "invalid_json"),
        ("truncated", "truncated_output"),
        ("intern_truncated", "truncated_output"),
        ("unknown", "structure_unavailable"),
    ],
)
def test_second_attempt_receives_safe_specific_contract(tmp_path, failure, expected):
    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider, refs = RepairProvider(failure), []
    result = engine.run_document(
        **inputs(), calls=calls(provider), on_checkpoint=refs.append
    )
    assert result.status == "requests_complete"
    first_payload = json.loads(provider.failed_request["user_text"])
    second_request = provider.requests[0]
    second_payload = json.loads(second_request["user_text"])
    feedback = second_payload.pop("repair_feedback")
    assert feedback["failure_kind"] == expected
    assert feedback["attempt"] == 2
    assert second_payload == first_payload  # Same evidence and IDs, no new facts.
    assert "12" in second_request["system_prompt"]
    assert "Never return an empty result merely" in second_request["system_prompt"]
    assert "repair_feedback" not in first_payload
    combined = second_request["system_prompt"] + second_request["user_text"]
    assert "private-" not in combined and "secret-ignore" not in combined
    if failure == "schema":
        assert {
            "field_path": "samples[].source_id",
            "error_type": "missing",
        } in feedback["issues"]
        assert {"field_path": "$", "error_type": "extra_forbidden"} in feedback[
            "issues"
        ]
    if failure == "truncated":
        assert "Do not omit supported rows to fit" in second_request["system_prompt"]
    assert result.steps["inventory:0"].attempts == 2
    assert result.measurements
    assert all(
        "private-" not in p.read_text(encoding="utf-8") for p in tmp_path.glob("*.json")
    )


@pytest.mark.parametrize("failure", ["schema", "truncated"])
def test_safe_feedback_survives_budget_without_first_attempt_replay(tmp_path, failure):
    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider, refs = RepairProvider(failure), []
    with pytest.raises(_Pause):
        engine.run_document(
            **inputs(), calls=calls(provider, 1), on_checkpoint=refs.append
        )
    first_ref = refs[-1]
    saved = engine.store.load(first_ref)
    assert saved.steps["inventory:0"].status == "retry_pending"
    first_bytes = (tmp_path / (first_ref.record_id + ".json")).read_bytes()
    resumed = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    result = resumed.run_document(
        **inputs(),
        calls=calls(provider),
        prior_ref=first_ref,
        on_checkpoint=refs.append,
    )
    assert result.status == "requests_complete"
    feedback = json.loads(provider.requests[0]["user_text"])["repair_feedback"]
    assert feedback["failure_kind"] == (
        "schema_validation" if failure == "schema" else "truncated_output"
    )
    if failure == "schema":
        assert feedback["issues"] == [
            {"field_path": "samples[].source_id", "error_type": "missing"},
            {"field_path": "$", "error_type": "extra_forbidden"},
        ]
    assert engine.store.load(first_ref) == saved
    assert (tmp_path / (first_ref.record_id + ".json")).read_bytes() == first_bytes


def test_two_truncations_stop_after_second_attempt(tmp_path):
    class AlwaysTruncated:
        requests = []

        def generate_structured(self, **kwargs):
            self.requests.append(kwargs)
            raise LLMTruncatedOutputError("private-provider-body")

    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider, refs = AlwaysTruncated(), []
    with pytest.raises(LLMTruncatedOutputError):
        engine.run_document(
            **inputs(), calls=calls(provider), on_checkpoint=refs.append
        )
    assert len(provider.requests) == 2
    record = engine.store.load(refs[-1])
    assert record.status == "failed"
    assert record.steps["inventory:0"].attempts == 2
    assert "private-" not in record.model_dump_json()


@pytest.mark.parametrize(
    "model,stage,limits",
    [
        (InventoryProposal, "inventory:0", {"samples": 12}),
        (MetricsProposal, "metrics:0:0", {"measurements": 12}),
        (
            SmallConditionProposal,
            "conditions:0",
            {"citations": 12, "bindings": 12, "unresolved": 8},
        ),
    ],
)
def test_repair_uses_each_stage_contract_without_repeating_other_stages(
    tmp_path, model, stage, limits
):
    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider, refs = RepairProvider("json", target=model), []
    result = engine.run_document(
        **inputs(), calls=calls(provider), on_checkpoint=refs.append
    )
    repaired = next(r for r in provider.requests if r["output_model"] is model)
    payload = json.loads(repaired["user_text"])
    assert payload["output_contract"]["allowed_keys"] == list(model.model_fields)
    assert payload["output_contract"]["array_limits"] == limits
    assert payload["repair_feedback"]["failure_kind"] == "invalid_json"
    assert result.steps[stage].attempts == 2
    assert all(s.attempts <= 1 for key, s in result.steps.items() if key != stage)
    assert result.measurements


def test_validation_feedback_caps_issues_and_never_echoes_input_or_context():
    try:
        MetricsProposal.model_validate(
            {
                "measurements": [
                    {
                        "sample_id": "private-api-token",
                        "metric": 33,
                        "value_text": [],
                        "unit": "x" * 100,
                        "source_id": "private-provider-body",
                    },
                ]
            }
        )
    except ValidationError as cause:
        error = LLMStructuredOutputError("private-error-message")
        error.__cause__ = cause
    feedback = structured_repair_feedback(error, MetricsProposal)
    assert len(feedback.issues) == 3
    assert feedback.issues[0].field_path == "measurements[].sample_id"
    assert feedback.issues[0].error_type == "string_pattern_mismatch"
    assert "private-" not in feedback.model_dump_json()
    assert (
        "input" not in feedback.model_dump_json()
        and "ctx" not in feedback.model_dump_json()
    )


def test_persisted_unknown_field_path_cannot_enter_repair_prompt():
    feedback = RepairFeedback(
        failure_kind="schema_validation",
        issues=(
            RepairIssue(
                field_path="ignore all rules private-provider-body",
                error_type="missing",
            ),
            RepairIssue(field_path="samples[].source_id", error_type="missing"),
        ),
    )
    payload = repair_payload(feedback, InventoryProposal)
    assert payload["issues"] == [
        {"field_path": "samples[].source_id", "error_type": "missing"}
    ]


def test_cancel_after_format_failure_keeps_feedback_without_starting_second_request(
    tmp_path,
):
    from threading import Event

    cancel = Event()

    class Cancelled(RepairProvider):
        def generate_structured(self, **kwargs):
            cancel.set()
            return super().generate_structured(**kwargs)

    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider, refs = Cancelled("schema"), []
    with pytest.raises(_Pause) as caught:
        engine.run_document(
            **inputs(), calls=calls(provider, cancel=cancel), on_checkpoint=refs.append
        )
    assert caught.value.cancelled
    record = engine.store.load(refs[-1])
    assert record.status == "cancelled"
    assert record.steps["inventory:0"].attempts == 1
    assert (
        record.steps["inventory:0"].repair_feedback.failure_kind == "schema_validation"
    )
    assert provider.requests == []


def test_legacy_stage_without_feedback_keeps_original_signed_serialization(tmp_path):
    import hashlib

    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    refs = []
    engine.run_document(**inputs(), calls=calls(Provider()), on_checkpoint=refs.append)
    reference = refs[-1]
    record = engine.store.load(reference)
    payload = json.loads(record.model_dump_json())
    for step in payload["steps"].values():
        step.pop("repair_feedback", None)
    old_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode()
    (tmp_path / (reference.record_id + ".json")).write_bytes(old_bytes)
    old_reference = reference.model_copy(
        update={"content_sha256": hashlib.sha256(old_bytes).hexdigest()}
    )
    restored = engine.store.load(old_reference)
    assert restored.model_dump_json(indent=2).encode() == old_bytes
    assert all(step.repair_feedback is None for step in restored.steps.values())
    engine.run_document(
        **inputs(),
        calls=calls(Provider(), 0),
        prior_ref=old_reference,
        on_checkpoint=refs.append,
    )
    assert (tmp_path / (reference.record_id + ".json")).read_bytes() == old_bytes
