import time
from dataclasses import replace
from threading import Event

import pytest

from materials_screening.llm.base import StructuredProviderResponse
from materials_screening.master.fulltext_preview import _Pause, _PreviewCalls
from materials_screening.master.staged_fulltext_engine import StagedFulltextEngine
from materials_screening.master.staged_fulltext_evidence import (
    InventoryProposal,
    MetricProposal,
    MetricsProposal,
    SampleProposal,
)
from materials_screening.master.staged_fulltext_store import StagedExtractionStore
from tests.unit.master.test_staged_fulltext_evidence import source


def inputs():
    document = "doc-" + "a" * 24
    chunks = tuple(
        replace(c, document_id=document)
        for c in (
            source("Sample NTO3 is the Nb-doped SnO2 film with 1.5 wt% Nb."),
            source(
                "The maximum average transmittance of around 82% was observed "
                "in NTO3 film at UV-visible region.",
                "optics",
            ),
        )
    )
    return dict(
        task_id="task-fulltext-" + "b" * 32,
        conversation_id="unit",
        document_id=document,
        pdf_sha256="a" * 64,
        title="Unit paper",
        requirements="提取透过率及电阻率、掺杂和制备条件",
        chunks=chunks,
        requested_metrics=("transmittance", "resistivity"),
    )


def test_metric_prompt_explicitly_separates_literal_exponent_and_physical_unit():
    from materials_screening.master.staged_fulltext_engine import _METRIC_PROMPT

    assert "value_text" in _METRIC_PROMPT and "unit" in _METRIC_PROMPT
    assert "16.130 × 10−4" in _METRIC_PROMPT


def test_inventory_hints_are_literal_own_descriptors_not_verified_aliases():
    from types import SimpleNamespace

    from materials_screening.master.staged_fulltext_engine import _inventory_payload
    from materials_screening.master.staged_fulltext_evidence import build_catalogue
    from tests.unit.master.test_staged_fulltext_evidence import _MIXED_CONTROL_RESULT

    quote = _MIXED_CONTROL_RESULT.replace("comparable", "compara\nble")
    cat = build_catalogue(
        (
            source(quote),
            source("Previous reports found pure ZnO transmittance of 90%.", "prior"),
        )
    )
    payload = _inventory_payload(cat, (SimpleNamespace(label="pure SnO2"),))
    assert payload["known_samples"] == ["pure SnO2"]
    assert payload["unreviewed_literal_descriptors"] == ["undoped SnO2"]
    assert len(payload["evidence"]) == 1
    assert payload["evidence"]["s0"] in quote
    assert "pure SnO2" not in payload["evidence"]["s0"]
    assert "sample_id" not in payload  # Hints cannot become verified samples.


class Provider:
    def __init__(self):
        self.requests = []
        self.cancel = None

    def generate_structured(self, **kwargs):
        import json

        self.requests.append(kwargs)
        payload = json.loads(kwargs["user_text"])
        if kwargs["output_model"] is InventoryProposal:
            key = next(k for k, v in payload["evidence"].items() if "NTO3" in v)
            out = InventoryProposal(
                samples=(SampleProposal(label="NTO3", source_id=key),)
            )
        elif kwargs["output_model"] is MetricsProposal:
            key = next(k for k, v in payload["evidence"].items() if "82%" in v)
            out = MetricsProposal(
                measurements=(
                    MetricProposal(
                        sample_id=payload["samples"][0]["sample_id"],
                        metric="transmittance",
                        value_text="82",
                        unit="%",
                        source_id=key,
                    ),
                )
            )
        else:
            from materials_screening.master.fulltext_conditions import ConditionProposal

            out = ConditionProposal()
        if self.cancel:
            self.cancel.set()
        return StructuredProviderResponse(
            parsed=out,
            provider="offline",
            model="unit",
            request_id=None,
            latency_ms=0,
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            raw_output_sha256=None,
        )


def calls(provider, budget=100, cancel=None):
    return _PreviewCalls(lambda: provider, budget, cancel, time.monotonic() + 60)


def test_independent_stage_storage_resume_skips_confirmed_inventory(tmp_path):
    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider = Provider()
    refs = []
    with pytest.raises(_Pause):
        engine.run_document(
            **inputs(), calls=calls(provider, 1), on_checkpoint=refs.append
        )
    first_ref = refs[-1]
    first = engine.store.load(first_ref)
    assert first.samples and not first.measurements
    result = engine.run_document(
        **inputs(), calls=calls(provider), prior_ref=refs[-1], on_checkpoint=refs.append
    )
    assert result.measurements[0].value_text == "around 82"
    assert sum(r["output_model"] is InventoryProposal for r in provider.requests) == 1
    assert engine.store.load(first_ref) == first
    assert result.coverage["resistivity"] == "not_found_current_evidence"


def test_cancel_after_reply_saves_inventory_and_never_starts_metric_request(tmp_path):
    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider, cancel, refs = Provider(), Event(), []
    provider.cancel = cancel
    with pytest.raises(_Pause) as caught:
        engine.run_document(
            **inputs(), calls=calls(provider, cancel=cancel), on_checkpoint=refs.append
        )
    assert caught.value.cancelled
    assert engine.store.load(refs[-1]).samples
    assert len(provider.requests) == 1


def test_cached_sources_and_bytes_revalidated_without_requests(tmp_path):
    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider, refs = Provider(), []
    engine.run_document(**inputs(), calls=calls(provider), on_checkpoint=refs.append)
    count = len(provider.requests)
    value = inputs()
    value["chunks"] = (replace(value["chunks"][0], text="changed"), value["chunks"][1])
    with pytest.raises(ValueError):
        engine.run_document(
            **value,
            calls=calls(provider),
            prior_ref=refs[-1],
            on_checkpoint=refs.append,
        )
    assert len(provider.requests) == count
    path = tmp_path / (refs[-1].record_id + ".json")
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        engine.store.load(refs[-1])


def test_model_failure_saved_as_failure_not_budget_and_bounded_to_two(tmp_path):
    from materials_screening.llm.errors import LLMStructuredOutputError

    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )

    class Failing:
        count = 0

        def generate_structured(self, **kwargs):
            self.count += 1
            raise LLMStructuredOutputError("invalid JSON")

    provider, refs = Failing(), []
    with pytest.raises(LLMStructuredOutputError):
        engine.run_document(
            **inputs(), calls=calls(provider), on_checkpoint=refs.append
        )
    assert provider.count == 2
    assert engine.store.load(refs[-1]).status == "failed"


def test_oversized_or_flat_table_explicitly_pending_not_submitted_whole(tmp_path):
    args = inputs()
    c = replace(
        source(
            "Table 1\nParameter\nNTO3\nNTO4\nResistivity (Ω cm)\n"
            "6.45 \x02 10 4\n9.0e-4",
            "table",
        ),
        document_id=args["document_id"],
    )
    args["chunks"] += (c,)
    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider, refs = Provider(), []
    result = engine.run_document(
        **args, calls=calls(provider), on_checkpoint=refs.append
    )
    assert result.coverage["resistivity"] == "pending_verification"
    assert any(r.reason == "table_layout_unverified" for r in result.unresolved)
    assert not any(
        "6.45" in r["user_text"]
        for r in provider.requests
        if r["output_model"] is MetricsProposal
    )


def test_new_policy_keeps_history_but_does_not_reuse_cached_facts(tmp_path):
    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider, refs = Provider(), []
    engine.run_document(**inputs(), calls=calls(provider), on_checkpoint=refs.append)
    first = refs[-1]
    args = inputs()
    args["requirements"] += "；补充核对"
    result = engine.run_document(
        **args, calls=calls(provider), prior_ref=first, on_checkpoint=refs.append
    )
    assert result.policy_sha256 != engine.store.load(first).policy_sha256
    assert sum(r["output_model"] is InventoryProposal for r in provider.requests) == 2


def test_schema_attempt_count_survives_a_budget_boundary(tmp_path):
    from materials_screening.llm.errors import LLMStructuredOutputError

    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )

    class Failing:
        count = 0

        def generate_structured(self, **kwargs):
            self.count += 1
            raise LLMStructuredOutputError("schema")

    provider, refs = Failing(), []
    with pytest.raises(_Pause):
        engine.run_document(
            **inputs(), calls=calls(provider, 1), on_checkpoint=refs.append
        )
    assert engine.store.load(refs[-1]).steps["inventory:0"].attempts == 1
    with pytest.raises(LLMStructuredOutputError):
        engine.run_document(
            **inputs(),
            calls=calls(provider, 1),
            prior_ref=refs[-1],
            on_checkpoint=refs.append,
        )
    assert provider.count == 2
    assert engine.store.load(refs[-1]).status == "failed"


def test_transport_failure_persists_step_and_safe_cause_without_retry(tmp_path):
    from materials_screening.llm.errors import LLMTimeoutError
    from materials_screening.master.fulltext_preview import _ProviderFailure

    class Timeout:
        count = 0

        def generate_structured(self, **kwargs):
            self.count += 1
            raise LLMTimeoutError("secret request or provider details")

    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider, refs = Timeout(), []
    with pytest.raises(_ProviderFailure):
        engine.run_document(
            **inputs(), calls=calls(provider), on_checkpoint=refs.append
        )
    record = engine.store.load(refs[-1])
    assert record.status == "failed"
    failed = record.steps["inventory:0"]
    assert failed.status == "failed" and failed.attempts == 1
    assert failed.error_kind == "LLMTimeoutError"
    assert provider.count == 1
    assert "secret" not in record.model_dump_json()


def test_rejected_metric_retains_literal_candidate_and_specific_reason(tmp_path):
    class BadMetric(Provider):
        def generate_structured(self, **kwargs):
            reply = super().generate_structured(**kwargs)
            if kwargs["output_model"] is MetricsProposal:
                measurement = reply.parsed.measurements[0].model_copy(
                    update={"value_text": "99"}
                )
                reply = reply.model_copy(
                    update={"parsed": MetricsProposal(measurements=(measurement,))}
                )
            return reply

    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider, refs = BadMetric(), []
    record = engine.run_document(
        **inputs(), calls=calls(provider), on_checkpoint=refs.append
    )
    step = record.steps["metrics:0:0"]
    assert step.rejected == 1 and not record.measurements
    detail = step.rejected_details[0]
    assert detail.candidate["value_text"] == "99"
    assert "literal" in detail.reason
    assert record.coverage["transmittance"] == "pending_verification"


def test_metric_requests_exclude_generic_abstract_with_competing_values(tmp_path):
    args = inputs()
    abstract = replace(
        source("Films exhibited transmittance of 82% and low resistivity.", "abstract"),
        document_id=args["document_id"],
    )
    args["chunks"] = (abstract, *args["chunks"])
    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider, refs = Provider(), []
    record = engine.run_document(
        **args, calls=calls(provider), on_checkpoint=refs.append
    )
    requests = [r for r in provider.requests if r["output_model"] is MetricsProposal]
    assert requests and all("Films exhibited" not in r["user_text"] for r in requests)
    assert record.measurements[0].source.chunk_id == "optics"
    assert any(u.reason == "sample_binding_unverified" for u in record.unresolved)


def test_conditions_do_not_submit_reference_list_as_experimental_protocol(tmp_path):
    from materials_screening.master.staged_fulltext_engine import SmallConditionProposal

    args = inputs()
    references = replace(
        source(
            "References\n[1] Example, Doped SnO2 deposition temperature study, "
            "Journal 10 (2020) 1-9.",
            "references",
        ),
        document_id=args["document_id"],
        page_from=9,
        page_to=9,
    )
    args["chunks"] += (references,)
    provider, refs = Provider(), []
    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    result = engine.run_document(
        **args, calls=calls(provider), on_checkpoint=refs.append
    )
    requests = [
        r for r in provider.requests if r["output_model"] is SmallConditionProposal
    ]
    assert requests and all("Journal 10" not in r["user_text"] for r in requests)
    assert result.measurements and result.status == "requests_complete"


@pytest.mark.parametrize("include_identity", [False, True])
def test_condition_plan_includes_the_verified_measurement_anchor_without_keyword(
    tmp_path,
    include_identity,
):
    import json

    from materials_screening.master.staged_fulltext_engine import SmallConditionProposal

    args = inputs()
    args["chunks"] = (
        args["chunks"][0],
        replace(
            source(
                "The maximum average transmittance of around 82% was observed "
                "in NTO3 film at 550 nm.",
                "optics",
            ),
            document_id=args["document_id"],
        ),
    )

    class Wavelength(Provider):
        def generate_structured(self, **kwargs):
            if kwargs["output_model"] is not SmallConditionProposal:
                return super().generate_structured(**kwargs)
            self.requests.append(kwargs)
            payload = json.loads(kwargs["user_text"])
            anchor = next(
                (c for c in payload["evidence"] if "550 nm" in c["text"]), None
            )
            out = SmallConditionProposal()
            if anchor:
                out = SmallConditionProposal(
                    citations=[
                        dict(
                            key="s0",
                            chunk_key=anchor["chunk_key"],
                            quote=anchor["text"],
                        )
                    ],
                    bindings=[
                        dict(
                            measurement_keys=["m0"],
                            kind="condition",
                            key="transmittance_wavelength",
                            value_text="550 nm",
                            unit="nm",
                            numeric_value=550,
                            value_source="s0",
                            applicability="per_sample",
                            applicability_source="s0",
                        )
                    ],
                )
                if include_identity:
                    identity = dict(
                        out.bindings[0],
                        key="sample_type",
                        value_text="NTO3",
                        numeric_value=None,
                        unit=None,
                    )
                    out = out.model_copy(update={"bindings": (*out.bindings, identity)})
            return StructuredProviderResponse(
                parsed=out,
                provider="offline",
                model="unit",
                request_id=None,
                latency_ms=0,
                input_tokens=1,
                output_tokens=1,
                reasoning_tokens=0,
                raw_output_sha256=None,
            )

    refs = []
    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    record = engine.run_document(
        **args, calls=calls(Wavelength()), on_checkpoint=refs.append
    )
    assert record.measurements[0].value_text == "around 82"
    assert any(
        b.value_text == "550 nm" for p in record.condition_plans for b in p.bindings
    )
    assert not any(
        b.key == "sample_type" for p in record.condition_plans for b in p.bindings
    )
    if include_identity:
        assert any(
            r.candidate.get("key") == "sample_type" and "identity" in r.reason
            for s in record.steps.values()
            for r in s.rejected_details
        )
    from materials_screening.data_analysis.dataset_store import DatasetStore
    from materials_screening.master.staged_fulltext_handoff import (
        render_staged_result,
        staged_trial_handoff,
    )

    datasets = DatasetStore(tmp_path / "datasets")
    handoff = staged_trial_handoff(
        record,
        reference=refs[-1],
        chunks=args["chunks"],
        dataset_factory=lambda: datasets,
    )
    row = datasets.load_dataframe(handoff.dataset_id).iloc[0]
    assert row["numeric_value"] == 82
    assert json.loads(row["conditions"]) == {"transmittance_wavelength": "550 nm"}
    text = render_staged_result(record, handoff)
    assert "透过率测试波长 = 550 nm" in text
    assert "无法直接排名" in text


@pytest.mark.parametrize(
    "damage",
    [
        "malformed_reference",
        "duplicate_citation",
        "duplicate_unknown",
        "invented_quote",
    ],
)
def test_bad_condition_candidates_are_isolated_and_never_approved(tmp_path, damage):
    import json

    from materials_screening.master.staged_fulltext_engine import SmallConditionProposal

    class Conditions(Provider):
        def generate_structured(self, **kwargs):
            if kwargs["output_model"] is not SmallConditionProposal:
                return super().generate_structured(**kwargs)
            self.requests.append(kwargs)
            payload = json.loads(kwargs["user_text"])
            key = next(
                c["chunk_key"] for c in payload["evidence"] if "1.5 wt%" in c["text"]
            )
            quote = next(
                c["text"] for c in payload["evidence"] if c["chunk_key"] == key
            )
            good = dict(
                measurement_keys=["m0"],
                kind="preparation",
                key="dopant_concentration",
                value_text="1.5 wt%",
                unit="wt%",
                numeric_value=1.5,
                value_source="s0",
                applicability="per_sample",
                applicability_source="s0",
            )
            citations = [
                dict(key="s0", chunk_key=key, quote=quote),
                dict(key="s2", chunk_key="c999", quote="Unrelated bad citation"),
            ]
            if damage == "duplicate_citation":
                citations.append(dict(key="s0", chunk_key=key, quote=quote))
            elif damage == "duplicate_unknown":
                citations.append(dict(key="s0", chunk_key="c999", quote=quote))
            elif damage == "invented_quote":
                citations[0]["quote"] = "Sample NTO3 contains invented 1.5 wt% Nb."
            out = SmallConditionProposal(
                citations=citations,
                bindings=[good, {**good, "key": "bad", "value_source": "s0,s1"}],
            )
            return StructuredProviderResponse(
                parsed=out,
                provider="offline",
                model="unit",
                request_id=None,
                latency_ms=0,
                input_tokens=1,
                output_tokens=1,
                reasoning_tokens=0,
                raw_output_sha256=None,
            )

    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider, refs = Conditions(), []
    record = engine.run_document(
        **inputs(), calls=calls(provider), on_checkpoint=refs.append
    )
    step = record.steps["conditions:0"]
    assert step.status == "done"
    if damage == "malformed_reference":
        assert len(step.conditions.bindings) == 1
        assert step.conditions.bindings[0].value_text == "1.5 wt%"
        assert step.rejected == 2
    else:
        assert not step.conditions.bindings
        assert step.rejected >= 3
    assert len(step.rejected_details) == step.rejected
    assert record.measurements and record.status == "requests_complete"


@pytest.mark.parametrize(
    "history", ["valid", "newly_eligible_rejection", "bare_formula"]
)
def test_v5_inventory_migration_replays_identity_and_re_requests_changed_steps(
    tmp_path, history
):
    import hashlib
    import json
    from uuid import uuid4

    from materials_screening.master.staged_fulltext_engine import (
        _PREVIOUS_INVENTORY,
        _PREVIOUS_METRIC_PROMPT,
        CONDITION_PROMPT,
    )
    from materials_screening.master.staged_fulltext_evidence import build_catalogue
    from materials_screening.master.staged_fulltext_store import RejectedCandidate

    args = inputs()
    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider, refs = Provider(), []
    with pytest.raises(_Pause):
        engine.run_document(**args, calls=calls(provider, 1), on_checkpoint=refs.append)
    saved = engine.store.load(refs[-1])
    step = saved.steps["inventory:0"]
    catalogue = build_catalogue(args["chunks"])
    source_key = next(k for k, v in catalogue.items() if "1.5 wt%" in v.quote)
    if history == "newly_eligible_rejection":
        # Add an actually literal descriptor to the old rejection list. It must
        # cause a fresh request, never be promoted automatically to evidence.
        args["chunks"] = (
            replace(
                source("Sample NTO3 is the pure SnO2 film."),
                document_id=args["document_id"],
            ),
            args["chunks"][1],
        )
        provider, refs = Provider(), []
        with pytest.raises(_Pause):
            engine.run_document(
                **args, calls=calls(provider, 1), on_checkpoint=refs.append
            )
        saved = engine.store.load(refs[-1])
        step = saved.steps["inventory:0"].model_copy(
            update={
                "rejected": 1,
                "rejected_details": (
                    RejectedCandidate(
                        candidate={"label": "pure SnO2", "source_id": source_key},
                        reason="old generic descriptor rejection",
                    ),
                ),
            }
        )
    elif history == "bare_formula":
        step = step.model_copy(
            update={"samples": (step.samples[0].model_copy(update={"label": "SnO2"}),)}
        )
    base = dict(
        version="master-staged-extraction-v1",
        requirements=args["requirements"],
        requested_metrics=args["requested_metrics"],
        model_profile="offline/unit",
        inventory_prompt=_PREVIOUS_INVENTORY,
        metric_prompt=_PREVIOUS_METRIC_PROMPT,
        condition_prompt=CONDITION_PROMPT,
        output_tokens=4096,
        source_chars=3000,
        candidates=12,
        validator_policy="literal-composite-v5",
    )
    legacy = saved.model_copy(
        update={
            "record_id": "staged-" + uuid4().hex,
            "policy_sha256": hashlib.sha256(
                json.dumps(base, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
            "steps": {"inventory:0": step},
            "samples": step.samples,
        }
    )
    prior = engine.store.save(legacy)
    provider.requests.clear()
    result = engine.run_document(
        **args, calls=calls(provider), prior_ref=prior, on_checkpoint=refs.append
    )
    requests = [r for r in provider.requests if r["output_model"] is InventoryProposal]
    assert len(requests) == (0 if history == "valid" else 1)
    assert all(s.label == "NTO3" for s in result.samples)
    assert engine.store.load(prior) == legacy


@pytest.mark.parametrize("mixed_comparison", [False, True])
@pytest.mark.parametrize("version", [8, 11, 12])
def test_prior_policy_migration_keeps_safe_inventory_but_reasks_mixed_attribution_pack(
    tmp_path, mixed_comparison, version
):
    import hashlib
    import json
    from uuid import uuid4

    from materials_screening.master.staged_fulltext_engine import (
        _CONDITION_PROMPT,
        _INVENTORY,
        _METRIC_PROMPT,
        _PREVIOUS_METRIC_PROMPT,
        _PREVIOUS_SPECIFIC_INVENTORY,
    )
    from tests.unit.master.test_staged_fulltext_evidence import _MIXED_CONTROL_RESULT

    args = inputs()
    if mixed_comparison:
        args["chunks"] += (
            replace(
                source(_MIXED_CONTROL_RESULT, "mixed"), document_id=args["document_id"]
            ),
        )
    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    provider, refs = Provider(), []
    old = engine.run_document(**args, calls=calls(provider), on_checkpoint=refs.append)
    if version == 12:
        from materials_screening.master.fulltext_conditions import (
            AttributeBinding,
            ConditionPlan,
        )

        metric = old.measurements[0]
        mistaken = ConditionPlan(
            bindings=(
                AttributeBinding(
                    measurement_ids=(metric.measurement_id,),
                    kind="condition",
                    key="sample_type",
                    value_text="NTO3",
                    value_source=metric.source,
                    applicability="per_sample",
                    applicability_source=metric.source,
                ),
            )
        )
        old = old.model_copy(
            update={
                "condition_plans": (mistaken,),
                "steps": {
                    **old.steps,
                    "conditions:0": old.steps["conditions:0"].model_copy(
                        update={"conditions": mistaken}
                    ),
                },
            }
        )
    policy = dict(
        version="master-staged-extraction-v1",
        requirements=args["requirements"],
        requested_metrics=args["requested_metrics"],
        model_profile="offline/unit",
        inventory_prompt=_PREVIOUS_SPECIFIC_INVENTORY if version == 8 else _INVENTORY,
        metric_prompt=_PREVIOUS_METRIC_PROMPT if version == 8 else _METRIC_PROMPT,
        condition_prompt=_CONDITION_PROMPT,
        output_tokens=4096,
        source_chars=3000,
        candidates=12,
        validator_policy=f"literal-composite-v{version}",
    )
    old = old.model_copy(
        update={
            "record_id": "staged-" + uuid4().hex,
            "policy_sha256": hashlib.sha256(
                json.dumps(policy, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
        }
    )
    # Round-trip the synthetic historical row just as the actual JSON cache is
    # read: AttributeBinding uses SourceSpan, not its catalogue-only subclass.
    old = type(old).model_validate_json(old.model_dump_json())
    prior = engine.store.save(old)
    provider.requests.clear()
    current = engine.run_document(
        **args, calls=calls(provider), prior_ref=prior, on_checkpoint=refs.append
    )
    assert sum(
        r["output_model"] is InventoryProposal for r in provider.requests
    ) == int(mixed_comparison)
    assert sum(r["output_model"] is MetricsProposal for r in provider.requests) == 1
    assert not any(
        b.key == "sample_type" for p in current.condition_plans for b in p.bindings
    )
    assert (
        current.parent_record_id != old.record_id
    )  # append-only intermediate checkpoints
    assert engine.store.load(prior) == old


def test_prior_report_gap_is_saved_without_aborting_current_paper(tmp_path):
    args = inputs()
    args["chunks"] += (
        replace(
            source("Previous reports found pure SnO2 transmittance of 99%.", "prior"),
            document_id=args["document_id"],
        ),
    )
    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path), model_profile="offline/unit"
    )
    refs = []
    record = engine.run_document(
        **args, calls=calls(Provider()), on_checkpoint=refs.append
    )
    assert record.status == "requests_complete" and record.measurements
    assert any(u.reason == "prior_report_scope_unverified" for u in record.unresolved)
    assert all("Previous reports" not in m.source.quote for m in record.measurements)
    assert engine.store.load(refs[-1]) == record
