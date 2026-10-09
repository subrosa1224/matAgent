"""Actual checked PDF/statistics graph; only LLM location/value proposals mocked."""

import threading
from types import SimpleNamespace

from materials_screening.master.fulltext_condition_locator import (
    LiteralValues,
    SentenceLocations,
    locate_shared_conditions,
    read_located_values,
)
from tests.unit.master.test_fulltext_analysis import analysis_env
from tests.unit.master.test_fulltext_preview import run


def environment(tmp_path):
    env = analysis_env(
        tmp_path,
        body=(
            "Methods\nAll the samples were tested in water at 25 C.\n"
            "Sample A at 25 C had strength 12.0 MPa.\nResults and Discussion\n"
        ),
    )
    processor = env.processor.extraction_processor.analysis_processor
    processor.condition_locator = locate_shared_conditions
    processor.condition_reader = read_located_values
    prior = env.matrix_llm.generate_structured
    env.locate_calls = env.value_calls = 0

    def generate(**kwargs):
        if kwargs["output_model"] is SentenceLocations:
            env.locate_calls += 1
            if getattr(env, "cancel_after_location", None) is not None:
                env.cancel_after_location.set()
            return SimpleNamespace(
                parsed=SentenceLocations(
                    shared_scope="s0", temperature="s0", test_medium="s0"
                )
            )
        if kwargs["output_model"] is LiteralValues:
            env.value_calls += 1
            return SimpleNamespace(
                parsed=LiteralValues(
                    temperature="25 C", test_medium="water", irradiation_time="1 h"
                )
            )
        return prior(**kwargs)

    env.matrix_llm.generate_structured = generate
    return env


def test_budget_between_location_and_values_saves_location_then_resumes(tmp_path):
    env = environment(tmp_path)
    result = run(env, env.task, budget=2)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_BUDGET"
    first = env.saved[-1]
    record = env.analysis_store.load(first.fulltext_analysis_ref)
    assert record.condition_locations and not record.condition_plans
    assert env.locate_calls == 1 and env.value_calls == 0
    result = run(env, first)
    assert result.status == "completed" and result.model_call_count == 4
    record = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
    assert env.locate_calls == env.value_calls == 1
    assert record.condition_plans and record.analysis_ids
    assert next(iter(record.condition_plans.values())).isolated_attributes
    assert "隔离的补充属性" in result.response_text


def test_two_stage_analysis_uses_six_shared_calls_and_zero_replay(tmp_path):
    env = environment(tmp_path)
    result = run(env, env.task, budget=6)
    assert result.status == "completed" and result.model_call_count == 6
    task = env.saved[-1]
    result = run(env, task)
    assert result.status == "completed" and result.model_call_count == 0
    assert env.locate_calls == env.value_calls == 1


def test_cancel_after_location_keeps_it_but_starts_no_value_or_agent(tmp_path):
    env = environment(tmp_path)
    event = threading.Event()
    env.cancel_after_location = event
    result = run(env, env.task, cancel=event)
    assert result.status == "cancelled" and result.tool_call_count == 0
    record = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
    assert record.condition_locations and not record.condition_plans
    assert env.value_calls == 0


def test_normal_factory_exposes_opt_in_two_stage_gate(tmp_path, monkeypatch):
    from materials_screening.master.artifact_registry import ArtifactRegistry
    from materials_screening.sub_agents.literature.fulltext_factory import (
        create_fulltext_preview_processor,
    )

    monkeypatch.setenv("MASTER_SHARED_TEST_CONDITIONS", "true")
    processor = create_fulltext_preview_processor(
        ArtifactRegistry(tmp_path / "artifacts"),
        enable_remote=True,
        analysis_agent_factory=lambda: None,
    ).extraction_processor.analysis_processor
    assert processor.condition_locator is locate_shared_conditions
    assert processor.condition_reader is read_located_values
    assert processor.condition_planner is None


def test_dense_annotations_use_bounded_references_not_repeated_full_citations():
    import json

    from materials_screening.master.fulltext_analysis import _supplementary_attributes
    from materials_screening.master.fulltext_conditions import (
        AttributeBinding,
        ConditionPlan,
        SourceSpan,
    )

    quote = "All the samples were tested in water. " + "Source sentence. " * 90
    span = SourceSpan(
        chunk_id="chunk-unit", text_sha256="a" * 64, page_from=3, page_to=3, quote=quote
    )
    bindings = tuple(
        AttributeBinding(
            measurement_ids=("measurement-unit",),
            kind="condition",
            key=f"field-{index}",
            value_text="water",
            value_source=span,
            applicability="reported_common_protocol",
            applicability_source=span,
        )
        for index in range(10)
    )
    plan = ConditionPlan(bindings=bindings)
    assert len(json.dumps([row.model_dump(mode="json") for row in bindings])) > 10000
    packed = _supplementary_attributes(plan, "measurement-unit", "refs-v2")
    assert len(packed) < 10000
    assert all(
        "binding_sha256" in row and "value_source_sha256" in row
        for row in json.loads(packed)
    )
    assert plan.bindings[0].value_source.quote == quote


def test_representation_only_revision_preserves_validated_stage_and_old_record(
    tmp_path,
):
    from uuid import uuid4

    env = environment(tmp_path)
    run(env, env.task, budget=2)
    first = env.saved[-1]
    record = env.analysis_store.load(first.fulltext_analysis_ref)
    doc, locations = next(iter(record.condition_locations.items()))
    import json

    rows = json.loads(
        env.datasets.load_dataframe(first.measurement_handoffs[doc].dataset_id).to_json(
            orient="records"
        )
    )
    plan = read_located_values(
        calls=SimpleNamespace(
            generate_structured=lambda **kwargs: SimpleNamespace(
                parsed=LiteralValues(temperature="25 C")
            )
        ),
        locations=locations,
        candidates={row["measurement_id"]: row for row in rows},
        chunks=env.store.chunks,
    )
    old = record.model_copy(
        update={
            "record_id": "fulltext-analysis-" + uuid4().hex,
            "parent_record_id": record.record_id,
            "annotation_format": "inline-v1",
            "condition_plans": {doc: plan},
        }
    )
    old_ref = env.analysis_store.save(old)
    result = run(env, first.model_copy(update={"fulltext_analysis_ref": old_ref}))
    assert result.status == "completed" and result.model_call_count == 3
    new = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
    assert (
        new.annotation_format == "refs-v2"
        and new.condition_plans == old.condition_plans
    )
    assert env.analysis_store.load(old_ref) == old
    assert env.locate_calls == 1 and env.value_calls == 0
