"""Offline scope/condition fixtures, actual bounded statistics graph and storage."""

import json
import threading
from types import SimpleNamespace

from materials_screening.master.fulltext_conditions import (
    AttributeProposal,
    CitationProposal,
    ConditionProposal,
    request_condition_plan,
)
from tests.unit.master.test_fulltext_analysis import analysis_env
from tests.unit.master.test_fulltext_preview import run


def conditioned_env(tmp_path):
    env = analysis_env(tmp_path)
    processor = env.processor.extraction_processor.analysis_processor
    processor.condition_planner = request_condition_plan
    env.condition_calls = 0
    prior = env.matrix_llm.generate_structured

    def generate(**kwargs):
        if kwargs["output_model"] is not ConditionProposal:
            return prior(**kwargs)
        env.condition_calls += 1
        payload = json.loads(kwargs["user_text"])
        source = payload["evidence"][0]
        proposal = ConditionProposal(
            citations=(
                CitationProposal(
                    key="s0", chunk_key=source["chunk_key"], quote=source["text"]
                ),
            ),
            bindings=(
                AttributeProposal(
                    measurement_keys=tuple(
                        row["measurement_key"]
                        for row in payload["selected_measurements"]
                    ),
                    kind="condition",
                    key="temperature",
                    value_text="25 C",
                    numeric_value=25,
                    unit="C",
                    value_source="s0",
                    applicability="per_sample",
                    applicability_source="s0",
                ),
            ),
        )
        if getattr(env, "bad_condition", False):
            proposal = proposal.model_copy(
                update={
                    "bindings": (
                        proposal.bindings[0].model_copy(
                            update={"value_text": "30 C", "numeric_value": 30}
                        ),
                    )
                }
            )
        return SimpleNamespace(parsed=proposal)

    env.matrix_llm.generate_structured = generate
    return env


def test_condition_checkpoint_and_new_dataset_keep_original_handoff(tmp_path):
    env = conditioned_env(tmp_path)
    handoff = next(iter(env.task.measurement_handoffs.values()))
    original = env.datasets.load_dataframe(handoff.dataset_id).to_json(orient="records")
    result = run(env, env.task)
    assert result.status == "completed" and result.model_call_count == 5
    record = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
    doc, plan = next(iter(record.condition_plans.items()))
    assert plan.review_status == "pending" and plan.bindings[0].numeric_value == 25
    assert env.condition_calls == 1
    frame = env.datasets.load_dataframe(record.dataset_ids[doc])
    attributes = json.loads(frame.iloc[0]["supplementary_attributes"])
    assert attributes[0]["value_source"]["quote"] == env.store.chunks[0].text
    assert attributes[0]["key"] == "temperature"
    assert frame.iloc[0]["supplementary_review_status"] == "pending"
    assert (
        env.datasets.load_dataframe(handoff.dataset_id).to_json(orient="records")
        == original
    )
    assert (
        "补充条件证据" in result.response_text
        and "不代表条件可比" in result.response_text
    )


def test_budget_saves_conditions_before_statistics_and_resume_does_not_repeat(tmp_path):
    env = conditioned_env(tmp_path)
    result = run(env, env.task, budget=2)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_BUDGET"
    task = env.saved[-1]
    record = env.analysis_store.load(task.fulltext_analysis_ref)
    assert record.condition_plans and not record.analysis_ids
    assert env.condition_calls == 1
    result = run(env, task)
    assert result.status == "completed" and result.model_call_count == 3
    assert env.condition_calls == 1 and env.scope_calls == 1


def test_invalid_condition_blocks_agent_but_preserves_scope_and_extraction(tmp_path):
    env = conditioned_env(tmp_path)
    env.bad_condition = True
    result = run(env, env.task)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_FAILED"
    assert result.tool_call_count == 0
    record = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
    assert (
        record.scope is not None
        and not record.analysis_ids
        and not record.condition_plans
    )
    assert env.saved[-1].extraction_snapshots == env.task.extraction_snapshots


def test_saved_conditions_are_revalidated_with_zero_call_replay(tmp_path):
    env = conditioned_env(tmp_path)
    run(env, env.task)
    task = env.saved[-1]
    result = run(env, task)
    assert result.status == "completed" and result.model_call_count == 0
    assert env.saved[-1].fulltext_analysis_ref == task.fulltext_analysis_ref
    assert env.condition_calls == 1


def test_supplementary_input_tampering_is_not_replayed(tmp_path):
    env = conditioned_env(tmp_path)
    run(env, env.task)
    task = env.saved[-1]
    record = env.analysis_store.load(task.fulltext_analysis_ref)
    # Corrupt only this test's isolated dataset through its canonical path.
    dataset = env.datasets.get(next(iter(record.dataset_ids.values())))
    path = env.datasets.resolve_path(dataset.dataset_id)
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[0]["supplementary_attributes"] = "[]"
    path.write_text(json.dumps(rows), encoding="utf-8")
    result = run(env, task)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_FAILED"
    assert env.condition_calls == 1


def test_saved_condition_value_is_rechecked_not_just_checkpoint_hash(tmp_path):
    env = conditioned_env(tmp_path)
    run(env, env.task, budget=2)
    task = env.saved[-1]
    record = env.analysis_store.load(task.fulltext_analysis_ref)
    doc, plan = next(iter(record.condition_plans.items()))
    binding = plan.bindings[0].model_copy(
        update={"value_text": "30 C", "numeric_value": 30}
    )
    # Construct a validly hashed test checkpoint with an invalid evidence value.
    from uuid import uuid4

    ref = env.analysis_store.save(
        record.model_copy(
            update={
                "record_id": "fulltext-analysis-" + uuid4().hex,
                "parent_record_id": record.record_id,
                "condition_plans": {
                    doc: plan.model_copy(update={"bindings": (binding,)})
                },
            }
        )
    )
    result = run(env, task.model_copy(update={"fulltext_analysis_ref": ref}))
    assert result.error["code"] == "FULLTEXT_ANALYSIS_FAILED"
    assert result.model_call_count == 0 and result.tool_call_count == 0
    assert env.condition_calls == 1


def test_cancellation_after_condition_checkpoint_starts_no_agent(tmp_path):
    env = conditioned_env(tmp_path)
    cancel = threading.Event()
    prior = env.matrix_llm.generate_structured

    def generate(**kwargs):
        result = prior(**kwargs)
        if kwargs["output_model"] is ConditionProposal:
            cancel.set()
        return result

    env.matrix_llm.generate_structured = generate
    result = run(env, env.task, cancel=cancel)
    assert result.status == "cancelled" and result.tool_call_count == 0
    record = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
    assert record.condition_plans and not record.analysis_ids


def test_normal_factory_does_not_enable_unqualified_condition_provider(tmp_path):
    from materials_screening.master.artifact_registry import ArtifactRegistry
    from materials_screening.sub_agents.literature.fulltext_factory import (
        create_fulltext_preview_processor,
    )

    processor = create_fulltext_preview_processor(
        ArtifactRegistry(tmp_path / "artifacts"),
        enable_remote=True,
        analysis_agent_factory=lambda: None,
    )
    assert processor.extraction_processor.analysis_processor.condition_planner is None
