"""Local proposals with actual checked PDF and real statistics tool graph."""

import json
import threading
from types import SimpleNamespace

import pytest

from materials_screening.master.fulltext_targeted_evidence import (
    SampleSources,
    TargetedEvidencePlan,
    TargetedProposal,
    request_targeted_evidence,
)
from tests.unit.master.test_fulltext_preview import run
from tests.unit.master.test_fulltext_two_stage_continuation import environment


def test_targeted_stage_checkpoints_before_statistics_and_is_not_repeated(tmp_path):
    env = environment(tmp_path)
    processor = env.processor.extraction_processor.analysis_processor
    calls = []

    def targeted(**kwargs):
        calls.append(True)
        return TargetedEvidencePlan()

    processor.targeted_planner = targeted
    first_result = run(env, env.task, budget=2)
    first = env.saved[-1]
    assert first_result.error["code"] == "FULLTEXT_ANALYSIS_BUDGET"
    run(env, first, budget=1)
    second = env.saved[-1]
    record = env.analysis_store.load(second.fulltext_analysis_ref)
    assert record.targeted_plans and not record.analysis_ids
    assert calls == [True]
    result = run(env, second, budget=3)
    assert result.status == "completed" and calls == [True]
    record = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
    assert "逐样品时间与报告比例" in record.report_markdown
    assert not record.targeted_plans[
        next(iter(record.targeted_plans))
    ].attributes.bindings


def test_normal_factory_targeted_feature_requires_explicit_flag(monkeypatch, tmp_path):
    from materials_screening.master.artifact_registry import ArtifactRegistry
    from materials_screening.sub_agents.literature.fulltext_factory import (
        create_fulltext_preview_processor,
    )

    monkeypatch.setenv("MASTER_SHARED_TEST_CONDITIONS", "true")
    monkeypatch.delenv("MASTER_TARGETED_TEST_EVIDENCE", raising=False)
    args = dict(enable_remote=True, analysis_agent_factory=lambda: None)
    artifacts = ArtifactRegistry(root=tmp_path / "artifacts")
    preview = create_fulltext_preview_processor(artifacts, **args)
    assert preview.extraction_processor.analysis_processor.targeted_planner is None
    monkeypatch.setenv("MASTER_TARGETED_TEST_EVIDENCE", "true")
    preview = create_fulltext_preview_processor(artifacts, **args)
    assert (
        preview.extraction_processor.analysis_processor.targeted_planner
        is request_targeted_evidence
    )


def bound_environment(tmp_path):
    from materials_screening.master.fulltext_condition_locator import (
        locate_shared_conditions,
        read_located_values,
    )
    from tests.unit.master.test_fulltext_analysis import analysis_env

    def configure(env):
        env.scope_metric = "Degradation"
        env.scope_quote = "全文提取降解率。"
        from materials_screening.sub_agents.literature.matrix_automation import (
            MatrixExtractionBatch,
        )

        original = env.matrix_llm.generate_structured

        def generate(**kwargs):
            response = original(**kwargs)
            if kwargs["output_model"] is not MatrixExtractionBatch:
                return response
            batch = response.parsed
            groups = tuple(
                group.model_copy(
                    update={"label": "7% BiVO4/g-C3N4", "material": "7% BiVO4/g-C3N4"}
                )
                for group in batch.groups
            )
            measurements = tuple(
                row.model_copy(update={"metric": "Degradation", "unit": "%"})
                for row in batch.measurements
            )
            return response.model_copy(
                update={
                    "parsed": batch.model_copy(
                        update={"groups": groups, "measurements": measurements}
                    )
                }
            )

        env.matrix_llm.generate_structured = generate

    env = analysis_env(
        tmp_path,
        body=(
            "Preparation of BiVO4/g-C3N4 composites\n"
            "7% BiVO4/g-C3N4 was prepared by adjusting the mass ratios.\n"
            "Table 1. Degradation\n7% BiVO4/g-C3N4\n12.0 %\n"
            "The degradation of 7% BiVO4/g-C3N4 under visible light "
            "for 4 h was 12.0 %.\n"
        ),
        configure=configure,
    )
    processor = env.processor.extraction_processor.analysis_processor
    processor.condition_locator = locate_shared_conditions
    processor.condition_reader = read_located_values
    processor.targeted_planner = request_targeted_evidence
    original = env.matrix_llm.generate_structured
    env.targeted_calls = 0

    def generate(**kwargs):
        if kwargs["output_model"] is not TargetedProposal:
            return original(**kwargs)
        env.targeted_calls += 1
        evidence = json.loads(kwargs["user_text"])["evidence"]

        def key(kind):
            return next(key for key, row in evidence.items() if row["kind"] == kind)

        if getattr(env, "cancel_after_targeted", None) is not None:
            env.cancel_after_targeted.set()
        return SimpleNamespace(
            parsed=TargetedProposal(
                samples={
                    "m0": SampleSources(
                        time_source=key("time"),
                        sample_source=key("sample"),
                        preparation_source=key("preparation"),
                        preparation_context=key("preparation_context"),
                    )
                }
            )
        )

    env.matrix_llm.generate_structured = generate
    return env


def test_bound_evidence_annotations_are_pending_cached_and_do_not_merge_conditions(
    tmp_path,
):
    env = bound_environment(tmp_path)
    result = run(env, env.task, budget=2)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_BUDGET"
    first = env.saved[-1]
    record = env.analysis_store.load(first.fulltext_analysis_ref)
    assert len(next(iter(record.targeted_plans.values())).attributes.bindings) == 2
    assert env.targeted_calls == 1
    result = run(env, first, budget=3)
    assert result.status == "completed" and result.model_call_count == 3
    record = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
    frame = env.datasets.load_dataframe(next(iter(record.dataset_ids.values())))
    row = frame.iloc[0]
    assert row["targeted_review_status"] == "pending"
    assert {item["key"] for item in json.loads(row["targeted_attributes"])} == {
        "irradiation_time",
        "reported_loading_percent",
    }
    assert json.loads(row["variables"]) == {}  # No inferred variable merging.
    assert "质量分数" in record.report_markdown and "未执行" in record.report_markdown
    result = run(env, env.saved[-1])
    assert result.model_call_count == 0 and env.targeted_calls == 1


def test_cancel_after_targeted_request_saves_evidence_but_starts_no_statistics(
    tmp_path,
):
    env = bound_environment(tmp_path)
    cancelled = threading.Event()
    env.cancel_after_targeted = cancelled
    result = run(env, env.task, cancel=cancelled)
    assert result.status == "cancelled" and result.tool_call_count == 0
    record = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
    assert record.targeted_plans and not record.analysis_ids


@pytest.mark.parametrize("tamper", ["plan", "dataset"])
def test_saved_targeted_plan_and_input_tamper_are_rejected(tmp_path, tamper):
    env = bound_environment(tmp_path)
    run(env, env.task)
    first = env.saved[-1]
    record = env.analysis_store.load(first.fulltext_analysis_ref)
    if tamper == "plan":
        from uuid import uuid4

        doc, plan = next(iter(record.targeted_plans.items()))
        invalid = plan.model_copy(update={"preparation_sources": {}})
        revised = record.model_copy(
            update={
                "record_id": "fulltext-analysis-" + uuid4().hex,
                "targeted_plans": {doc: invalid},
            }
        )
        first = first.model_copy(
            update={"fulltext_analysis_ref": env.analysis_store.save(revised)}
        )
    else:
        dataset_id = next(iter(record.dataset_ids.values()))
        original = env.datasets.load_dataframe

        def changed(identifier):
            frame = original(identifier)
            if identifier == dataset_id:
                frame["targeted_attributes"] = "[]"
            return frame

        env.datasets.load_dataframe = changed
    result = run(env, first)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_FAILED"
    assert env.targeted_calls == 1 and result.tool_call_count == 0
