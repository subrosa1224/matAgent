"""Task-scoped analysis through the actual data-analysis graph, offline only."""

import json
import threading
from unittest.mock import Mock

import pytest

from materials_screening.master.fulltext_analysis import FulltextAnalysisProcessor
from materials_screening.master.fulltext_analysis_contracts import (
    AnalysisScopePlan,
    MetricScope,
)
from materials_screening.master.fulltext_analysis_store import FulltextAnalysisStore
from materials_screening.sub_agents.data_analysis.fulltext_adapter import (
    bounded_description_agent,
)
from tests.unit.master.test_fulltext_extraction import extraction_env
from tests.unit.master.test_fulltext_preview import run


def test_experimental_requirement_options_remain_exact_original_quotes():
    from materials_screening.master.fulltext_analysis import _experimental_requirements

    text = (
        "从 Materials Project 查询计算带隙；全文提取响应度和增益。"
        "不要将器件指标归给纯相。"
    )
    options = _experimental_requirements(text)
    assert isinstance(options, list)
    assert options and all(option in text for option in options)
    assert not any("计算带隙" in option for option in options)


def test_scope_cannot_silently_omit_explicit_gain_candidate():
    requirements = "查询 Materials Project 计算带隙；全文提取增益。"
    scope = AnalysisScopePlan(
        metrics=(MetricScope(name="增益", task_quote="全文提取增益。"),),
        requested_operations=("describe",),
    )
    candidates = {"measurement-a": {"metric": "gain"}}
    with pytest.raises(ValueError, match="omitted"):
        FulltextAnalysisProcessor._validate_scope(scope, candidates, requirements)


def test_scope_accepts_explicit_exclusion_not_auto_inclusion():
    requirements = "全文提取增益。"
    scope = AnalysisScopePlan(
        metrics=(MetricScope(name="增益", task_quote=requirements),),
        requested_operations=("describe",),
        excluded_measurements={
            "measurement-a": "当前候选来自背景器件，暂不纳入目标样品。"
        },
    )
    FulltextAnalysisProcessor._validate_scope(
        scope, {"measurement-a": {"metric": "gain"}}, requirements
    )
    assert scope.metrics[0].measurement_ids == ()


def test_scope_rejects_computed_only_task_quote():
    requirements = "从 Materials Project 查询计算带隙和密度；全文提取增益。"
    scope = AnalysisScopePlan(
        metrics=(
            MetricScope(
                name="带隙", task_quote="从 Materials Project 查询计算带隙和密度"
            ),
        ),
        requested_operations=("describe",),
    )
    with pytest.raises(ValueError, match="computed"):
        FulltextAnalysisProcessor._validate_scope(scope, {}, requirements)


def analysis_env(tmp_path, count=1, body=None, configure=None):
    env = extraction_env(tmp_path, count=count, body=body)
    if configure is not None:
        configure(env)
    metric_name = getattr(env, "scope_metric", "reported strength")
    metric_quote = getattr(env, "scope_quote", "全文提取强度。")
    env.task = env.task.model_copy(
        update={"user_instructions": (*env.task.user_instructions, metric_quote)}
    )
    run(env)
    env.task = env.saved[-1]
    env.saved.clear()
    env.scope_calls = 0
    original = env.matrix_llm.generate_structured

    def scoped(**kwargs):
        from materials_screening.master.fulltext_scope_stages import (
            SupplementaryScopePlan,
        )

        if kwargs["output_model"] is SupplementaryScopePlan:
            from materials_screening.llm.base import StructuredProviderResponse

            parsed = (
                env.supplement_plan(kwargs)
                if hasattr(env, "supplement_plan")
                else SupplementaryScopePlan()
            )
            return StructuredProviderResponse(
                parsed=parsed,
                provider="offline",
                model="unit",
                request_id="unit",
                latency_ms=0,
                input_tokens=1,
                output_tokens=1,
                reasoning_tokens=0,
                raw_output_sha256=None,
            )
        if kwargs["output_model"] is not AnalysisScopePlan:
            return original(**kwargs)
        from materials_screening.llm.base import StructuredProviderResponse

        candidates = json.loads(kwargs["user_text"])["candidate_metadata"]
        env.scope_calls += 1
        policy = AnalysisScopePlan(
            metrics=(
                MetricScope(
                    name=metric_name,
                    task_quote=metric_quote,
                    measurement_ids=tuple(row["measurement_id"] for row in candidates),
                ),
            ),
            requested_operations=("describe", "trend"),
            required_conditions=("temperature", "light_source"),
            limitations=("No independent replication; trend unavailable",),
        )
        if hasattr(env, "mutate_plan"):
            policy = env.mutate_plan(policy)
        return StructuredProviderResponse(
            parsed=policy,
            provider="offline",
            model="unit",
            request_id="unit",
            latency_ms=0,
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            raw_output_sha256=None,
        )

    env.matrix_llm.generate_structured = scoped

    # Real DataAnalysisRoutingModel + real quality/statistics tools. The fallback
    # model is never called because the handoff is an explicit typed operation.
    def agent():
        return bounded_description_agent(
            data_root=tmp_path / "datasets",
            workflow_runner=Mock(),
            result_reader=Mock(),
        )

    env.analysis_store = FulltextAnalysisStore(tmp_path / "analysis_snapshots")
    env.processor.extraction_processor.analysis_processor = FulltextAnalysisProcessor(
        snapshots=env.snapshots,
        analyses=env.analysis_store,
        dataset_factory=lambda: env.datasets,
        agent_factory=agent,
        query_factory=None,
        model_profile="offline-unit",
    )
    return env


def test_scope_prompt_requires_exact_operation_tokens():
    from materials_screening.master.fulltext_analysis import _SCOPE_PROMPT

    assert '"requested_operations": ["describe"]' in _SCOPE_PROMPT
    assert "Do not translate" in _SCOPE_PROMPT


@pytest.mark.parametrize("always_invalid", [False, True])
def test_unrequested_metric_requires_bounded_model_repair(tmp_path, always_invalid):
    env = analysis_env(tmp_path)
    provider = env.matrix_llm.generate_structured
    requests = []

    def proposed(**kwargs):
        response = provider(**kwargs)
        if kwargs["output_model"] is AnalysisScopePlan:
            requests.append(json.loads(kwargs["user_text"]))
            if always_invalid or len(requests) == 1:
                metric = response.parsed.metrics[0].model_copy(
                    update={"name": "Bandgap"}
                )
                response = response.model_copy(
                    update={
                        "parsed": response.parsed.model_copy(
                            update={"metrics": (metric,)}
                        )
                    }
                )
        return response

    env.matrix_llm.generate_structured = proposed
    result = run(env, env.task)
    assert len(requests) == 2
    assert "explicitly requested" in requests[1]["validation_issue"]
    assert requests[1]["metric_scope_feedback"]["issues"][0] == {
        "metric_name": "Bandgap",
        "task_quote": "全文提取强度。",
        "reason": "quote_does_not_request_this_literature_metric",
        "action": "remove_scope_entry",
    }
    if always_invalid:
        assert result.error["code"] == "FULLTEXT_ANALYSIS_FAILED"
        record = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
        assert not record.analysis_ids and not record.report_markdown
    else:
        assert result.status == "completed"
        record = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
        assert all(metric.name != "Bandgap" for metric in record.scope.metrics)


def test_scope_schema_repair_identifies_operations_without_echoing_input(tmp_path):
    from pydantic import ValidationError

    from materials_screening.llm.errors import LLMStructuredOutputError

    env = analysis_env(tmp_path)
    provider = env.matrix_llm.generate_structured
    requests = []

    def invalid_first(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            try:
                AnalysisScopePlan.model_validate(
                    {
                        "metrics": [{"name": "strength", "task_quote": "test"}],
                        "requested_operations": ["private-unknown-operation"],
                    }
                )
            except ValidationError as error:
                raise LLMStructuredOutputError("private-provider-message") from error
        return provider(**kwargs)

    env.matrix_llm.generate_structured = invalid_first
    result = run(env, env.task)
    assert result.status == "completed" and len(requests) == 2
    repair = json.loads(requests[1]["user_text"])
    assert requests[0]["system_prompt"] != requests[1]["system_prompt"]
    assert "complete replacement AnalysisScopePlan" in requests[1]["system_prompt"]
    assert "Do not return the input or feedback" in requests[1]["system_prompt"]
    contract = repair["output_contract"]
    assert contract["required_fields"] == ["metrics", "requested_operations"]
    assert set(contract["allowed_fields"]) == set(AnalysisScopePlan.model_fields)
    assert contract["allowed_operations"] == ["describe", "compare", "trend"]
    assert list(repair)[-1] == "output_contract"
    assert repair["validation_fields"][0] == {
        "location": ["requested_operations", 0],
        "error_code": "literal_error",
        "allowed_values": ["describe", "compare", "trend"],
    }
    assert "private-" not in requests[1]["user_text"]


def test_unknown_operations_are_not_automatically_converted():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AnalysisScopePlan.model_validate(
            {
                "metrics": [{"name": "strength", "task_quote": "test"}],
                "requested_operations": ["描述"],
            }
        )


def test_scope_schema_feedback_filters_untrusted_paths_and_values():
    from pydantic import ValidationError

    from materials_screening.llm.errors import LLMStructuredOutputError
    from materials_screening.master.fulltext_analysis import _scope_validation_fields

    cause = ValidationError.from_exception_data(
        "private-title",
        [
            {
                "type": "string_type",
                "loc": ("private-field", "requested_operations", -1),
                "input": "private-value",
            }
        ],
    )
    error = LLMStructuredOutputError("private-message")
    error.__cause__ = cause
    feedback = _scope_validation_fields(error)
    assert feedback == [
        {
            "location": ["other", "requested_operations", "other"],
            "error_code": "string_type",
        }
    ]
    assert "private" not in json.dumps(feedback)
    assert _scope_validation_fields(LLMStructuredOutputError("private")) == []


@pytest.mark.parametrize("always_invalid", [False, True])
def test_scope_schema_failure_has_only_one_bounded_repair(tmp_path, always_invalid):
    from materials_screening.llm.errors import LLMStructuredOutputError

    env = analysis_env(tmp_path)
    provider = env.matrix_llm.generate_structured
    attempts = []

    def invalid_first(**kwargs):
        if kwargs["output_model"] is AnalysisScopePlan:
            attempts.append(kwargs["user_text"])
            if always_invalid or len(attempts) == 1:
                raise LLMStructuredOutputError("private malformed response detail")
        return provider(**kwargs)

    env.matrix_llm.generate_structured = invalid_first
    result = run(env, env.task)
    assert len(attempts) == 2
    assert "private malformed" not in attempts[1] + result.response_text
    if always_invalid:
        assert env.saved[-1].stage == "failed" and result.tool_call_count == 0
    else:
        assert env.saved[-1].stage == "finished" and result.tool_call_count == 2


def test_scope_schema_repair_cannot_exceed_turn_budget(tmp_path):
    from materials_screening.llm.errors import LLMStructuredOutputError

    env = analysis_env(tmp_path)
    attempts = []

    def invalid(**kwargs):
        attempts.append(kwargs["schema_name"])
        raise LLMStructuredOutputError("private response")

    env.matrix_llm.generate_structured = invalid
    result = run(env, env.task, budget=1)
    assert attempts == ["fulltext_analysis_scope_v1"]
    assert env.saved[-1].stage == "ready_to_resume"
    assert result.model_call_count == 1 and result.tool_call_count == 0


@pytest.mark.parametrize("cancelled", [False, True])
def test_scope_transport_failure_or_cancellation_does_not_retry(tmp_path, cancelled):
    from materials_screening.llm.errors import (
        LLMConnectionError,
        LLMStructuredOutputError,
    )

    env = analysis_env(tmp_path)
    cancel = threading.Event()
    attempts = []

    def failed(**kwargs):
        attempts.append(kwargs["schema_name"])
        if cancelled:
            cancel.set()
            raise LLMStructuredOutputError("private response detail")
        raise LLMConnectionError("private connection detail")

    env.matrix_llm.generate_structured = failed
    result = run(env, env.task, cancel=cancel)
    assert attempts == ["fulltext_analysis_scope_v1"]
    assert result.model_call_count == 1 and result.tool_call_count == 0
    assert "private" not in result.response_text
    assert result.status == ("cancelled" if cancelled else "error")


def test_normal_continuation_calls_real_agent_and_saves_partial_report(tmp_path):
    env = analysis_env(tmp_path)
    result = run(env, env.task)
    task = env.saved[-1]
    assert result.status == "completed" and result.final_status == "completed"
    assert task.stage == "finished"
    record = env.analysis_store.load(task.fulltext_analysis_ref)
    assert record.status == "complete" and record.scientific_coverage == "partial"
    assert env.scope_calls == 1 and result.model_call_count == 4
    assert result.tool_call_count == 2
    assert record.analysis_ids
    for doc, analysis_id in record.analysis_ids.items():
        analysis = env.datasets.get_analysis(analysis_id)
        assert analysis.parameters["group_by"] == "measurement_context"
        assert analysis.summary["statistics"][0]["mean"] == 12.0
        assert analysis.dataset_id == record.dataset_ids[doc]
    assert "pending" in result.response_text and "partial" in result.response_text
    assert "未执行" in result.response_text and "趋势" in result.response_text
    assert "temperature" in result.response_text and "缺失 1/1" in result.response_text
    assert "light_source" in result.response_text and "缺失 0/1" in result.response_text


def test_tiny_output_limit_is_respected(tmp_path):
    env = analysis_env(tmp_path)
    env.processor.extraction_processor.analysis_processor.max_output_bytes = 12
    result = run(env, env.task)
    assert len(result.response_text.encode()) <= 12
    assert env.analysis_store.load(env.saved[-1].fulltext_analysis_ref).report_markdown


def test_report_lists_explicit_conditions_even_when_scope_omits_them(tmp_path):
    def configure(env):
        env.scope_quote = "全文提取强度及光照和偏压条件。"

    env = analysis_env(tmp_path, configure=configure)
    env.mutate_plan = lambda policy: policy.model_copy(
        update={"required_conditions": ()}
    )
    result = run(env, env.task)
    assert result.status == "completed"
    record = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
    assert record.scope.required_conditions == ()
    assert "任务明确要求：光照" in record.report_markdown
    assert "任务明确要求：偏压；未绑定／未知 1/1" in record.report_markdown
    assert "不代表完整实验协议或条件可比" in record.report_markdown


def test_explicit_condition_display_does_not_infer_from_general_goal():
    from materials_screening.master.fulltext_analysis import _explicit_report_conditions

    assert _explicit_report_conditions("判断ZnO是否适合紫外探测器。") == []
    assert _explicit_report_conditions("不要提取光照或偏压条件。") == []
    assert (
        len(_explicit_report_conditions("全文提取响应度、增益及光照和偏压条件。")) == 2
    )


def test_cached_report_rejects_changed_database_query_snapshot(tmp_path):
    from materials_screening.models import MaterialRecord

    env = analysis_env(tmp_path)
    query = Mock()
    query.load_records.return_value = (
        MaterialRecord(
            source="unit",
            material_id="mp-1",
            formula_pretty="ZnO",
            elements=("Zn", "O"),
        ),
    )
    env.task = env.task.model_copy(
        update={"database_query_ids": ("query-" + "a" * 32,)}
    )
    processor = env.processor.extraction_processor.analysis_processor
    processor.query_factory = lambda: query
    run(env, env.task)
    first = env.saved[-1]
    query.load_records.return_value = (
        query.load_records.return_value[0].model_copy(update={"material_id": "mp-2"}),
    )
    result = run(env, first)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_FAILED"
    assert env.scope_calls == 1


def test_cached_report_revalidates_without_scope_or_agent_calls(tmp_path, monkeypatch):
    env = analysis_env(tmp_path)
    run(env, env.task)
    first = env.saved[-1]
    processor = env.processor.extraction_processor.analysis_processor
    monkeypatch.setattr(
        processor, "agent_factory", lambda: pytest.fail("Repeated agent")
    )
    files = set((tmp_path / "analysis_snapshots").iterdir())
    result = run(env, first)
    assert result.model_call_count == result.tool_call_count == 0
    assert env.scope_calls == 1
    assert env.saved[-1].fulltext_analysis_ref == first.fulltext_analysis_ref
    assert set((tmp_path / "analysis_snapshots").iterdir()) == files


def test_budget_after_scope_persists_plan_and_resume_skips_scope_model(tmp_path):
    env = analysis_env(tmp_path)
    result = run(env, env.task, budget=1)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_BUDGET"
    first = env.saved[-1]
    assert first.resume_stage == "analyzing"
    record = env.analysis_store.load(first.fulltext_analysis_ref)
    assert record.scope is not None and not record.analysis_ids
    result = run(env, first)
    assert result.status == "completed" and result.model_call_count == 3
    assert env.scope_calls == 1


@pytest.mark.parametrize("invalid", ["measurement", "task_quote"])
def test_unknown_measurement_or_invented_requirement_quote_fails_closed(
    tmp_path, invalid
):
    env = analysis_env(tmp_path)
    env.mutate_plan = lambda plan: plan.model_copy(
        update={
            "metrics": (
                plan.metrics[0].model_copy(
                    update={"measurement_ids": ("measurement-" + "a" * 24,)}
                    if invalid == "measurement"
                    else {"task_quote": "invented requirement"}
                ),
            )
        }
    )
    result = run(env, env.task)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_FAILED"
    assert not list((tmp_path / "datasets/analyses").glob("*.json"))


def test_documents_are_separate_agent_inputs_and_budget_resumes_second(tmp_path):
    env = analysis_env(tmp_path, count=2)
    result = run(env, env.task)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_BUDGET"
    first = env.saved[-1]
    record = env.analysis_store.load(first.fulltext_analysis_ref)
    assert len(record.analysis_ids) == 1 and result.model_call_count == 4
    result = run(env, first)
    assert result.status == "completed" and result.model_call_count == 3
    record = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
    assert len(set(record.dataset_ids.values())) == 2
    for doc, dataset_id in record.dataset_ids.items():
        assert set(env.datasets.load_dataframe(dataset_id)["document_id"]) == {doc}
    assert env.scope_calls == 1


def test_empty_scope_skips_agent_not_empty_table_statistics(tmp_path, monkeypatch):
    env = analysis_env(tmp_path)
    env.mutate_plan = lambda plan: plan.model_copy(
        update={
            "metrics": (plan.metrics[0].model_copy(update={"measurement_ids": ()}),)
        }
    )
    monkeypatch.setattr(
        env.processor.extraction_processor.analysis_processor,
        "agent_factory",
        lambda: pytest.fail("Empty data agent"),
    )
    result = run(env, env.task)
    assert result.status == "completed"
    record = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
    assert not record.analysis_ids and not record.dataset_ids
    assert record.scientific_coverage == "partial"


def test_cancel_before_analysis_preserves_extraction_and_starts_no_scope(tmp_path):
    env = analysis_env(tmp_path)
    cancel = threading.Event()
    cancel.set()
    result = run(env, env.task, cancel=cancel)
    assert result.status == "cancelled"
    assert env.scope_calls == 0


def test_changed_scope_never_reuses_old_report(tmp_path):
    env = analysis_env(tmp_path)
    run(env, env.task)
    first = env.saved[-1]
    old = env.analysis_store.load(first.fulltext_analysis_ref)
    modified = first.model_copy(update={"user_instructions": ("仅核对报告值",)})
    run(env, modified)
    second = env.saved[-1]
    assert second.fulltext_analysis_ref != first.fulltext_analysis_ref
    assert env.analysis_store.load(first.fulltext_analysis_ref) == old


def test_report_or_analysis_tampering_cannot_replay_as_success(tmp_path):
    env = analysis_env(tmp_path)
    run(env, env.task)
    first = env.saved[-1]
    record = env.analysis_store.load(first.fulltext_analysis_ref)
    # Test output corruption, never a user file.
    (tmp_path / "analysis_snapshots" / f"{record.record_id}.md").write_text(
        "changed", encoding="utf-8"
    )
    result = run(env, first)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_FAILED"
    assert env.scope_calls == 1


def test_saved_statistics_tampering_is_rejected_without_rerun(tmp_path):
    env = analysis_env(tmp_path)
    run(env, env.task)
    first = env.saved[-1]
    record = env.analysis_store.load(first.fulltext_analysis_ref)
    analysis_id = next(iter(record.analysis_ids.values()))
    path = tmp_path / "datasets/analyses" / f"{analysis_id}.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    content["summary"]["statistics"][0]["std"] = 999.0
    path.write_text(json.dumps(content), encoding="utf-8")
    result = run(env, first)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_FAILED"
    assert env.scope_calls == 1


def test_agent_failure_has_no_complete_report_and_preserves_scope(
    tmp_path, monkeypatch
):
    from materials_screening.agent.models import AgentResult

    env = analysis_env(tmp_path)
    fake = Mock()
    fake.ask.return_value = AgentResult(
        conversation_id="unit",
        user_turn_id="unit",
        status="error",
        final_status="error",
        response_text="private failure",
        model_call_count=3,
        tool_call_count=1,
    )
    monkeypatch.setattr(
        env.processor.extraction_processor.analysis_processor,
        "agent_factory",
        lambda: fake,
    )
    result = run(env, env.task)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_FAILED"
    assert result.model_call_count == 4 and result.tool_call_count == 1
    assert "private failure" not in result.response_text
    record = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
    assert record.scope is not None and record.status == "collecting"
    assert not record.report_markdown


def test_literal_sample_gate_excludes_other_series_not_composite_components():
    from materials_screening.master.fulltext_analysis import _sample_scope

    rows = {
        "gd": {"material": "GdVO4"},
        "comp": {"material": "40% GdVO4/g-C3N4"},
        "la": {"material": "20% LaVO4/g-C3N4"},
        "control": {"material": "g-C3N4"},
    }
    selected, excluded = _sample_scope(
        rows, "研究GdVO4/g-C3N4的降解率，保留纯GdVO4对照。"
    )
    assert set(selected) == {"gd", "comp"}
    assert set(excluded) == {"la", "control"}
    selected, _ = _sample_scope(
        rows, "比较GdVO4/g-C3N4与LaVO4/g-C3N4，纯g-C3N4作对照。"
    )
    assert set(selected) == {"comp", "la", "control"}


def test_general_material_question_does_not_invent_literal_formula_constraints():
    from materials_screening.master.fulltext_analysis import _sample_scope

    rows = {"sample": {"material": "Hydroxyapatite"}}
    selected, excluded = _sample_scope(rows, "比较文中全部材料的强度")
    assert selected == rows and not excluded
