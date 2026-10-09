from unittest.mock import Mock

import pytest

from materials_screening.master.staged_fulltext_processor import StagedFulltextProcessor
from materials_screening.master.staged_fulltext_store import StagedExtractionStore
from materials_screening.sub_agents.data_analysis.fulltext_adapter import (
    bounded_description_agent,
)
from tests.unit.master.test_fulltext_extraction import extraction_env
from tests.unit.master.test_fulltext_preview import run
from tests.unit.master.test_staged_fulltext_engine import Provider, calls


def env(tmp_path, *, body=None, count=1):
    e = extraction_env(
        tmp_path,
        count=count,
        body=body
        or (
            "Sample NTO3 is a film. NTO3 transmittance was around 82% "
            "in UV-visible region."
        ),
    )
    run(e, e.task.model_copy(update={"user_instructions": ("只预览这些论文",)}))
    e.task = e.saved[-1].model_copy(
        update={"user_instructions": ("确认详细分析全部论文，提取透过率和电阻率",)}
    )
    from materials_screening.master.fulltext_tasks import PreviewDecision

    e.task = e.task.model_copy(
        update={
            "preview_decisions": {
                doc: PreviewDecision(action="extract", reason="用户确认")
                for doc in e.task.document_ids
            }
        }
    )
    e.stages = StagedExtractionStore(tmp_path / "stages")
    e.stat_calls = []

    def agent():
        e.stat_calls.append(1)
        return bounded_description_agent(
            data_root=tmp_path / "datasets",
            workflow_runner=Mock(),
            result_reader=Mock(),
        )

    e.staged_processor = StagedFulltextProcessor(
        artifacts=e.processor.artifacts,
        stages=e.stages,
        dataset_factory=lambda: e.datasets,
        agent_factory=agent,
        model_profile="offline/unit",
    )
    return e


def invoke(e, task=None, budget=100, provider=None):
    chunks = tuple(e.store.chunks)
    return e.staged_processor.run(
        task or e.task,
        chunks_by_document={
            doc: tuple(c for c in chunks if c.document_id == doc)
            for doc in e.task.document_ids
        },
        refresh_sources=lambda doc: tuple(c for c in chunks if c.document_id == doc),
        calls=calls(provider or Provider(), budget),
        save_task=e.saved.append,
        user_turn_id="unit-turn",
    )


def test_normal_master_contract_independent_refs_real_statistics_and_cached_resume(
    tmp_path,
):
    e = env(tmp_path)
    result = invoke(e)
    assert result.status == "completed" and not result.error
    task = e.saved[-1]
    assert task.staged_extractions and task.staged_handoffs
    assert not task.extraction_snapshots
    assert len(e.stat_calls) == 1
    handoff = next(iter(task.staged_handoffs.values()))
    assert handoff.analysis_id and handoff.analysis_sha256
    assert "约82" in result.response_text and "无法直接排名" in result.response_text
    assert "已完成的描述统计" in result.response_text
    second = invoke(e, task)
    assert second.model_call_count == second.tool_call_count == 0
    assert len(e.stat_calls) == 1
    assert e.saved[-1].staged_handoffs == task.staged_handoffs
    assert "已完成的描述统计" in second.response_text
    assert "本轮未执行描述统计" not in second.response_text


def test_budget_before_statistics_keeps_dataset_and_resumes_without_extraction_calls(
    tmp_path,
):
    e = env(tmp_path)
    result = invoke(e, budget=3)
    task = e.saved[-1]
    assert result.error["code"] == "FULLTEXT_ANALYSIS_BUDGET"
    assert task.staged_handoffs and not e.stat_calls
    assert "统计结果已经保存" not in result.response_text
    handoff = next(iter(task.staged_handoffs.values()))
    assert handoff.dataset_id and not handoff.analysis_id
    second = invoke(e, task)
    assert second.status == "completed"
    assert second.model_call_count == 3
    assert len(e.stat_calls) == 1


def test_continuation_words_do_not_invalidate_scope_and_repeat_models(tmp_path):
    e = env(tmp_path)
    invoke(e)
    task = e.saved[-1]
    repeated = task.model_copy(
        update={"user_instructions": (*task.user_instructions, "继续", "重试")}
    )
    result = invoke(e, repeated)
    assert result.model_call_count == 0 and len(e.stat_calls) == 1


def test_old_task_json_and_legacy_snapshots_stay_readable(tmp_path):
    from materials_screening.master.fulltext_tasks import FulltextTask

    e = env(tmp_path)
    raw = e.task.model_dump(mode="json")
    for name in (
        "staged_extractions",
        "staged_extraction_history",
        "staged_handoffs",
        "staged_handoff_history",
    ):
        raw.pop(name)
    old = FulltextTask.model_validate(raw)
    assert not old.staged_extractions and not old.staged_handoffs
    invoke(e, old)
    assert not e.saved[-1].extraction_snapshots


def test_bound_only_report_never_claims_statistics_or_runs_empty_analysis(tmp_path):
    e = env(
        tmp_path, body="Sample NTO3 is a film. NTO3 transmittance was more than 82%."
    )
    e.staged_processor.agent_factory = lambda: pytest.fail(
        "Bounds cannot start statistics"
    )
    result = invoke(e)
    assert result.status == "completed" and not result.error
    assert "大于82%" in result.response_text
    assert "本轮仅整理文献证据，未执行描述统计" in result.response_text
    assert "本次统计" not in result.response_text
    assert "独立重复实验的统计推断" not in result.response_text
    assert not e.stat_calls and result.tool_call_count == 0
    handoff = next(iter(e.saved[-1].staged_handoffs.values()))
    assert handoff.dataset_id is None and handoff.analysis_id is None
    second = invoke(e, e.saved[-1])
    assert second.model_call_count == second.tool_call_count == 0
    assert "未执行描述统计" in second.response_text


def test_mixed_report_distinguishes_statistical_and_evidence_only_papers(tmp_path):
    import json

    from materials_screening.llm.base import StructuredProviderResponse
    from materials_screening.master.staged_fulltext_evidence import (
        MetricProposal,
        MetricsProposal,
        SampleProposal,
        build_catalogue,
        checked_sample,
    )

    e = env(
        tmp_path,
        count=2,
        body="Sample NTO3 is a film. NTO3 transmittance was around 82%. "
        "NTO3 transmittance was more than 85%.",
    )
    bound_doc = e.task.document_ids[1]
    bound_catalogue = build_catalogue(
        tuple(c for c in e.store.chunks if c.document_id == bound_doc)
    )
    bound_sample = checked_sample(
        SampleProposal(label="NTO3", source_id="s0"),
        bound_catalogue,
        document_id=bound_doc,
    )

    class MixedProvider(Provider):
        def generate_structured(self, **kwargs):
            payload = json.loads(kwargs["user_text"])
            if (
                kwargs["output_model"] is MetricsProposal
                and payload["samples"][0]["sample_id"] == bound_sample.sample_id
            ):
                return StructuredProviderResponse(
                    parsed=MetricsProposal(
                        measurements=(
                            MetricProposal(
                                sample_id=bound_sample.sample_id,
                                metric="transmittance",
                                value_text="85",
                                unit="%",
                                source_id=next(
                                    k
                                    for k, v in payload["evidence"].items()
                                    if "85%" in v
                                ),
                            ),
                        )
                    ),
                    provider="offline",
                    model="unit",
                    request_id=None,
                    latency_ms=0,
                    input_tokens=1,
                    output_tokens=1,
                    reasoning_tokens=0,
                    raw_output_sha256=None,
                )
            return super().generate_structured(**kwargs)

    result = invoke(e, provider=MixedProvider())
    assert result.status == "completed" and not result.error
    assert len(e.stat_calls) == 1 and result.tool_call_count == 2
    assert "已完成描述统计的论文：1篇；另有1篇仅展示文献证据" in result.response_text
    assert "本轮未执行描述统计" not in result.response_text
    second = invoke(e, e.saved[-1])
    assert second.model_call_count == second.tool_call_count == 0
    assert "已完成描述统计的论文：1篇；另有1篇仅展示文献证据" in second.response_text
