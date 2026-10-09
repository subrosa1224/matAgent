from __future__ import annotations

import json

import pytest

from materials_screening.agent.models import AgentResult
from materials_screening.master.literature_evidence_trial import (
    LiteratureEvidenceHandoff,
    LiteratureReportNotFoundError,
)
from materials_screening.sub_agents.literature.evidence_runner import (
    LITERATURE_DATASET_HANDOFF_MARKER,
    EvidenceAwareLiteratureRunner,
)
from materials_screening.sub_agents.literature.metric_coverage import (
    RequiredMetric,
    RequiredMetricCheck,
    RequiredMetricCoverage,
)


class _BaseRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def ask(self, *, message: str, conversation_id: str | None = None) -> AgentResult:
        self.calls.append((message, conversation_id))
        return AgentResult(
            conversation_id=conversation_id or "fresh-search",
            user_turn_id="fresh-turn",
            status="completed",
            response_text="fresh literature search",
        )


class _EvidenceService:
    def __init__(self, handoff: LiteratureEvidenceHandoff | None) -> None:
        self.handoff = handoff

    def prepare(self, *, topic: str):
        if self.handoff is None:
            raise LiteratureReportNotFoundError(topic)
        return self.handoff


def _handoff() -> LiteratureEvidenceHandoff:
    return LiteratureEvidenceHandoff(
        topic="任务5",
        report_id="user-lit-test",
        document_ids=("doc-1", "doc-2"),
        dataset_id="dataset-test",
        record_count=8,
        evidence_ids=("E-1",),
        literature_markdown="## 主题概述\n已有证据。",
        warnings=("仅供试运行。",),
    )


def test_exact_report_is_reused_and_exposes_dataset_handoff() -> None:
    base = _BaseRunner()
    runner = EvidenceAwareLiteratureRunner(
        base_runner=base,
        evidence_service=_EvidenceService(_handoff()),  # type: ignore[arg-type]
    )

    result = runner.ask(message="任务5", conversation_id="master-child")

    assert base.calls == []
    assert result.status == "completed"
    assert result.conversation_id == "master-child"
    assert LITERATURE_DATASET_HANDOFF_MARKER in result.response_text
    assert "dataset_id=dataset-test" in result.response_text
    assert result.evidence_ids == ("E-1",)


def test_nonmatching_topic_runs_fresh_literature_search() -> None:
    base = _BaseRunner()
    runner = EvidenceAwareLiteratureRunner(
        base_runner=base,
        evidence_service=_EvidenceService(None),  # type: ignore[arg-type]
    )

    result = runner.ask(message="另一个科学问题", conversation_id="master-child")

    assert result.response_text == "fresh literature search"
    assert base.calls == [("另一个科学问题", "master-child")]


def test_pool_prescreen_is_not_treated_as_reusable_full_text() -> None:
    base = _BaseRunner()
    runner = EvidenceAwareLiteratureRunner(
        base_runner=base, evidence_service=_EvidenceService(_handoff())
    )  # type: ignore[arg-type]
    result = runner.ask(message="执行候选池文献预检：1:ZnO", conversation_id="pool")
    assert result.response_text == "fresh literature search"
    assert base.calls == [("执行候选池文献预检：1:ZnO", "pool")]


def test_reuse_visibly_reports_missing_required_metric(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    coverage = RequiredMetricCoverage(
        status="incomplete",
        checks=(
            RequiredMetricCheck(
                requirement=RequiredMetric(
                    metric="capacity retention after 50 cycles", unit="%"
                ),
                status="missing",
            ),
        ),
    )
    handoff = _handoff().model_copy(update={"required_metric_coverage": coverage})
    runner = EvidenceAwareLiteratureRunner(
        base_runner=_BaseRunner(), evidence_service=_EvidenceService(handoff)
    )
    result = runner.ask(message="任务5")
    assert "必需指标覆盖检查" in result.response_text
    assert "缺失：capacity retention after 50 cycles" in result.response_text
    assert result.response_text.index("必需指标覆盖检查") < result.response_text.index(
        "主题概述"
    )


def test_no_policy_does_not_claim_coverage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = EvidenceAwareLiteratureRunner(
        base_runner=_BaseRunner(), evidence_service=_EvidenceService(_handoff())
    )
    assert "未提供必需指标清单" in runner.ask(message="任务5").response_text


class _PolicyService(_EvidenceService):
    def __init__(self):
        super().__init__(_handoff())
        self.policy = None

    def prepare(self, *, topic, metric_requirements=None):
        self.policy = metric_requirements
        return super().prepare(topic=topic)


def write_policy(tmp_path, topic="任务5", name="policy.json"):
    root = tmp_path / "data/literature_metric_requirements"
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(
        json.dumps(
            {
                "topic": topic,
                "required_metrics": [{"metric": "test metric", "unit": "Ω"}],
            }
        ),
        encoding="utf-8",
    )


def test_only_exact_normalized_policy_is_passed_to_prepare(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_policy(tmp_path)
    write_policy(tmp_path, topic="任务51", name="another.json")
    service = _PolicyService()
    runner = EvidenceAwareLiteratureRunner(
        base_runner=_BaseRunner(), evidence_service=service
    )
    runner.ask(message=" 任务5 ")
    assert service.policy is not None
    assert service.policy.topic == "任务5"
    assert service.policy.required_metrics[0].metric == "test metric"


def test_different_topic_does_not_inherit_metric_requirements(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_policy(tmp_path)
    service = _PolicyService()
    runner = EvidenceAwareLiteratureRunner(
        base_runner=_BaseRunner(), evidence_service=service
    )
    runner.ask(message="另一个任务")
    assert service.policy is None


@pytest.mark.parametrize("bad", ["duplicate", "invalid", "oversized"])
def test_bad_policy_is_not_silently_ignored(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path)
    write_policy(tmp_path)
    root = tmp_path / "data/literature_metric_requirements"
    if bad == "duplicate":
        write_policy(tmp_path, name="duplicate.json")
    elif bad == "invalid":
        (root / "bad.json").write_text(
            '{"topic":"任务5","required_metrics":[],"answer":42}', encoding="utf-8"
        )
    else:
        (root / "huge.json").write_text(" " * 65537, encoding="utf-8")
    runner = EvidenceAwareLiteratureRunner(
        base_runner=_BaseRunner(), evidence_service=_PolicyService()
    )
    with pytest.raises(ValueError):
        runner.ask(message="任务5")
