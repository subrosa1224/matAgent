"""Offline reuse checks: synthetic source text, no database or model calls."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.master.literature_evidence_trial import (
    LiteratureEvidenceTrialService,
    _review_saved_report_sources,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord
from materials_screening.sub_agents.literature.user_report import (
    LiteratureUserReportStore,
    UserReportEvidence,
    UserReportNarrative,
)
from tests.unit.master.test_literature_evidence_trial import (
    _group,
    _MatrixStore,
    _measurement,
    _report,
    _ReportStore,
)


def _chunk(text: str, page: int = 4, doc: str = "doc-1", key: str = "chunk-4"):
    return ChunkRecord(
        key, doc, None, page, page, text, hashlib.sha256(text.encode()).hexdigest()
    )


def _evidence(
    key: str = "E-body", quote: str = "Our sample retained 80% capacity.", page: int = 4
):
    return UserReportEvidence(
        evidence_id=key,
        category="results",
        summary=f"待核对摘要 {key}",
        page=page,
        source_quote=quote,
        risk_level="low",
        display_status="reliable",
    )


def _saved(evidence, narrative: str | None = None):
    base = _report()
    return base.model_copy(
        update={
            "papers": (
                base.papers[0].model_copy(
                    update={
                        "evidence": tuple(evidence),
                        "low_risk_count": len(evidence),
                    }
                ),
            ),
            "narrative": UserReportNarrative(markdown=narrative) if narrative else None,
        }
    )


class _Sources(_MatrixStore):
    def __init__(self, chunks, error: Exception | None = None):
        super().__init__([], [])
        self.chunks, self.error, self.calls = chunks, error, []

    def get_document_chunks(self, document_id):
        self.calls.append(document_id)
        if self.error:
            raise self.error
        return self.chunks


def _prepare(tmp_path, report, sources, explicit: bool = False):
    return LiteratureEvidenceTrialService(
        report_store=_ReportStore(report),
        matrix_store=sources,
        dataset_store=DatasetStore(tmp_path / "analysis"),
    ).prepare(topic=report.topic, report_id=report.report_id if explicit else None)


@pytest.mark.parametrize("explicit", [False, True])
def test_reuse_isolates_reference_entries_before_handoff(tmp_path, explicit):
    body, reference = (
        _evidence(),
        _evidence("E-ref", "Synthesis of Li2FeSiO4/C nanocomposite.", 12),
    )
    report = _saved([body, reference])
    before = report.model_dump_json()
    sources = _Sources(
        [
            _chunk(body.source_quote),
            _chunk("References\n1. " + reference.source_quote, 12, key="refs"),
        ]
    )
    result = _prepare(tmp_path, report, sources, explicit)
    assert result.evidence_ids == (body.evidence_id,)
    assert body.source_quote in result.literature_markdown
    assert reference.source_quote not in result.literature_markdown
    assert any("E-ref" in w and "参考文献" in w for w in result.warnings)
    assert report.model_dump_json() == before
    assert result.document_ids == ("doc-1",)
    assert result.dataset_id is None
    assert not (tmp_path / "analysis").exists()
    assert sources.calls == ["doc-1"]


def test_reference_continuation_without_repeated_heading_is_isolated(tmp_path):
    item = _evidence("E-ref", "Large-scale synthesis of nanorods.", 12)
    sources = _Sources(
        [
            _chunk("References\n1. Other citation.", 11, key="heading"),
            _chunk(item.source_quote, 12),
        ]
    )
    result = _prepare(tmp_path, _saved([item]), sources)
    assert result.evidence_ids == ()
    assert any("参考文献" in w for w in result.warnings)


@pytest.mark.parametrize(
    "context",
    [
        "Li et al. reported that Our sample retained 80% capacity.",
        "Previous studies showed: Our sample retained 80% capacity.",
        "已有研究：Our sample retained 80% capacity.",
    ],
)
def test_recognizable_prior_work_is_not_reused_as_current_results(tmp_path, context):
    result = _prepare(tmp_path, _saved([_evidence()]), _Sources([_chunk(context)]))
    assert result.evidence_ids == ()
    assert any("他人研究" in w for w in result.warnings)


@pytest.mark.parametrize(
    "sources",
    [
        _Sources([]),
        _Sources([_chunk("Unrelated source text.")]),
        _Sources([_chunk("Our sample retained 80% capacity.", page=5)]),
        _Sources([_chunk("Our sample retained 80% capacity.", doc="other-document")]),
        _MatrixStore([], []),
    ],
)
def test_unlocatable_sources_do_not_count_as_effective_evidence(tmp_path, sources):
    result = _prepare(tmp_path, _saved([_evidence()]), sources)
    assert result.evidence_ids == ()
    assert "待核对摘要 E-body" not in result.literature_markdown
    assert any("无法核对原文" in w for w in result.warnings)
    assert result.document_ids == ("doc-1",)


def test_source_read_failure_is_distinguished_without_leaking_error_payload(tmp_path):
    result = _prepare(
        tmp_path, _saved([_evidence()]), _Sources([], RuntimeError("secret payload"))
    )
    assert result.evidence_ids == ()
    assert any("原文读取失败" in w for w in result.warnings)
    assert not any("secret payload" in w for w in result.warnings)


def test_source_read_failure_must_not_enable_pending_numeric_handoff(tmp_path):
    sources = _Sources([], RuntimeError("secret payload"))
    sources.groups = [_group()]
    sources.measurements = [_measurement()]
    with pytest.raises(ValueError, match="原文读取失败"):
        _prepare(tmp_path, _saved([_evidence()]), sources)
    assert not (tmp_path / "analysis").exists()


@pytest.mark.parametrize(
    "narrative",
    [
        "## 综合结论\n错误引用参考文献支持实验结论 [E-ref]，这是旧的综合文字。",
        "## 综合结论\n旧综合文字没有完整引用标记，仍可能包含已隔离内容。",
    ],
)
def test_any_removal_pauses_old_narrative_and_shows_only_retained_evidence(
    tmp_path, narrative
):
    body, reference = _evidence(), _evidence("E-ref", "Reference title.", 12)
    result = _prepare(
        tmp_path,
        _saved([body, reference], narrative),
        _Sources(
            [
                _chunk(body.source_quote),
                _chunk("References\n" + reference.source_quote, 12, key="refs"),
            ]
        ),
    )
    assert narrative not in result.literature_markdown
    assert body.source_quote in result.literature_markdown
    assert any("旧综合文字" in w and "暂停复用" in w for w in result.warnings)


@pytest.mark.parametrize(
    "context",
    [
        "Our sample retained 80% capacity.\nReferences\n1. A citation.",
        "In this work, Our sample retained 80% capacity. [12]",
        "References\n1. A citation.\nAppendix A\nOur sample retained 80% capacity.",
    ],
)
def test_own_work_before_references_or_after_appendix_is_kept(tmp_path, context):
    result = _prepare(tmp_path, _saved([_evidence()]), _Sources([_chunk(context)]))
    assert result.evidence_ids == ("E-body",)


def test_unchanged_evidence_can_still_reuse_old_narrative(tmp_path):
    text = "## 综合结论\n保留已定位的原文证据，并且仍需人工复核 [E-body]。"
    result = _prepare(
        tmp_path,
        _saved([_evidence()], text),
        _Sources([_chunk(_evidence().source_quote)]),
    )
    assert result.literature_markdown == text


def test_ambiguous_own_and_reference_matches_are_not_cherry_picked(tmp_path):
    quote = _evidence().source_quote
    result = _prepare(
        tmp_path,
        _saved([_evidence()]),
        _Sources(
            [
                _chunk(quote, key="body"),
                _chunk("References\n" + quote, key="ref"),
            ]
        ),
    )
    assert result.evidence_ids == ()


def test_blank_quote_is_not_located_by_empty_string_matching(tmp_path):
    result = _prepare(
        tmp_path, _saved([_evidence(quote=" ")]), _Sources([_chunk("Text.")])
    )
    assert result.evidence_ids == ()


def test_quote_matching_does_not_change_unit_case(tmp_path):
    result = _prepare(
        tmp_path,
        _saved([_evidence(quote="Resistance was 5 mΩ.")]),
        _Sources([_chunk("Resistance was 5 MΩ.")]),
    )
    assert result.evidence_ids == ()


def test_whitespace_differences_and_multi_page_chunks_can_be_located(tmp_path):
    text = "Our sample\n retained 80% capacity."
    original = _chunk(text, page=3)
    chunk = ChunkRecord(
        original.chunk_id,
        original.document_id,
        None,
        3,
        5,
        original.text,
        original.text_sha256,
    )
    result = _prepare(tmp_path, _saved([_evidence()]), _Sources([chunk]))
    assert result.evidence_ids == ("E-body",)


def test_medium_risk_is_not_promoted_and_counts_are_recalculated_in_memory():
    body = _evidence().model_copy(
        update={
            "risk_level": "medium",
            "display_status": "check_recommended",
        }
    )
    reference = _evidence("E-ref", "Reference title.", 12)
    report = _saved([body, reference])
    original = report.model_dump_json()
    filtered, _ = _review_saved_report_sources(
        report,
        _Sources(
            [
                _chunk(body.source_quote),
                _chunk("References\n" + reference.source_quote, 12, key="refs"),
            ]
        ),
    )
    paper = filtered.papers[0]
    assert paper.low_risk_count == 0
    assert paper.medium_risk_count == 1
    assert paper.excluded_unsafe_count == 1
    assert paper.evidence == (body,)
    assert report.model_dump_json() == original


def test_retained_sources_are_reused_once_without_changing_numeric_handoff(tmp_path):
    group, row = _group(), _measurement()
    source = _chunk(group.source_quote, 2, key="chunk-1")
    group = group.model_copy(update={"source_text_sha256": source.text_sha256})
    row = row.model_copy(update={"source_text_sha256": source.text_sha256})
    item = _evidence(quote=group.source_quote, page=2)
    sources = _Sources([source])
    sources.groups, sources.measurements = [group], [row]
    before = (group.model_dump_json(), row.model_dump_json())
    datasets = DatasetStore(tmp_path / "analysis")
    result = LiteratureEvidenceTrialService(
        report_store=_ReportStore(_saved([item])),
        matrix_store=sources,
        dataset_store=datasets,
    ).prepare(topic="任务5")
    assert result.record_count == 1
    assert sources.calls == ["doc-1"]
    actual = datasets.load_dataframe(result.dataset_id).iloc[0]
    assert actual["numeric_value"] == 60.0
    assert actual["numeric_lower"] == 50.0
    assert actual["numeric_upper"] == 70.0
    assert result.required_metric_coverage.status == "not_checked"
    assert before == (group.model_dump_json(), row.model_dump_json())


def test_isolating_qualitative_evidence_does_not_remove_separate_safe_measurement(
    tmp_path,
):
    group, row = _group(), _measurement()
    reference = _evidence("E-ref", "Reference title.", 12)
    sources = _Sources(
        [
            _chunk(group.source_quote, 2, key="chunk-1"),
            _chunk("References\n" + reference.source_quote, 12, key="refs"),
        ]
    )
    sources.groups, sources.measurements = [group], [row]
    result = _prepare(tmp_path, _saved([reference]), sources)
    assert "E-ref" not in result.evidence_ids
    assert result.record_count == 1
    assert result.dataset_id is not None
    assert "Reference title." not in result.literature_markdown


def test_missing_one_paper_does_not_discard_other_papers_sources(tmp_path):
    item = _evidence()
    report = _saved([item])
    missing = report.papers[0].model_copy(
        update={
            "document_id": "doc-missing",
            "title": "Missing original",
            "evidence": (_evidence("E-missing"),),
        }
    )
    report = report.model_copy(update={"papers": (*report.papers, missing)})
    result = _prepare(tmp_path, report, _Sources([_chunk(item.source_quote)]))
    assert result.document_ids == ("doc-1", "doc-missing")
    assert result.evidence_ids == ("E-body",)
    assert any("E-missing" in w and "无法核对原文" in w for w in result.warnings)


def test_reuse_does_not_rewrite_saved_report_file(tmp_path: Path):
    store = LiteratureUserReportStore(tmp_path / "reports")
    item = _evidence("E-ref", "Reference title.")
    report = _saved([item])
    store.save(report)
    files = {
        path: path.read_bytes()
        for path in (tmp_path / "reports").rglob("*")
        if path.is_file()
    }
    result = LiteratureEvidenceTrialService(
        report_store=store,
        matrix_store=_Sources([_chunk("References\n" + item.source_quote)]),
        dataset_store=DatasetStore(tmp_path / "analysis"),
    ).prepare(topic=report.topic)
    assert result.evidence_ids == ()
    assert files == {
        path: path.read_bytes()
        for path in (tmp_path / "reports").rglob("*")
        if path.is_file()
    }
