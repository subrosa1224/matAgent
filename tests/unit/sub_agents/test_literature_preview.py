from __future__ import annotations

from pathlib import Path
from typing import Any

from materials_screening.llm.base import StructuredProviderResponse
from materials_screening.llm.errors import LLMStructuredOutputError
from materials_screening.sub_agents.literature.preview import (
    BoundaryFinding,
    PaperPreviewExtractor,
    PaperPreviewStore,
    PreviewBoundaryPolicy,
    PreviewCandidate,
    infer_preview_boundary_policy,
    preview_batch_id,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord


class FakeLlm:
    def __init__(self, candidate: PreviewCandidate) -> None:
        self.candidate = candidate
        self.calls = 0

    def generate_structured(self, **_: Any) -> StructuredProviderResponse[Any]:
        self.calls += 1
        return StructuredProviderResponse(
            parsed=self.candidate,
            provider="fake",
            model="fake",
            request_id="preview-1",
            latency_ms=0,
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            raw_output_sha256="a" * 64,
        )


class SequenceLlm:
    def __init__(self, outcomes: list[PreviewCandidate | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def generate_structured(self, **_: Any) -> StructuredProviderResponse[Any]:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return StructuredProviderResponse(
            parsed=outcome,
            provider="fake",
            model="fake",
            request_id=f"preview-{self.calls}",
            latency_ms=0,
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            raw_output_sha256="a" * 64,
        )


def _chunk() -> ChunkRecord:
    text = "Results: The porous scaffold promoted osteogenic differentiation."
    return ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        page_from=8,
        page_to=8,
        text=text,
        text_sha256="b" * 64,
    )


def _candidate(quote: str | None = None) -> PreviewCandidate:
    return PreviewCandidate(
        article_type="research",
        topic_relevance="core",
        research_question="研究多孔支架的成骨作用。",
        methods="采用细胞成骨分化实验。",
        key_findings="多孔支架促进成骨分化。",
        recommendation="deep_analyze",
        reason="与目标主题直接相关。",
        evidence_quote=quote or _chunk().text,
        chunk_id="chunk-1",
    )


def test_preview_uses_one_call_and_persists(tmp_path: Path) -> None:
    llm = FakeLlm(_candidate())
    preview = PaperPreviewExtractor(llm).extract(
        document_id="doc-1",
        title="Paper",
        topic="多孔支架成骨",
        chunks=(_chunk(),),
    )
    store = PaperPreviewStore(tmp_path)
    store.save(preview)

    assert llm.calls == 1
    assert preview.recommendation == "deep_analyze"
    assert preview.review_status == "preview_only"
    assert preview.evidence_quality == "verbatim"
    assert store.load("doc-1", "多孔支架成骨") == preview
    assert store.load("doc-1", "另一个主题") is None


def test_preview_degrades_nonverbatim_evidence_to_source_chunk() -> None:
    preview = PaperPreviewExtractor(FakeLlm(_candidate("invented quote"))).extract(
        document_id="doc-1",
        title="Paper",
        topic="多孔支架成骨",
        chunks=(_chunk(),),
    )

    assert preview.evidence_quality == "fallback_chunk"
    assert preview.evidence_quote == _chunk().text


def test_prompt_separates_chinese_summary_from_contiguous_source_quote() -> None:
    class CapturingLlm(FakeLlm):
        def generate_structured(self, **kwargs):
            self.prompt = kwargs["system_prompt"]
            return super().generate_structured(**kwargs)

    llm = CapturingLlm(_candidate())
    preview = PaperPreviewExtractor(llm).extract(
        document_id="doc-1", title="Paper", topic="多孔支架成骨", chunks=(_chunk(),)
    )
    assert "Summaries may paraphrase" in llm.prompt
    assert "one continuous passage from ONE supplied chunk" in " ".join(
        llm.prompt.split()
    )
    assert "Do not stitch" in llm.prompt
    assert "units" in llm.prompt and "typographical" in llm.prompt
    assert preview.evidence_quality == "verbatim"
    assert preview.key_findings == "多孔支架促进成骨分化。"


def test_rewritten_quote_rejected_without_rejecting_chinese_summary() -> None:
    from dataclasses import replace

    source = replace(_chunk(), text="Notable, the device responds at 113 AW − 1.")
    candidate = _candidate("Notably, the device responds at 113 AW^-1.")
    preview = PaperPreviewExtractor(FakeLlm(candidate)).extract(
        document_id="doc-1", title="Paper", topic="多孔支架成骨", chunks=(source,)
    )
    assert preview.evidence_quality == "fallback_chunk"
    assert preview.proposed_evidence_quote == candidate.evidence_quote
    assert preview.evidence_quote == source.text
    assert preview.key_findings == candidate.key_findings


def test_numbered_passage_selection_copies_source_not_generated_quote():
    from dataclasses import replace

    from materials_screening.sub_agents.literature.preview import (
        PreviewSelectionCandidate,
    )

    source = replace(_chunk(), text="Notable, the device responds at 113 AW − 1.")

    class SelectingLlm:
        def generate_structured(self, **kwargs):
            assert kwargs["output_model"] is PreviewSelectionCandidate
            assert "passage-0001" in kwargs["user_text"]
            assert source.text in kwargs["user_text"]
            return StructuredProviderResponse(
                parsed=PreviewSelectionCandidate(
                    **_candidate().model_dump(exclude={"evidence_quote", "chunk_id"}),
                    passage_id="passage-0001",
                ),
                provider="fake",
                model="fake",
                request_id="selection",
                latency_ms=0,
                input_tokens=1,
                output_tokens=1,
                reasoning_tokens=0,
                raw_output_sha256=None,
            )

    preview = PaperPreviewExtractor(SelectingLlm()).extract(
        document_id="doc-1", title="Paper", topic="多孔支架成骨", chunks=(source,)
    )
    assert preview.evidence_quote == source.text
    assert preview.evidence_quality == "verbatim"
    assert preview.evidence_origin == "selected_passage"
    assert preview.selected_passage_id == "passage-0001"
    assert preview.chunk_id == source.chunk_id
    assert preview.key_findings == _candidate().key_findings


def test_unknown_passage_is_rejected_not_replaced_by_default_source():
    import pytest

    from materials_screening.sub_agents.literature.preview import (
        PreviewSelectionCandidate,
    )

    candidate = PreviewSelectionCandidate(
        **_candidate().model_dump(exclude={"evidence_quote", "chunk_id"}),
        passage_id="passage-9999",
    )
    with pytest.raises(ValueError, match="passage"):
        PaperPreviewExtractor(FakeLlm(candidate)).extract(
            document_id="doc-1", title="Paper", topic="多孔支架成骨", chunks=(_chunk(),)
        )


def test_preview_batch_id_is_order_independent() -> None:
    assert preview_batch_id("topic", ("doc-a", "doc-b")) == preview_batch_id(
        "topic", ("doc-b", "doc-a")
    )


def test_preview_retries_invalid_structured_output() -> None:
    llm = SequenceLlm([LLMStructuredOutputError("not valid JSON"), _candidate()])

    preview = PaperPreviewExtractor(llm).extract(
        document_id="doc-1",
        title="Paper",
        topic="多孔支架成骨",
        chunks=(_chunk(),),
    )
    assert llm.calls == 2
    assert preview.research_question == "研究多孔支架的成骨作用。"


def test_passage_budget_never_exposes_partial_or_hidden_identifier():
    from dataclasses import replace

    from materials_screening.sub_agents.literature.preview import _numbered_passages

    source = replace(_chunk(), text="A" * 700 + "B" * 700)
    passages, evidence = _numbered_passages((source,), 800)
    assert tuple(passages) == ("passage-0001",)
    assert passages["passage-0001"][1] == "A" * 700
    assert "passage-0002" not in evidence and "B" not in evidence
    assert len(evidence) <= 800


def test_preview_retries_english_summary() -> None:
    english = _candidate().model_copy(
        update={
            "research_question": "How does pore geometry influence osteogenesis?",
            "methods": "Cell differentiation assays were used.",
            "key_findings": "Porous scaffolds promoted osteogenesis.",
            "reason": "Directly relevant to the topic.",
        }
    )
    llm = SequenceLlm([english, _candidate()])

    PaperPreviewExtractor(llm).extract(
        document_id="doc-1",
        title="Paper",
        topic="多孔支架成骨",
        chunks=(_chunk(),),
    )

    assert llm.calls == 2


def test_preview_falls_back_to_safe_chinese_after_two_english_outputs() -> None:
    english = _candidate().model_copy(
        update={
            "research_question": "What is the research question?",
            "methods": "Directional freeze-casting was used.",
            "key_findings": "The scaffold affected bone regeneration.",
            "reason": "Potentially relevant.",
        }
    )
    llm = SequenceLlm([english, english])

    preview = PaperPreviewExtractor(llm).extract(
        document_id="doc-1",
        title="Paper",
        topic="多孔支架成骨",
        chunks=(_chunk(),),
    )

    assert llm.calls == 2
    assert preview.recommendation == "background_only"
    assert "安全中文回退" in preview.reason
    assert preview.generation_status == "safe_fallback"
    assert "不生成未经核验" in preview.key_findings


def test_preview_retries_method_copied_from_user_topic() -> None:
    misleading = _candidate().model_copy(update={"methods": "采用3D打印制备支架。"})
    llm = SequenceLlm([misleading, _candidate()])

    preview = PaperPreviewExtractor(llm).extract(
        document_id="doc-1",
        title="Directionally frozen scaffold",
        topic="3D打印支架成骨",
        chunks=(_chunk(),),
    )

    assert llm.calls == 2
    assert "3D打印" not in preview.methods


def test_preview_preserves_digits_in_assay_and_scientific_terms() -> None:
    candidate = _candidate().model_copy(
        update={
            "methods": "采用CCK-8与3D培养评价细胞增殖。",
        }
    )

    preview = PaperPreviewExtractor(FakeLlm(candidate)).extract(
        document_id="doc-1",
        title="Paper",
        topic="多孔支架成骨",
        chunks=(_chunk(),),
    )

    assert "CCK-8" in preview.methods
    assert "3D" in preview.methods


def test_material_scope_violation_forces_exclusion() -> None:
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        page_from=1,
        page_to=1,
        text=(
            "The hydroxyapatite contains Mg, Na, Sr, Fe and other trace "
            "minerals naturally present in bone."
        ),
        text_sha256="c" * 64,
    )
    candidate = _candidate(chunk.text).model_copy(
        update={"key_findings": "产物含有多种天然微量元素。"}
    )

    preview = PaperPreviewExtractor(
        FakeLlm(candidate),
        boundary_policy=PreviewBoundaryPolicy(
            excluded_material_term_groups=(("微量元素", "trace minerals"),),
        ),
    ).extract(
        document_id="doc-1",
        title="Trace-mineral hydroxyapatite",
        topic="仅研究纯、未掺杂羟基磷灰石",
        chunks=(chunk,),
    )

    assert preview.recommendation == "exclude"
    assert preview.topic_relevance == "low"
    assert preview.boundary_findings[0].finding_type == "material_scope"
    assert "硬边界" in preview.reason


def test_route_scope_violation_caps_preview_at_background() -> None:
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        page_from=1,
        page_to=1,
        text="Hydroxyapatite was deposited on titanium by electrophoretic deposition.",
        text_sha256="d" * 64,
    )
    candidate = _candidate(chunk.text).model_copy(
        update={"methods": "采用电泳沉积法在钛基底上制备。"}
    )

    preview = PaperPreviewExtractor(
        FakeLlm(candidate),
        boundary_policy=PreviewBoundaryPolicy(
            allowed_route_terms=("沉淀法", "水热法", "溶胶-凝胶法"),
        ),
    ).extract(
        document_id="doc-1",
        title="Electrophoretic deposition",
        topic="仅限沉淀法、水热法或溶胶-凝胶法",
        chunks=(chunk,),
    )

    assert preview.recommendation == "background_only"
    assert preview.topic_relevance == "extended"


def test_uncorroborated_model_boundary_finding_is_not_enforced() -> None:
    finding = BoundaryFinding(
        finding_type="material_scope",
        severity="exclude",
        reason="声称存在掺杂，但没有原文证据。",
        evidence_quote="invented dopant evidence",
        chunk_id="chunk-1",
    )
    candidate = _candidate().model_copy(update={"boundary_findings": (finding,)})

    preview = PaperPreviewExtractor(
        FakeLlm(candidate),
        boundary_policy=PreviewBoundaryPolicy(
            excluded_material_term_groups=(("微量元素", "trace minerals"),),
        ),
    ).extract(
        document_id="doc-1",
        title="Paper",
        topic="仅研究纯材料",
        chunks=(_chunk(),),
    )

    assert preview.recommendation == "deep_analyze"
    assert preview.boundary_findings == ()


def test_allowed_route_in_title_prevents_false_background_cap() -> None:
    candidate = _candidate().model_copy(
        update={"methods": "论文制备方法需结合下方代表原文核对。"}
    )

    preview = PaperPreviewExtractor(
        FakeLlm(candidate),
        boundary_policy=PreviewBoundaryPolicy(
            allowed_route_terms=("水热", "hydrothermal"),
        ),
    ).extract(
        document_id="doc-1",
        title="Hydrothermal synthesis of hydroxyapatite",
        topic="仅限水热法",
        chunks=(_chunk(),),
    )

    assert preview.recommendation == "deep_analyze"
    assert preview.boundary_findings == ()


def test_hallucinated_allowed_route_in_summary_does_not_bypass_policy() -> None:
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        page_from=1,
        page_to=1,
        text=(
            "Room-temperature phosphorylation and subsequent calcination "
            "produce pure hydroxyapatite."
        ),
        text_sha256="e" * 64,
    )
    candidate = _candidate(chunk.text).model_copy(
        update={"methods": "使用沉淀法合成纯羟基磷灰石。"}
    )

    preview = PaperPreviewExtractor(
        FakeLlm(candidate),
        boundary_policy=PreviewBoundaryPolicy(
            allowed_route_terms=("沉淀", "precipitation", "水热", "hydrothermal"),
        ),
    ).extract(
        document_id="doc-1",
        title="Pure hydroxyapatite synthesis from amorphous calcium carbonate",
        topic="路线仅限沉淀法或水热法",
        chunks=(chunk,),
    )

    assert preview.recommendation == "background_only"
    assert preview.boundary_findings[0].finding_type == "route_scope"


def test_task_text_infers_material_and_route_boundary_policy() -> None:
    policy = infer_preview_boundary_policy(
        "仅研究纯、未掺杂nHA，路线限于沉淀法、水热法或溶胶-凝胶法。"
    )

    assert policy is not None
    assert "水热" in policy.allowed_route_terms
    assert any(
        "微量元素" in aliases for aliases in policy.excluded_material_term_groups
    )


def test_unconstrained_topic_does_not_infer_boundary_policy() -> None:
    assert infer_preview_boundary_policy("羟基磷灰石研究进展") is None
