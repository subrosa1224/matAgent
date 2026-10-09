"""MA-1 tests for pluggable unified-result presentation."""

import pytest

from materials_screening.master import (
    ResultRendererRegistry,
    UnifiedResultEnvelope,
    default_renderer_registry,
    render_markdown_result,
)


def _envelope(result_type: str = "materials_answer") -> UnifiedResultEnvelope:
    return UnifiedResultEnvelope(
        agent_name="materials_database",
        status="completed",
        result_type=result_type,
        result={"markdown": "回答"},
    )


def test_default_registry_renders_both_domain_answers() -> None:
    registry = default_renderer_registry()
    assert registry.render(_envelope()) == "回答"
    assert registry.render(_envelope("literature_answer")) == "回答"
    assert registry.render(_envelope("analysis_answer")) == "回答"


def test_default_registry_renders_cross_agent_source_boundaries() -> None:
    envelope = UnifiedResultEnvelope(
        agent_name="master",
        status="partial",
        result_type="cross_agent_answer",
        result={
            "status": "failed",
            "clues": [],
            "candidates": [],
            "query_ids": [],
            "warnings": ["no validated clue"],
        },
    )
    rendered = default_renderer_registry().render(envelope)
    assert "论文中的组成仅作为检索线索" in rendered
    assert "Materials Project 候选" in rendered
    assert "no validated clue" in rendered


def test_renderer_registry_rejects_duplicate_and_unknown_types() -> None:
    registry = ResultRendererRegistry((("materials_answer", render_markdown_result),))
    with pytest.raises(ValueError, match="duplicate"):
        registry.register("materials_answer", render_markdown_result)
    with pytest.raises(KeyError, match="unknown"):
        registry.render(_envelope("missing_renderer"))


def test_markdown_renderer_requires_non_blank_payload() -> None:
    envelope = _envelope().model_copy(update={"result": {"markdown": ""}})
    with pytest.raises(ValueError, match="non-blank"):
        render_markdown_result(envelope)
