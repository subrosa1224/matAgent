from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from materials_screening.sub_agents.literature.models import (
    ExperimentalFactReview,
    KnowledgeEdge,
    LiteratureSearchInput,
)
from materials_screening.sub_agents.literature.providers import (
    normalize_doi,
    reconstruct_abstract,
)


def test_search_input_normalizes_and_validates() -> None:
    request = LiteratureSearchInput(
        topic="  titanium   dioxide photocatalysis ",
        material_keywords=(" TiO2 ", "TiO2"),
        year_from=2020,
        year_to=2025,
    )
    assert request.topic == "titanium dioxide photocatalysis"
    assert request.material_keywords == ("TiO2",)


def test_search_input_rejects_reversed_years_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LiteratureSearchInput(topic="TiO2", year_from=2025, year_to=2020)
    with pytest.raises(ValidationError):
        LiteratureSearchInput.model_validate({"topic": "TiO2", "unexpected": True})


def test_doi_and_openalex_abstract_normalization() -> None:
    assert normalize_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert (
        reconstruct_abstract({"photocatalysis": [2], "TiO2": [0], "for": [1]})
        == "TiO2 for photocatalysis"
    )


def test_review_and_knowledge_edge_contracts_are_strict() -> None:
    now = datetime.now(UTC)
    review = ExperimentalFactReview(
        review_id="review-1",
        fact_id="fact-1",
        previous_status="pending",
        decision="approved",
        reviewer="project-owner",
        reviewed_at=now,
    )
    assert review.decision == "approved"
    edge = KnowledgeEdge(
        edge_id="edge-1",
        fact_id="fact-1",
        document_id="doc-1",
        subject="CaP scaffold",
        predicate="has_parameter_performance_relation",
        object="porosity=70 % -> new bone formation=highest rate",
        variable_name="porosity",
        variable_value="70 %",
        performance_metric="new bone formation",
        performance_value="highest rate",
        chunk_id="chunk-1",
        page_from=11,
        page_to=11,
        source_quote="porosity of 70 % had the highest rate",
        source_text_sha256="a" * 64,
        created_at=now,
    )
    assert edge.review_status == "approved"
    with pytest.raises(ValidationError):
        KnowledgeEdge.model_validate({**edge.model_dump(), "review_status": "pending"})
