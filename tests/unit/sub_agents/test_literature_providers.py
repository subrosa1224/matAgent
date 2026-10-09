from __future__ import annotations

from collections.abc import Mapping

from materials_screening.sub_agents.literature.models import LiteratureSearchInput
from materials_screening.sub_agents.literature.providers import (
    LiteratureProviderError,
    OpenAlexProvider,
    SemanticScholarProvider,
)


def test_openalex_maps_bounded_request_and_record() -> None:
    captured: dict[str, object] = {}

    def transport(
        url: str, params: Mapping[str, str], headers: Mapping[str, str]
    ) -> dict[str, object]:
        captured.update(url=url, params=dict(params), headers=dict(headers))
        return {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "doi": "https://doi.org/10.1/TEST",
                    "title": "TiO2 photocatalysis",
                    "publication_year": 2024,
                    "authorships": [{"author": {"display_name": "A. Author"}}],
                    "primary_location": {
                        "landing_page_url": "https://example.test/paper",
                        "source": {"display_name": "Journal"},
                    },
                    "abstract_inverted_index": {"TiO2": [0], "works": [1]},
                    "cited_by_count": 7,
                    "open_access": {"is_oa": True},
                }
            ]
        }

    papers = OpenAlexProvider(transport=transport).search(
        LiteratureSearchInput(topic="photocatalysis", year_from=2020, max_papers=5)
    )
    assert captured["url"] == "https://api.openalex.org/works"
    assert captured["params"]["per-page"] == "5"  # type: ignore[index]
    assert "from_publication_date:2020-01-01" in captured["params"]["filter"]  # type: ignore[index]
    assert papers[0].doi == "10.1/test"
    assert papers[0].abstract == "TiO2 works"


def test_semantic_scholar_key_is_header_only() -> None:
    captured: dict[str, object] = {}

    def transport(
        url: str, params: Mapping[str, str], headers: Mapping[str, str]
    ) -> dict[str, object]:
        captured.update(params=dict(params), headers=dict(headers))
        return {
            "data": [
                {
                    "paperId": "S1",
                    "title": "Doped oxide",
                    "abstract": "Measured performance.",
                    "year": 2023,
                    "authors": [{"name": "Researcher"}],
                    "venue": "Venue",
                    "citationCount": 3,
                    "externalIds": {"DOI": "10.2/S2"},
                    "openAccessPdf": None,
                    "url": "https://semanticscholar.org/paper/S1",
                }
            ]
        }

    papers = SemanticScholarProvider(
        transport=transport,
        api_key="secret",
        min_request_interval_seconds=0,
    ).search(LiteratureSearchInput(topic="doped oxide"))
    assert captured["headers"]["x-api-key"] == "secret"  # type: ignore[index]
    assert "secret" not in captured["params"].values()  # type: ignore[union-attr]
    assert papers[0].doi == "10.2/s2"


def test_openalex_key_is_header_only() -> None:
    captured: dict[str, object] = {}

    def transport(
        url: str, params: Mapping[str, str], headers: Mapping[str, str]
    ) -> dict[str, object]:
        captured.update(params=dict(params), headers=dict(headers), url=url)
        return {"results": []}

    OpenAlexProvider(transport=transport, api_key="test-secret").search(
        LiteratureSearchInput(topic="ZnO ultraviolet detector")
    )
    assert captured["headers"]["Authorization"] == "Bearer test-secret"  # type: ignore[index]
    assert "test-secret" not in captured["url"]  # type: ignore[operator]
    assert "test-secret" not in captured["params"].values()  # type: ignore[union-attr]


def test_semantic_scholar_maps_json_rate_limit_response() -> None:
    def transport(
        url: str, params: Mapping[str, str], headers: Mapping[str, str]
    ) -> dict[str, object]:
        return {"message": "Too Many Requests", "code": "429"}

    provider = SemanticScholarProvider(
        transport=transport, min_request_interval_seconds=0
    )

    try:
        provider.search(LiteratureSearchInput(topic="doped oxide"))
    except LiteratureProviderError as exc:
        assert exc.code == "PROVIDER_RATE_LIMIT"
    else:
        raise AssertionError("rate-limit payload must raise")


def test_semantic_scholar_accepts_zero_result_response_without_data() -> None:
    def transport(
        url: str, params: Mapping[str, str], headers: Mapping[str, str]
    ) -> dict[str, object]:
        return {"total": 0, "offset": 0}

    papers = SemanticScholarProvider(
        transport=transport, min_request_interval_seconds=0
    ).search(LiteratureSearchInput(topic="没有匹配结果的主题"))

    assert papers == ()
