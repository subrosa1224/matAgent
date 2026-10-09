from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from materials_screening.agent.context import AgentToolContext
from materials_screening.agent.ledger import ToolExecutionLedger
from materials_screening.agent.tool_registry import AgentToolRegistry
from materials_screening.sub_agents.literature.models import (
    LiteratureSearchInput,
    PaperProvenance,
    PaperRecord,
)
from materials_screening.sub_agents.literature.service import LiteratureSearchService
from materials_screening.sub_agents.literature.store import LiteratureQueryStore
from materials_screening.sub_agents.literature.tools import (
    OpenAlexSearchTool,
    build_tool_registry,
)


class StubIdGenerator:
    def new_id(self) -> str:
        return "evidence-literature-1"


class StubProvider:
    name = "openalex"

    def search(self, request: LiteratureSearchInput) -> tuple[PaperRecord, ...]:
        provenance = PaperProvenance(
            provider="openalex",
            provider_id="W1",
            retrieved_at=datetime.now(UTC),
            raw_record_sha256="a" * 64,
        )
        return (
            PaperRecord(
                paper_id="p2",
                title="Unrelated",
                doi="10.1/b",
                year=2020,
                provenance=(provenance,),
            ),
            PaperRecord(
                paper_id="p1",
                title="TiO2 photocatalysis",
                doi="10.1/a",
                year=2024,
                abstract="TiO2 photocatalysis performance",
                cited_by_count=10,
                provenance=(provenance,),
            ),
            PaperRecord(
                paper_id="duplicate",
                title="Duplicate",
                doi="10.1/a",
                year=2024,
                provenance=(provenance,),
            ),
        )


class StubLedger:
    def __init__(self) -> None:
        self.recorded: dict[str, object] = {}

    def record(self, **kwargs: object) -> None:
        self.recorded = kwargs


def test_service_ranks_deduplicates_and_persists(tmp_path: Path) -> None:
    service = LiteratureSearchService((StubProvider(),), LiteratureQueryStore(tmp_path))
    result = service.search(
        "openalex", LiteratureSearchInput(topic="TiO2 photocatalysis")
    )
    assert [paper.paper_id for paper in result.papers] == ["p1", "p2"]
    assert service.store.load(result.query_id).papers == result.papers
    assert (
        service.search("openalex", LiteratureSearchInput(topic="TiO2 photocatalysis"))
        == result
    )


def test_tool_registry_and_evidence_recording(tmp_path: Path) -> None:
    service = LiteratureSearchService((StubProvider(),), LiteratureQueryStore(tmp_path))
    registry = build_tool_registry(service)
    assert registry.names() == (
        "literature_search",
        "openalex_search",
        "s2_search",
        "screen_candidate_literature",
    )
    ledger = StubLedger()
    context = AgentToolContext(
        workflow_runner=cast(object, object()),  # type: ignore[arg-type]
        workflow_result_reader=cast(object, object()),  # type: ignore[arg-type]
        clock=lambda: datetime.now(UTC),
        id_generator=StubIdGenerator(),
        ledger=cast(ToolExecutionLedger, ledger),
        call_id="call-1",
        user_turn_id="turn-1",
        conversation_id="conversation-1",
    )
    result = OpenAlexSearchTool(service).execute(
        LiteratureSearchInput(topic="TiO2"), context
    )
    assert result.evidence_id
    assert ledger.recorded["tool_name"] == "openalex_search"


def test_global_registry_accepts_literature_tools(tmp_path: Path) -> None:
    service = LiteratureSearchService((StubProvider(),), LiteratureQueryStore(tmp_path))
    assert isinstance(
        AgentToolRegistry((OpenAlexSearchTool(service),)), AgentToolRegistry
    )
