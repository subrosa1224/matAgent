"""Whitelisted tools for the literature sub-agent."""

from __future__ import annotations

from typing import cast

from materials_screening.agent.context import AgentToolContext
from materials_screening.agent.tool_base import AgentTool, ToolSideEffect
from materials_screening.agent.tool_registry import AgentToolRegistry

from .extraction import ExperimentalExtractionService
from .integration import LiteratureIntegrationService
from .models import (
    AssembleLiteratureResultInput,
    CandidateLiteratureScreenInput,
    CandidateLiteratureScreenOutput,
    ExtractExperimentalDataInput,
    ExtractExperimentalDataOutput,
    IngestDocumentsInput,
    IngestDocumentsOutput,
    LiteratureResult,
    LiteratureSearchInput,
    LiteratureSearchOutput,
    RagRetrieveInput,
    RagRetrieveOutput,
)
from .providers import LiteratureProviderError
from .rag import LiteratureRagService
from .service import LiteratureSearchService


class _SearchTool:
    name: str
    provider_name: str
    side_effect = ToolSideEffect.READ_ONLY
    input_model = LiteratureSearchInput
    output_model = LiteratureSearchOutput

    def __init__(self, service: LiteratureSearchService) -> None:
        self._service = service

    def execute(
        self, arguments: LiteratureSearchInput, context: AgentToolContext
    ) -> LiteratureSearchOutput:
        result = self._service.search(self.provider_name, arguments)
        evidence_id = context.id_generator.new_id()
        result = result.model_copy(update={"evidence_id": evidence_id})
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=evidence_id,
            result_json=result.model_dump_json(),
            side_effect=self.side_effect,
        )
        return result


class OpenAlexSearchTool(_SearchTool):
    name = "openalex_search"
    provider_name = "openalex"
    description = (
        "Search OpenAlex for paper metadata using a bounded topic, material keywords, "
        "optional year range, and result limit."
    )


class SemanticScholarSearchTool(_SearchTool):
    name = "s2_search"
    provider_name = "semantic_scholar"
    description = (
        "Search Semantic Scholar for paper metadata and abstracts using a "
        "bounded topic, material keywords, optional year range, and result limit."
    )

    def execute(
        self, arguments: LiteratureSearchInput, context: AgentToolContext
    ) -> LiteratureSearchOutput:
        try:
            return super().execute(arguments, context)
        except LiteratureProviderError as exc:
            if exc.code not in {"PROVIDER_RATE_LIMIT", "PROVIDER_UNAVAILABLE"}:
                raise
            result = self._service.degraded_result(
                self.provider_name,
                arguments,
                "Semantic Scholar is temporarily unavailable; use OpenAlex results.",
            )
            evidence_id = context.id_generator.new_id()
            result = result.model_copy(update={"evidence_id": evidence_id})
            context.ledger.record(
                call_id=context.call_id,
                tool_name=self.name,
                evidence_id=evidence_id,
                result_json=result.model_dump_json(),
                side_effect=self.side_effect,
            )
            return result


class UnifiedLiteratureSearchTool:
    name = "literature_search"
    description = (
        "Expand a materials-science topic, search OpenAlex and Semantic Scholar, "
        "merge duplicate papers, rank a unified candidate list, and save one "
        "reproducible query snapshot. Prefer this tool for user topic searches."
    )
    input_model = LiteratureSearchInput
    output_model = LiteratureSearchOutput
    side_effect = ToolSideEffect.READ_ONLY

    def __init__(self, service: LiteratureSearchService) -> None:
        self._service = service

    def execute(
        self, arguments: LiteratureSearchInput, context: AgentToolContext
    ) -> LiteratureSearchOutput:
        result = self._service.search_unified(arguments)
        evidence_id = context.id_generator.new_id()
        result = result.model_copy(update={"evidence_id": evidence_id})
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=evidence_id,
            result_json=result.model_dump_json(),
            side_effect=self.side_effect,
        )
        return result


class CandidateLiteratureScreenTool:
    name = "screen_candidate_literature"
    description = (
        "Pre-screen 1 to 100 ordered material formulas for one application, grade "
        "formula-matched literature as A/B/C/NONE, rerank candidates by evidence, "
        "and return a bounded final shortlist plus paper download links. Use this "
        "for an explicit 候选池文献预检 task instead of ordinary topic search."
    )
    input_model = CandidateLiteratureScreenInput
    output_model = CandidateLiteratureScreenOutput
    side_effect = ToolSideEffect.READ_ONLY

    def __init__(self, service: LiteratureSearchService) -> None:
        self._service = service

    def execute(
        self, arguments: CandidateLiteratureScreenInput, context: AgentToolContext
    ) -> CandidateLiteratureScreenOutput:
        result = self._service.screen_candidates(arguments)
        # Full abstracts remain in the immutable snapshot. Keep tool handoffs
        # compact, rather than sending 100 multi-paper bundles into the model.
        result = result.model_copy(
            update={
                "candidates": tuple(
                    row.model_copy(update={"papers": ()}) for row in result.candidates
                ),
                "finalists": tuple(
                    row.model_copy(update={"papers": ()}) for row in result.finalists
                ),
                "supplementary_finalists": tuple(
                    row.model_copy(update={"papers": ()})
                    for row in result.supplementary_finalists
                ),
                "download_candidates": tuple(
                    paper.model_copy(update={"abstract": None})
                    for paper in result.download_candidates
                ),
            }
        )
        evidence_id = context.id_generator.new_id()
        result = result.model_copy(update={"evidence_id": evidence_id})
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=evidence_id,
            result_json=result.model_dump_json(),
            side_effect=self.side_effect,
        )
        return result


class IngestLiteratureDocumentsTool:
    name = "ingest_literature_documents"
    description = (
        "Index user-authorized local PDF files for literature RAG. Paths must be "
        "inside configured ingestion roots. Never use this tool without an explicit "
        "user request to ingest or index files."
    )
    input_model = IngestDocumentsInput
    output_model = IngestDocumentsOutput
    side_effect = ToolSideEffect.CREATE_WORKFLOW_RUN

    def __init__(self, service: LiteratureRagService) -> None:
        self._service = service

    def execute(
        self, arguments: IngestDocumentsInput, context: AgentToolContext
    ) -> IngestDocumentsOutput:
        documents = self._service.ingest(arguments.paths, arguments.paper_id)
        evidence_id = context.id_generator.new_id()
        result = IngestDocumentsOutput(
            documents=documents,
            evidence_id=evidence_id,
        )
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=evidence_id,
            result_json=result.model_dump_json(),
            side_effect=self.side_effect,
        )
        return result


class RagRetrieveTool:
    name = "rag_retrieve"
    description = (
        "Retrieve evidence chunks from previously indexed authorized PDFs using "
        "bge-m3 vector search and optional reranking."
    )
    input_model = RagRetrieveInput
    output_model = RagRetrieveOutput
    side_effect = ToolSideEffect.READ_ONLY

    def __init__(self, service: LiteratureRagService) -> None:
        self._service = service

    def execute(
        self, arguments: RagRetrieveInput, context: AgentToolContext
    ) -> RagRetrieveOutput:
        result = self._service.retrieve(arguments)
        evidence_id = context.id_generator.new_id()
        result = result.model_copy(update={"evidence_id": evidence_id})
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=evidence_id,
            result_json=result.model_dump_json(),
            side_effect=self.side_effect,
        )
        return result


class ExtractExperimentalDataTool:
    name = "extract_experimental_data"
    description = (
        "Validate LLM-proposed experimental rows against exact indexed PDF "
        "evidence and stage accepted rows for human review."
    )
    input_model = ExtractExperimentalDataInput
    output_model = ExtractExperimentalDataOutput
    side_effect = ToolSideEffect.CREATE_WORKFLOW_RUN

    def __init__(self, service: ExperimentalExtractionService) -> None:
        self._service = service

    def execute(
        self, arguments: ExtractExperimentalDataInput, context: AgentToolContext
    ) -> ExtractExperimentalDataOutput:
        result = self._service.validate_and_save(
            document_id=arguments.document_id,
            candidates=arguments.rows,
        )
        evidence_id = context.id_generator.new_id()
        result = result.model_copy(update={"evidence_id": evidence_id})
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=evidence_id,
            result_json=result.model_dump_json(),
            side_effect=self.side_effect,
        )
        return result


class AssembleLiteratureResultTool:
    name = "assemble_literature_result"
    description = (
        "Assemble saved paper searches, human-approved experimental tables, and "
        "approved knowledge edges into the strict LiteratureResult schema."
    )
    input_model = AssembleLiteratureResultInput
    output_model = LiteratureResult
    side_effect = ToolSideEffect.READ_ONLY

    def __init__(self, service: LiteratureIntegrationService) -> None:
        self._service = service

    def execute(
        self, arguments: AssembleLiteratureResultInput, context: AgentToolContext
    ) -> LiteratureResult:
        result = self._service.assemble(
            query_ids=arguments.query_ids,
            document_id=arguments.document_id,
        )
        evidence_id = context.id_generator.new_id()
        result = result.model_copy(update={"evidence_id": evidence_id})
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=evidence_id,
            result_json=result.model_dump_json(),
            side_effect=self.side_effect,
        )
        return result


def build_tool_registry(
    service: LiteratureSearchService,
    rag_service: LiteratureRagService | None = None,
    extraction_service: ExperimentalExtractionService | None = None,
    integration_service: LiteratureIntegrationService | None = None,
) -> AgentToolRegistry:
    tools: list[AgentTool] = [
        cast(AgentTool, CandidateLiteratureScreenTool(service)),
        cast(AgentTool, UnifiedLiteratureSearchTool(service)),
        cast(AgentTool, OpenAlexSearchTool(service)),
        cast(AgentTool, SemanticScholarSearchTool(service)),
    ]
    if rag_service is not None:
        tools.extend(
            (
                cast(AgentTool, IngestLiteratureDocumentsTool(rag_service)),
                cast(AgentTool, RagRetrieveTool(rag_service)),
            )
        )
    if extraction_service is not None:
        tools.append(cast(AgentTool, ExtractExperimentalDataTool(extraction_service)))
    if integration_service is not None:
        tools.append(cast(AgentTool, AssembleLiteratureResultTool(integration_service)))
    return AgentToolRegistry(tools)
