"""Factory for the LiteratureAgent SubAgentSpec."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import SecretStr

from materials_screening.agent.conversation_store import SqliteConversationStore
from materials_screening.agent.intern_model import InternAgentModel
from materials_screening.agent.model_base import MaterialAgentModel
from materials_screening.agent.runner import MaterialAgentRunner
from materials_screening.agent.settings import AgentSettings, shared_intern_settings
from materials_screening.agent_tools.result_reader import FileWorkflowResultReader
from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.master import SubAgentSpec
from materials_screening.master.literature_evidence_trial import (
    LiteratureEvidenceTrialService,
)

from .evidence_runner import EvidenceAwareLiteratureRunner
from .extraction import ExperimentalExtractionService
from .integration import LiteratureIntegrationService
from .mock_model import LiteratureMockModel
from .pgvector_store import PgVectorLiteratureStore
from .prompt import SYSTEM_PROMPT
from .providers import (
    OfflineLiteratureProvider,
    OpenAlexProvider,
    SemanticScholarProvider,
)
from .rag import (
    BgeM3EmbeddingProvider,
    FlagEmbeddingReranker,
    LiteratureRagService,
    PyMuPdfParser,
)
from .service import LiteratureSearchService
from .store import LiteratureQueryStore
from .tools import build_tool_registry
from .user_report import LiteratureUserReportStore


def create_spec(
    *,
    workflow_runner: object,
    result_reader: FileWorkflowResultReader,
    query_root: Path = Path("data/literature_queries"),
    user_report_root: Path = Path("data/literature_user_reports"),
    data_analysis_root: Path = Path("data/data_analysis"),
    intern_api_key: str | None = None,
    s2_api_key: str | None = None,
    literature_database_url: str | None = None,
    ingestion_roots: tuple[Path, ...] = (),
    enable_reranker: bool = True,
    enable_remote_search: bool | None = None,
    enable_rag: bool = True,
) -> SubAgentSpec:
    settings = AgentSettings(
        agent_system_prompt=SYSTEM_PROMPT,
        agent_max_model_calls_per_turn=4,
        agent_max_tool_calls_per_turn=3,
        agent_allow_multi_step_tools=True,
        agent_max_tool_output_bytes=262_144,
        agent_max_input_bytes=524_288,
    )
    remote_search = (
        intern_api_key is not None
        if enable_remote_search is None
        else enable_remote_search
    )
    providers = (
        (
            OpenAlexProvider(mailto=os.getenv("OPENALEX_MAILTO")),
            SemanticScholarProvider(api_key=s2_api_key),
        )
        if remote_search
        else (
            OfflineLiteratureProvider("openalex"),
            OfflineLiteratureProvider("semantic_scholar"),
        )
    )
    query_store = LiteratureQueryStore(query_root)
    service = LiteratureSearchService(providers, query_store)
    database_url = literature_database_url or os.getenv("LITERATURE_DATABASE_URL")
    configured_roots = ingestion_roots or _ingestion_roots_from_env()
    rag_service: LiteratureRagService | None = None
    extraction_service: ExperimentalExtractionService | None = None
    integration_service: LiteratureIntegrationService | None = None
    vector_store: PgVectorLiteratureStore | None = None
    if database_url:
        vector_store = PgVectorLiteratureStore(database_url)
        integration_service = LiteratureIntegrationService(query_store, vector_store)
    if enable_rag and database_url and configured_roots:
        embeddings = BgeM3EmbeddingProvider(
            revision=os.getenv("LITERATURE_EMBEDDING_REVISION") or None,
            device=os.getenv("LITERATURE_MODEL_DEVICE", "cpu"),
        )
        vector_store = PgVectorLiteratureStore(
            database_url, dimensions=embeddings.dimensions
        )
        reranker = (
            FlagEmbeddingReranker(device=os.getenv("LITERATURE_MODEL_DEVICE", "cpu"))
            if enable_reranker
            else None
        )
        rag_service = LiteratureRagService(
            parser=PyMuPdfParser(
                max_pages=int(os.getenv("LITERATURE_MAX_PDF_PAGES", "500"))
            ),
            embeddings=embeddings,
            store=vector_store,
            reranker=reranker,
            ingestion_roots=configured_roots,
            max_pdf_bytes=int(os.getenv("LITERATURE_MAX_PDF_MB", "50")) * 1024 * 1024,
        )
        extraction_service = ExperimentalExtractionService(vector_store)
    registry = build_tool_registry(
        service, rag_service, extraction_service, integration_service
    )

    evidence_service = (
        LiteratureEvidenceTrialService(
            report_store=LiteratureUserReportStore(user_report_root),
            matrix_store=vector_store,
            dataset_store=DatasetStore(data_analysis_root),
        )
        if vector_store is not None
        else None
    )

    def runner_factory() -> MaterialAgentRunner | EvidenceAwareLiteratureRunner:
        store = SqliteConversationStore(Path("data/literature_conversations.sqlite"))
        checkpoint = Path("data/literature_checkpoints.sqlite")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(checkpoint), check_same_thread=False)
        saver = SqliteSaver(connection)
        model: MaterialAgentModel
        if intern_api_key:
            model = InternAgentModel(
                shared_intern_settings(settings), api_key=SecretStr(intern_api_key)
            )
        else:
            model = LiteratureMockModel()
        base_runner = MaterialAgentRunner(
            settings=settings,
            store=store,
            workflow_runner=workflow_runner,  # type: ignore[arg-type]
            workflow_result_reader=result_reader,
            tool_registry=registry,
            agent_model=model,
            checkpointer=saver,
        )
        if evidence_service is None:
            return base_runner
        return EvidenceAwareLiteratureRunner(
            base_runner=base_runner,
            evidence_service=evidence_service,
        )

    return SubAgentSpec(
        name="literature",
        description=(
            "Search OpenAlex and Semantic Scholar for materials-science papers, "
            "deduplicate and rank metadata, reuse exact-topic local full-text "
            "evidence when available, and stage numeric measurements for the "
            "data-analysis agent."
        ),
        system_prompt=SYSTEM_PROMPT,
        tool_definitions=registry.definitions(),
        runner_factory=runner_factory,
    )


def _ingestion_roots_from_env() -> tuple[Path, ...]:
    raw = os.getenv("LITERATURE_INGEST_ROOTS", "").strip()
    if not raw:
        return ()
    return tuple(
        Path(value.strip()) for value in raw.split(os.pathsep) if value.strip()
    )
