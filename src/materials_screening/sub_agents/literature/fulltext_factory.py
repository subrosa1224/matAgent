"""Lazy normal-entry factories; metadata startup loads no vector or LLM model."""

from __future__ import annotations

import os
from pathlib import Path

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.llm.factory import create_llm_provider
from materials_screening.master.artifact_registry import ArtifactRegistry
from materials_screening.master.fulltext_analysis import FulltextAnalysisProcessor
from materials_screening.master.fulltext_analysis_store import FulltextAnalysisStore
from materials_screening.master.fulltext_condition_locator import (
    locate_shared_conditions,
    read_located_values,
)
from materials_screening.master.fulltext_extraction import FulltextExtractionProcessor
from materials_screening.master.fulltext_preview import FulltextPreviewProcessor
from materials_screening.master.fulltext_snapshots import ExtractionSnapshotStore
from materials_screening.master.fulltext_targeted_evidence import (
    request_targeted_evidence,
)
from materials_screening.planner.settings import Settings

from .pgvector_store import PgVectorLiteratureStore
from .rag import (
    BgeM3EmbeddingProvider,
    LiteratureRagService,
    PyMuPdfParser,
    RagConfigurationError,
)


def _roots():
    return tuple(
        Path(value.strip())
        for value in os.getenv("LITERATURE_INGEST_ROOTS", "").split(os.pathsep)
        if value.strip()
    )


class _ConfiguredParser:
    def parse(self, path):
        source = Path(path).resolve()
        if not any(source.is_relative_to(root.resolve()) for root in _roots()):
            raise ValueError("PDF is outside configured ingestion roots")
        if (
            path.stat().st_size
            > int(os.getenv("LITERATURE_MAX_PDF_MB", "50")) * 1024 * 1024
        ):
            raise ValueError("PDF exceeds configured byte limit")
        return PyMuPdfParser(
            max_pages=int(os.getenv("LITERATURE_MAX_PDF_PAGES", "500"))
        ).parse(path)


def extraction_model(settings=None, *, default_model=None):
    settings = settings or Settings()
    return (
        (settings.literature_extraction_model or "").strip()
        or default_model
        or settings.intern_model
    )


class _ConfiguredLlm:
    def __init__(self, *, default_model=None):
        settings = Settings()
        timeout = float(os.getenv("FULLTEXT_LLM_TIMEOUT_SECONDS", "120"))
        if not 0 < timeout <= 300:
            raise ValueError("Fulltext request timeout must be within (0, 300] seconds")
        self.max_tokens = settings.llm_max_output_tokens
        self.provider = create_llm_provider(
            settings.model_copy(
                update={
                    "llm_provider": "intern",
                    "intern_model": extraction_model(
                        settings, default_model=default_model
                    ),
                    "intern_thinking_mode": False,
                    "llm_timeout_seconds": timeout,
                }
            )
        )

    def generate_structured(self, **kwargs):
        kwargs["max_output_tokens"] = min(kwargs["max_output_tokens"], self.max_tokens)
        return self.provider.generate_structured(**kwargs)


def create_fulltext_preview_processor(
    artifacts: ArtifactRegistry,
    *,
    enable_remote: bool,
    max_output_bytes: int = 65536,
    analysis_agent_factory=None,
    query_factory=None,
    figure_review_service=None,
) -> FulltextPreviewProcessor | None:
    if not enable_remote:
        return None  # Mock mode never silently sends uploaded PDFs externally.
    shared_conditions = os.getenv(
        "MASTER_SHARED_TEST_CONDITIONS", ""
    ).strip().casefold() in {
        "1",
        "true",
        "yes",
    }
    targeted_evidence = shared_conditions and os.getenv(
        "MASTER_TARGETED_TEST_EVIDENCE", ""
    ).strip().casefold() in {"1", "true", "yes"}

    def store_factory():
        database_url = os.getenv("LITERATURE_DATABASE_URL", "").strip()
        if not database_url or not _roots():
            raise RagConfigurationError(
                "Fulltext database and ingestion roots are required"
            )
        return PgVectorLiteratureStore(database_url)

    def rag_factory(store):
        embeddings = BgeM3EmbeddingProvider(
            revision=os.getenv("LITERATURE_EMBEDDING_REVISION") or None,
            device=os.getenv("LITERATURE_MODEL_DEVICE", "cpu"),
        )
        return LiteratureRagService(
            parser=_ConfiguredParser(),
            embeddings=embeddings,
            store=store,
            ingestion_roots=_roots(),
            max_pdf_bytes=int(os.getenv("LITERATURE_MAX_PDF_MB", "50")) * 1024 * 1024,
        )

    processor = FulltextPreviewProcessor(
        artifacts=artifacts,
        store_factory=store_factory,
        rag_factory=rag_factory,
        llm_factory=_ConfiguredLlm,
        parser=_ConfiguredParser(),
        max_output_bytes=max_output_bytes,
        auto_batches=True,
        batch_seconds=float(os.getenv("FULLTEXT_BATCH_SECONDS", "600")),
        max_batches=int(os.getenv("FULLTEXT_MAX_AUTO_BATCHES", "40")),
        batch_model_budget=int(os.getenv("FULLTEXT_MODEL_CALLS_PER_BATCH", "6")),
        require_preview_confirmation=True,
        extraction_processor=FulltextExtractionProcessor(
            artifacts=artifacts,
            snapshots=ExtractionSnapshotStore(Path("data/fulltext_snapshots")),
            dataset_factory=lambda: DatasetStore(Path("data/data_analysis")),
            model_profile="intern/" + extraction_model(),
            max_output_tokens=min(8192, Settings().llm_max_output_tokens),
            max_output_bytes=max_output_bytes,
            figure_review_service=figure_review_service,
            analysis_processor=FulltextAnalysisProcessor(
                snapshots=ExtractionSnapshotStore(Path("data/fulltext_snapshots")),
                analyses=FulltextAnalysisStore(Path("data/fulltext_analyses")),
                dataset_factory=lambda: DatasetStore(Path("data/data_analysis")),
                agent_factory=analysis_agent_factory,
                query_factory=query_factory,
                model_profile="intern/" + extraction_model(),
                max_output_bytes=max_output_bytes,
                figure_review_service=figure_review_service,
                condition_locator=locate_shared_conditions
                if shared_conditions
                else None,
                condition_reader=read_located_values if shared_conditions else None,
                targeted_planner=request_targeted_evidence
                if targeted_evidence
                else None,
            )
            if analysis_agent_factory is not None
            else None,
        ),
    )
    # Opt-in only until the original ten papers finish real acceptance. Preview,
    # standalone Literature and the legacy snapshot path keep their defaults.
    if os.getenv("MASTER_STAGED_FULLTEXT_EXTRACTION", "").strip().casefold() in {
        "1",
        "true",
        "yes",
    }:
        from materials_screening.master.staged_fulltext_processor import (
            StagedFulltextProcessor,
        )
        from materials_screening.master.staged_fulltext_store import (
            StagedExtractionStore,
        )

        model = extraction_model()
        processor.extraction_processor = StagedFulltextProcessor(
            artifacts=artifacts,
            stages=StagedExtractionStore(Path("data/fulltext_stages")),
            dataset_factory=lambda: DatasetStore(Path("data/data_analysis")),
            agent_factory=analysis_agent_factory,
            model_profile="intern/" + model,
            llm_factory=lambda: _ConfiguredLlm(default_model=Settings().intern_model),
            max_output_bytes=max_output_bytes,
            max_output_tokens=Settings().llm_max_output_tokens,
        )
    return processor
