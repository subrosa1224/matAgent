"""Command-line interface for materials-screening-core."""

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, cast
from uuid import uuid4

import typer
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from materials_screening import __version__
from materials_screening.errors import (
    ConfigurationError,
    ExportError,
    InvalidRequestError,
    RepositoryError,
    RepositoryMappingError,
    ValidationFailedError,
)
from materials_screening.evaluation.planner_evaluator import (
    PlannerEvaluator,
    load_cases,
)
from materials_screening.fingerprints import request_fingerprint
from materials_screening.inspection import main as inspect_main
from materials_screening.llm.base import StructuredLLM
from materials_screening.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMModelNotSupportedError,
    LLMPermissionError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMServiceUnavailableError,
    LLMStructuredOutputError,
    LLMTimeoutError,
    LLMTruncatedOutputError,
)
from materials_screening.llm.factory import LLMProviderName, create_llm_provider
from materials_screening.llm.mock_provider import (
    MockErrorKind,
    MockStructuredProvider,
)
from materials_screening.master.figure_evidence_cli import register_figure_review_command
from materials_screening.models import MaterialRecord, ScreeningRequest
from materials_screening.planner.errors import PlannerError, PlannerQueryError
from materials_screening.planner.facade import NaturalLanguageScreeningFacade
from materials_screening.planner.models import (
    PlannerDraft,
    PlannerResult,
    PlannerStatus,
)
from materials_screening.planner.service import PlannerService
from materials_screening.planner.settings import Settings
from materials_screening.repositories.base import MaterialsRepository
from materials_screening.repositories.materials_project import (
    MaterialsProjectRepository,
    map_summary_document,
)
from materials_screening.repositories.mock import (
    FIXED_TEST_TIME,
    MockMaterialsRepository,
)
from materials_screening.services.export_service import ExportService
from materials_screening.services.filter_service import FilterService
from materials_screening.services.ranking_service import RankingService
from materials_screening.services.screening_service import (
    ScreeningRunOutput,
    ScreeningService,
)
from materials_screening.services.validation_service import ValidationService
from materials_screening.workflow.input_output import (
    WorkflowInput,
    WorkflowOutput,
)
from materials_screening.workflow.state import WorkflowStateView, WorkflowStatus

app = typer.Typer(
    name="materials-screen",
    help="Deterministic inorganic semiconductor screening core (stage 1).",
)

workflow_app = typer.Typer(
    name="workflow",
    help="Run and inspect the fixed LangGraph workflow (stage 3).",
)
app.add_typer(workflow_app)

agent_app = typer.Typer(
    name="agent",
    help="Ask the single MaterialAgent and inspect conversations (stage 3.5).",
)
app.add_typer(agent_app)

master_app = typer.Typer(
    name="master",
    help="Multi-agent orchestrator with sub-agent delegation (stage 3.5).",
)
app.add_typer(master_app)

literature_app = typer.Typer(
    name="literature",
    help="Diagnose and migrate LiteratureAgent PostgreSQL/pgvector infrastructure.",
)
app.add_typer(literature_app)


@app.callback()
def _app_callback() -> None:
    """Materials screening CLI entry point."""


def _literature_ingestion_roots() -> tuple[Path, ...]:
    raw = os.getenv("LITERATURE_INGEST_ROOTS", "").strip()
    if not raw:
        return ()
    return tuple(
        Path(value.strip()) for value in raw.split(os.pathsep) if value.strip()
    )


@literature_app.command("doctor")
def literature_doctor_command(
    skip_database: bool = typer.Option(
        False,
        "--skip-database",
        help="Check local dependencies and paths without connecting to PostgreSQL.",
    ),
) -> None:
    """Report whether metadata search and PDF RAG are ready."""
    from materials_screening.sub_agents.literature.diagnostics import (
        diagnose_literature_environment,
    )

    report = diagnose_literature_environment(
        database_url=os.getenv("LITERATURE_DATABASE_URL", "").strip() or None,
        ingestion_roots=_literature_ingestion_roots(),
        check_database=not skip_database,
    )
    for check in report.checks:
        typer.echo(f"{check.name}: {check.status} - {check.detail}")
    typer.echo(
        f"metadata search ready: {str(report.ready_for_metadata_search).lower()}"
    )
    typer.echo(f"PDF RAG ready: {str(report.ready_for_pdf_rag).lower()}")
    if not report.ready_for_pdf_rag:
        raise typer.Exit(3)


@literature_app.command("search")
def literature_search_command(
    topic: str = typer.Argument(..., help="Natural-language literature topic."),
    material: list[str] | None = typer.Option(
        None, "--material", help="Material keyword; may be repeated."
    ),
    year_from: int | None = typer.Option(None, "--year-from", min=1900, max=2100),
    year_to: int | None = typer.Option(None, "--year-to", min=1900, max=2100),
    max_papers: int = typer.Option(20, "--max-papers", min=1, max=100),
    sort_mode: str = typer.Option(
        "balanced",
        "--sort",
        help="Ranking mode: balanced, recent, or relevance.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the strict JSON result instead of a summary."
    ),
    show_abstract: bool = typer.Option(
        False,
        "--show-abstract",
        help="Include a bounded abstract preview for each candidate.",
    ),
) -> None:
    """Search both literature providers and return one ranked candidate list."""
    from materials_screening.sub_agents.literature.models import (
        LiteratureSearchInput,
    )
    from materials_screening.sub_agents.literature.providers import (
        LiteratureProviderError,
        OpenAlexProvider,
        SemanticScholarProvider,
    )
    from materials_screening.sub_agents.literature.service import (
        LiteratureSearchService,
    )
    from materials_screening.sub_agents.literature.store import LiteratureQueryStore

    try:
        request = LiteratureSearchInput(
            topic=topic,
            material_keywords=tuple(material or ()),
            year_from=year_from,
            year_to=year_to,
            max_papers=max_papers,
            sort_mode=sort_mode,
        )
        service = LiteratureSearchService(
            (
                OpenAlexProvider(mailto=os.getenv("OPENALEX_MAILTO") or None),
                SemanticScholarProvider(api_key=os.getenv("S2_API_KEY") or None),
            ),
            LiteratureQueryStore(Path("data/literature_queries")),
        )
        result = service.search_unified(request)
    except ValidationError as exc:
        _abort(f"invalid literature search ({exc.errors()[0]['msg']})", 2)
    except LiteratureProviderError as exc:
        _abort(f"literature search failed ({exc.code})", 4)
    except Exception as exc:
        _abort(f"literature search failed ({type(exc).__name__})", 4)
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"统一检索ID：{result.query_id}")
    if result.expanded_query is not None:
        typer.echo("检索式：")
        for query in result.expanded_query.search_queries:
            typer.echo(f"  - {query}")
    typer.echo(f"候选文献：{result.returned_count}篇")
    for status in result.provider_statuses:
        label = "正常" if status.status == "ok" else "降级"
        typer.echo(
            f"来源状态：{status.provider}={label}（返回 {status.records_returned} 条）"
        )
    access_labels = {
        "metadata_only": "仅元数据",
        "abstract_available": "有摘要",
        "open_access_reported": "来源报告开放获取",
    }
    for index, paper in enumerate(result.papers, 1):
        year = paper.year if paper.year is not None else "年份未知"
        doi = paper.doi or "无DOI"
        typer.echo(f"{index}. {paper.title}（{year}）")
        relevance_label = {
            "core": "核心相关",
            "high": "高度相关",
            "extended": "扩展阅读",
            None: "未分级",
        }[paper.relevance_level]
        typer.echo(f"   相关层级：{relevance_label}")
        typer.echo(f"   DOI：{doi}")
        if paper.authors:
            authors = "、".join(paper.authors[:5])
            suffix = " 等" if len(paper.authors) > 5 else ""
            typer.echo(f"   作者：{authors}{suffix}")
        if paper.venue:
            typer.echo(f"   期刊/会议：{paper.venue}")
        citations = (
            str(paper.cited_by_count) if paper.cited_by_count is not None else "未知"
        )
        typer.echo(f"   引用数：{citations}")
        typer.echo(f"   状态：{access_labels[paper.access_status]}")
        if paper.landing_page_url:
            typer.echo(f"   文献入口：{paper.landing_page_url}")
        typer.echo(f"   入选原因：{paper.selection_reason or '主题相关'}")
        if show_abstract:
            if paper.abstract:
                preview = " ".join(paper.abstract.split())
                if len(preview) > 500:
                    preview = f"{preview[:497]}..."
                typer.echo(f"   摘要：{preview}")
            else:
                typer.echo("   摘要：数据源未提供")
    for warning in result.warnings:
        typer.echo(f"注意：{warning}")


@literature_app.command("process", hidden=True)
@literature_app.command("dossier-extract")
def literature_process_command(
    pdf: Path = typer.Option(..., "--pdf", exists=True, dir_okay=False),
    title: str | None = typer.Option(None, "--title"),
) -> None:
    """Ingest one text PDF and create an evidence-bound pending dossier."""
    from materials_screening.sub_agents.literature.automation import (
        AutomatedDossierExtractor,
    )
    from materials_screening.sub_agents.literature.dossier import PaperDossierStore
    from materials_screening.sub_agents.literature.rag import (
        BgeM3EmbeddingProvider,
        LiteratureRagService,
        PyMuPdfParser,
        parse_pdf_chunks,
    )

    resolved = pdf.resolve()
    roots = tuple(path.resolve() for path in _literature_ingestion_roots())
    if not roots or not any(
        resolved == root or root in resolved.parents for root in roots
    ):
        _abort("PDF path is outside LITERATURE_INGEST_ROOTS", 2)
    file_sha = hashlib.sha256(resolved.read_bytes()).hexdigest()
    document_id = f"doc-{file_sha[:24]}"
    store = _literature_pgvector_store()
    extraction_warnings: list[str] = []
    try:
        database_available = True
        try:
            indexed = store.document_exists(document_id)
        except Exception:
            indexed = False
            database_available = False
            extraction_warnings.append(
                "pgvector unavailable; extracted directly from the authorized PDF"
            )
        if database_available and not indexed:
            typer.echo("正在加载文本向量模型并索引PDF……")
            embeddings = BgeM3EmbeddingProvider(
                revision=os.getenv("LITERATURE_EMBEDDING_REVISION") or None,
                device=os.getenv("LITERATURE_MODEL_DEVICE", "cpu"),
            )
            rag = LiteratureRagService(
                parser=PyMuPdfParser(
                    max_pages=int(os.getenv("LITERATURE_MAX_PDF_PAGES", "500"))
                ),
                embeddings=embeddings,
                store=store,
                ingestion_roots=roots,
                max_pdf_bytes=int(os.getenv("LITERATURE_MAX_PDF_MB", "50"))
                * 1024
                * 1024,
            )
            rag.ingest((str(resolved),), None)
        if database_available:
            chunks = store.get_document_chunks(document_id)
            metadata = store.get_document_metadata(document_id)
        else:
            chunks = list(
                parse_pdf_chunks(
                    resolved,
                    parser=PyMuPdfParser(
                        max_pages=int(os.getenv("LITERATURE_MAX_PDF_PAGES", "500"))
                    ),
                    document_id=document_id,
                )
            )
            metadata = None
        paper_title = (
            title or (metadata.title if metadata is not None else None) or resolved.stem
        )
        settings = Settings()
        if settings.llm_provider == "mock":
            _abort(
                "automatic extraction requires LLM_PROVIDER=intern and an API key",
                3,
            )
        typer.echo(f"正在从 {len(chunks)} 个证据分块自动生成论文档案……")
        extraction_model = (settings.literature_extraction_model or "").strip()
        extraction_settings = settings.model_copy(
            update={"intern_model": extraction_model or settings.intern_model}
        )
        dossier, warnings = AutomatedDossierExtractor(
            create_llm_provider(extraction_settings)
        ).extract(
            document_id=document_id,
            title=paper_title,
            chunks=chunks,
            max_output_tokens=settings.llm_max_output_tokens,
        )
        if extraction_warnings:
            warnings = tuple(extraction_warnings) + warnings
            dossier = dossier.model_copy(update={"warnings": warnings})
        PaperDossierStore(Path("data/literature_dossier_candidates")).save_pending(
            dossier
        )
    except (ValueError, OSError) as exc:
        _abort(f"literature process failed ({exc})", 4)
    except Exception as exc:
        _abort(f"literature process failed ({type(exc).__name__})", 4)
    risk_counts = {
        level: sum(item.risk_level == level for item in dossier.items)
        for level in ("low", "medium", "high")
    }
    typer.echo(f"文档ID：{document_id}")
    typer.echo(
        f"档案候选：{len(dossier.items)} 条；低风险 {risk_counts['low']}，"
        f"中风险 {risk_counts['medium']}，高风险 {risk_counts['high']}。"
    )
    rejected = sum("rejected" in warning for warning in warnings)
    repaired = sum("replaced" in warning for warning in warnings)
    removed = sum("removed" in warning for warning in warnings)
    other_warnings = len(warnings) - rejected - repaired - removed
    typer.echo(
        f"证据门处理：拒绝 {rejected} 条，安全修复 {repaired} 条，"
        f"去重/裁剪 {removed} 条，其他警告 {other_warnings} 条。"
    )
    typer.echo(f"查看：materials-screen literature dossier-pending {document_id}")


@literature_app.command("batch-analyze")
def literature_batch_analyze_command(
    pdfs: list[Path] | None = typer.Option(
        None,
        "--pdf",
        exists=True,
        dir_okay=False,
        help="Authorized PDF path; may be repeated.",
    ),
    directory: Path | None = typer.Option(
        None,
        "--directory",
        exists=True,
        file_okay=False,
        help="Analyze every PDF directly inside this authorized directory.",
    ),
    retry_incomplete: bool = typer.Option(
        False,
        "--retry-incomplete",
        help="Retry only matrices that exist but contain zero measurements.",
    ),
    refresh_dossier: bool = typer.Option(
        False,
        "--refresh-dossier",
        help="Regenerate pending dossiers with the current extractor version.",
    ),
    extract_matrix: bool = typer.Option(
        False,
        "--extract-matrix/--skip-matrix",
        help="Also run the slow experimental-matrix extraction (default: skip).",
    ),
) -> None:
    """Build paper dossiers; optionally extract advanced experiment matrices."""
    from materials_screening.sub_agents.literature.batch import (
        BatchPaperResult,
        LiteratureBatchStore,
        document_id_for_pdf,
        matrix_analysis_complete,
        missing_index_paths,
        should_extract_matrix,
    )
    from materials_screening.sub_agents.literature.dossier import PaperDossierStore

    selected = list(pdfs or ())
    if directory is not None:
        selected.extend(sorted(directory.glob("*.pdf")))
    paths = tuple(dict.fromkeys(path.resolve() for path in selected))
    if not paths:
        _abort("at least one --pdf or --directory is required", 2)
    if len(paths) > 20:
        _abort("a literature batch may contain at most 20 PDFs", 2)
    roots = tuple(path.resolve() for path in _literature_ingestion_roots())
    if not roots or any(
        not any(path == root or root in path.parents for root in roots)
        for path in paths
    ):
        _abort("every PDF must be inside LITERATURE_INGEST_ROOTS", 2)

    pending_store = PaperDossierStore(Path("data/literature_dossier_candidates"))
    approved_store = PaperDossierStore(Path("data/literature_dossiers"))
    matrix_store = _literature_pgvector_store()
    _ensure_literature_batch_indexed(paths, roots, matrix_store, missing_index_paths)
    rows: list[BatchPaperResult] = []
    for index, path in enumerate(paths, 1):
        document_id = document_id_for_pdf(path)
        typer.echo(f"\n[{index}/{len(paths)}] {path.name}")
        errors: list[str] = []
        approved_dossier = approved_store.load(document_id)
        pending_dossier = pending_store.load(document_id)
        if refresh_dossier and approved_dossier is not None:
            typer.echo("论文档案已批准，受保护且不会重新生成。")
        dossier = approved_dossier or (None if refresh_dossier else pending_dossier)
        if dossier is None:
            try:
                literature_process_command(pdf=path, title=None)
            except Exception as exc:
                errors.append(f"dossier: {type(exc).__name__}")
            dossier = approved_store.load(document_id) or pending_store.load(
                document_id
            )
        else:
            typer.echo("复用已有论文档案。")

        try:
            counts = matrix_store.matrix_counts(document_id)
        except Exception as exc:
            counts = {}
            errors.append(f"matrix status: {type(exc).__name__}")
        matrix_requested = extract_matrix or retry_incomplete
        if (
            matrix_requested
            and dossier is not None
            and should_extract_matrix(counts, retry_incomplete=retry_incomplete)
        ):
            try:
                literature_matrix_extract_command(
                    document_id=document_id, show_warnings=False
                )
            except Exception as exc:
                errors.append(f"matrix: {type(exc).__name__}")
            try:
                counts = matrix_store.matrix_counts(document_id)
            except Exception as exc:
                errors.append(f"matrix status: {type(exc).__name__}")
        elif matrix_requested and any(counts.values()):
            typer.echo("复用已有实验矩阵。")
        elif not matrix_requested:
            typer.echo("普通模式已跳过实验矩阵；需要时使用 --extract-matrix。")

        measurements = counts.get("measurements", 0)
        status = (
            "failed"
            if dossier is None
            else "completed"
            if (not matrix_requested or matrix_analysis_complete(counts)) and not errors
            else "partial"
        )
        warnings = tuple(dossier.warnings if dossier is not None else ())
        rows.append(
            BatchPaperResult(
                document_id=document_id,
                file_name=path.name,
                title=dossier.title if dossier is not None else path.stem,
                status=status,
                dossier_status=(
                    dossier.review_status if dossier is not None else "missing"
                ),
                dossier_items=len(dossier.items) if dossier is not None else 0,
                completeness_score=(
                    dossier.completeness_score if dossier is not None else None
                ),
                matrix_groups=counts.get("groups", 0),
                matrix_measurements=measurements,
                matrix_claims=counts.get("claims", 0),
                claim_checks=counts.get("claim_evidence_links", 0),
                verified_claim_checks=counts.get("verified_claim_checks", 0),
                pending_measurements=counts.get("pending_measurements", 0),
                warnings=warnings,
                error="; ".join(errors) or None,
            )
        )

    report = LiteratureBatchStore(Path("data/literature_batches")).save(tuple(rows))
    _print_literature_batch(report)


def _ensure_literature_batch_indexed(
    paths: tuple[Path, ...],
    roots: tuple[Path, ...],
    store: Any,
    missing_paths_fn: Any,
) -> None:
    try:
        missing_paths = missing_paths_fn(paths, store.document_exists)
    except Exception as exc:
        _abort(f"could not inspect PDF index ({type(exc).__name__})", 4)
    if not missing_paths:
        typer.echo("全部 PDF 已索引，跳过向量模型加载。")
    else:
        from materials_screening.sub_agents.literature.rag import (
            BgeM3EmbeddingProvider,
            LiteratureRagService,
            PyMuPdfParser,
        )

        typer.echo(f"正在一次性加载向量模型并索引 {len(missing_paths)} 篇新 PDF……")
        try:
            embeddings = BgeM3EmbeddingProvider(
                revision=os.getenv("LITERATURE_EMBEDDING_REVISION") or None,
                device=os.getenv("LITERATURE_MODEL_DEVICE", "cpu"),
            )
            LiteratureRagService(
                parser=PyMuPdfParser(
                    max_pages=int(os.getenv("LITERATURE_MAX_PDF_PAGES", "500"))
                ),
                embeddings=embeddings,
                store=store,
                ingestion_roots=roots,
                max_pdf_bytes=int(os.getenv("LITERATURE_MAX_PDF_MB", "50"))
                * 1024
                * 1024,
            ).ingest(tuple(str(path) for path in missing_paths), None)
        except Exception as exc:
            _abort(f"batch PDF indexing failed ({type(exc).__name__})", 4)
    from materials_screening.sub_agents.literature.batch import document_id_for_pdf
    from materials_screening.sub_agents.literature.rag import infer_pdf_title

    try:
        for path in paths:
            document_id = document_id_for_pdf(path)
            metadata = store.get_document_metadata(document_id)
            inferred_title = infer_pdf_title(path)
            if metadata is None or not inferred_title:
                continue
            if metadata.title and metadata.title.casefold() not in {
                metadata.file_name.casefold(),
                Path(metadata.file_name).stem.casefold(),
                path.name.casefold(),
                path.stem.casefold(),
            }:
                continue
            store.set_document_metadata(
                document_id=document_id,
                title=inferred_title,
                doi=metadata.doi,
                year=metadata.year,
                paper_id=metadata.paper_id,
            )
    except Exception as exc:
        _abort(f"batch PDF title detection failed ({type(exc).__name__})", 4)


@literature_app.command("batch-preview")
def literature_batch_preview_command(
    topic: str = typer.Option(..., "--topic", help="Topic used to screen the PDFs."),
    pdfs: list[Path] | None = typer.Option(
        None, "--pdf", exists=True, dir_okay=False, help="PDF; may be repeated."
    ),
    directory: Path | None = typer.Option(
        None, "--directory", exists=True, file_okay=False
    ),
) -> None:
    """Index a batch and create one-call previews before deep analysis."""
    from materials_screening.sub_agents.literature.batch import (
        document_id_for_pdf,
        missing_index_paths,
    )
    from materials_screening.sub_agents.literature.preview import (
        PaperPreviewExtractor,
        PaperPreviewStore,
        infer_preview_boundary_policy,
        preview_batch_id,
    )

    selected = list(pdfs or ())
    if directory is not None:
        selected.extend(sorted(directory.glob("*.pdf")))
    paths = tuple(dict.fromkeys(path.resolve() for path in selected))
    if not paths:
        _abort("at least one --pdf or --directory is required", 2)
    if len(paths) > 20:
        _abort("a literature preview batch may contain at most 20 PDFs", 2)
    roots = tuple(path.resolve() for path in _literature_ingestion_roots())
    if not roots or any(
        not any(path == root or root in path.parents for root in roots)
        for path in paths
    ):
        _abort("every PDF must be inside LITERATURE_INGEST_ROOTS", 2)
    store = _literature_pgvector_store()
    _ensure_literature_batch_indexed(paths, roots, store, missing_index_paths)
    settings = Settings()
    if settings.llm_provider == "mock":
        _abort("paper preview requires LLM_PROVIDER=intern and an API key", 3)
    extraction_model = (settings.literature_extraction_model or "").strip()
    provider = create_llm_provider(
        settings.model_copy(
            update={
                "intern_model": extraction_model or settings.intern_model,
                "intern_thinking_mode": False,
            }
        )
    )
    preview_store = PaperPreviewStore(Path("data/literature_previews"))
    previews = []
    for index, path in enumerate(paths, 1):
        document_id = document_id_for_pdf(path)
        preview = preview_store.load(document_id, topic)
        if preview is None:
            typer.echo(f"[{index}/{len(paths)}] 正在快速阅读 {path.name}……")
            try:
                metadata = store.get_document_metadata(document_id)
                preview = PaperPreviewExtractor(
                    provider,
                    boundary_policy=infer_preview_boundary_policy(topic),
                ).extract(
                    document_id=document_id,
                    title=(
                        metadata.title
                        if metadata is not None and metadata.title
                        else path.stem
                    ),
                    topic=topic,
                    chunks=store.get_document_chunks(document_id),
                    max_output_tokens=min(settings.llm_max_output_tokens, 2048),
                )
                preview_store.save(preview)
            except Exception as exc:
                detail = str(exc).strip()
                if len(detail) > 300:
                    detail = f"{detail[:297]}..."
                suffix = f"：{detail}" if detail else ""
                typer.echo(f"   快速阅读失败：{type(exc).__name__}{suffix}")
                continue
        else:
            typer.echo(f"[{index}/{len(paths)}] 复用已有预览：{path.name}")
        previews.append(preview)
    batch_id = preview_batch_id(topic, [item.document_id for item in previews])
    typer.echo(f"\n快速阅读批次：{batch_id}；成功 {len(previews)}/{len(paths)} 篇")
    labels = {
        "core": "核心相关",
        "high": "高度相关",
        "extended": "扩展阅读",
        "low": "低相关",
    }
    recommendations = {
        "deep_analyze": "建议深度分析",
        "background_only": "仅作背景阅读",
        "exclude": "建议排除",
    }
    for item in previews:
        typer.echo(f"\n{item.title}")
        typer.echo(f"  文档ID：{item.document_id}")
        typer.echo(f"  相关性：{labels[item.topic_relevance]}")
        typer.echo(f"  研究问题：{item.research_question}")
        typer.echo(f"  方法：{item.methods}")
        typer.echo(f"  主要结论：{item.key_findings}")
        typer.echo(f"  建议：{recommendations[item.recommendation]}；{item.reason}")
        if item.evidence_quality == "fallback_chunk":
            typer.echo("  注意：代表引文已降级为原始 chunk，预览仅用于选文。")
    typer.echo("\n请只把选中的 PDF 传给 literature batch-analyze 做深度分析。")


@literature_app.command("batch-show")
def literature_batch_show_command(batch_id: str = typer.Argument(...)) -> None:
    """Show a saved multi-PDF analysis summary."""
    from materials_screening.sub_agents.literature.batch import LiteratureBatchStore

    report = LiteratureBatchStore(Path("data/literature_batches")).load(batch_id)
    if report is None:
        _abort(f"literature batch not found: {batch_id}", 2)
    _print_literature_batch(report)


def _print_literature_batch(report: Any) -> None:
    typer.echo(f"\n批次ID：{report.batch_id}")
    typer.echo(
        f"论文：{len(report.papers)} 篇；完成 {report.completed_count} 篇；"
        f"分析异常或不完整 {report.attention_count} 篇；"
        f"待人工审核 {report.review_required_count} 篇。"
    )
    labels = {"completed": "分析完成", "partial": "部分完成", "failed": "失败"}
    for index, row in enumerate(report.papers, 1):
        typer.echo(f"{index}. {row.title} [{labels[row.status]}]")
        typer.echo(f"   文档ID：{row.document_id}")
        completeness = (
            f"{row.completeness_score:.0%}"
            if row.completeness_score is not None
            else "未知"
        )
        typer.echo(
            f"   档案：{row.dossier_items} 条，完整度 {completeness}；"
            f"实验矩阵：{row.matrix_groups} 组、{row.matrix_measurements} 条测量、"
            f"{row.matrix_claims} 条结论、{row.claim_checks} 条证据核验"
            f"（有效 {row.verified_claim_checks} 条）；"
            f"待审核测量：{row.pending_measurements} 条。"
        )
        if row.error:
            typer.echo(f"   注意：{row.error}")
    typer.echo("\n已批准档案：materials-screen literature dossier-show <document_id>")
    typer.echo("待审档案：materials-screen literature dossier-pending <document_id>")
    typer.echo(
        "矩阵查看：materials-screen literature matrix-show <document_id> --summary"
    )


@literature_app.command("migrate")
def literature_migrate_command(
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm creation of the pgvector extension, tables, and indexes.",
    ),
) -> None:
    """Create the version-1 LiteratureAgent pgvector schema."""
    from materials_screening.sub_agents.literature.pgvector_store import (
        PgVectorLiteratureStore,
    )
    from materials_screening.sub_agents.literature.rag import RagConfigurationError

    database_url = os.getenv("LITERATURE_DATABASE_URL", "").strip()
    if not database_url:
        _abort("LITERATURE_DATABASE_URL is not configured", 3)
    if not yes and not typer.confirm(
        "Create or update the LiteratureAgent pgvector schema?"
    ):
        raise typer.Abort()
    try:
        PgVectorLiteratureStore(database_url).migrate()
    except RagConfigurationError:
        _abort("psycopg is not installed; install the literature extra", 3)
    except Exception as exc:
        _abort(f"literature database migration failed ({type(exc).__name__})", 4)
    typer.echo("LiteratureAgent pgvector schema is ready.")


def _literature_pgvector_store() -> Any:
    from materials_screening.sub_agents.literature.pgvector_store import (
        PgVectorLiteratureStore,
    )

    database_url = os.getenv("LITERATURE_DATABASE_URL", "").strip()
    if not database_url:
        _abort("LITERATURE_DATABASE_URL is not configured", 3)
    return PgVectorLiteratureStore(database_url)


@literature_app.command("facts-list")
def literature_facts_list_command(
    status: str | None = typer.Option(
        None, "--status", help="Filter by pending, approved, or rejected."
    ),
    document_id: str | None = typer.Option(None, "--document-id"),
) -> None:
    """List extracted facts with evidence and current review status."""
    if status not in {None, "pending", "approved", "rejected"}:
        _abort("status must be pending, approved, or rejected", 2)
    try:
        rows = _literature_pgvector_store().list_experimental_facts(
            status=status, document_id=document_id
        )
    except Exception as exc:
        _abort(f"could not list literature facts ({type(exc).__name__})", 4)
    typer.echo(json.dumps([row.model_dump(mode="json") for row in rows], indent=2))


@literature_app.command("facts-review")
def literature_facts_review_command(
    fact_id: str = typer.Argument(...),
    decision: str = typer.Option(..., "--decision", help="approved or rejected"),
    reviewer: str = typer.Option(..., "--reviewer"),
    reason: str | None = typer.Option(None, "--reason"),
    yes: bool = typer.Option(False, "--yes", help="Confirm the review decision."),
) -> None:
    """Approve or reject one fact and append an immutable audit record."""
    if decision not in {"approved", "rejected"}:
        _abort("decision must be approved or rejected", 2)
    if not reviewer.strip():
        _abort("reviewer must not be blank", 2)
    if not yes and not typer.confirm(f"Mark {fact_id} as {decision}?"):
        raise typer.Abort()
    try:
        review = _literature_pgvector_store().review_experimental_fact(
            fact_id=fact_id,
            decision=decision,
            reviewer=reviewer,
            reason=reason,
        )
    except KeyError:
        _abort(f"experimental fact not found: {fact_id}", 4)
    except Exception as exc:
        _abort(f"could not review literature fact ({type(exc).__name__})", 4)
    typer.echo(review.model_dump_json(indent=2))


@literature_app.command("edges-build")
def literature_edges_build_command(
    document_id: str | None = typer.Option(None, "--document-id"),
) -> None:
    """Build provenance-bound knowledge edges from approved facts only."""
    try:
        edges = _literature_pgvector_store().build_knowledge_edges(
            document_id=document_id
        )
    except Exception as exc:
        _abort(f"could not build literature edges ({type(exc).__name__})", 4)
    typer.echo(json.dumps([edge.model_dump(mode="json") for edge in edges], indent=2))


@literature_app.command("edges-list")
def literature_edges_list_command(
    document_id: str | None = typer.Option(None, "--document-id"),
) -> None:
    """List generated approved knowledge edges and their evidence."""
    try:
        edges = _literature_pgvector_store().list_knowledge_edges(
            document_id=document_id
        )
    except Exception as exc:
        _abort(f"could not list literature edges ({type(exc).__name__})", 4)
    typer.echo(json.dumps([edge.model_dump(mode="json") for edge in edges], indent=2))


@literature_app.command("result-build")
def literature_result_build_command(
    document_ids: list[str] | None = typer.Option(
        None,
        "--document-id",
        help="Reviewed document id; may be repeated for a multi-paper report.",
    ),
    query_ids: list[str] | None = typer.Option(
        None, "--query-id", help="Saved literature query id; may be repeated."
    ),
    summary: bool = typer.Option(
        False, "--summary", help="Show a concise human-readable result summary."
    ),
) -> None:
    """Build the strict LiteratureResult JSON from saved approved evidence."""
    if not query_ids and not document_ids:
        _abort("at least one --query-id or --document-id is required", 2)
    from materials_screening.sub_agents.literature.dossier import PaperDossierStore
    from materials_screening.sub_agents.literature.integration import (
        LiteratureIntegrationService,
    )
    from materials_screening.sub_agents.literature.store import LiteratureQueryStore

    try:
        service = LiteratureIntegrationService(
            LiteratureQueryStore(Path("data/literature_queries")),
            _literature_pgvector_store(),
            PaperDossierStore(Path("data/literature_dossiers")),
        )
        selected_documents = tuple(document_ids or ())
        result = (
            service.assemble(
                query_ids=tuple(query_ids or ()),
                document_id=selected_documents[0] if selected_documents else None,
            )
            if len(selected_documents) <= 1
            else service.assemble_many(
                query_ids=tuple(query_ids or ()), document_ids=selected_documents
            )
        )
    except (KeyError, ValueError) as exc:
        _abort(f"invalid literature evidence selection ({exc})", 2)
    except Exception as exc:
        _abort(f"could not build literature result ({type(exc).__name__})", 4)
    if summary:
        from materials_screening.sub_agents.literature.models import (
            ExperimentMatrixDataTable,
        )

        typer.echo("文献知识整合报告")
        typer.echo(
            f"概览：{len(result.papers)} 篇论文，"
            f"{len(result.paper_dossiers)} 份已审核论文档案，"
            f"{len(result.data_tables)} 张实验数据表，"
            f"{len(result.kp_edges)} 条可信知识关系。"
        )
        for paper in result.papers:
            year = paper.year if paper.year is not None else "年份未知"
            typer.echo(f"原文题名：{paper.title}（{year}）")
        for dossier in result.paper_dossiers:
            _print_literature_dossier(dossier)
        for table in result.data_tables:
            if not isinstance(table, ExperimentMatrixDataTable):
                typer.echo(f"实验数据表：{table.document_id}（{len(table.rows)} 行）")
                continue
            typer.echo(f"\n已审核实验矩阵：{table.document_id}")
            measurements_by_group: dict[str, list[str]] = {}
            for measurement in table.measurements:
                unit = measurement.unit or ""
                value = measurement.value_text
                if unit and unit not in value:
                    value += unit
                measurements_by_group.setdefault(measurement.group_id, []).append(
                    f"{_literature_term(measurement.metric)}={value}"
                )
            for group in table.groups:
                values = "；".join(measurements_by_group.get(group.group_id, ()))
                role = {"control": "对照组", "treatment": "实验组"}.get(
                    group.role, "参考组"
                )
                typer.echo(f"  {group.label}（{role}）：{values or '无测量值'}")
            links_by_claim = {
                link.claim_id: link for link in table.claim_evidence_links
            }
            for claim in table.claims:
                assessment = links_by_claim.get(claim.claim_id)
                status = assessment.assessment if assessment is not None else "unlinked"
                status_label = {
                    "supported": "正文证据支持",
                    "partially_supported": "正文部分支持",
                    "contradicted": "正文证据矛盾",
                    "not_verifiable": "暂不可核验",
                    "unsupported": "正文未支持",
                    "unlinked": "尚未关联证据",
                }[status]
                typer.echo(f"  摘要结论核验：{status_label}")
        if result.paper_dossiers:
            typer.echo("\n跨论文证据对照")
            dimensions = (
                ("performance_result", "性能与生物学结果"),
                ("mechanism", "可能机制"),
                ("limitations", "适用边界"),
            )
            for category, label in dimensions:
                rows = []
                for dossier in result.paper_dossiers:
                    item = next(
                        (
                            candidate
                            for candidate in dossier.items
                            if candidate.category == category
                        ),
                        None,
                    )
                    if item is not None:
                        rows.append((dossier.title, item.summary, item.page))
                if not rows:
                    continue
                typer.echo(f"  {label}：")
                for title, summary_text, page in rows:
                    typer.echo(f"    - {title}：{summary_text}（第 {page} 页）")
            common_categories = set.intersection(
                *(
                    {item.category for item in dossier.items}
                    for dossier in result.paper_dossiers
                )
            )
            labels = {
                "research_problem": "研究问题",
                "innovation": "核心创新",
                "materials": "材料体系",
                "preparation": "材料制备",
                "device_fabrication": "样品制备",
                "characterization": "表征方法",
                "structural_result": "结构结果",
                "optical_result": "光学结果",
                "performance_result": "性能结果",
                "mechanism": "作用机理",
                "limitations": "局限性",
                "reproducibility": "复现参数",
            }
            shared = "、".join(
                labels[category] for category in labels if category in common_categories
            )
            if shared:
                typer.echo(f"  共同覆盖的证据维度：{shared}。")
        for edge in result.kp_edges:
            relation_value = edge.performance_value
            object_result = edge.object.rsplit("=", maxsplit=1)[-1]
            if " to " in object_result:
                relation_value = object_result.replace(" to ", " → ")
            typer.echo(
                f"可信关系：{_literature_term(edge.variable_name)}="
                f"{edge.variable_value} → "
                f"{_literature_term(edge.performance_metric)}="
                f"{relation_value}"
            )
        for warning in result.warnings:
            warning_text = warning.replace(
                "No human-approved experimental evidence was found.",
                "未发现经过人工审核的实验数据。",
            )
            typer.echo(f"注意：{warning_text}")
        return
    typer.echo(result.model_dump_json(indent=2))


@literature_app.command("user-report")
def literature_user_report_command(
    topic: str = typer.Option(..., "--topic", help="Topic for the user report."),
    document_ids: list[str] | None = typer.Option(
        None,
        "--document-id",
        help="Analyzed document id; may be repeated.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the strict JSON report."
    ),
    narrative: bool = typer.Option(
        True,
        "--narrative/--evidence-only",
        help="Create one cited narrative synthesis or show the evidence view.",
    ),
) -> None:
    """Build a risk-filtered report without knowledge-base approval."""
    from materials_screening.llm.errors import LLMError
    from materials_screening.sub_agents.literature.dossier import PaperDossierStore
    from materials_screening.sub_agents.literature.user_report import (
        LiteratureUserReportService,
        LiteratureUserReportStore,
        LiteratureUserReportSynthesizer,
    )

    selected = tuple(document_ids or ())
    if not selected:
        _abort("at least one --document-id is required", 2)
    if len(selected) > 20:
        _abort("a user report may contain at most 20 documents", 2)
    try:
        report = LiteratureUserReportService(
            metadata_store=_literature_pgvector_store(),
            pending_store=PaperDossierStore(Path("data/literature_dossier_candidates")),
            approved_store=PaperDossierStore(Path("data/literature_dossiers")),
        ).build(topic=topic, document_ids=selected)
        report_store = LiteratureUserReportStore(Path("data/literature_user_reports"))
        cached_report = report_store.load(report.report_id)
        if cached_report is not None and cached_report.narrative is not None:
            report = cached_report
        if narrative:
            settings = Settings()
            if report.narrative is not None:
                pass
            elif settings.llm_provider == "mock":
                report = report.model_copy(
                    update={
                        "warnings": (
                            *report.warnings,
                            "当前为 mock 模式，已回退到证据视图。",
                        )
                    }
                )
            else:
                try:
                    synthesis_settings = settings.model_copy(
                        update={"intern_thinking_mode": False}
                    )
                    report = LiteratureUserReportSynthesizer(
                        create_llm_provider(synthesis_settings)
                    ).synthesize(
                        report,
                        max_output_tokens=min(settings.llm_max_output_tokens, 3072),
                    )
                except (LLMError, ValueError) as exc:
                    detail = " ".join(str(exc).split())[:240] or type(exc).__name__
                    report = report.model_copy(
                        update={
                            "warnings": (
                                *report.warnings,
                                "叙事综合未通过引用或数字校验，"
                                f"已安全回退到证据视图（{detail}）。",
                            )
                        }
                    )
        path = report_store.save(report)
    except ValueError as exc:
        _abort(str(exc), 2)
    except Exception as exc:
        _abort(f"could not build user literature report ({type(exc).__name__})", 4)
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    typer.echo("普通用户文献报告")
    typer.echo(f"报告ID：{report.report_id}")
    typer.echo(f"主题：{report.topic}")
    typer.echo(f"纳入论文：{len(report.papers)} 篇")
    typer.echo(
        "说明：低风险证据自动纳入；中风险证据标记为建议核对；"
        "高风险及不安全内容已自动隔离；实验矩阵不参与本报告。"
    )
    if report.narrative is not None:
        _print_user_report_narrative(report)
        for warning in report.warnings:
            typer.echo(f"注意：{warning}")
        typer.echo(f"\n详细证据与报告JSON已保存：{path}")
        return
    if narrative:
        typer.echo("\n叙述型综合暂不可用，以下为自动压缩的安全要点。")
        for warning in report.warnings:
            typer.echo(f"注意：{warning}")
        _print_user_report_compact_fallback(report)
        typer.echo(f"\n详细证据与报告JSON已保存：{path}")
        return
    for paper in report.papers:
        year = paper.year if paper.year is not None else "年份未知"
        typer.echo(f"\n论文：{paper.title}（{year}）")
        if paper.doi:
            typer.echo(f"DOI：{paper.doi}")
        typer.echo(
            f"证据：可靠 {paper.low_risk_count} 条，建议核对 "
            f"{paper.medium_risk_count} 条；已隔离高风险 "
            f"{paper.excluded_high_risk_count} 条、不安全 "
            f"{paper.excluded_unsafe_count} 条。"
        )
        by_category: dict[str, list[Any]] = {}
        for item in paper.evidence:
            by_category.setdefault(item.category, []).append(item)
        for category, label in _literature_dossier_labels().items():
            items = by_category.get(category, ())
            if not items:
                continue
            typer.echo(f"  {label}：")
            for item in items:
                marker = "可靠" if item.display_status == "reliable" else "建议核对"
                typer.echo(f"    - [{marker}] {item.summary}（第 {item.page} 页）")
    if len(report.papers) > 1:
        typer.echo("\n跨论文证据对照")
        for category, label in _literature_dossier_labels().items():
            rows = [
                (paper.title, item)
                for paper in report.papers
                for item in paper.evidence
                if item.category == category
            ]
            covered_papers = {title for title, _ in rows}
            if len(covered_papers) < 2:
                continue
            typer.echo(f"  {label}：")
            for title, item in rows:
                marker = "可靠" if item.display_status == "reliable" else "建议核对"
                typer.echo(
                    f"    - {title} [{marker}]：{item.summary}（第 {item.page} 页）"
                )
    for warning in report.warnings:
        typer.echo(f"注意：{warning}")
    typer.echo(f"\n报告JSON已保存：{path}")


def _print_user_report_compact_fallback(report: Any) -> None:
    priority = {
        "performance_result": 0,
        "mechanism": 1,
        "structural_result": 2,
        "innovation": 3,
        "materials": 4,
        "limitations": 5,
        "research_question": 6,
    }
    typer.echo("\n逐篇要点")
    for paper in report.papers:
        year = paper.year if paper.year is not None else "年份未知"
        typer.echo(f"\n{paper.title}（{year}）")
        selected = sorted(
            paper.evidence,
            key=lambda item: (
                priority.get(item.category, 20),
                item.risk_level != "low",
                item.page,
            ),
        )[:5]
        if not selected:
            typer.echo("  暂无通过安全筛选的正文证据。")
            continue
        for item in selected:
            marker = "可靠" if item.display_status == "reliable" else "建议核对"
            typer.echo(f"  - {item.summary}（第 {item.page} 页，{marker}）")
        typer.echo(
            "  证据覆盖："
            f"可靠 {paper.low_risk_count} 条，建议核对 {paper.medium_risk_count} 条；"
            f"隔离 {paper.excluded_high_risk_count + paper.excluded_unsafe_count} 条。"
        )


def _print_user_report_narrative(report: Any) -> None:
    typer.echo(f"\n{report.narrative.markdown.strip()}")
    typer.echo("\n证据覆盖")
    for paper in report.papers:
        typer.echo(
            f"  - {paper.title}：可靠 {paper.low_risk_count} 条，"
            f"建议核对 {paper.medium_risk_count} 条；隔离高风险 "
            f"{paper.excluded_high_risk_count} 条、不安全 "
            f"{paper.excluded_unsafe_count} 条。"
        )


@literature_app.command("ui")
def literature_ui_command(
    server_port: int = typer.Option(8503, "--port", min=1, max=65535),
) -> None:
    """Launch the local ordinary-user LiteratureAgent workbench."""
    try:
        import gradio  # noqa: F401
    except ImportError as exc:
        _abort(f"gradio not installed: {exc}; run `uv sync --extra web-ui`", 3)
    ui_path = Path(__file__).parent / "literature_ui_gradio.py"
    env = os.environ.copy()
    env["GRADIO_SERVER_PORT"] = str(server_port)
    try:
        subprocess.run([sys.executable, str(ui_path)], check=True, env=env)
    except subprocess.CalledProcessError as exc:
        _abort(f"literature UI exited with code {exc.returncode}", 4)


@literature_app.command("dossier-show")
def literature_dossier_show_command(document_id: str = typer.Argument(...)) -> None:
    """Display the evidence-grounded single-paper dossier."""
    from materials_screening.sub_agents.literature.dossier import PaperDossierStore

    try:
        dossier = PaperDossierStore(Path("data/literature_dossiers")).load(document_id)
    except Exception as exc:
        _abort(f"could not read paper dossier ({type(exc).__name__})", 4)
    if dossier is None:
        _abort("paper dossier has not been generated", 2)
    _print_literature_dossier(dossier)


@literature_app.command("dossier-pending")
def literature_dossier_pending_command(
    document_id: str = typer.Argument(...),
    risky_only: bool = typer.Option(
        False, "--risky-only", help="Hide low-risk candidates."
    ),
    debug_warnings: bool = typer.Option(
        False, "--debug-warnings", help="Display internal evidence-gate logs."
    ),
) -> None:
    """Display pending automatic candidates, with risky evidence first."""
    from materials_screening.sub_agents.literature.dossier import PaperDossierStore

    try:
        dossier = PaperDossierStore(Path("data/literature_dossier_candidates")).load(
            document_id
        )
    except Exception as exc:
        _abort(f"could not read pending dossier ({type(exc).__name__})", 4)
    if dossier is None:
        _abort("pending paper dossier has not been generated", 2)
    priorities = {"high": 0, "medium": 1, "low": 2}
    items = sorted(
        dossier.items,
        key=lambda item: (priorities[item.risk_level], item.page, item.category),
    )
    hidden_translation_failures = sum(
        item.summary.startswith("该候选摘要未能安全翻译") for item in items
    )
    items = [
        item for item in items if not item.summary.startswith("该候选摘要未能安全翻译")
    ]
    if risky_only:
        items = [item for item in items if item.risk_level != "low"]
    typer.echo(
        f"待审核档案：{dossier.title}；显示 {len(items)}/{len(dossier.items)} 条"
    )
    if dossier.completeness_score is not None:
        typer.echo(f"核心档案完整度：{dossier.completeness_score:.0%}")
    if dossier.narrative_summary:
        typer.echo("\n论文档案叙述：")
        narrative = dossier.narrative_summary.replace(
            "；该候选摘要未能安全翻译，请根据下方原文人工核对", ""
        ).replace("该候选摘要未能安全翻译，请根据下方原文人工核对。", "")
        typer.echo(narrative)
    if dossier.missing_core_categories:
        typer.echo("\n缺失核心栏目：" + "、".join(dossier.missing_core_categories))
    typer.echo("\n证据明细（按风险排序）：")
    if hidden_translation_failures:
        typer.echo(
            f"已隔离 {hidden_translation_failures} 条翻译失败候选，无需人工处理。"
        )
    for item in items:
        typer.echo(
            f"[{item.risk_level}] {item.category}（第 {item.page} 页）\n"
            f"  摘要：{item.summary}\n  原文：{item.source_quote}"
        )
    if dossier.warnings:
        rejected = sum("rejected" in warning for warning in dossier.warnings)
        removed = sum("removed" in warning for warning in dossier.warnings)
        repaired = sum("replaced" in warning for warning in dossier.warnings)
        typer.echo(
            f"\n证据门处理：丢弃 {rejected} 条，清理 {removed} 条，"
            f"安全修复 {repaired} 条。"
        )
    if debug_warnings:
        for warning in dossier.warnings:
            typer.echo(f"调试：{warning}")


@literature_app.command("dossier-review")
def literature_dossier_review_command(
    document_id: str = typer.Argument(...),
    decision: str = typer.Option(..., "--decision"),
    reviewer: str = typer.Option(..., "--reviewer"),
    reason: str | None = typer.Option(None, "--reason"),
    yes: bool = typer.Option(False, "--yes", help="Confirm the dossier decision."),
) -> None:
    """Approve or reject a paper dossier and preserve an immutable audit event."""
    if decision not in {"approved", "rejected"}:
        _abort("decision must be approved or rejected", 2)
    if not yes and not typer.confirm(f"Mark dossier {document_id} {decision}?"):
        raise typer.Abort()
    from materials_screening.sub_agents.literature.dossier import PaperDossierStore

    candidate_store = PaperDossierStore(Path("data/literature_dossier_candidates"))
    approved_store = PaperDossierStore(Path("data/literature_dossiers"))
    review_store = (
        candidate_store
        if candidate_store.load(document_id) is not None
        else approved_store
    )
    try:
        review = review_store.review(
            document_id=document_id,
            decision=cast(Literal["approved", "rejected"], decision),
            reviewer=reviewer,
            reason=reason,
        )
        if decision == "approved" and review_store is candidate_store:
            reviewed_dossier = candidate_store.load(document_id)
            if reviewed_dossier is None:
                raise ValueError("reviewed dossier could not be reloaded")
            approved_store.save_reviewed(reviewed_dossier)
    except ValueError as exc:
        _abort(str(exc), 2)
    except Exception as exc:
        _abort(f"could not review paper dossier ({type(exc).__name__})", 4)
    typer.echo(review.model_dump_json(indent=2))


def _print_literature_dossier(dossier: Any) -> None:
    labels = _literature_dossier_labels()
    status = {"approved": "已批准", "pending": "待审核", "rejected": "已拒绝"}[
        dossier.review_status
    ]
    typer.echo(f"\n单篇论文档案（{status}）：{dossier.title}")
    items_by_category: dict[str, list[Any]] = {}
    for item in dossier.items:
        items_by_category.setdefault(item.category, []).append(item)
    for category, label in labels.items():
        items = items_by_category.get(category, ())
        if not items:
            continue
        typer.echo(f"  {label}：")
        seen: set[tuple[str, int]] = set()
        for item in items:
            identity = (" ".join(item.summary.split()), item.page)
            if identity in seen:
                continue
            seen.add(identity)
            typer.echo(f"    - {item.summary}（第 {item.page} 页）")


def _literature_dossier_labels() -> dict[str, str]:
    return {
        "research_problem": "研究问题",
        "innovation": "核心创新",
        "materials": "材料体系",
        "preparation": "材料制备",
        "device_fabrication": "样品/器件制备",
        "characterization": "表征方法",
        "structural_result": "结构结果",
        "optical_result": "光学/能带",
        "performance_result": "性能结果",
        "mechanism": "作用机理",
        "limitations": "局限性",
        "reproducibility": "复现参数",
    }


def _literature_term(value: str) -> str:
    return {
        "Nb doping": "Nb 掺杂量",
        "PbI2/FAI molar ratio": "PbI₂/FAI 摩尔比",
        "photoconversion efficiency": "光电转换效率",
        "PCE": "光电转换效率",
        "Jsc": "短路电流密度 Jsc",
        "Voc": "开路电压 Voc",
        "FF": "填充因子 FF",
        "optical bandgap": "光学带隙",
        "ECBM-EF": "导带底与费米能级差",
        "excess PbI2": "过量 PbI₂",
        "nonradiative lifetime": "非辐射复合寿命",
    }.get(value, value)


@literature_app.command("document-metadata")
def literature_document_metadata_command(
    document_id: str = typer.Argument(...),
    title: str = typer.Option(..., "--title"),
    doi: str | None = typer.Option(None, "--doi"),
    year: int | None = typer.Option(None, "--year", min=1900, max=2100),
    paper_id: str | None = typer.Option(None, "--paper-id"),
    yes: bool = typer.Option(False, "--yes", help="Confirm metadata update."),
) -> None:
    """Bind verified bibliographic metadata to one indexed PDF."""
    if not title.strip():
        _abort("title must not be blank", 2)
    if not yes and not typer.confirm(f"Update metadata for {document_id}?"):
        raise typer.Abort()
    try:
        metadata = _literature_pgvector_store().set_document_metadata(
            document_id=document_id,
            title=title.strip(),
            doi=doi.strip().lower() if doi else None,
            year=year,
            paper_id=paper_id,
        )
    except KeyError:
        _abort(f"literature document not found: {document_id}", 4)
    except Exception as exc:
        _abort(f"could not update document metadata ({type(exc).__name__})", 4)
    typer.echo(metadata.model_dump_json(indent=2))


@literature_app.command("matrix-extract")
def literature_matrix_extract_command(
    document_id: str = typer.Argument(...),
    show_warnings: bool = typer.Option(
        True,
        "--show-warnings/--no-show-warnings",
        help="Display every evidence-gate warning.",
    ),
    requirements: Annotated[
        Path | None,
        typer.Option(
            "--requirements", help="Task-specific required-metric JSON policy."
        ),
    ] = None,
) -> None:
    """Extract a generic evidence-bound pending matrix from an indexed PDF."""
    from materials_screening.sub_agents.literature.matrix_automation import (
        AutomatedMatrixExtractor,
    )
    from materials_screening.sub_agents.literature.metric_coverage import (
        load_task_metric_requirements,
        render_required_metric_coverage,
    )

    try:
        metric_requirements = (
            load_task_metric_requirements(requirements) if requirements else None
        )
        store = _literature_pgvector_store()
        chunks = store.get_document_chunks(document_id)
        if not chunks:
            _abort(f"literature document has no indexed chunks: {document_id}", 4)
        settings = Settings()
        if settings.llm_provider == "mock":
            _abort(
                "automatic matrix extraction requires LLM_PROVIDER=intern "
                "and an API key",
                3,
            )
        extraction_model = (settings.literature_extraction_model or "").strip()
        extraction_settings = settings.model_copy(
            update={"intern_model": extraction_model or settings.intern_model}
        )
        result = AutomatedMatrixExtractor(
            create_llm_provider(extraction_settings), store
        ).extract(
            document_id=document_id,
            chunks=chunks,
            max_output_tokens=settings.llm_max_output_tokens,
            required_metrics=(
                metric_requirements.required_metrics if metric_requirements else ()
            ),
        )
        diagnostic_root = Path("data/literature_extractions")
        diagnostic_root.mkdir(parents=True, exist_ok=True)
        diagnostic_path = diagnostic_root / f"matrix-{uuid4().hex}.json"
        diagnostic_path.write_text(
            json.dumps(
                {
                    "document_id": document_id,
                    "created_at": datetime.now().isoformat(),
                    "diagnostics": result.diagnostics.model_dump(),
                    "warnings": result.warnings,
                    "metric_requirements": (
                        metric_requirements.model_dump(mode="json")
                        if metric_requirements else None
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        store.save_experiment_matrix(result.groups, result.measurements)
        store.save_comparisons(result.comparisons)
        store.save_claims_and_links(result.claims, result.claim_evidence_links)
    except (ValueError, OSError) as exc:
        _abort(f"matrix extraction failed ({exc})", 4)
    except Exception as exc:
        _abort(f"matrix extraction failed ({type(exc).__name__})", 4)
    typer.echo(f"Document: {document_id}")
    typer.echo(f"Extraction diagnostics: {diagnostic_path}")
    typer.echo(
        render_required_metric_coverage(
            result.diagnostics.required_metric_coverage,
            scope_label="本次提取最终保留的测量记录",
        )
    )
    typer.echo(
        f"Pending candidates: {len(result.groups)} groups, "
        f"{len(result.measurements)} measurements, "
        f"{len(result.comparisons)} comparisons, {len(result.claims)} claims, "
        f"{len(result.claim_evidence_links)} claim checks."
    )
    if show_warnings:
        for warning in result.warnings:
            typer.echo(f"Warning: {warning}")
    elif result.warnings:
        typer.echo(f"Evidence-gate warnings: {len(result.warnings)} (details hidden).")


@literature_app.command("matrix-status")
def literature_matrix_status_command(
    document_id: str = typer.Argument(...),
) -> None:
    """Show experiment-matrix and claim-evidence record counts for one PDF."""
    try:
        counts = _literature_pgvector_store().matrix_counts(document_id)
    except Exception as exc:
        _abort(f"could not read matrix status ({type(exc).__name__})", 4)
    typer.echo(json.dumps({"document_id": document_id, **counts}, indent=2))


@literature_app.command("matrix-review")
def literature_matrix_review_command(
    document_id: str = typer.Argument(...),
    decision: str = typer.Option(..., "--decision", help="approved or rejected"),
    reviewer: str = typer.Option(..., "--reviewer"),
    reason: str | None = typer.Option(None, "--reason"),
    yes: bool = typer.Option(False, "--yes", help="Confirm the matrix decision."),
) -> None:
    """Review every matrix and claim record with per-record immutable audit."""
    if decision not in {"approved", "rejected"}:
        _abort("decision must be approved or rejected", 2)
    if not reviewer.strip():
        _abort("reviewer must not be blank", 2)
    if not yes and not typer.confirm(f"Mark the full matrix {document_id} {decision}?"):
        raise typer.Abort()
    try:
        result = _literature_pgvector_store().review_document_matrix(
            document_id=document_id,
            decision=decision,
            reviewer=reviewer,
            reason=reason,
        )
    except ValueError as exc:
        _abort(str(exc), 2)
    except Exception as exc:
        _abort(f"could not review matrix ({type(exc).__name__})", 4)
    typer.echo(result.model_dump_json(indent=2))


@literature_app.command("matrix-show")
def literature_matrix_show_command(
    document_id: str = typer.Argument(...),
    status: str = typer.Option("pending", "--status"),
    summary: bool = typer.Option(
        False,
        "--summary",
        help="Show only groups, linked key comparisons, and claim assessments.",
    ),
) -> None:
    """Display matrix groups, measurements, comparisons, and claim assessments."""
    if status not in {"pending", "approved", "rejected"}:
        _abort("status must be pending, approved, or rejected", 2)
    try:
        groups, measurements, comparisons, claims, links = (
            _literature_pgvector_store().load_matrix(document_id, status=status)
        )
    except Exception as exc:
        _abort(f"could not read matrix ({type(exc).__name__})", 4)
    if summary:
        typer.echo(f"Document: {document_id}")
        typer.echo(f"Status: {status}")
        typer.echo(
            "Records: "
            f"{len(groups)} groups, {len(measurements)} measurements, "
            f"{len(comparisons)} comparisons, {len(claims)} claims"
        )
        typer.echo("Groups:")
        measurements_by_group: dict[str, list[Any]] = {}
        for measurement in measurements:
            measurements_by_group.setdefault(measurement.group_id, []).append(
                measurement
            )
        for group in groups:
            variables = ", ".join(
                f"{key}={value}" for key, value in group.variables.items()
            )
            detail = f"; {variables}" if variables else ""
            typer.echo(f"- {group.label} [{group.role}]: {group.material}{detail}")
            group_measurements = measurements_by_group.get(group.group_id, ())
            if not group_measurements:
                typer.echo("  Measurements: none")
            for measurement in group_measurements:
                unit = measurement.unit or ""
                value = measurement.value_text
                if unit and unit not in value:
                    value = f"{value} {unit}"
                typer.echo(
                    f"  Measurement: {measurement.metric}={value} "
                    f"(page {measurement.page_from})"
                )
        comparisons_by_id = {row.comparison_id: row for row in comparisons}
        claims_by_id = {row.claim_id: row for row in claims}
        typer.echo("Claim checks:")
        if not links:
            typer.echo("- No claim-evidence assessment in this status.")
        for link in links:
            claim = claims_by_id.get(link.claim_id)
            claim_text = claim.claim_text if claim is not None else link.claim_id
            claim_text = " ".join(claim_text.split())
            typer.echo(f"- [{link.assessment}] {claim_text}")
            comparison = comparisons_by_id.get(link.evidence_id)
            if comparison is not None:
                unit = comparison.unit or ""
                typer.echo(
                    "  Evidence: "
                    f"{comparison.metric}: {comparison.baseline_value}{unit} -> "
                    f"{comparison.target_value}{unit} "
                    f"({comparison.direction})"
                )
            typer.echo(f"  Reason: {link.explanation}")
        return
    payload = {
        "document_id": document_id,
        "status": status,
        "groups": [row.model_dump(mode="json") for row in groups],
        "measurements": [row.model_dump(mode="json") for row in measurements],
        "comparisons": [row.model_dump(mode="json") for row in comparisons],
        "claims": [row.model_dump(mode="json") for row in claims],
        "claim_evidence_links": [row.model_dump(mode="json") for row in links],
    }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _installed_version() -> str:
    """Return the installed package version, falling back to the module constant."""
    try:
        return package_version("materials-screening-core")
    except PackageNotFoundError:
        return __version__


def _abort(message: str, code: int) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code)


def _error_code(exc: Exception) -> int:
    if isinstance(exc, InvalidRequestError):
        return 2
    if isinstance(exc, ConfigurationError):
        return 3
    if isinstance(exc, RepositoryError):
        return 4
    if isinstance(exc, ValidationFailedError):
        return 5
    if isinstance(exc, ExportError):
        return 6
    if isinstance(exc, PlannerQueryError):
        return 2
    return _llm_error_code(exc)


def _llm_error_code(exc: Exception) -> int:
    if isinstance(exc, (LLMConfigurationError, LLMModelNotSupportedError)):
        return 30
    if isinstance(exc, (LLMAuthenticationError, LLMPermissionError)):
        return 31
    if isinstance(
        exc,
        (
            LLMRateLimitError,
            LLMTimeoutError,
            LLMConnectionError,
            LLMServiceUnavailableError,
        ),
    ):
        return 32
    if isinstance(exc, LLMRefusalError):
        return 33
    if isinstance(exc, (LLMStructuredOutputError, LLMTruncatedOutputError)):
        return 34
    return 10


def _read_file_or_abort(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        _abort(f"cannot read {label} file {path}: {exc}", 2)


def _validation_error_lines(exc: ValidationError) -> list[str]:
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        lines.append(f"- {location}: {error.get('msg', 'invalid')}")
    return lines


def _load_request(path: Path) -> ScreeningRequest:
    raw = _read_file_or_abort(path, "request")
    try:
        return ScreeningRequest.model_validate_json(raw)
    except ValidationError as exc:
        typer.echo("Request is invalid")
        for line in _validation_error_lines(exc):
            typer.echo(line)
        raise typer.Exit(2) from exc


def _resolve_api_key() -> str | None:
    api_key = os.getenv("MP_API_KEY", "").strip()
    if api_key:
        return api_key
    legacy_key = os.getenv("PMG_MAPI_KEY", "").strip()
    return legacy_key or None


def _load_mock_records(fixture: Path | None) -> tuple[MaterialRecord, ...]:
    if fixture is None:
        return ()
    raw = _read_file_or_abort(fixture, "fixture")
    if fixture.suffix.casefold() == ".jsonl":
        try:
            return tuple(
                _normalized_snapshot_record(json.loads(line))
                for line in raw.splitlines()
                if line.strip()
            )
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            _abort(f"invalid fixture JSONL record: {exc}", 2)
    try:
        documents = json.loads(raw)
    except json.JSONDecodeError as exc:
        _abort(f"invalid fixture JSON: {exc}", 2)
    if not isinstance(documents, list):
        _abort("fixture must contain a JSON list of summary documents", 2)
    records: list[MaterialRecord] = []
    for document in documents:
        try:
            records.append(
                map_summary_document(
                    document,
                    database_version="fixture-v1",
                    retrieved_at=FIXED_TEST_TIME,
                )
            )
        except (RepositoryMappingError, ValidationError) as exc:
            _abort(f"invalid fixture document: {exc}", 2)
    return tuple(records)


def _normalized_snapshot_record(value: object) -> MaterialRecord:
    """Adapt the research benchmark's flat JSONL rows to ``MaterialRecord``."""

    if not isinstance(value, dict):
        raise TypeError("snapshot record must be an object")
    record = {
        field: value[field] for field in MaterialRecord.model_fields if field in value
    }
    record["symmetry"] = {
        "crystal_system": value.get("crystal_system"),
        "symbol": value.get("spacegroup_symbol"),
        "number": value.get("spacegroup_number"),
    }
    return MaterialRecord.model_validate(record)


def _build_repository(repository: str, fixture: Path | None) -> MaterialsRepository:
    if repository == "mock":
        records = _load_mock_records(fixture)
        return MockMaterialsRepository(records=records)
    elif repository == "materials-project":
        api_key = _resolve_api_key()
        if api_key is None:
            raise ConfigurationError(
                "MP_API_KEY is not set; set MP_API_KEY (or PMG_MAPI_KEY) "
                "before using the materials-project repository"
            )
        return MaterialsProjectRepository(api_key=api_key)
    raise InvalidRequestError(
        f"unknown repository {repository!r}; expected 'materials-project' or 'mock'"
    )


def _build_service_from_repository(repo: MaterialsRepository) -> ScreeningService:
    return ScreeningService(
        repository=repo,
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=ExportService(),
    )


def _build_service(repository: str, fixture: Path | None) -> ScreeningService:
    return _build_service_from_repository(_build_repository(repository, fixture))


def _fmt(value: object) -> str:
    return "" if value is None else str(value)


def _read_queries(path: Path) -> list[str]:
    text = _read_file_or_abort(path, "query")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        _abort("query file contains no queries", 2)
    return lines


def _load_planner_fixtures(
    path: Path | None,
) -> tuple[dict[str, PlannerDraft], dict[str, MockErrorKind]]:
    if path is None:
        return {}, {}
    raw = _read_file_or_abort(path, "planner fixture")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _abort(f"invalid planner fixture JSON: {exc}", 2)
    fixture_entries: list[object]
    error_entries: list[object]
    if isinstance(data, list):
        fixture_entries, error_entries = data, []
    elif isinstance(data, dict):
        fixture_entries = data.get("fixtures", [])
        error_entries = data.get("errors", [])
    else:
        _abort(
            "planner fixture must be a list or an object with fixtures/errors",
            2,
        )
    fixtures: dict[str, PlannerDraft] = {}
    for entry in fixture_entries:
        if not isinstance(entry, dict):
            _abort("invalid planner fixture entry", 2)
        query = entry.get("query")
        if not isinstance(query, str):
            _abort("invalid planner fixture entry", 2)
        try:
            draft = PlannerDraft.model_validate(entry["draft"])
        except ValidationError as exc:
            _abort(f"invalid planner fixture draft: {exc}", 2)
        fixtures[query] = draft
    error_fixtures: dict[str, MockErrorKind] = {}
    for entry in error_entries:
        if not isinstance(entry, dict):
            _abort("invalid planner error fixture entry", 2)
        query = entry.get("query")
        if not isinstance(query, str):
            _abort("invalid planner error fixture entry", 2)
        kind_value = entry.get("kind")
        if not isinstance(kind_value, str):
            _abort("invalid planner error fixture kind", 2)
        error_fixtures[query] = MockErrorKind(kind_value)
    return fixtures, error_fixtures


def _build_planner(
    provider: Literal["mock", "intern"],
    fixture: Path | None = None,
) -> PlannerService:
    settings = Settings(llm_provider=provider)
    if LLMProviderName(settings.llm_provider) is LLMProviderName.MOCK:
        fixtures, error_fixtures = _load_planner_fixtures(fixture)
        llm_provider: StructuredLLM = MockStructuredProvider(
            fixtures=fixtures,
            error_fixtures=error_fixtures,
        )
    else:
        llm_provider = create_llm_provider(settings)
    return PlannerService(settings=settings, provider=llm_provider)


def _print_planner_details(result: PlannerResult) -> None:
    if result.status is PlannerStatus.READY and result.request is not None:
        typer.echo(f"Fingerprint: {request_fingerprint(result.request)}")
        typer.echo(f"Request: {result.request.model_dump_json()}")
    elif result.status is PlannerStatus.NEEDS_CLARIFICATION:
        typer.echo(f"Clarification: {result.clarification_question}")
    elif result.status is PlannerStatus.INVALID:
        for reason in result.invalid_reasons:
            typer.echo(f"- {reason}")
    elif result.status is PlannerStatus.UNSUPPORTED:
        for item in result.unsupported_requirements:
            typer.echo(f"- {item}")
    metadata = result.provider_metadata
    if metadata is not None:
        typer.echo(
            f"Provider: {metadata.provider} model={metadata.model} "
            f"latency_ms={metadata.latency_ms}"
        )


def _print_planner_result(result: PlannerResult) -> None:
    typer.echo(f"Status: {result.status.value}")
    _print_planner_details(result)


def _planner_status_code(status: PlannerStatus) -> int:
    if status is PlannerStatus.NEEDS_CLARIFICATION:
        return 20
    if status is PlannerStatus.INVALID:
        return 21
    if status is PlannerStatus.UNSUPPORTED:
        return 22
    return 0


def _worst_status(statuses: list[PlannerStatus]) -> PlannerStatus:
    priority = {
        PlannerStatus.READY: 0,
        PlannerStatus.NEEDS_CLARIFICATION: 1,
        PlannerStatus.UNSUPPORTED: 2,
        PlannerStatus.INVALID: 3,
    }
    return max(statuses, key=lambda status: priority[status])


def _screen_query(
    query: str,
    output_root: Path,
    repository: str,
    fixture: Path | None,
    planner_fixture: Path | None,
    include_cif: bool,
    provider: Literal["mock", "intern"],
) -> ScreeningRunOutput:
    planner_service = _build_planner(provider, planner_fixture)
    planner_result = planner_service.parse(query)
    if planner_result.status is not PlannerStatus.READY:
        _print_planner_result(planner_result)
        raise typer.Exit(_planner_status_code(planner_result.status))
    facade = NaturalLanguageScreeningFacade(
        planner_service=planner_service,
        screening_service=_build_service_from_repository(
            _build_repository(repository, fixture)
        ),
    )
    result = facade.screen_ready(
        planner_result,
        output_root,
        include_cif=include_cif,
    )
    if result.screening is None:
        raise PlannerError("READY plan must produce a screening output")
    return result.screening


def _print_summary(run_output: ScreeningRunOutput) -> None:
    result = run_output.result
    console = Console()
    table = Table(title="Ranked candidates")
    for column in (
        "rank",
        "material_id",
        "formula_pretty",
        "band_gap_ev",
        "total_score",
    ):
        table.add_column(column)
    for item in result.ranked_materials[:20]:
        record = item.record
        table.add_row(
            str(item.rank),
            record.material_id,
            record.formula_pretty,
            _fmt(record.band_gap_ev),
            str(item.total_score),
        )
    console.print(table)
    console.print("Task completed")
    console.print(f"Retrieved: {result.retrieved_count}")
    console.print(f"Passed hard filters: {result.passed_filter_count}")
    console.print(f"Returned: {len(result.ranked_materials)}")
    validation = "PASSED" if result.validation.passed else "FAILED"
    console.print(f"Validation: {validation}")
    console.print(f"Output: {run_output.exports.run_dir}")


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(_installed_version())


@app.command()
def validate_request(
    request: Path = typer.Option(
        ..., "--request", "-r", help="Path to the request JSON file."
    ),
) -> None:
    """Validate a request JSON file without network access."""
    screening_request = _load_request(request)
    typer.echo("Request is valid")
    typer.echo(f"Fingerprint: {request_fingerprint(screening_request)}")


@app.command()
def inspect_mp() -> None:
    """Inspect installed mp-api signatures and fields (offline without a key)."""
    try:
        inspect_main()
    except Exception as exc:
        _abort(str(exc), _error_code(exc))


@app.command()
def parse(
    query: str = typer.Option(
        ..., "--query", "-q", help="Natural language screening request."
    ),
    provider: Literal["mock", "intern"] = typer.Option(
        "mock", "--provider", help="Planner provider: mock or intern."
    ),
    fixture: Path | None = typer.Option(
        None, "--fixture", help="Planner mock fixtures JSON."
    ),
) -> None:
    """Parse a natural language request into a planner plan."""
    try:
        planner = _build_planner(provider, fixture)
        result = planner.parse(query)
    except typer.Exit:
        raise
    except Exception as exc:
        _abort(str(exc), _error_code(exc))
    _print_planner_result(result)
    if result.status is not PlannerStatus.READY:
        raise typer.Exit(_planner_status_code(result.status))


@app.command()
def parse_file(
    input: Path = typer.Option(
        ..., "--input", "-i", help="Path to a UTF-8 file with one query per line."
    ),
    provider: Literal["mock", "intern"] = typer.Option(
        "mock", "--provider", help="Planner provider: mock or intern."
    ),
    fixture: Path | None = typer.Option(
        None, "--fixture", help="Planner mock fixtures JSON."
    ),
) -> None:
    """Parse multiple queries, one per line, and print a summary."""
    try:
        queries = _read_queries(input)
        planner = _build_planner(provider, fixture)
    except typer.Exit:
        raise
    except Exception as exc:
        _abort(str(exc), _error_code(exc))
    statuses: list[PlannerStatus] = []
    for index, query in enumerate(queries, start=1):
        try:
            result = planner.parse(query)
        except Exception as exc:
            _abort(f"query {index} failed: {exc}", _error_code(exc))
        typer.echo(f"#{index}: {result.status.value}")
        _print_planner_details(result)
        statuses.append(result.status)
    counts = {
        status: statuses.count(status)
        for status in (
            PlannerStatus.READY,
            PlannerStatus.NEEDS_CLARIFICATION,
            PlannerStatus.INVALID,
            PlannerStatus.UNSUPPORTED,
        )
    }
    typer.echo(
        f"Total: {len(queries)} "
        f"ready={counts[PlannerStatus.READY]} "
        f"needs_clarification={counts[PlannerStatus.NEEDS_CLARIFICATION]} "
        f"invalid={counts[PlannerStatus.INVALID]} "
        f"unsupported={counts[PlannerStatus.UNSUPPORTED]}"
    )
    worst = _worst_status(statuses)
    if worst is not PlannerStatus.READY:
        raise typer.Exit(_planner_status_code(worst))


@app.command()
def evaluate_planner(
    input: Path = typer.Option(
        ..., "--input", "-i", help="Path to JSONL planner eval cases."
    ),
    provider: Literal["mock", "intern"] = typer.Option(
        "mock", "--provider", help="Planner provider: mock or intern."
    ),
    fixture: Path | None = typer.Option(
        None, "--fixture", help="Planner mock fixtures JSON."
    ),
) -> None:
    """Evaluate the planner against JSONL cases."""
    try:
        cases = load_cases(input)
        planner = _build_planner(provider, fixture)
        metrics = PlannerEvaluator(planner).evaluate(cases)
    except typer.Exit:
        raise
    except Exception as exc:
        _abort(str(exc), _error_code(exc))
    typer.echo(f"Total cases: {metrics.total}")
    typer.echo(
        f"Schema success rate: {metrics.schema_success_rate:.3f} "
        f"({metrics.schema_success}/{metrics.total})"
    )
    typer.echo(
        f"Status accuracy: {metrics.status_accuracy:.3f} "
        f"({metrics.status_accurate}/{metrics.total})"
    )
    typer.echo(f"Ready precision: {metrics.ready_precision:.3f}")
    typer.echo(f"Ready recall: {metrics.ready_recall:.3f}")
    typer.echo(
        f"Request field exact match: {metrics.request_exact_match_rate:.3f} "
        f"({metrics.request_exact_match}/{metrics.request_exact_instances})"
    )
    typer.echo(
        f"Numeric constraint accuracy: {metrics.numeric_constraint_accuracy:.3f} "
        f"({metrics.numeric_matches}/{metrics.numeric_instances})"
    )
    typer.echo(
        f"Element constraint accuracy: {metrics.element_constraint_accuracy:.3f} "
        f"({metrics.element_matches}/{metrics.element_instances})"
    )
    typer.echo(f"Ambiguity recall: {metrics.ambiguity_recall:.3f}")
    typer.echo(f"Conflict recall: {metrics.conflict_recall:.3f}")
    typer.echo(f"Unsupported accuracy: {metrics.unsupported_accuracy:.3f}")
    typer.echo(f"Injection resilience: {metrics.injection_resilience:.3f}")
    typer.echo(f"Avg latency ms: {metrics.average_latency_ms:.3f}")
    typer.echo(f"Avg input tokens: {metrics.average_input_tokens:.3f}")
    typer.echo(f"Avg output tokens: {metrics.average_output_tokens:.3f}")
    typer.echo(f"Avg reasoning tokens: {metrics.average_reasoning_tokens:.3f}")


@app.command()
def screen(
    request: Path | None = typer.Option(
        None, "--request", "-r", help="Path to the request JSON file."
    ),
    query: str | None = typer.Option(
        None, "--query", "-q", help="Natural language screening request."
    ),
    output: Path = typer.Option(
        Path("data/exports"),
        "--output",
        "-o",
        help="Root directory for run exports.",
    ),
    repository: str = typer.Option(
        "materials-project",
        "--repository",
        help="Repository to use: materials-project or mock.",
    ),
    fixture: Path | None = typer.Option(
        None, "--fixture", help="Fixture JSON for the mock repository."
    ),
    planner_fixture: Path | None = typer.Option(
        None, "--planner-fixture", help="Planner mock fixtures JSON."
    ),
    no_cif: bool = typer.Option(False, "--no-cif", help="Skip CIF export."),
    provider: Literal["mock", "intern"] = typer.Option(
        "mock", "--provider", help="Planner provider for --query."
    ),
) -> None:
    """Run a screening pipeline; input is exactly one of --request or --query."""
    if (request is None) == (query is None):
        _abort("exactly one of --request or --query must be provided", 2)
    try:
        if query is not None:
            run_output = _screen_query(
                query,
                output,
                repository,
                fixture,
                planner_fixture,
                not no_cif,
                provider,
            )
        else:
            assert request is not None
            screening_request = _load_request(request)
            service = _build_service(repository, fixture)
            run_output = service.run(screening_request, output, include_cif=not no_cif)
    except typer.Exit:
        raise
    except Exception as exc:
        _abort(str(exc), _error_code(exc))
    _print_summary(run_output)


def _utc_now() -> datetime:
    from datetime import UTC

    return datetime.now(UTC)


_BUNDLED_PLANNER_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "planner_cli_fixtures.json"
)
_BUNDLED_MATERIALS_FIXTURE = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "mp_documents.json"
)


def _resolve_provider_defaults(
    llm_provider: Literal["mock", "intern"],
    materials_repository: str | None,
    planner_fixture: Path | None,
    materials_fixture: Path | None,
) -> tuple[str, Path | None, Path | None]:
    """Apply usage defaults: Intern uses the real MP repository; mock mode
    auto-uses the bundled planner and materials fixtures."""
    if materials_repository is None:
        materials_repository = (
            "materials-project" if llm_provider == "intern" else "mock"
        )
    if llm_provider == "mock" and planner_fixture is None:
        planner_fixture = _BUNDLED_PLANNER_FIXTURE
    if materials_repository == "mock" and materials_fixture is None:
        materials_fixture = _BUNDLED_MATERIALS_FIXTURE
    return materials_repository, planner_fixture, materials_fixture


class _UuidIdGenerator:
    def new_id(self) -> str:
        import uuid

        return str(uuid.uuid4())


@contextmanager
def _open_workflow_runner(
    *,
    run_root: Path,
    llm_provider: Literal["mock", "intern"],
    planner_fixture: Path | None,
    materials_repository: str,
    materials_fixture: Path | None,
) -> Iterator[Any]:
    """Open a WorkflowRunner with configured providers (lazy langgraph imports)."""
    from materials_screening.workflow.artifact_store import FileRunArtifactStore
    from materials_screening.workflow.checkpointer import create_checkpointer_handle
    from materials_screening.workflow.context import WorkflowContext
    from materials_screening.workflow.export_adapter import WorkflowExportAdapter
    from materials_screening.workflow.runner import WorkflowRunner
    from materials_screening.workflow.settings import WorkflowSettings

    settings = WorkflowSettings()
    planner_service = _build_planner(llm_provider, planner_fixture)
    repository = _build_repository(materials_repository, materials_fixture)
    context = WorkflowContext(
        planner_service=planner_service,
        materials_repository=repository,
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=WorkflowExportAdapter(run_root),
        artifact_store=FileRunArtifactStore(run_root),
        clock=_utc_now,
        id_generator=_UuidIdGenerator(),
    )
    with create_checkpointer_handle(settings=settings) as handle:
        runner = WorkflowRunner(
            settings=settings,
            context=context,
            checkpointer=handle,
        )
        yield runner


def _print_workflow_output(output: WorkflowOutput) -> None:
    typer.echo(f"Run: {output.run_id}")
    typer.echo(f"Thread: {output.thread_id}")
    typer.echo(f"Status: {output.status.value}")
    if output.planner_status is not None:
        typer.echo(f"Planner: {output.planner_status}")
    if output.clarification_question:
        typer.echo(f"Clarification: {output.clarification_question}")
    typer.echo(f"Retrieved: {output.retrieved_count}")
    typer.echo(f"Filtered: {output.filtered_count}")
    typer.echo(f"Returned: {output.returned_count}")
    if output.validation_passed is not None:
        typer.echo(f"Validation: {'PASSED' if output.validation_passed else 'FAILED'}")
    if output.exports:
        typer.echo(f"Exports: {len(output.exports)} file(s)")
    if output.error is not None:
        typer.echo(f"Error: {output.error['code']}: {output.error['message']}")


def _workflow_exit_code(output: WorkflowOutput) -> int:
    if output.status is WorkflowStatus.NEEDS_CLARIFICATION:
        return 20
    if output.status is WorkflowStatus.INVALID_REQUEST:
        return 21
    if output.status is WorkflowStatus.UNSUPPORTED_REQUEST:
        return 22
    if output.status is WorkflowStatus.FAILED:
        return 10
    return 0


def _print_state_view(view: WorkflowStateView) -> None:
    typer.echo(f"Run: {view.run_id or '(none)'}")
    typer.echo(f"Thread: {view.thread_id}")
    typer.echo(f"Status: {view.status or '(none)'}")
    if view.current_node:
        typer.echo(f"Current node: {view.current_node}")
    if view.started_at:
        typer.echo(f"Started: {view.started_at}")
    if view.finished_at:
        typer.echo(f"Finished: {view.finished_at}")
    if view.planner_status:
        typer.echo(f"Planner: {view.planner_status}")
    typer.echo(f"Retrieved: {view.retrieved_count}")
    typer.echo(f"Filtered: {view.filtered_count}")
    typer.echo(f"Returned: {view.returned_count}")
    if view.validation_passed is not None:
        typer.echo(f"Validation: {'PASSED' if view.validation_passed else 'FAILED'}")
    typer.echo(f"Exports: {len(view.exports)} file(s)")
    typer.echo(f"Warnings: {len(view.warnings)}")
    if view.error is not None:
        typer.echo(f"Error: {view.error['code']}: {view.error['message']}")


def _workflow_mermaid() -> str:
    from langgraph.checkpoint.memory import InMemorySaver

    from materials_screening.workflow.graph_builder import compile_workflow

    compiled = compile_workflow(checkpointer=InMemorySaver())
    edges = [(edge.source, edge.target) for edge in compiled.get_graph().edges]
    lines = ["flowchart LR"]
    for source, target in edges:
        lines.append(f"    {source} --> {target}")
    return "\n".join(lines) + "\n"


@workflow_app.command("run")
def workflow_run_command(
    query: str | None = typer.Option(
        None, "--query", "-q", help="Natural language workflow input."
    ),
    request: Path | None = typer.Option(
        None, "--request", "-r", help="Path to a request JSON file."
    ),
    llm_provider: Literal["mock", "intern"] = typer.Option(
        "mock", "--llm-provider", help="Planner provider: mock or intern."
    ),
    materials_repository: str | None = typer.Option(
        None,
        "--materials-repository",
        help=(
            "Repository: mock or materials-project. Defaults to "
            "materials-project with --llm-provider intern, else mock."
        ),
    ),
    planner_fixture: Path | None = typer.Option(
        None, "--planner-fixture", help="Planner mock fixtures JSON."
    ),
    materials_fixture: Path | None = typer.Option(
        None, "--materials-fixture", help="Mock repository fixtures JSON."
    ),
    output: Path = typer.Option(
        Path("data/workflow_runs"),
        "--output",
        "-o",
        help="Workflow run root directory.",
    ),
    stream: bool = typer.Option(False, "--stream", help="Stream node-level summaries."),
    no_cif: bool = typer.Option(False, "--no-cif", help="Skip CIF export."),
    thread_id: str | None = typer.Option(
        None,
        "--thread-id",
        help="Run on an explicit new thread id (advanced).",
    ),
) -> None:
    """Run the fixed LangGraph workflow; exactly one of --query or --request."""
    if (query is None) == (request is None):
        _abort("exactly one of --query or --request must be provided", 2)
    materials_repository, planner_fixture, materials_fixture = (
        _resolve_provider_defaults(
            llm_provider,
            materials_repository,
            planner_fixture,
            materials_fixture,
        )
    )
    exit_code = 0
    try:
        raw_request: dict[str, Any] | None = None
        if request is not None:
            raw_text = _read_file_or_abort(request, "request")
            try:
                raw_request = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                _abort(f"invalid request JSON: {exc}", 2)
        with _open_workflow_runner(
            run_root=output,
            llm_provider=llm_provider,
            planner_fixture=planner_fixture,
            materials_repository=materials_repository,
            materials_fixture=materials_fixture,
        ) as runner:
            if thread_id is not None and runner.thread_exists(thread_id):
                _abort(
                    f"thread {thread_id!r} already exists; refusing a new run",
                    2,
                )
            workflow_input = WorkflowInput(
                query=query,
                request=raw_request,
                output_root=str(output),
                export_cif=not no_cif,
            )
            if stream:
                for event in runner.stream(workflow_input, thread_id=thread_id):
                    typer.echo(f"[{event.node}] {event.message}")
                run_output = runner.get_output(runner.last_thread_id or "")
            else:
                run_output = runner.run(workflow_input, thread_id=thread_id)
            _print_workflow_output(run_output)
            exit_code = _workflow_exit_code(run_output)
    except typer.Exit:
        raise
    except ImportError as exc:
        _abort(
            f"workflow extra not installed: {exc}; run `uv sync --extra workflow`",
            3,
        )
    except Exception as exc:
        _abort(str(exc), _error_code(exc))
    if exit_code:
        raise typer.Exit(exit_code)


@workflow_app.command("status")
def workflow_status_command(
    thread_id: str = typer.Option(..., "--thread-id", help="Workflow thread id."),
) -> None:
    """Show a safe summary of one workflow thread."""
    try:
        with _open_workflow_runner(
            run_root=Path("data/workflow_runs"),
            llm_provider="mock",
            planner_fixture=None,
            materials_repository="mock",
            materials_fixture=None,
        ) as runner:
            _print_state_view(runner.get_state(thread_id))
    except typer.Exit:
        raise
    except ImportError as exc:
        _abort(
            f"workflow extra not installed: {exc}; run `uv sync --extra workflow`",
            3,
        )
    except Exception as exc:
        _abort(str(exc), _error_code(exc))


@workflow_app.command("history")
def workflow_history_command(
    thread_id: str = typer.Option(..., "--thread-id", help="Workflow thread id."),
    limit: int | None = typer.Option(
        None, "--limit", help="Maximum number of checkpoints to show."
    ),
) -> None:
    """Show safe checkpoint summaries for one workflow thread."""
    try:
        with _open_workflow_runner(
            run_root=Path("data/workflow_runs"),
            llm_provider="mock",
            planner_fixture=None,
            materials_repository="mock",
            materials_fixture=None,
        ) as runner:
            for view in runner.get_history(thread_id, limit):
                next_nodes = ",".join(view.next_nodes)
                typer.echo(
                    f"{view.step}\t{view.checkpoint_id}\t{view.source}\t"
                    f"{view.current_node or ''}\t{view.status}\t{next_nodes}"
                )
    except typer.Exit:
        raise
    except ImportError as exc:
        _abort(
            f"workflow extra not installed: {exc}; run `uv sync --extra workflow`",
            3,
        )
    except Exception as exc:
        _abort(str(exc), _error_code(exc))


@workflow_app.command("replay")
def workflow_replay_command(
    thread_id: str = typer.Option(..., "--thread-id", help="Workflow thread id."),
    checkpoint_id: str = typer.Option(
        ..., "--checkpoint-id", help="Checkpoint id from workflow history."
    ),
    confirm_remote_calls: bool = typer.Option(
        False,
        "--confirm-remote-calls",
        help="Confirm replay may re-trigger remote calls.",
    ),
    llm_provider: Literal["mock", "intern"] = typer.Option(
        "mock", "--llm-provider", help="Planner provider: mock or intern."
    ),
    materials_repository: str | None = typer.Option(
        None,
        "--materials-repository",
        help=(
            "Repository: mock or materials-project. Defaults to "
            "materials-project with --llm-provider intern, else mock."
        ),
    ),
    planner_fixture: Path | None = typer.Option(
        None, "--planner-fixture", help="Planner mock fixtures JSON."
    ),
    materials_fixture: Path | None = typer.Option(
        None, "--materials-fixture", help="Mock repository fixtures JSON."
    ),
    output: Path = typer.Option(
        Path("data/workflow_runs"),
        "--output",
        "-o",
        help="Workflow run root directory.",
    ),
) -> None:
    """Replay a workflow run from an existing checkpoint (advanced)."""
    materials_repository, planner_fixture, materials_fixture = (
        _resolve_provider_defaults(
            llm_provider,
            materials_repository,
            planner_fixture,
            materials_fixture,
        )
    )
    exit_code = 0
    try:
        with _open_workflow_runner(
            run_root=output,
            llm_provider=llm_provider,
            planner_fixture=planner_fixture,
            materials_repository=materials_repository,
            materials_fixture=materials_fixture,
        ) as runner:
            run_output = runner.replay(
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                confirm_remote_calls=confirm_remote_calls,
            )
            _print_workflow_output(run_output)
            exit_code = _workflow_exit_code(run_output)
    except typer.Exit:
        raise
    except ImportError as exc:
        _abort(
            f"workflow extra not installed: {exc}; run `uv sync --extra workflow`",
            3,
        )
    except Exception as exc:
        _abort(str(exc), _error_code(exc))
    if exit_code:
        raise typer.Exit(exit_code)


@workflow_app.command("draw")
def workflow_draw_command() -> None:
    """Print Mermaid text for the fixed workflow graph."""
    try:
        typer.echo(_workflow_mermaid())
    except ImportError as exc:
        _abort(
            f"workflow extra not installed: {exc}; run `uv sync --extra workflow`",
            3,
        )
    except Exception as exc:
        _abort(str(exc), _error_code(exc))


@workflow_app.command("inspect")
def workflow_inspect_command() -> None:
    """Print safe workflow configuration and installed LangGraph version."""
    try:
        from materials_screening.workflow.settings import WorkflowSettings

        settings = WorkflowSettings()
        typer.echo(f"enabled: {str(settings.workflow_enabled).lower()}")
        typer.echo(f"version: {settings.workflow_version}")
        typer.echo(f"checkpointer backend: {settings.workflow_checkpointer_backend}")
        typer.echo(f"checkpoint db: {settings.workflow_checkpoint_db}")
        typer.echo(f"run root: {settings.workflow_run_root}")
        typer.echo(f"recursion limit: {settings.workflow_recursion_limit}")
        typer.echo(f"history limit: {settings.workflow_history_limit}")
        typer.echo(f"stream mode: {settings.workflow_stream_mode}")
        typer.echo(f"allow replay: {str(settings.workflow_allow_replay).lower()}")
        try:
            typer.echo(f"langgraph: {package_version('langgraph')}")
        except PackageNotFoundError:
            typer.echo("langgraph: not installed")
    except Exception as exc:
        _abort(str(exc), _error_code(exc))


def _agent_tool_instances() -> tuple[Any, ...]:
    """Instantiate the five whitelisted agent tools (no network)."""
    from materials_screening.agent_tools import (
        CompareRankedMaterialsTool,
        GetScreeningResultTool,
        GetWorkflowHistoryTool,
        GetWorkflowStatusTool,
        RunScreeningWorkflowTool,
    )

    return (
        RunScreeningWorkflowTool(),
        GetWorkflowStatusTool(),
        GetWorkflowHistoryTool(),
        GetScreeningResultTool(),
        CompareRankedMaterialsTool(),
    )


def _agent_tool_definitions() -> tuple[Any, ...]:
    """Whitelisted tool definitions, sorted by name."""
    from materials_screening.agent.tool_registry import AgentToolRegistry

    return AgentToolRegistry(_agent_tool_instances()).definitions()


def _print_agent_tools() -> None:
    """Print only the whitelisted tool names, descriptions and input fields."""
    for definition in _agent_tool_definitions():
        typer.echo(f"{definition.name}: {definition.description}")
        properties = definition.parameters.get("properties", {})
        if isinstance(properties, dict) and properties:
            typer.echo("  inputs: " + ", ".join(sorted(properties)))


def _build_agent_model(llm_provider: Literal["mock", "intern"]) -> Any:
    """Build the Intern agent model or the offline mock."""
    from pydantic import SecretStr

    from materials_screening.agent.intern_model import InternAgentModel
    from materials_screening.agent.mock_model import WorkflowDrivenMockAgentModel
    from materials_screening.agent.settings import AgentSettings, shared_intern_settings

    if llm_provider == "intern":
        api_key = os.getenv("INTERN_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigurationError("INTERN_API_KEY is not set")
        return InternAgentModel(
            shared_intern_settings(AgentSettings()),
            api_key=SecretStr(api_key),
        )
    # Offline demo: the mock model drives the real (mock) workflow so CLI
    # answers carry genuine screening results instead of a canned reply.
    return WorkflowDrivenMockAgentModel()


def _build_master_model(llm_provider: Literal["mock", "intern"]) -> Any:
    """Build the Master model; mock mode supports deterministic delegation."""
    if llm_provider == "mock":
        from materials_screening.master.mock_model import MasterMockAgentModel

        return MasterMockAgentModel()
    return _build_agent_model(llm_provider)


@contextmanager
def _open_agent_runner(
    *,
    run_root: Path,
    llm_provider: Literal["mock", "intern"],
    planner_fixture: Path | None,
    materials_repository: str,
    materials_fixture: Path | None,
    conversation_db: Path | None = None,
    checkpoint_db: Path | None = None,
) -> Iterator[Any]:
    """Open a MaterialAgentRunner; never constructs real clients for mock."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    from materials_screening.agent.conversation_store import SqliteConversationStore
    from materials_screening.agent.runner import MaterialAgentRunner
    from materials_screening.agent.settings import AgentSettings
    from materials_screening.agent.tool_registry import AgentToolRegistry
    from materials_screening.agent_tools.result_reader import FileWorkflowResultReader

    settings = AgentSettings()
    store = SqliteConversationStore(
        conversation_db or Path("data/agent_conversations.sqlite")
    )
    try:
        with _open_workflow_runner(
            run_root=run_root,
            llm_provider=llm_provider,
            planner_fixture=planner_fixture,
            materials_repository=materials_repository,
            materials_fixture=materials_fixture,
        ) as workflow_runner:
            checkpoint_path = checkpoint_db or settings.agent_checkpoint_db
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                str(checkpoint_path),
                check_same_thread=False,
            )
            saver = SqliteSaver(connection)
            runner = MaterialAgentRunner(
                settings=settings,
                store=store,
                workflow_runner=workflow_runner,
                workflow_result_reader=FileWorkflowResultReader(run_root),
                tool_registry=AgentToolRegistry(_agent_tool_instances()),
                agent_model=_build_agent_model(llm_provider),
                checkpointer=saver,
            )
            try:
                yield runner
            finally:
                connection.close()
    finally:
        store.close()


def _print_agent_result(result: Any) -> None:
    typer.echo(f"Conversation: {result.conversation_id}")
    typer.echo(f"Status: {result.status}")
    if result.active_workflow_thread_id:
        typer.echo(f"Active thread: {result.active_workflow_thread_id}")
    if result.selected_tools:
        typer.echo(f"Tools: {', '.join(result.selected_tools)}")
    typer.echo(f"Agent> {result.response_text}")
    if result.error is not None:
        typer.echo(
            f"Error: {result.error.get('code', 'ERROR')}: "
            f"{result.error.get('message', 'agent failed')}"
        )


def _run_agent_with_progress(
    runner: Any,
    *,
    message: str,
    conversation_id: str | None = None,
    artifact_refs: tuple[str, ...] = (),
    task_id: str | None = None,
) -> None:
    """Run agent.ask_stream and print real-time progress."""
    arguments: dict[str, Any] = {"message": message, "conversation_id": conversation_id}
    if artifact_refs:
        arguments["artifact_refs"] = artifact_refs
    if task_id is not None:
        arguments["task_id"] = task_id
    for event in runner.ask_stream(**arguments):
        if event.is_final:
            if event.result is not None:
                typer.echo("")  # blank line before final output
                _print_agent_result(event.result)
        else:
            typer.echo(f"  [{event.node}] {event.message}")


def _chat_loop(
    runner: Any, conversation_id: str | None, *, progress: bool = False
) -> None:
    """Interactive loop; only /exit, /new, /id and /tools are supported."""
    current_id = conversation_id
    if current_id is None:
        current_id = runner.start_conversation()
        typer.echo(f"Conversation: {current_id}")
    while True:
        try:
            line = input("You> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.startswith("/"):
            command = line.lower()
            if command == "/exit":
                break
            if command == "/new":
                current_id = runner.start_conversation()
                typer.echo(f"Conversation: {current_id}")
                continue
            if command == "/id":
                typer.echo(f"Conversation: {current_id}")
                continue
            if command == "/tools":
                _print_agent_tools()
                continue
            typer.echo("Unknown command; supported: /exit /new /id /tools")
            continue
        if progress:
            for event in runner.ask_stream(message=line, conversation_id=current_id):
                if event.is_final:
                    if event.result is not None:
                        typer.echo("")
                        typer.echo(f"Agent> {event.result.response_text}")
                        if event.result.error is not None:
                            typer.echo(
                                f"Error: {event.result.error.get('code', 'ERROR')}: "
                                f"{event.result.error.get('message', 'agent failed')}"
                            )
                else:
                    typer.echo(f"  [{event.node}] {event.message}")
        else:
            result = runner.ask(message=line, conversation_id=current_id)
            typer.echo(f"Agent> {result.response_text}")
            if result.error is not None:
                typer.echo(
                    f"Error: {result.error.get('code', 'ERROR')}: "
                    f"{result.error.get('message', 'agent failed')}"
                )


@agent_app.command("ask")
def agent_ask_command(
    message: str = typer.Option(
        ...,
        "--message",
        "-m",
        help="User message for the agent.",
    ),
    conversation_id: str | None = typer.Option(
        None,
        "--conversation-id",
        "-c",
        help="Continue an existing conversation.",
    ),
    llm_provider: Literal["mock", "intern"] = typer.Option(
        "mock", "--llm-provider", help="Agent model provider: mock or intern."
    ),
    materials_repository: str | None = typer.Option(
        None,
        "--materials-repository",
        help=(
            "Repository: mock or materials-project. Defaults to "
            "materials-project with --llm-provider intern, else mock."
        ),
    ),
    planner_fixture: Path | None = typer.Option(
        None, "--planner-fixture", help="Planner mock fixtures JSON."
    ),
    materials_fixture: Path | None = typer.Option(
        None, "--materials-fixture", help="Mock repository fixtures JSON."
    ),
    run_root: Path = typer.Option(
        Path("data/workflow_runs"),
        "--output",
        "-o",
        help="Workflow run root directory.",
    ),
    progress: bool = typer.Option(
        False,
        "--progress",
        "-p",
        help="Show execution progress in real time.",
    ),
) -> None:
    """Ask the agent once and print the conversation id and safe response."""
    materials_repository, planner_fixture, materials_fixture = (
        _resolve_provider_defaults(
            llm_provider,
            materials_repository,
            planner_fixture,
            materials_fixture,
        )
    )
    try:
        with _open_agent_runner(
            run_root=run_root,
            llm_provider=llm_provider,
            planner_fixture=planner_fixture,
            materials_repository=materials_repository,
            materials_fixture=materials_fixture,
        ) as runner:
            if progress:
                _run_agent_with_progress(
                    runner, message=message, conversation_id=conversation_id
                )
            else:
                result = runner.ask(message=message, conversation_id=conversation_id)
                _print_agent_result(result)
    except typer.Exit:
        raise
    except ImportError as exc:
        _abort(
            f"agent extra not installed: {exc}; run `uv sync --extra workflow`",
            3,
        )
    except Exception as exc:
        _abort(str(exc), _error_code(exc))


@agent_app.command("chat")
def agent_chat_command(
    conversation_id: str | None = typer.Option(
        None,
        "--conversation-id",
        "-c",
        help="Continue an existing conversation.",
    ),
    llm_provider: Literal["mock", "intern"] = typer.Option(
        "mock", "--llm-provider", help="Agent model provider: mock or intern."
    ),
    materials_repository: str | None = typer.Option(
        None,
        "--materials-repository",
        help=(
            "Repository: mock or materials-project. Defaults to "
            "materials-project with --llm-provider intern, else mock."
        ),
    ),
    planner_fixture: Path | None = typer.Option(
        None, "--planner-fixture", help="Planner mock fixtures JSON."
    ),
    materials_fixture: Path | None = typer.Option(
        None, "--materials-fixture", help="Mock repository fixtures JSON."
    ),
    run_root: Path = typer.Option(
        Path("data/workflow_runs"),
        "--output",
        "-o",
        help="Workflow run root directory.",
    ),
    progress: bool = typer.Option(
        False,
        "--progress",
        "-p",
        help="Show execution progress in real time.",
    ),
) -> None:
    """Interactive terminal; commands: /exit, /new, /id, /tools."""
    materials_repository, planner_fixture, materials_fixture = (
        _resolve_provider_defaults(
            llm_provider,
            materials_repository,
            planner_fixture,
            materials_fixture,
        )
    )
    try:
        with _open_agent_runner(
            run_root=run_root,
            llm_provider=llm_provider,
            planner_fixture=planner_fixture,
            materials_repository=materials_repository,
            materials_fixture=materials_fixture,
        ) as runner:
            _chat_loop(runner, conversation_id, progress=progress)
    except typer.Exit:
        raise
    except ImportError as exc:
        _abort(
            f"agent extra not installed: {exc}; run `uv sync --extra workflow`",
            3,
        )
    except Exception as exc:
        _abort(str(exc), _error_code(exc))


@agent_app.command("show")
def agent_show_command(
    conversation_id: str = typer.Argument(..., help="Conversation id."),
) -> None:
    """Show a safe summary of one conversation."""
    try:
        with _open_agent_runner(
            run_root=Path("data/workflow_runs"),
            llm_provider="mock",
            planner_fixture=None,
            materials_repository="mock",
            materials_fixture=None,
        ) as runner:
            view = runner.get_conversation(conversation_id)
            typer.echo(f"Conversation: {view.conversation_id}")
            typer.echo(f"Created: {view.created_at}")
            typer.echo(f"Updated: {view.updated_at}")
            typer.echo(f"Turns: {view.turn_count}")
            typer.echo(f"Active thread: {view.active_workflow_thread_id or '(none)'}")
            typer.echo(
                "Last status: "
                f"{runner.conversation_status(conversation_id) or '(none)'}"
            )
            if view.workflow_threads:
                typer.echo("Workflows: " + ", ".join(view.workflow_threads))
    except typer.Exit:
        raise
    except ImportError as exc:
        _abort(
            f"agent extra not installed: {exc}; run `uv sync --extra workflow`",
            3,
        )
    except Exception as exc:
        _abort(str(exc), _error_code(exc))


@agent_app.command("tools")
def agent_tools_command() -> None:
    """Show the whitelisted agent tools (names, descriptions, inputs)."""
    try:
        _print_agent_tools()
    except Exception as exc:
        _abort(str(exc), _error_code(exc))


@agent_app.command("inspect")
def agent_inspect_command() -> None:
    """Print safe agent configuration; no network access by default."""
    try:
        from materials_screening.agent.settings import (
            AgentSettings,
            shared_intern_settings,
        )

        settings = shared_intern_settings(AgentSettings())
        typer.echo(f"enabled: {str(settings.agent_enabled).lower()}")
        typer.echo(f"version: {settings.agent_version}")
        typer.echo(f"prompt version: {settings.agent_prompt_version}")
        typer.echo(f"base url: {settings.agent_base_url}")
        typer.echo(f"model: {settings.agent_model}")
        typer.echo(f"reasoning effort: {settings.agent_reasoning_effort}")
        typer.echo(f"temperature: {settings.agent_temperature}")
        typer.echo(
            f"max model calls per turn: {settings.agent_max_model_calls_per_turn}"
        )
        typer.echo(f"max tool calls per turn: {settings.agent_max_tool_calls_per_turn}")
        typer.echo(f"max conversation turns: {settings.agent_max_conversation_turns}")
        typer.echo(f"checkpoint db: {settings.agent_checkpoint_db}")
        typer.echo(f"allow web search: {str(settings.agent_allow_web_search).lower()}")
        key_configured = bool(os.getenv("INTERN_API_KEY", "").strip())
        typer.echo(f"INTERN_API_KEY configured: {str(key_configured).lower()}")
        try:
            typer.echo(f"langgraph: {package_version('langgraph')}")
        except PackageNotFoundError:
            typer.echo("langgraph: not installed")
    except Exception as exc:
        _abort(str(exc), _error_code(exc))


@agent_app.command("ui")
def agent_ui_command(
    server_port: int = typer.Option(8501, "--port", help="Server port."),
) -> None:
    """Launch the Gradio web UI for the agent."""
    _launch_gradio_ui(server_port)


@master_app.command("db-ui")
def materials_database_ui_command(
    server_port: int = typer.Option(8502, "--port", help="Server port."),
) -> None:
    """Launch the dedicated Materials Database Agent demonstration UI."""
    _launch_materials_database_ui(server_port)


@master_app.command("ui")
def unified_multi_agent_ui_command(
    server_port: int = typer.Option(8501, "--port", min=1, max=65535),
) -> None:
    """Launch the MA-1 unified explicit-mode multi-agent UI."""
    try:
        import gradio  # noqa: F401
    except ImportError as exc:
        _abort(f"gradio not installed: {exc}; run `uv sync --extra web-ui`", 3)
    ui_path = Path(__file__).parent / "unified_ui_gradio.py"
    env = os.environ.copy()
    env["GRADIO_SERVER_PORT"] = str(server_port)
    try:
        subprocess.run([sys.executable, str(ui_path)], check=True, env=env)
    except subprocess.CalledProcessError as exc:
        _abort(f"unified multi-agent UI exited with code {exc.returncode}", 4)


def _launch_materials_database_ui(server_port: int) -> None:
    """Launch the standalone database sub-agent Gradio UI."""
    try:
        import gradio  # noqa: F401
    except ImportError as exc:
        _abort(
            f"gradio not installed: {exc}; run `uv sync --extra web-ui`",
            3,
        )

    ui_path = Path(__file__).parent / "materials_database_ui_gradio.py"
    env = os.environ.copy()
    env["GRADIO_SERVER_PORT"] = str(server_port)
    typer.echo(f"Starting Materials Database Agent UI: http://localhost:{server_port}")
    try:
        subprocess.run([sys.executable, str(ui_path)], check=True, env=env)
    except subprocess.CalledProcessError as exc:
        _abort(f"Database Agent UI exited with code {exc.returncode}", 10)
    except KeyboardInterrupt:
        typer.echo("")
        typer.echo("Database Agent UI stopped.")


def _launch_gradio_ui(server_port: int) -> None:
    """Launch the Gradio web UI."""
    try:
        import gradio  # noqa: F401
    except ImportError as exc:
        _abort(
            f"gradio not installed: {exc}; run `uv pip install gradio`",
            3,
        )

    ui_path = Path(__file__).parent / "agent_ui_gradio.py"

    args = [
        sys.executable,
        str(ui_path),
    ]
    env = os.environ.copy()
    env.setdefault("GRADIO_SERVER_PORT", str(server_port))

    typer.echo(f"鍚姩 Gradio Web UI: http://localhost:{server_port}")
    typer.echo("Tip: Gradio uses WebSocket for responsive updates.")
    try:
        subprocess.run(args, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        _abort(f"Gradio exited with code {exc.returncode}", 10)
    except KeyboardInterrupt:
        typer.echo("")
        typer.echo("Web UI stopped.")


# Master Agent CLI


@contextmanager
def _open_master_runner(
    *,
    run_root: Path,
    llm_provider: Literal["mock", "intern"],
    planner_fixture: Path | None,
    materials_repository: str,
    materials_fixture: Path | None,
) -> Iterator[Any]:
    """Open a MasterAgentRunner with the materials_screening sub-agent."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    from materials_screening.agent.conversation_store import SqliteConversationStore
    from materials_screening.agent_tools.result_reader import FileWorkflowResultReader
    from materials_screening.data_analysis.dataset_store import DatasetStore
    from materials_screening.master import (
        ArtifactRegistry,
        DataAnalysisCrossAgentCoordinator,
        MasterAgentRunner,
        MasterAgentSettings,
        SubAgentRegistry,
        SubAgentSpec,
    )
    from materials_screening.services.query_result_store import QueryResultStore
    from materials_screening.sub_agents.data_analysis.fulltext_adapter import (
        bounded_description_agent,
    )
    from materials_screening.sub_agents.data_analysis.spec_factory import (
        create_spec as data_analysis_create_spec,
    )
    from materials_screening.sub_agents.literature.fulltext_factory import (
        create_fulltext_preview_processor,
    )
    from materials_screening.sub_agents.literature.spec_factory import (
        create_spec as literature_create_spec,
    )
    from materials_screening.sub_agents.materials_database.spec_factory import (
        create_spec as db_create_spec,
    )

    master_settings = MasterAgentSettings()
    store = SqliteConversationStore(master_settings.master_conversation_db)
    try:
        with _open_workflow_runner(
            run_root=run_root,
            llm_provider=llm_provider,
            planner_fixture=planner_fixture,
            materials_repository=materials_repository,
            materials_fixture=materials_fixture,
        ) as workflow_runner:
            # Only pass a real API key when the master itself is running in
            # real mode; otherwise the sub-agents would silently switch from
            # their offline mock models to Intern despite --llm-provider mock.
            intern_api_key = (
                os.getenv("INTERN_API_KEY", "").strip() or None
                if llm_provider == "intern"
                else None
            )
            result_reader = FileWorkflowResultReader(run_root)
            repository = _build_repository(materials_repository, materials_fixture)
            db_spec = db_create_spec(
                repository=repository,
                workflow_runner=workflow_runner,
                result_reader=result_reader,
                intern_api_key=intern_api_key,
            )
            literature_spec = literature_create_spec(
                workflow_runner=workflow_runner,
                result_reader=result_reader,
                intern_api_key=intern_api_key,
                s2_api_key=(
                    os.getenv("S2_API_KEY", "").strip() or None
                    if llm_provider == "intern"
                    else None
                ),
                enable_remote_search=llm_provider == "intern",
                # Keep metadata routing lazy. Master fulltext task attachments
                # use a separate staged processor; do not enable heavy RAG here.
                enable_rag=False,
            )
            data_analysis_spec = data_analysis_create_spec(
                workflow_runner=workflow_runner,
                result_reader=result_reader,
                data_root=Path("data/data_analysis"),
                intern_api_key=intern_api_key,
            )
            screening_compat = SubAgentSpec(
                name="materials_screening",
                description=(
                    "Deprecated compatibility alias for materials_database. "
                    "Use delegate_to_materials_database for new requests."
                ),
                system_prompt=db_spec.system_prompt,
                tool_definitions=db_spec.tool_definitions,
                runner_factory=db_spec.runner_factory,
            )
            outlier_compat = SubAgentSpec(
                name="outlier_detection",
                description=(
                    "Deprecated compatibility alias for materials_database. "
                    "Use delegate_to_materials_database for new requests."
                ),
                system_prompt=db_spec.system_prompt,
                tool_definitions=db_spec.tool_definitions,
                runner_factory=db_spec.runner_factory,
            )
            registry = SubAgentRegistry(
                [
                    db_spec,
                    literature_spec,
                    data_analysis_spec,
                    screening_compat,
                    outlier_compat,
                ]
            )
            master_model = _build_master_model(llm_provider)

            checkpoint_path = master_settings.master_checkpoint_db
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
            saver = SqliteSaver(connection)

            artifact_registry = ArtifactRegistry(
                next(iter(_literature_ingestion_roots()), Path("data/literature_pdfs"))
            )
            from materials_screening.master.figure_evidence_ocr import WindowsFigureOcr
            from materials_screening.master.figure_evidence_review import FigureEvidenceService
            from materials_screening.master.figure_evidence_store import FigureEvidenceStore
            from materials_screening.master.fulltext_snapshots import ExtractionSnapshotStore
            figure_service = FigureEvidenceService(
                FigureEvidenceStore(Path("data/fulltext_figures")), artifact_registry,
                ExtractionSnapshotStore(Path("data/fulltext_snapshots")), ocr=WindowsFigureOcr(),
            )
            runner = MasterAgentRunner(
                settings=master_settings,
                store=store,
                sub_agent_registry=registry,
                master_model=master_model,
                checkpointer=saver,
                data_analysis_coordinator=DataAnalysisCrossAgentCoordinator(
                    dataset_store=DatasetStore(Path("data/data_analysis")),
                    query_store=QueryResultStore(Path("data/material_queries")),
                ),
                artifact_registry=artifact_registry,
                figure_review_service=figure_service,
                figure_review_enabled=os.getenv("MASTER_FIGURE_EVIDENCE_REVIEW", "").strip().casefold() in {"1","true","yes"},
                preview_processor=create_fulltext_preview_processor(
                    artifact_registry, enable_remote=llm_provider == "intern",
                    figure_review_service=figure_service,
                    max_output_bytes=master_settings.master_max_sub_agent_output_bytes,
                    analysis_agent_factory=lambda: bounded_description_agent(
                        data_root=Path("data/data_analysis"),
                        workflow_runner=workflow_runner, result_reader=result_reader,
                    ),
                    query_factory=lambda: QueryResultStore(
                        Path("data/material_queries")
                    ),
                ),
            )
            try:
                yield runner
            finally:
                connection.close()
    finally:
        store.close()


register_figure_review_command(master_app, _open_master_runner)


@master_app.command("ask")
def master_ask_command(
    message: str = typer.Option(
        ..., "--message", "-m", help="User message for the master agent."
    ),
    conversation_id: str | None = typer.Option(
        None, "--conversation-id", "-c", help="Continue an existing conversation."
    ),
    llm_provider: Literal["mock", "intern"] = typer.Option(
        "mock", "--llm-provider", help="Agent model provider."
    ),
    materials_repository: str | None = typer.Option(
        None,
        "--materials-repository",
        help="Repository: mock or materials-project.",
    ),
    planner_fixture: Path | None = typer.Option(
        None, "--planner-fixture", help="Planner mock fixtures JSON."
    ),
    materials_fixture: Path | None = typer.Option(
        None, "--materials-fixture", help="Mock repository fixtures JSON."
    ),
    run_root: Path = typer.Option(
        Path("data/workflow_runs"),
        "--output",
        "-o",
        help="Workflow run root directory.",
    ),
    progress: bool = typer.Option(
        False, "--progress", "-p", help="Show execution progress in real time."
    ),
    pdf: list[Path] | None = typer.Option(
        None,
        "--pdf",
        help="Attach PDF fulltexts to this Master task; may be repeated.",
    ),
    task_id: str | None = typer.Option(
        None, "--task-id", help="Select a saved fulltext task in this conversation.",
    ),
) -> None:
    """Ask the master agent once; delegates to sub-agents as needed."""
    materials_repository, planner_fixture, materials_fixture = (
        _resolve_provider_defaults(
            llm_provider,
            materials_repository,
            planner_fixture,
            materials_fixture,
        )
    )
    try:
        with _open_master_runner(
            run_root=run_root,
            llm_provider=llm_provider,
            planner_fixture=planner_fixture,
            materials_repository=materials_repository,
            materials_fixture=materials_fixture,
        ) as runner:
            arguments: dict[str, Any] = {
                "message": message, "conversation_id": conversation_id,
            }
            if pdf:
                if len(pdf) > 100:
                    _abort("一次最多绑定100篇全文。", 2)
                resolved_id = conversation_id or runner.start_conversation()
                artifacts = [
                    runner.register_pdf_attachment(path, conversation_id=resolved_id)
                    for path in pdf
                ]
                arguments["conversation_id"] = resolved_id
                arguments["artifact_refs"] = tuple(
                    dict.fromkeys(item.artifact_id for item in artifacts)
                )
            if task_id is not None:
                arguments["task_id"] = task_id
            if progress:
                _run_agent_with_progress(runner, **arguments)
            else:
                result = runner.ask(**arguments)
                _print_agent_result(result)
    except typer.Exit:
        raise
    except ImportError as exc:
        _abort(
            f"agent extra not installed: {exc}; run `uv sync --extra workflow`",
            3,
        )
    except Exception as exc:
        _abort(str(exc), _error_code(exc))


@master_app.command("sub-agents")
def master_sub_agents_command(
    run_root: Path = typer.Option(
        Path("data/workflow_runs"),
        "--output",
        "-o",
    ),
) -> None:
    """List registered sub-agents."""
    try:
        with _open_master_runner(
            run_root=run_root,
            llm_provider="mock",
            planner_fixture=None,
            materials_repository="mock",
            materials_fixture=None,
        ) as runner:
            for name in runner.sub_agent_registry.names():
                spec = runner.sub_agent_registry.get(name)
                typer.echo(f"{spec.delegate_function_name}: {spec.description}")
    except Exception as exc:
        _abort(str(exc), _error_code(exc))


@master_app.command("evidence-trial")
def master_evidence_trial_command(
    message: str = typer.Option(
        ..., "--message", "-m", help="Scientific question for the pilot report."
    ),
    report_id: str | None = typer.Option(
        None,
        "--report-id",
        help=(
            "Saved literature report ID. Omit when the message exactly "
            "matches its topic."
        ),
    ),
    data_root: Path = typer.Option(
        Path("data/data_analysis"),
        "--data-root",
        help="Private data-analysis artifact root.",
    ),
    output: Path = typer.Option(
        Path("outputs/task5_nha_trial/system_evidence_trial.md"),
        "--output",
        "-o",
        help="Human-readable Markdown report path.",
    ),
    requirements: Annotated[
        Path | None,
        typer.Option(
            "--requirements", help="Same task-specific required-metric JSON policy."
        ),
    ] = None,
) -> None:
    """Run the literature-evidence-to-analysis pilot without inventing data."""

    from materials_screening.data_analysis.dataset_store import DatasetStore
    from materials_screening.master.literature_evidence_trial import (
        LiteratureEvidenceTrialService,
        render_literature_evidence_trial,
    )
    from materials_screening.sub_agents.literature.metric_coverage import (
        load_task_metric_requirements,
    )
    from materials_screening.sub_agents.literature.user_report import (
        LiteratureUserReportStore,
    )

    try:
        metric_requirements = (
            load_task_metric_requirements(requirements) if requirements else None
        )
        if metric_requirements is not None:
            metric_requirements.ensure_topic(message)
        service = LiteratureEvidenceTrialService(
            report_store=LiteratureUserReportStore(
                Path("data/literature_user_reports")
            ),
            matrix_store=_literature_pgvector_store(),
            dataset_store=DatasetStore(data_root),
        )
        result = service.run(
            topic=message, report_id=report_id, metric_requirements=metric_requirements
        )
        markdown = render_literature_evidence_trial(result)
        target = output.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_text(markdown, encoding="utf-8")
        os.replace(temporary, target)
        typer.echo(markdown)
        typer.echo(f"Report path: {target}")
        if result.dataset_id:
            typer.echo(f"Dataset ID: {result.dataset_id}")
        if result.report_artifact_id:
            typer.echo(f"Report artifact ID: {result.report_artifact_id}")
    except Exception as exc:
        _abort(str(exc), _error_code(exc))


def main() -> None:
    """Run the CLI application."""
    load_dotenv()
    # Windows consoles often default to GBK and crash on model answers that
    # contain subscript/superscript characters; emit UTF-8 instead.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    app()


if __name__ == "__main__":
    main()
