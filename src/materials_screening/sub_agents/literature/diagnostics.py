"""Read-only environment diagnostics for LiteratureAgent."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class DiagnosticCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    status: str
    detail: str


class LiteratureDiagnosticReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ready_for_metadata_search: bool
    ready_for_pdf_rag: bool
    checks: tuple[DiagnosticCheck, ...]


def diagnose_literature_environment(
    *,
    database_url: str | None,
    ingestion_roots: tuple[Path, ...],
    check_database: bool = True,
) -> LiteratureDiagnosticReport:
    checks: list[DiagnosticCheck] = []
    modules = {
        "PyMuPDF": "pymupdf",
        "psycopg": "psycopg",
        "sentence-transformers": "sentence_transformers",
        "FlagEmbedding": "FlagEmbedding",
    }
    module_ready: dict[str, bool] = {}
    for label, module in modules.items():
        present = importlib.util.find_spec(module) is not None
        module_ready[label] = present
        checks.append(
            DiagnosticCheck(
                name=label,
                status="ok" if present else "missing",
                detail="installed"
                if present
                else "optional dependency is not installed",
            )
        )

    roots_ready = bool(ingestion_roots) and all(
        root.expanduser().resolve().is_dir() for root in ingestion_roots
    )
    checks.append(
        DiagnosticCheck(
            name="ingestion_roots",
            status="ok" if roots_ready else "missing",
            detail=(
                ", ".join(str(root.expanduser().resolve()) for root in ingestion_roots)
                if ingestion_roots
                else "LITERATURE_INGEST_ROOTS is not configured"
            ),
        )
    )

    database_ready = False
    if not database_url:
        checks.append(
            DiagnosticCheck(
                name="pgvector",
                status="missing",
                detail="LITERATURE_DATABASE_URL is not configured",
            )
        )
    elif not module_ready["psycopg"]:
        checks.append(
            DiagnosticCheck(
                name="pgvector",
                status="blocked",
                detail="psycopg is required before the database can be checked",
            )
        )
    elif not check_database:
        checks.append(
            DiagnosticCheck(
                name="pgvector",
                status="skipped",
                detail="database connection check was disabled",
            )
        )
    else:
        database_ready, detail = _check_pgvector(database_url)
        checks.append(
            DiagnosticCheck(
                name="pgvector",
                status="ok" if database_ready else "error",
                detail=detail,
            )
        )

    rag_ready = all(module_ready.values()) and roots_ready and database_ready
    return LiteratureDiagnosticReport(
        ready_for_metadata_search=True,
        ready_for_pdf_rag=rag_ready,
        checks=tuple(checks),
    )


def _check_pgvector(database_url: str) -> tuple[bool, str]:
    try:
        psycopg = importlib.import_module("psycopg")

        with (
            psycopg.connect(database_url, connect_timeout=5) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            row = cursor.fetchone()
        if row is None:
            return (
                False,
                "PostgreSQL is reachable but the vector extension is not enabled",
            )
        return True, f"PostgreSQL reachable; pgvector version {row[0]}"
    except Exception as exc:
        return False, f"database check failed ({type(exc).__name__})"
