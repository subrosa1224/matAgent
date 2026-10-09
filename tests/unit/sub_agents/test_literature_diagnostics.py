from __future__ import annotations

from pathlib import Path

from materials_screening.sub_agents.literature.diagnostics import (
    diagnose_literature_environment,
)


def test_diagnostics_reports_missing_optional_runtime(tmp_path: Path) -> None:
    report = diagnose_literature_environment(
        database_url=None,
        ingestion_roots=(tmp_path,),
        check_database=False,
    )
    assert report.ready_for_metadata_search is True
    assert report.ready_for_pdf_rag is False
    checks = {check.name: check for check in report.checks}
    assert checks["ingestion_roots"].status == "ok"
    assert checks["pgvector"].status == "missing"


def test_diagnostics_does_not_expose_database_url(tmp_path: Path) -> None:
    secret_url = "postgresql://user:very-secret@127.0.0.1:1/database"
    report = diagnose_literature_environment(
        database_url=secret_url,
        ingestion_roots=(tmp_path,),
        check_database=False,
    )
    assert "very-secret" not in report.model_dump_json()
