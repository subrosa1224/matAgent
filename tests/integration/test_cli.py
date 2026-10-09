"""CLI tests with Typer CliRunner (M6)."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import materials_screening.cli as cli_module
from materials_screening.cli import app
from materials_screening.errors import ExportError, RepositoryError
from materials_screening.models import MaterialRecord, ScreeningRequest
from materials_screening.repositories.base import RetrievalResult
from materials_screening.repositories.mock import MockMaterialsRepository
from materials_screening.services.export_service import ExportResult, ExportService
from materials_screening.services.filter_service import FilterService
from materials_screening.services.ranking_service import RankingService
from materials_screening.services.screening_service import ScreeningService
from materials_screening.services.validation_service import ValidationService

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def _bare_record() -> MaterialRecord:
    return MaterialRecord(
        source="mock",
        material_id="mp-1",
        formula_pretty="Fe2O3",
        elements=("Fe", "O"),
        band_gap_ev=1.5,
        energy_above_hull_ev_atom=0.01,
        is_metal=False,
    )


class _FailingRepository:
    def search(self, request: ScreeningRequest) -> RetrievalResult:
        raise RepositoryError("repository exploded")

    def healthcheck(self) -> bool:
        return False


class _FailingExport:
    def export(
        self,
        result: object,
        output_root: Path,
        *,
        include_cif: bool = True,
    ) -> ExportResult:
        raise ExportError("disk full")


class TestVersionCommand:
    def test_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert result.stdout.strip() == "0.1.0"


class TestValidateRequest:
    def test_valid_request(self) -> None:
        request_path = EXAMPLES_DIR / "semiconductor_request.json"
        result = runner.invoke(
            app, ["validate-request", "--request", str(request_path)]
        )
        assert result.exit_code == 0
        assert "Request is valid" in result.output
        assert "Fingerprint:" in result.output

    def test_invalid_request(self) -> None:
        request_path = EXAMPLES_DIR / "invalid_request.json"
        result = runner.invoke(
            app, ["validate-request", "--request", str(request_path)]
        )
        assert result.exit_code == 2
        assert "Request is invalid" in result.output

    def test_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.json"
        result = runner.invoke(app, ["validate-request", "--request", str(missing)])
        assert result.exit_code == 2
        assert "cannot read request file" in result.output


class TestInspectMp:
    def test_inspect_mp(self) -> None:
        result = runner.invoke(app, ["inspect-mp"])
        assert result.exit_code == 0
        assert "mp-api:" in result.output
        assert "query param band_gap: present" in result.output


class TestScreen:
    def test_mock_full_run(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "screen",
                "--repository",
                "mock",
                "--fixture",
                str(FIXTURES_DIR / "mp_documents.json"),
                "--request",
                str(EXAMPLES_DIR / "semiconductor_request.json"),
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        assert "Task completed" in result.output
        assert "Validation: PASSED" in result.output
        run_dirs = list(tmp_path.glob("run_*"))
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "result.json").exists()
        assert (run_dirs[0] / "candidates.csv").exists()
        assert (run_dirs[0] / "report.md").exists()

    def test_mock_default_exports_cif(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "screen",
                "--repository",
                "mock",
                "--fixture",
                str(FIXTURES_DIR / "mp_documents.json"),
                "--request",
                str(EXAMPLES_DIR / "li_fe_o_request.json"),
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        run_dirs = list(tmp_path.glob("run_*"))
        assert (run_dirs[0] / "cif" / "mp-1.cif").exists()

    def test_mock_no_cif(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "screen",
                "--repository",
                "mock",
                "--fixture",
                str(FIXTURES_DIR / "mp_documents.json"),
                "--request",
                str(EXAMPLES_DIR / "li_fe_o_request.json"),
                "--output",
                str(tmp_path),
                "--no-cif",
            ],
        )
        assert result.exit_code == 0
        run_dirs = list(tmp_path.glob("run_*"))
        assert not (run_dirs[0] / "cif").exists()

    def test_missing_api_key_clear_error(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("MP_API_KEY", raising=False)
        monkeypatch.delenv("PMG_MAPI_KEY", raising=False)
        result = runner.invoke(
            app,
            [
                "screen",
                "--request",
                str(EXAMPLES_DIR / "semiconductor_request.json"),
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 3
        assert "MP_API_KEY is not set" in result.output

    def test_unknown_repository(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "screen",
                "--repository",
                "bogus",
                "--request",
                str(EXAMPLES_DIR / "semiconductor_request.json"),
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 2
        assert "unknown repository" in result.output

    def test_missing_request_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.json"
        result = runner.invoke(
            app,
            [
                "screen",
                "--request",
                str(missing),
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 2
        assert "cannot read request file" in result.output

    def test_invalid_request_schema(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "screen",
                "--request",
                str(EXAMPLES_DIR / "invalid_request.json"),
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 2
        assert "Request is invalid" in result.output


class TestScreenErrorMapping:
    def test_repository_error_exit_code_4(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = ScreeningService(
            repository=_FailingRepository(),
            filter_service=FilterService(),
            ranking_service=RankingService(),
            validation_service=ValidationService(),
            export_service=ExportService(),
        )
        monkeypatch.setattr(
            cli_module, "_build_service", lambda repository, fixture: service
        )
        result = runner.invoke(
            app,
            [
                "screen",
                "--request",
                str(EXAMPLES_DIR / "semiconductor_request.json"),
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 4
        assert "repository exploded" in result.output

    def test_validation_failure_exit_code_5(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = ScreeningService(
            repository=MockMaterialsRepository(records=(_bare_record(),)),
            filter_service=FilterService(),
            ranking_service=RankingService(),
            validation_service=ValidationService(),
            export_service=ExportService(),
        )
        monkeypatch.setattr(
            cli_module, "_build_service", lambda repository, fixture: service
        )
        result = runner.invoke(
            app,
            [
                "screen",
                "--request",
                str(EXAMPLES_DIR / "semiconductor_request.json"),
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 5
        assert "validation failed" in result.output

    def test_export_failure_exit_code_6(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = ScreeningService(
            repository=MockMaterialsRepository(records=()),
            filter_service=FilterService(),
            ranking_service=RankingService(),
            validation_service=ValidationService(),
            export_service=_FailingExport(),
        )
        monkeypatch.setattr(
            cli_module, "_build_service", lambda repository, fixture: service
        )
        result = runner.invoke(
            app,
            [
                "screen",
                "--request",
                str(EXAMPLES_DIR / "semiconductor_request.json"),
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 6
        assert "disk full" in result.output
