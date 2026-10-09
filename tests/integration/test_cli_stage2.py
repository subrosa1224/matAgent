"""CLI tests for the stage-2 natural language commands (D2-M6)."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import materials_screening.cli as cli_module
from materials_screening.cli import app
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
from materials_screening.planner.errors import PlannerQueryError

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
PLANNER_FIXTURE = FIXTURES_DIR / "planner_cli_fixtures.json"
MP_FIXTURE = FIXTURES_DIR / "mp_documents.json"
EVAL_CASES = FIXTURES_DIR / "planner_eval_cases.jsonl"


class TestParseCommand:
    def _invoke(self, query: str, *extra: str):
        return runner.invoke(
            app,
            [
                "parse",
                "--query",
                query,
                "--fixture",
                str(PLANNER_FIXTURE),
                *extra,
            ],
        )

    def test_ready_exit_0(self) -> None:
        result = self._invoke("ready query")
        assert result.exit_code == 0
        assert "Status: ready" in result.output
        assert "Fingerprint:" in result.output
        assert "Provider: mock" in result.output

    def test_default_provider_is_mock(self) -> None:
        result = self._invoke("ready query")
        assert "Provider: mock" in result.output

    def test_needs_clarification_exit_20(self) -> None:
        result = self._invoke("clarify query")
        assert result.exit_code == 20
        assert "Status: needs_clarification" in result.output
        assert "Clarification:" in result.output

    def test_invalid_exit_21(self) -> None:
        result = self._invoke("invalid query")
        assert result.exit_code == 21
        assert "Status: invalid" in result.output

    def test_unsupported_exit_22(self) -> None:
        result = self._invoke("unsupported query")
        assert result.exit_code == 22
        assert "Status: unsupported" in result.output

    def test_provider_error_exit_32(self) -> None:
        result = self._invoke("timeout query")
        assert result.exit_code == 32
        assert "mock timeout" in result.output

    def test_intern_config_error_exit_30(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _no_key_provider(settings: object) -> None:
            raise LLMConfigurationError("INTERN_API_KEY is not set")

        monkeypatch.setattr(cli_module, "create_llm_provider", _no_key_provider)
        result = self._invoke("ready query", "--provider", "intern")
        assert result.exit_code == 30
        assert "INTERN_API_KEY is not set" in result.output

    def test_unknown_provider_exit_2(self) -> None:
        result = self._invoke("ready query", "--provider", "bogus")
        assert result.exit_code == 2
        assert "is not one of 'mock', 'intern'" in result.output


class TestExitCodeMapping:
    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (PlannerQueryError("empty query"), 2),
            (LLMConfigurationError("config"), 30),
            (LLMModelNotSupportedError("model"), 30),
            (LLMAuthenticationError("auth"), 31),
            (LLMPermissionError("permission"), 31),
            (LLMRateLimitError("rate limit"), 32),
            (LLMTimeoutError("timeout"), 32),
            (LLMConnectionError("connection"), 32),
            (LLMServiceUnavailableError("service"), 32),
            (LLMRefusalError("refusal"), 33),
            (LLMStructuredOutputError("structure"), 34),
            (LLMTruncatedOutputError("truncated"), 34),
        ],
    )
    def test_parse_maps_error_to_exit_code(
        self,
        error: Exception,
        expected: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _broken_planner(provider: str, fixture: Path | None = None):
            raise error

        monkeypatch.setattr(cli_module, "_build_planner", _broken_planner)
        result = runner.invoke(
            app,
            ["parse", "--query", "ready query"],
        )
        assert result.exit_code == expected


class TestParseFileCommand:
    def _query_file(self, tmp_path: Path, lines: list[str]) -> Path:
        path = tmp_path / "queries.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_mixed_statuses_uses_worst_exit_code(self, tmp_path: Path) -> None:
        query_file = self._query_file(
            tmp_path,
            ["ready query", "clarify query", "invalid query", "unsupported query"],
        )
        result = runner.invoke(
            app,
            [
                "parse-file",
                "--input",
                str(query_file),
                "--fixture",
                str(PLANNER_FIXTURE),
            ],
        )
        assert result.exit_code == 21
        assert "ready=1 needs_clarification=1 invalid=1 unsupported=1" in result.output
        assert "#1: ready" in result.output
        assert "#4: unsupported" in result.output

    def test_provider_error_aborts_with_exit_32(self, tmp_path: Path) -> None:
        query_file = self._query_file(tmp_path, ["timeout query"])
        result = runner.invoke(
            app,
            [
                "parse-file",
                "--input",
                str(query_file),
                "--fixture",
                str(PLANNER_FIXTURE),
            ],
        )
        assert result.exit_code == 32
        assert "query 1 failed" in result.output

    def test_empty_file_exit_2(self, tmp_path: Path) -> None:
        query_file = tmp_path / "empty.txt"
        query_file.write_text("\n  \n", encoding="utf-8")
        result = runner.invoke(app, ["parse-file", "--input", str(query_file)])
        assert result.exit_code == 2
        assert "contains no queries" in result.output


class TestScreenQueryCommand:
    def _invoke_query(
        self,
        query: str,
        tmp_path: Path,
        *extra: str,
    ):
        return runner.invoke(
            app,
            [
                "screen",
                "--query",
                query,
                "--repository",
                "mock",
                "--fixture",
                str(MP_FIXTURE),
                "--planner-fixture",
                str(PLANNER_FIXTURE),
                "--output",
                str(tmp_path),
                *extra,
            ],
        )

    def test_ready_runs_full_pipeline(self, tmp_path: Path) -> None:
        result = self._invoke_query("ready query", tmp_path)
        assert result.exit_code == 0
        assert "Task completed" in result.output
        assert "Validation: PASSED" in result.output
        run_dirs = list(tmp_path.glob("run_*"))
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "result.json").exists()

    def test_needs_clarification_no_dir_no_mp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MP_API_KEY", raising=False)
        monkeypatch.delenv("PMG_MAPI_KEY", raising=False)
        result = runner.invoke(
            app,
            [
                "screen",
                "--query",
                "clarify query",
                "--repository",
                "materials-project",
                "--planner-fixture",
                str(PLANNER_FIXTURE),
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 20
        assert "Status: needs_clarification" in result.output
        assert list(tmp_path.glob("run_*")) == []

    def test_invalid_exit_21_no_dir(self, tmp_path: Path) -> None:
        result = self._invoke_query("invalid query", tmp_path)
        assert result.exit_code == 21
        assert list(tmp_path.glob("run_*")) == []

    def test_unsupported_exit_22_no_dir(self, tmp_path: Path) -> None:
        result = self._invoke_query("unsupported query", tmp_path)
        assert result.exit_code == 22
        assert list(tmp_path.glob("run_*")) == []

    def test_intern_config_error_exit_30(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _no_key_provider(settings: object) -> None:
            raise LLMConfigurationError("INTERN_API_KEY is not set")

        monkeypatch.setattr(cli_module, "create_llm_provider", _no_key_provider)
        result = runner.invoke(
            app,
            [
                "screen",
                "--query",
                "ready query",
                "--repository",
                "mock",
                "--fixture",
                str(MP_FIXTURE),
                "--planner-fixture",
                str(PLANNER_FIXTURE),
                "--output",
                str(tmp_path),
                "--provider",
                "intern",
            ],
        )
        assert result.exit_code == 30
        assert list(tmp_path.glob("run_*")) == []

    def test_request_and_query_are_exclusive(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "screen",
                "--request",
                str(EXAMPLES_DIR / "semiconductor_request.json"),
                "--query",
                "ready query",
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 2
        assert "exactly one of --request or --query" in result.output

    def test_neither_request_nor_query(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["screen", "--output", str(tmp_path)])
        assert result.exit_code == 2
        assert "exactly one of --request or --query" in result.output


class TestEvaluatePlannerCommand:
    def test_evaluate_planner(self) -> None:
        result = runner.invoke(
            app,
            [
                "evaluate-planner",
                "--input",
                str(EVAL_CASES),
                "--fixture",
                str(PLANNER_FIXTURE),
            ],
        )
        assert result.exit_code == 0
        assert "Total cases: 3" in result.output
        assert "Schema success rate: 0.667 (2/3)" in result.output
        assert "Status accuracy: 0.667 (2/3)" in result.output
        assert "Ready precision: 1.000" in result.output
        assert "Ready recall: 0.500" in result.output
        assert "Request field exact match: 1.000 (1/1)" in result.output
        assert "Numeric constraint accuracy: 0.000 (0/0)" in result.output
        assert "Element constraint accuracy: 0.000 (0/0)" in result.output
        assert "Ambiguity recall: 0.000" in result.output
        assert "Conflict recall: 1.000" in result.output
        assert "Unsupported accuracy: 0.000" in result.output
        assert "Injection resilience: 0.000" in result.output
        assert "Avg latency ms: 0.000" in result.output
        assert "Avg input tokens: 0.000" in result.output
        assert "Avg output tokens: 0.000" in result.output
        assert "Avg reasoning tokens: 0.000" in result.output

    def test_missing_eval_file_exit_2(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.jsonl"
        result = runner.invoke(app, ["evaluate-planner", "--input", str(missing)])
        assert result.exit_code == 2
        assert "cannot read eval file" in result.output
