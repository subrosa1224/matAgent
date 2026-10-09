"""Integration tests for the workflow CLI commands (S3-M6)."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from materials_screening.cli import app

_RUNNER = CliRunner()
_THREAD_RE = re.compile(r"^Thread: (.+)$", re.MULTILINE)


@pytest.fixture()
def offline_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run CLI with cwd inside tmp so data/ paths stay offline."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _materials_fixture(tmp_path: Path) -> Path:
    fixture = [
        {
            "material_id": "mp-1",
            "formula_pretty": "Si",
            "elements": ["Si"],
            "band_gap": 1.5,
            "is_metal": False,
        },
        {
            "material_id": "mp-2",
            "formula_pretty": "Ge",
            "elements": ["Ge"],
            "band_gap": 1.6,
            "is_metal": False,
        },
    ]
    path = tmp_path / "materials.json"
    _write_json(path, fixture)
    return path


def _request_file(tmp_path: Path) -> Path:
    path = tmp_path / "request.json"
    _write_json(
        path,
        {"limit": 10, "band_gap_ev": {"min": 1.0, "max": 2.0}},
    )
    return path


def _planner_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "planner.json"
    _write_json(
        path,
        {
            "fixtures": [
                {
                    "query": "find materials",
                    "draft": {
                        "status": "extracted",
                        "required_elements": [],
                        "excluded_elements": [],
                        "chemsys": None,
                        "formula": None,
                        "band_gap_min": None,
                        "band_gap_max": None,
                        "band_gap_unit": "unspecified",
                        "hull_min": None,
                        "hull_max": None,
                        "hull_unit": "unspecified",
                        "density_min": None,
                        "density_max": None,
                        "density_unit": "unspecified",
                        "crystal_system": None,
                        "spacegroup_numbers": [],
                        "is_metal": None,
                        "is_stable": None,
                        "theoretical": None,
                        "target_band_gap": None,
                        "target_band_gap_unit": "unspecified",
                        "limit": None,
                        "ambiguities": ["llm_flagged_ambiguity"],
                        "unsupported_requirements": [],
                        "conflicts": [],
                        "assumptions": [],
                        "clarification_question": "",
                        "evidence": [],
                    },
                }
            ]
        },
    )
    return path


def _run_request_success(
    offline_cwd: Path,
    *extra: str,
) -> Any:
    fixture = _materials_fixture(offline_cwd)
    request = _request_file(offline_cwd)
    args = [
        "workflow",
        "run",
        "--request",
        str(request),
        "--materials-repository",
        "mock",
        "--materials-fixture",
        str(fixture),
    ]
    return _RUNNER.invoke(app, args + list(extra))


class TestWorkflowRun:
    def test_request_mode_completes(self, offline_cwd: Path) -> None:
        result = _run_request_success(offline_cwd)
        assert result.exit_code == 0, result.output
        assert "Status: completed" in result.output
        assert "Validation: PASSED" in result.output
        assert "Retrieved: 2" in result.output
        assert "Exports:" in result.output
        match = _THREAD_RE.search(result.output)
        assert match is not None
        assert match.group(1)

    def test_stream_mode_prints_node_summaries(self, offline_cwd: Path) -> None:
        result = _run_request_success(offline_cwd, "--stream")
        assert result.exit_code == 0, result.output
        assert "[initialize_run] run started" in result.output
        assert "[resolve_request] request resolved: ready" in result.output
        assert "[retrieve_materials] retrieved=2" in result.output
        assert "[finalize_success] workflow completed" in result.output

    def test_explicit_new_thread_id(self, offline_cwd: Path) -> None:
        result = _run_request_success(offline_cwd, "--thread-id", "fresh-123")
        assert result.exit_code == 0, result.output
        assert "Thread: fresh-123" in result.output

    def test_existing_thread_id_rejected(self, offline_cwd: Path) -> None:
        first = _run_request_success(offline_cwd)
        assert first.exit_code == 0
        match = _THREAD_RE.search(first.output)
        assert match is not None
        thread_id = match.group(1)
        second = _run_request_success(offline_cwd, "--thread-id", thread_id)
        assert second.exit_code == 2
        assert "already exists" in second.output

    def test_query_and_request_exclusive(self, offline_cwd: Path) -> None:
        request = _request_file(offline_cwd)
        result = _RUNNER.invoke(
            app,
            [
                "workflow",
                "run",
                "--query",
                "find materials",
                "--request",
                str(request),
            ],
        )
        assert result.exit_code == 2
        assert "exactly one" in result.output

    def test_neither_query_nor_request(self, offline_cwd: Path) -> None:
        result = _RUNNER.invoke(app, ["workflow", "run"])
        assert result.exit_code == 2
        assert "exactly one" in result.output

    def test_invalid_request_json_exit_2(self, offline_cwd: Path) -> None:
        bad = offline_cwd / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        result = _RUNNER.invoke(
            app,
            [
                "workflow",
                "run",
                "--request",
                str(bad),
                "--materials-repository",
                "mock",
            ],
        )
        assert result.exit_code == 2
        assert "invalid request JSON" in result.output

    def test_query_planner_stop_exit_20(self, offline_cwd: Path) -> None:
        planner = _planner_fixture(offline_cwd)
        result = _RUNNER.invoke(
            app,
            [
                "workflow",
                "run",
                "--query",
                "find materials",
                "--planner-fixture",
                str(planner),
            ],
        )
        assert result.exit_code == 20
        assert "needs_clarification" in result.output
        assert "Clarification:" in result.output
        assert "Retrieved: 0" in result.output

    def test_mock_query_mode_auto_uses_bundled_fixtures(
        self, offline_cwd: Path
    ) -> None:
        result = _RUNNER.invoke(
            app,
            [
                "workflow",
                "run",
                "--query",
                "ready query",
                "--llm-provider",
                "mock",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Status: completed" in result.output
        assert "Retrieved: 7" in result.output

    def test_no_cif_flag_accepted(self, offline_cwd: Path) -> None:
        result = _run_request_success(offline_cwd, "--no-cif")
        assert result.exit_code == 0, result.output


class TestWorkflowStatusAndHistory:
    def _thread_id(self, offline_cwd: Path) -> str:
        result = _run_request_success(offline_cwd)
        assert result.exit_code == 0
        match = _THREAD_RE.search(result.output)
        assert match is not None
        return match.group(1)

    def test_status_shows_safe_summary(self, offline_cwd: Path) -> None:
        thread_id = self._thread_id(offline_cwd)
        result = _RUNNER.invoke(app, ["workflow", "status", "--thread-id", thread_id])
        assert result.exit_code == 0, result.output
        assert "Status: completed" in result.output
        assert "Validation: PASSED" in result.output
        assert "Retrieved: 2" in result.output
        assert "events" not in result.output
        assert "retrieval_ref" not in result.output

    def test_history_shows_checkpoints(self, offline_cwd: Path) -> None:
        thread_id = self._thread_id(offline_cwd)
        result = _RUNNER.invoke(app, ["workflow", "history", "--thread-id", thread_id])
        assert result.exit_code == 0, result.output
        assert "finalize_success" in result.output
        assert re.search(r"\t[0-9a-f-]{36}\t", result.output)
        assert "\tcompleted\t" in result.output


class TestWorkflowDrawAndInspect:
    def test_draw_outputs_mermaid(self) -> None:
        result = _RUNNER.invoke(app, ["workflow", "draw"])
        assert result.exit_code == 0, result.output
        assert "flowchart LR" in result.output
        assert "initialize_run --> resolve_request" in result.output
        assert "finalize_success --> __end__" in result.output

    def test_inspect_shows_safe_config(self) -> None:
        result = _RUNNER.invoke(app, ["workflow", "inspect"])
        assert result.exit_code == 0, result.output
        assert "recursion limit: 32" in result.output
        assert "stream mode: updates" in result.output
        assert "langgraph:" in result.output


class TestWorkflowReplay:
    def _run_and_get_checkpoint(self, offline_cwd: Path) -> tuple[str, str]:
        run_result = _run_request_success(offline_cwd)
        assert run_result.exit_code == 0
        thread_id = _THREAD_RE.search(run_result.output).group(1)
        history = _RUNNER.invoke(app, ["workflow", "history", "--thread-id", thread_id])
        assert history.exit_code == 0
        for line in history.output.splitlines():
            fields = line.split("\t")
            if len(fields) == 6 and "rank_materials" in fields[5]:
                return thread_id, fields[1]
        raise AssertionError("filter-after checkpoint not found")

    def test_replay_unconfirmed_rejected(self, offline_cwd: Path) -> None:
        thread_id, checkpoint_id = self._run_and_get_checkpoint(offline_cwd)
        result = _RUNNER.invoke(
            app,
            [
                "workflow",
                "replay",
                "--thread-id",
                thread_id,
                "--checkpoint-id",
                checkpoint_id,
            ],
        )
        assert result.exit_code == 10
        assert "confirm_remote_calls" in result.output

    def test_replay_confirmed_completes(self, offline_cwd: Path) -> None:
        thread_id, checkpoint_id = self._run_and_get_checkpoint(offline_cwd)
        result = _RUNNER.invoke(
            app,
            [
                "workflow",
                "replay",
                "--thread-id",
                thread_id,
                "--checkpoint-id",
                checkpoint_id,
                "--confirm-remote-calls",
                "--materials-repository",
                "mock",
                "--materials-fixture",
                str(_materials_fixture(offline_cwd)),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Status: completed" in result.output
