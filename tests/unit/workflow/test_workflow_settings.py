"""Unit tests for Stage 3 workflow settings (S3-M1)."""

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from materials_screening.workflow.settings import WorkflowSettings

_WORKFLOW_ENV_NAMES: tuple[str, ...] = (
    "WORKFLOW_ENABLED",
    "WORKFLOW_VERSION",
    "WORKFLOW_CHECKPOINTER_BACKEND",
    "WORKFLOW_CHECKPOINT_DB",
    "WORKFLOW_RUN_ROOT",
    "WORKFLOW_RECURSION_LIMIT",
    "WORKFLOW_HISTORY_LIMIT",
    "WORKFLOW_STREAM_MODE",
    "WORKFLOW_STORE_FULL_STATE_LOGS",
    "WORKFLOW_ALLOW_REPLAY",
)


def _clear_workflow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _WORKFLOW_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


class TestWorkflowSettingsDefaults:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_workflow_env(monkeypatch)
        settings = WorkflowSettings(_env_file=None)
        assert settings.workflow_enabled is True
        assert settings.workflow_version == "workflow-v1"
        assert settings.workflow_checkpointer_backend == "sqlite"
        assert settings.workflow_checkpoint_db == Path(
            "data/workflow_checkpoints.sqlite"
        )
        assert settings.workflow_run_root == Path("data/workflow_runs")
        assert settings.workflow_recursion_limit == 32
        assert settings.workflow_history_limit == 50
        assert settings.workflow_stream_mode == "updates"
        assert settings.workflow_store_full_state_logs is False
        assert settings.workflow_allow_replay is True


class TestWorkflowSettingsEnvOverride:
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_workflow_env(monkeypatch)
        monkeypatch.setenv("WORKFLOW_ENABLED", "false")
        monkeypatch.setenv("WORKFLOW_VERSION", "workflow-v2")
        monkeypatch.setenv("WORKFLOW_CHECKPOINTER_BACKEND", "sqlite")
        monkeypatch.setenv("WORKFLOW_CHECKPOINT_DB", "data/custom_checkpoints.sqlite")
        monkeypatch.setenv("WORKFLOW_RUN_ROOT", "data/custom_runs")
        monkeypatch.setenv("WORKFLOW_RECURSION_LIMIT", "40")
        monkeypatch.setenv("WORKFLOW_HISTORY_LIMIT", "80")
        monkeypatch.setenv("WORKFLOW_STREAM_MODE", "updates")
        monkeypatch.setenv("WORKFLOW_STORE_FULL_STATE_LOGS", "true")
        monkeypatch.setenv("WORKFLOW_ALLOW_REPLAY", "false")
        settings = WorkflowSettings()
        assert settings.workflow_enabled is False
        assert settings.workflow_version == "workflow-v2"
        assert settings.workflow_checkpointer_backend == "sqlite"
        assert settings.workflow_checkpoint_db == Path("data/custom_checkpoints.sqlite")
        assert settings.workflow_run_root == Path("data/custom_runs")
        assert settings.workflow_recursion_limit == 40
        assert settings.workflow_history_limit == 80
        assert settings.workflow_stream_mode == "updates"
        assert settings.workflow_store_full_state_logs is True
        assert settings.workflow_allow_replay is False


class TestWorkflowSettingsValidators:
    @pytest.mark.parametrize("limit", ["7", "57"])
    def test_recursion_limit_out_of_bounds_rejected(
        self, monkeypatch: pytest.MonkeyPatch, limit: str
    ) -> None:
        _clear_workflow_env(monkeypatch)
        monkeypatch.setenv("WORKFLOW_RECURSION_LIMIT", limit)
        with pytest.raises(ValidationError):
            WorkflowSettings()

    @pytest.mark.parametrize("limit", ["8", "32", "56"])
    def test_recursion_limit_bounds_accepted(
        self, monkeypatch: pytest.MonkeyPatch, limit: str
    ) -> None:
        _clear_workflow_env(monkeypatch)
        monkeypatch.setenv("WORKFLOW_RECURSION_LIMIT", limit)
        assert WorkflowSettings().workflow_recursion_limit == int(limit)

    @pytest.mark.parametrize("limit", ["0", "101"])
    def test_history_limit_out_of_bounds_rejected(
        self, monkeypatch: pytest.MonkeyPatch, limit: str
    ) -> None:
        _clear_workflow_env(monkeypatch)
        monkeypatch.setenv("WORKFLOW_HISTORY_LIMIT", limit)
        with pytest.raises(ValidationError):
            WorkflowSettings()

    @pytest.mark.parametrize("limit", ["1", "50", "100"])
    def test_history_limit_bounds_accepted(
        self, monkeypatch: pytest.MonkeyPatch, limit: str
    ) -> None:
        _clear_workflow_env(monkeypatch)
        monkeypatch.setenv("WORKFLOW_HISTORY_LIMIT", limit)
        assert WorkflowSettings().workflow_history_limit == int(limit)

    @pytest.mark.parametrize("mode", ["values", "messages", "debug"])
    def test_stream_mode_only_allows_updates(
        self, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        _clear_workflow_env(monkeypatch)
        monkeypatch.setenv("WORKFLOW_STREAM_MODE", mode)
        with pytest.raises(ValidationError):
            WorkflowSettings()

    @pytest.mark.parametrize(
        "backend",
        ["memory", "postgres", "redis", "inmemory"],
    )
    def test_checkpointer_backend_only_allows_sqlite(
        self, monkeypatch: pytest.MonkeyPatch, backend: str
    ) -> None:
        _clear_workflow_env(monkeypatch)
        monkeypatch.setenv("WORKFLOW_CHECKPOINTER_BACKEND", backend)
        with pytest.raises(ValidationError):
            WorkflowSettings()

    def test_checkpoint_db_outside_data_dir_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _clear_workflow_env(monkeypatch)
        monkeypatch.setenv(
            "WORKFLOW_CHECKPOINT_DB", str(tmp_path / "checkpoints.sqlite")
        )
        with pytest.raises(ValidationError):
            WorkflowSettings()

    def test_run_root_outside_data_dir_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _clear_workflow_env(monkeypatch)
        monkeypatch.setenv("WORKFLOW_RUN_ROOT", str(tmp_path / "runs"))
        with pytest.raises(ValidationError):
            WorkflowSettings()

    def test_empty_workflow_version_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_workflow_env(monkeypatch)
        monkeypatch.setenv("WORKFLOW_VERSION", "")
        with pytest.raises(ValidationError):
            WorkflowSettings()


class TestDirectModeIndependence:
    """Direct (stage 1/2) mode must not require the ``workflow`` extra."""

    @staticmethod
    def _assert_import_without_langgraph(module: str) -> None:
        code = f"import sys\nimport {module}\nassert 'langgraph' not in sys.modules"
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    def test_cli_import_does_not_load_langgraph(self) -> None:
        self._assert_import_without_langgraph("materials_screening.cli")

    def test_workflow_settings_do_not_load_langgraph(self) -> None:
        self._assert_import_without_langgraph("materials_screening.workflow.settings")
