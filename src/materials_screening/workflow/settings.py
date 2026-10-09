"""Stage 3 workflow configuration (S3-M1)."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

WORKFLOW_VERSION_DEFAULT = "workflow-v1"


def _validate_data_location(value: Path) -> Path:
    """Require checkpoint/run paths to stay inside the project data directory."""
    resolved = value.expanduser().resolve()
    data_root = Path("data").resolve()
    if resolved != data_root and data_root not in resolved.parents:
        raise ValueError(f"path {value!s} must be inside the project data directory")
    return value


class WorkflowSettings(BaseSettings):
    """Runtime configuration for the Stage 3 LangGraph workflow layer.

    This stage only supports the SQLite checkpoint backend and ``updates``
    streaming. Constructing this settings object does not require LangGraph,
    so direct (stage 1/2) mode remains usable without the ``workflow`` extra.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    workflow_enabled: bool = True
    workflow_version: str = Field(default=WORKFLOW_VERSION_DEFAULT, min_length=1)
    workflow_checkpointer_backend: Literal["sqlite"] = "sqlite"
    workflow_checkpoint_db: Path = Path("data/workflow_checkpoints.sqlite")
    workflow_run_root: Path = Path("data/workflow_runs")
    workflow_recursion_limit: int = Field(default=32, ge=8, le=56)
    workflow_history_limit: int = Field(default=50, ge=1, le=100)
    workflow_stream_mode: Literal["updates"] = "updates"
    workflow_store_full_state_logs: bool = False
    workflow_allow_replay: bool = True

    @field_validator("workflow_checkpoint_db", "workflow_run_root")
    @classmethod
    def _validate_checkpoint_and_run_paths(cls, value: Path) -> Path:
        return _validate_data_location(value)
