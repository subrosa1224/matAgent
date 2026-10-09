"""Stage 3.5 agent settings (S3.5-M1)."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Runtime configuration for the single MaterialAgent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    agent_enabled: bool = True
    agent_version: str = Field(default="material-agent-v1", min_length=1)
    agent_prompt_version: str = Field(default="material-agent-v1", min_length=1)
    # Empty means the built-in materials-screening prompt in nodes.py.
    # Sub-agent factories set this explicitly to avoid prompt/tool mismatch.
    agent_system_prompt: str = ""

    agent_checkpointer_backend: Literal["sqlite"] = "sqlite"
    agent_checkpoint_db: Path = Path("data/agent_checkpoints.sqlite")

    agent_max_model_calls_per_turn: int = Field(default=4, ge=1, le=10)
    agent_max_tool_calls_per_turn: int = Field(default=4, ge=1, le=10)
    agent_allow_multi_step_tools: bool = False
    agent_max_workflow_runs_per_turn: int = Field(default=1, ge=1, le=1)
    agent_max_conversation_turns: int = Field(default=20, ge=1, le=100)

    agent_max_argument_bytes: int = Field(default=8192, ge=1)
    agent_max_tool_output_bytes: int = Field(default=32768, ge=1)
    agent_max_input_bytes: int = Field(default=122880, ge=1)

    agent_reasoning_effort: Literal["none", "low", "high", "max"] = "none"
    agent_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    agent_store_raw_model_output: bool = False
    agent_allow_web_search: bool = False

    agent_base_url: str = "https://chat.intern-ai.org.cn/api/v1/"
    agent_model: str = "intern-s2-preview-35b"
    agent_thinking_mode: bool = True
    intern_use_system_proxy: bool = False
    agent_model_timeout_seconds: float = Field(default=120, gt=0, le=300)
    agent_model_max_attempts: int = Field(default=2, ge=1, le=3)

    @field_validator("agent_allow_web_search")
    @classmethod
    def _web_search_must_stay_disabled(cls, value: bool) -> bool:
        if value:
            raise ValueError("web search must stay disabled in this stage")
        return value

    @field_validator("agent_base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized:
            raise ValueError("agent base URL must not be empty")
        if "://" not in normalized:
            raise ValueError(
                "agent base URL must include a scheme (https:// or http://)"
            )
        return normalized

    @field_validator("agent_system_prompt")
    @classmethod
    def _validate_system_prompt(cls, value: str) -> str:
        return value.strip()

    @field_validator("agent_model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("agent model must not be empty")
        return value.strip()

    @field_validator("agent_checkpoint_db")
    @classmethod
    def _validate_checkpoint_db(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        data_root = Path("data").resolve()
        if resolved != data_root and data_root not in resolved.parents:
            raise ValueError(
                "agent checkpoint db must be inside the project data directory"
            )
        return value


def shared_intern_settings(
    settings: AgentSettings, *, thinking_mode: bool | None = None
) -> AgentSettings:
    """Map validated shared INTERN_* settings; retain agent-specific limits."""
    from materials_screening.planner.settings import Settings

    shared = Settings()
    return AgentSettings.model_validate(
        {
            **settings.model_dump(),
            "agent_base_url": shared.intern_base_url,
            "agent_model": shared.intern_model,
            "agent_thinking_mode": shared.intern_thinking_mode
            if thinking_mode is None
            else thinking_mode,
        }
    )
