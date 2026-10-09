"""Master agent settings (extends AgentSettings pattern)."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MasterAgentSettings(BaseSettings):
    """Configuration for the master orchestration agent.

    Same pattern as ``AgentSettings`` (settings.py:13-77).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    master_agent_enabled: bool = True
    master_agent_version: str = Field(default="master-agent-v1", min_length=1)
    master_agent_prompt_version: str = Field(default="master-agent-v1", min_length=1)

    master_checkpoint_db: Path = Path("data/master_agent_checkpoints.sqlite")
    master_conversation_db: Path = Path("data/master_agent_conversations.sqlite")

    master_max_model_calls_per_turn: int = Field(default=6, ge=1, le=15)
    master_max_sub_agent_calls_per_turn: int = Field(default=3, ge=1, le=10)
    master_max_conversation_turns: int = Field(default=20, ge=1, le=100)

    master_max_argument_bytes: int = Field(default=8192, ge=1)
    master_max_sub_agent_output_bytes: int = Field(default=65536, ge=1)
    master_max_input_bytes: int = Field(default=122880, ge=1)

    master_reasoning_effort: Literal["none", "low", "high", "max"] = "none"
    master_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    master_base_url: str = "https://chat.intern-ai.org.cn/api/v1/"
    master_model: str = "intern-s2-preview-35b"

    @field_validator("master_base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized:
            raise ValueError("master base URL must not be empty")
        if "://" not in normalized:
            raise ValueError(
                "master base URL must include a scheme (https:// or http://)"
            )
        return normalized

    @field_validator("master_model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("master model must not be empty")
        return value.strip()

    @field_validator("master_checkpoint_db")
    @classmethod
    def _validate_checkpoint_db(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        data_root = Path("data").resolve()
        if resolved != data_root and data_root not in resolved.parents:
            raise ValueError(
                "master checkpoint db must be inside the project data directory"
            )
        return value
