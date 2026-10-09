"""Stage 2 planner settings (D2-M1)."""

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Intern planner layer."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Literal["mock", "intern"] = "mock"

    intern_api_key: SecretStr | None = None
    intern_base_url: str = "https://chat.intern-ai.org.cn/api/v1/"
    intern_model: str = "intern-s2-preview-35b"
    literature_extraction_model: str | None = None
    intern_thinking_mode: bool = True
    intern_use_system_proxy: bool = False

    llm_timeout_seconds: float = Field(default=45, gt=0, le=300)
    llm_max_attempts: int = Field(default=2, ge=1, le=3)
    llm_max_output_tokens: int = Field(default=4096, ge=512, le=32768)

    planner_prompt_version: str = "planner-v2"
    planner_schema_version: str = "planner-draft-v1"
    planner_max_query_chars: int = Field(default=4000, ge=1, le=20000)
    planner_store_raw_io: bool = False

    @field_validator("intern_base_url")
    @classmethod
    def validate_intern_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/") + "/"
        if "://" not in normalized:
            raise ValueError("Intern base URL must include a scheme")
        return normalized

    @field_validator("intern_model")
    @classmethod
    def validate_intern_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Intern model name must not be empty")
        return value.strip()
