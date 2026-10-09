"""统一 LLM 与外部 API 配置。

在 .env 中配置书生模型::

    # .env
    LLM_PROVIDER=intern
    INTERN_BASE_URL=https://chat.intern-ai.org.cn/api/v1/
    INTERN_MODEL=intern-s2-preview-35b
    INTERN_API_KEY=your-token

用法::

    from materials_screening.config.llm_config import llm_config
    cfg = llm_config()
    print(cfg.base_url, cfg.model, cfg.api_key)
"""

import os

from pydantic import SecretStr


class LLMConfig:
    """从环境变量读取书生模型配置。"""

    def __init__(self) -> None:
        self._provider = self._read("LLM_PROVIDER", default="intern")

    # ── 当前生效的配置 ────────────────────────────────────────────────────

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def base_url(self) -> str:
        return self._provider_var(
            "BASE_URL", default="https://chat.intern-ai.org.cn/api/v1/"
        )

    @property
    def model(self) -> str:
        return self._provider_var("MODEL", default="intern-s2-preview-35b")

    @property
    def api_key(self) -> SecretStr | None:
        raw = self._provider_var("API_KEY", default="")
        return SecretStr(raw) if raw else None

    # ── 公共参数 ──────────────────────────────────────────────────────────

    @property
    def reasoning_effort(self) -> str:
        return self._read("LLM_REASONING_EFFORT", default="none")

    @property
    def temperature(self) -> float:
        return float(self._read("LLM_TEMPERATURE", default="0.0"))

    @property
    def timeout_seconds(self) -> float:
        return float(self._read("LLM_TIMEOUT_SECONDS", default="45"))

    @property
    def max_attempts(self) -> int:
        return int(self._read("LLM_MAX_ATTEMPTS", default="2"))

    @property
    def max_output_tokens(self) -> int:
        return int(self._read("LLM_MAX_OUTPUT_TOKENS", default="4096"))

    # ── 外部 API ──────────────────────────────────────────────────────────

    @property
    def mp_api_key(self) -> SecretStr | None:
        raw = self._read("MP_API_KEY", default="")
        return SecretStr(raw) if raw else None

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _provider_var(self, suffix: str, default: str) -> str:
        """读取 <PROVIDER_UPPER>_<SUFFIX> 环境变量。"""
        key = f"{self._provider.upper()}_{suffix}"
        value = os.getenv(key, "").strip()
        return value if value else default

    @staticmethod
    def _read(key: str, default: str) -> str:
        value = os.getenv(key, "").strip()
        return value if value else default

    def __repr__(self) -> str:
        return (
            f"LLMConfig(provider={self.provider!r}, "
            f"base_url={self.base_url!r}, model={self.model!r})"
        )


# ── 单例 ──────────────────────────────────────────────────────────────────────

_config_cache: LLMConfig | None = None


def llm_config() -> LLMConfig:
    """返回 LLMConfig 单例。"""
    global _config_cache
    if _config_cache is None:
        _config_cache = LLMConfig()
    return _config_cache
