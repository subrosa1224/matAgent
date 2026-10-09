"""Prompt assembly for the planner (D2-M3)."""

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class PromptSpec(BaseModel):
    """Versioned system prompt with a content hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    system_prompt: str
    sha256: str


def build_user_message(query: str) -> str:
    """Wrap the user query as pure data in a dedicated user message."""
    return f"请解析以下材料筛选请求：\n<user_query>\n{query}\n</user_query>"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_examples(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"planner examples file must contain a JSON list: {path}")
    return data


def _assemble_system_prompt(system_prompt: str, examples: list[dict[str, Any]]) -> str:
    if not examples:
        return system_prompt
    examples_json = json.dumps(examples, ensure_ascii=False, indent=2)
    return f"{system_prompt}\n\n## 示例\n{examples_json}"


class PromptBuilder:
    """Load versioned prompt files and assemble a hashable PromptSpec."""

    def __init__(
        self,
        system_prompt_path: Path | None = None,
        examples_path: Path | None = None,
        version: str = "planner-v2",
    ) -> None:
        self._system_prompt = _read_text(
            system_prompt_path or _PROMPTS_DIR / "planner_system_v2.txt"
        )
        self._examples = _load_examples(
            examples_path or _PROMPTS_DIR / "planner_examples_v2.json"
        )
        self._version = version

    def build(self) -> PromptSpec:
        """Return the assembled system prompt, version and SHA-256."""
        system_prompt = _assemble_system_prompt(self._system_prompt, self._examples)
        digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        return PromptSpec(
            version=self._version,
            system_prompt=system_prompt,
            sha256=digest,
        )
