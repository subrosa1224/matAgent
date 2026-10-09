"""Agent transcript: validated multi-turn input history (S3.5)."""

import json
from collections.abc import Iterable
from typing import Any

from materials_screening.agent.errors import AgentInvariantError
from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
)

TranscriptItem = AgentMessageItem | AgentFunctionCallItem | AgentFunctionOutputItem
_TRANSCRIPT_ITEM_TYPES = (
    AgentMessageItem,
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
)


def _item_char_count(item: TranscriptItem) -> int:
    if isinstance(item, AgentMessageItem):
        return len(item.content)
    if isinstance(item, AgentFunctionCallItem):
        return len(item.call_id) + len(item.name) + len(item.arguments)
    if isinstance(item, AgentFunctionOutputItem):
        return len(item.call_id) + len(item.output)
    return 0


class AgentTranscript:
    """Ordered, JSON-safe multi-turn input history for the agent model.

    Only message, function_call and function_call_output items are allowed
    (never reasoning or web_search_call). Call ids must be unique, every
    function call must have exactly one matching output, and orphan outputs
    are rejected. Total size is bounded by optional char/byte limits.
    """

    def __init__(
        self,
        *,
        max_chars: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        self._items: list[TranscriptItem] = []
        self._max_chars = max_chars
        self._max_bytes = max_bytes

    @staticmethod
    def user(content: str) -> AgentMessageItem:
        """Build a user message item."""
        return AgentMessageItem(role="user", content=content)

    @staticmethod
    def assistant(content: str) -> AgentMessageItem:
        """Build an assistant message item."""
        return AgentMessageItem(role="assistant", content=content)

    @staticmethod
    def function_call(
        *,
        call_id: str,
        name: str,
        arguments: str,
    ) -> AgentFunctionCallItem:
        """Build an assistant function call item."""
        return AgentFunctionCallItem(
            call_id=call_id,
            name=name,
            arguments=arguments,
        )

    @staticmethod
    def function_output(
        *,
        call_id: str,
        output: str,
    ) -> AgentFunctionOutputItem:
        """Build a tool output item matching a preceding call id."""
        return AgentFunctionOutputItem(call_id=call_id, output=output)

    def append(self, item: object) -> None:
        """Append one item; unknown or non-item values are rejected."""
        if not isinstance(item, _TRANSCRIPT_ITEM_TYPES):
            raise AgentInvariantError(f"unknown transcript item: {type(item).__name__}")
        self._items.append(item)

    def extend(self, items: Iterable[object]) -> None:
        for item in items:
            self.append(item)

    def items(self) -> tuple[TranscriptItem, ...]:
        """Return the full multi-turn item sequence, in order."""
        return tuple(self._items)

    def validate(self) -> None:
        """Enforce call/output pairing, uniqueness and size limits."""
        call_ids: set[str] = set()
        output_ids: set[str] = set()
        for item in self._items:
            if isinstance(item, AgentFunctionCallItem):
                if item.call_id in call_ids:
                    raise AgentInvariantError(
                        f"duplicate function call id: {item.call_id!r}"
                    )
                call_ids.add(item.call_id)
            elif isinstance(item, AgentFunctionOutputItem):
                if item.call_id not in call_ids:
                    raise AgentInvariantError(
                        f"orphan function output: {item.call_id!r}"
                    )
                if item.call_id in output_ids:
                    raise AgentInvariantError(
                        f"duplicate function output id: {item.call_id!r}"
                    )
                output_ids.add(item.call_id)
        missing = call_ids - output_ids
        if missing:
            raise AgentInvariantError(
                f"function calls missing output: {sorted(missing)}"
            )
        total_chars = sum(_item_char_count(item) for item in self._items)
        if self._max_chars is not None and total_chars > self._max_chars:
            raise AgentInvariantError(
                f"transcript exceeds {self._max_chars} characters"
            )
        total_bytes = len(self._serialize().encode("utf-8"))
        if self._max_bytes is not None and total_bytes > self._max_bytes:
            raise AgentInvariantError(f"transcript exceeds {self._max_bytes} bytes")

    def as_dicts(self) -> list[dict[str, Any]]:
        """JSON-native items for agent state (no validation required)."""
        return [item.model_dump(mode="json") for item in self._items]

    def to_json(self) -> str:
        """Validate then serialize as compact JSON for the next model call."""
        self.validate()
        return self._serialize()

    def _serialize(self) -> str:
        return json.dumps(
            self.as_dicts(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
