"""Stage 3.5 tool execution ledger (S3.5-M2)."""

from dataclasses import dataclass

from materials_screening.agent.errors import AgentInvariantError
from materials_screening.agent.tool_base import ToolSideEffect

_RUN_TOOL_NAME = "run_screening_workflow"


@dataclass(frozen=True)
class LedgerEntry:
    """One recorded tool execution for idempotency and evidence."""

    call_id: str
    tool_name: str
    evidence_id: str
    side_effect: ToolSideEffect
    result_json: str
    query_hash: str | None = None


class ToolExecutionLedger:
    """Deterministic, idempotent record of tool calls for one user turn."""

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []
        self._by_call_id: dict[str, LedgerEntry] = {}
        self._by_query_hash: dict[str, LedgerEntry] = {}

    def record(
        self,
        *,
        call_id: str,
        tool_name: str,
        evidence_id: str,
        result_json: str,
        side_effect: ToolSideEffect,
        query_hash: str | None = None,
    ) -> None:
        """Record one execution; duplicates are rejected, not overwritten."""
        if call_id in self._by_call_id:
            raise AgentInvariantError(f"duplicate call_id: {call_id!r}")
        if query_hash is not None and query_hash in self._by_query_hash:
            raise AgentInvariantError(f"duplicate query hash: {query_hash!r}")
        entry = LedgerEntry(
            call_id=call_id,
            tool_name=tool_name,
            evidence_id=evidence_id,
            side_effect=side_effect,
            result_json=result_json,
            query_hash=query_hash,
        )
        self._entries.append(entry)
        self._by_call_id[call_id] = entry
        if query_hash is not None:
            self._by_query_hash[query_hash] = entry

    def has(self, call_id: str) -> bool:
        return call_id in self._by_call_id

    def get_result(self, call_id: str) -> str | None:
        """Return the previously recorded result for a duplicate call id."""
        entry = self._by_call_id.get(call_id)
        return entry.result_json if entry is not None else None

    def get_entry(self, call_id: str) -> LedgerEntry | None:
        """Return the recorded entry for a call id, or None."""
        return self._by_call_id.get(call_id)

    def has_query_hash(self, query_hash: str) -> bool:
        return query_hash in self._by_query_hash

    def get_by_query_hash(self, query_hash: str) -> str | None:
        """Return the first result recorded for a query hash in this turn."""
        entry = self._by_query_hash.get(query_hash)
        return entry.result_json if entry is not None else None

    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(entry.evidence_id for entry in self._entries)

    def entries(self) -> tuple[LedgerEntry, ...]:
        """All recorded entries, in execution order."""
        return tuple(self._entries)

    def executed_call_ids(self) -> tuple[str, ...]:
        return tuple(entry.call_id for entry in self._entries)

    def executed_tool_names(self) -> tuple[str, ...]:
        return tuple(entry.tool_name for entry in self._entries)

    def tool_call_count(self) -> int:
        return len(self._entries)

    def workflow_run_count(self) -> int:
        return sum(1 for entry in self._entries if entry.tool_name == _RUN_TOOL_NAME)

    def side_effect_executed(self) -> bool:
        return any(
            entry.side_effect is ToolSideEffect.CREATE_WORKFLOW_RUN
            for entry in self._entries
        )
