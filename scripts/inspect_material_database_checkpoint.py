"""Print the latest materials-database checkpoint transcript for diagnostics."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    conversations = Path("data/material_database_conversations.sqlite")
    checkpoints = Path("data/material_database_checkpoints.sqlite")
    with sqlite3.connect(conversations) as connection:
        row = connection.execute(
            "SELECT conversation_id FROM conversations "
            "ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise SystemExit("no materials database conversation found")
    conversation_id = str(row[0])
    thread_id = f"agent_{conversation_id}"
    with sqlite3.connect(checkpoints, check_same_thread=False) as connection:
        saver = SqliteSaver(connection)
        saved = saver.get_tuple({"configurable": {"thread_id": thread_id}})
    if saved is None:
        raise SystemExit(f"no checkpoint found for {thread_id}")
    values = saved.checkpoint.get("channel_values", {})
    safe = {
        "thread_id": thread_id,
        "status": values.get("status"),
        "model_call_count": values.get("model_call_count"),
        "tool_call_count": values.get("tool_call_count"),
        "selected_tools": values.get("selected_tools"),
        "error": values.get("error"),
        "input_items": values.get("input_items"),
        "tool_results": values.get("tool_results"),
        "final_draft": values.get("final_draft"),
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
