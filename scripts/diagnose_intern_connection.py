"""Replay only a failed model request with sanitized transport tracing.

No database tools are executed, no production configuration is changed, and no
model content, reasoning, authorization headers or raw request body is stored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import getproxies

import httpx
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import OpenAI
from pydantic import SecretStr

from materials_screening.agent.intern_model import InternAgentModel
from materials_screening.agent.model_base import MaterialAgentRequest
from materials_screening.agent.models import AgentFinalDraft
from materials_screening.agent.nodes import _transcript_items
from materials_screening.agent.settings import AgentSettings
from materials_screening.sub_agents.materials_database.prompt import SYSTEM_PROMPT
from materials_screening.sub_agents.materials_database.tools import build_tool_registry

ROOT = Path(__file__).resolve().parents[1]


def comparison_case(index: int, comparison: str, thinking: bool) -> tuple[bool, str]:
    if comparison == "network":
        return thinking, "proxy" if index % 2 == 0 else "direct"
    if comparison == "thinking":
        return thinking if index % 2 == 0 else not thinking, "default"
    raise ValueError("Unknown comparison")


def sanitized_exception_chain(exc: BaseException, secret: str) -> list[dict]:
    chain, seen = [], set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).replace(secret, "[REDACTED]") if secret else str(current)
        message = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", message, flags=re.I)
        message = re.sub(r"https?://\S+", "[URL]", message)
        chain.append(
            {
                "type": type(current).__name__,
                "message": message[:400],
                "errno": getattr(current, "errno", None),
            }
        )
        current = current.__cause__ or current.__context__
    return chain


class TracedClient(httpx.Client):
    def __init__(
        self, *, network_path: str = "default", proxy_url: str | None = None
    ) -> None:
        if network_path == "default":
            super().__init__(timeout=120.0)
        elif network_path in {"proxy", "direct"}:
            if network_path == "proxy" and not proxy_url:
                raise ValueError(
                    "Explicit proxy path requires an observed system proxy"
                )
            super().__init__(
                timeout=120.0,
                trust_env=False,
                proxy=proxy_url if network_path == "proxy" else None,
            )
        else:
            raise ValueError("Unknown network path")
        self.trace_events: list[dict] = []
        self.started = time.perf_counter()

    def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
        request.extensions["trace"] = self.trace
        return super().send(request, **kwargs)

    def trace(self, name: str, info: dict) -> None:
        # Never serialize trace info: it can contain headers/socket objects.
        error = info.get("exception")
        self.trace_events.append(
            {
                "event": name,
                "elapsed_s": round(time.perf_counter() - self.started, 3),
                "exception_type": type(error).__name__ if error else None,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conversation-id",
        required=True,
        help="Failed database sub-agent conversation, not Master ID",
    )
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--comparison", choices=("thinking", "network"), default="thinking"
    )
    parser.add_argument(
        "--allow-live-replay",
        action="store_true",
        help="User approval required; default is local preflight only",
    )
    args = parser.parse_args()
    if not 1 <= args.attempts <= 6:
        raise ValueError("Use between one and six bounded diagnostic requests")
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to((ROOT / "outputs/connection_diagnostics").resolve()):
        raise ValueError("Output must be inside outputs/connection_diagnostics")
    if output.exists():
        raise ValueError("Use a new output directory; do not overwrite evidence")
    output.mkdir(parents=True)
    load_dotenv(ROOT / ".env")
    secret = os.getenv("INTERN_API_KEY", "").strip()
    if not secret:
        raise ValueError("INTERN_API_KEY not configured")
    checkpoint = ROOT / "data/material_database_checkpoints.sqlite"
    with sqlite3.connect(
        checkpoint.as_uri() + "?mode=ro", uri=True, check_same_thread=False
    ) as connection:
        item = SqliteSaver(connection).get_tuple(
            {
                "configurable": {
                    "thread_id": "agent_" + args.conversation_id,
                }
            }
        )
    if item is None:
        raise ValueError("Database checkpoint not found")
    state = item.checkpoint["channel_values"]
    if state.get("model_call_count") != 0 or state.get("tool_call_count") != 0:
        raise ValueError("This probe only supports a failed first database model call")
    if "connection failed" not in (state.get("error") or {}).get("message", ""):
        raise ValueError("Expected a connection-failed checkpoint")
    # The registry is used ONLY for tool schemas; None guarantees no live service.
    definitions = build_tool_registry(None).definitions()
    request = MaterialAgentRequest(
        instructions=SYSTEM_PROMPT,
        input_items=_transcript_items(state.get("input_items", [])),
        tool_definitions=definitions,
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        max_output_tokens=8192,
        temperature=0.0,
        allow_tool_calls=True,
    )
    configured_thinking = os.getenv("INTERN_THINKING_MODE", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    settings = AgentSettings(
        # Each diagnostic probe must remain exactly one HTTP attempt, even after
        # production connection retries are enabled.
        agent_model_max_attempts=1,
        agent_base_url=os.getenv(
            "INTERN_BASE_URL", "https://chat.intern-ai.org.cn/api/v1/"
        ),
        agent_model=os.getenv("INTERN_MODEL", "intern-s2-preview-35b"),
        agent_thinking_mode=configured_thinking,
    )
    proxy_url = getproxies().get("https") or getproxies().get("all")
    if args.comparison == "network":
        if not proxy_url:
            raise ValueError(
                "No observed HTTPS proxy; cannot run a proxy/direct comparison"
            )
        # Keep TLS trust identical; do not alter custom certificate configuration.
        if os.getenv("SSL_CERT_FILE") or os.getenv("SSL_CERT_DIR"):
            raise ValueError(
                "Custom TLS trust present: adapt the test before comparing paths"
            )
    parsed_proxy = urlsplit(proxy_url) if proxy_url else None
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "messages": InternAgentModel._messages(request),
                "tools": [d.model_dump() for d in definitions],
                "max_tokens": request.max_output_tokens,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_conversation_id": args.conversation_id,
        "source_checkpoint_id": item.checkpoint["id"],
        "request_sha256": fingerprint,
        "tools_count": len(definitions),
        "model": settings.agent_model,
        "timeout_seconds": 120,
        "max_retries": 0,
        "model_max_attempts": settings.agent_model_max_attempts,
        "max_tokens": 8192,
        "stream": False,
        "note": "Same first model request/tool schemas, new client each attempt."
        " Network comparison fixes thinking and alternates paths;"
        " otherwise alternates thinking."
        " No tools executed; TLS verification remains enabled."
        " Small sample, not an acceptance run or a reliable failure-rate estimate.",
        "comparison": args.comparison,
        "configured_thinking_mode": configured_thinking,
        "observed_proxy": {
            "scheme": parsed_proxy.scheme,
            "host": parsed_proxy.hostname,
            "port": parsed_proxy.port,
        }
        if parsed_proxy
        else None,
        "live_replay_enabled": args.allow_live_replay,
        "input_items_count": len(request.input_items),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=True), flush=True)
    if not args.allow_live_replay:
        print("Local preflight only: no network requests sent.", flush=True)
        return
    observations = []
    for index in range(args.attempts):
        thinking, network_path = comparison_case(
            index, args.comparison, configured_thinking
        )
        transport = TracedClient(network_path=network_path, proxy_url=proxy_url)
        client = OpenAI(
            api_key=secret,
            base_url=settings.agent_base_url,
            timeout=120.0,
            max_retries=0,
            http_client=transport,
        )
        model = InternAgentModel(
            settings.model_copy(update={"agent_thinking_mode": thinking}),
            client=client,
            api_key=SecretStr(secret),
        )
        started = time.perf_counter()
        try:
            # Call only the HTTP path. Never execute tools or parse a final answer.
            result = model._call_api(request)
            choice = result.choices[0] if result.choices else None
            observation = {
                "success": True,
                "request_id": getattr(result, "id", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "returned_tool_names": [
                    c.function.name
                    for c in (getattr(choice.message, "tool_calls", None) or [])
                ]
                if choice
                else [],
                "content_chars": len(choice.message.content or "") if choice else 0,
            }
        except Exception as exc:
            observation = {
                "success": False,
                "exception_chain": sanitized_exception_chain(exc, secret),
            }
        finally:
            client.close()
        observation.update(
            attempt=index + 1,
            thinking_mode=thinking,
            network_path=network_path,
            elapsed_s=round(time.perf_counter() - started, 3),
            transport_trace=transport.trace_events,
        )
        observations.append(observation)
        (output / "observations.json").write_text(
            json.dumps(observations, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(observation, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
