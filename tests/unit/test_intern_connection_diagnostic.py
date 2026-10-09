"""Offline checks for the diagnostic script; not real connection acceptance."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


@pytest.fixture
def diagnostic():
    path = Path(__file__).resolve().parents[2] / "scripts/diagnose_intern_connection.py"
    spec = importlib.util.spec_from_file_location("connection_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("configured_thinking", [True, False])
def test_network_comparison_changes_only_path(diagnostic, configured_thinking):
    cases = [
        diagnostic.comparison_case(i, "network", configured_thinking) for i in range(4)
    ]
    assert cases == [
        (configured_thinking, "proxy"),
        (configured_thinking, "direct"),
        (configured_thinking, "proxy"),
        (configured_thinking, "direct"),
    ]


def test_existing_thinking_comparison_keeps_default_network(diagnostic):
    assert [diagnostic.comparison_case(i, "thinking", True) for i in range(4)] == [
        (True, "default"),
        (False, "default"),
        (True, "default"),
        (False, "default"),
    ]


def test_network_clients_disable_implicit_proxy_lookup(diagnostic, monkeypatch):
    settings = []
    monkeypatch.setattr(
        httpx.Client, "__init__", lambda self, **kwargs: settings.append(kwargs)
    )
    diagnostic.TracedClient(network_path="proxy", proxy_url="http://127.0.0.1:12450")
    diagnostic.TracedClient(network_path="direct")
    assert settings == [
        {"timeout": 120.0, "trust_env": False, "proxy": "http://127.0.0.1:12450"},
        {"timeout": 120.0, "trust_env": False, "proxy": None},
    ]


def test_exception_chain_sanitizes_key_bearer_url_and_preserves_type(diagnostic):
    inner = httpx.RemoteProtocolError(
        "Server disconnected; diagnostic-secret Bearer other-token "
        "https://example.com/private?token=url-token"
    )
    outer = RuntimeError("connection failed")
    outer.__cause__ = inner
    chain = diagnostic.sanitized_exception_chain(outer, "diagnostic-secret")
    text = json.dumps(chain)
    assert [r["type"] for r in chain] == ["RuntimeError", "RemoteProtocolError"]
    assert "Server disconnected" in text
    for secret in ("diagnostic-secret", "other-token", "url-token", "example.com"):
        assert secret not in text


def test_transport_trace_does_not_store_headers_or_error_body(diagnostic):
    with diagnostic.TracedClient() as client:
        client.trace(
            "http11.receive_response_headers.failed",
            {
                "headers": {"Authorization": "Bearer forbidden-token"},
                "exception": httpx.RemoteProtocolError("private-response-body"),
            },
        )
        serialized = json.dumps(client.trace_events)
        assert "RemoteProtocolError" in serialized
        assert "forbidden-token" not in serialized
        assert "private-response-body" not in serialized


def test_default_preflight_cannot_send_network_request(
    diagnostic, tmp_path, monkeypatch
):
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "load_dotenv", lambda *args: None)
    monkeypatch.setenv("INTERN_API_KEY", "offline-test-key")
    (tmp_path / "data").mkdir()
    connection = diagnostic.sqlite3.connect(
        tmp_path / "data/material_database_checkpoints.sqlite"
    )
    connection.close()
    checkpoint = {
        "id": "test-checkpoint",
        "channel_values": {
            "model_call_count": 0,
            "tool_call_count": 0,
            "error": {"message": "Intern connection failed"},
            "input_items": [
                {"type": "message", "role": "user", "content": "Test question"}
            ],
        },
    }
    monkeypatch.setattr(
        diagnostic,
        "SqliteSaver",
        lambda *args: SimpleNamespace(
            get_tuple=lambda config: SimpleNamespace(checkpoint=checkpoint)
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must never construct a network client")

    monkeypatch.setattr(diagnostic, "OpenAI", forbidden)
    monkeypatch.setattr(diagnostic, "TracedClient", forbidden)
    monkeypatch.setattr(
        diagnostic.os.sys,
        "argv",
        [
            "diagnostic",
            "--conversation-id",
            "fixture",
            "--output-dir",
            "outputs/connection_diagnostics/offline",
        ],
    )
    diagnostic.main()
    output = tmp_path / "outputs/connection_diagnostics/offline"
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["live_replay_enabled"] is False
    assert metadata["model_max_attempts"] == 1
    assert not (output / "observations.json").exists()
    assert "offline-test-key" not in json.dumps(metadata)
