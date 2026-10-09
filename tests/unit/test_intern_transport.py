"""Connection-layer regressions; never execute Agent tools or call live APIs."""

import importlib
import ssl
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, SecretStr, ValidationError

from materials_screening.agent.errors import AgentModelError
from materials_screening.agent.intern_model import InternAgentModel
from materials_screening.agent.settings import AgentSettings
from materials_screening.llm.intern_provider import InternProvider
from materials_screening.planner.settings import Settings


class Probe(BaseModel):
    answer: str


def response():
    return SimpleNamespace(
        id="test",
        model="intern-test",
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content='{"answer":"ok"}')
            )
        ],
    )


def connection_error(timeout=False):
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    return (
        APITimeoutError(request=request)
        if timeout
        else APIConnectionError(request=request)
    )


def call_entry(kind, create, *, attempts=2):
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    if kind == "agent":
        model = InternAgentModel(
            AgentSettings(_env_file=None, agent_model_max_attempts=attempts),
            client=client,
            api_key=SecretStr("test-only"),
        )
        return model._create_completion(
            {"model": "test", "messages": [], "stream": False}
        )
    provider = InternProvider(
        Settings(_env_file=None, intern_api_key="test-only", llm_max_attempts=attempts),
        client=client,
    )
    return provider.generate_structured(
        system_prompt="probe",
        user_text="probe",
        output_model=Probe,
        schema_name="probe",
        max_output_tokens=512,
    )


@pytest.mark.parametrize("kind", ["agent", "structured"])
def test_default_clients_do_not_inherit_system_proxy(kind, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    monkeypatch.delenv("INTERN_USE_SYSTEM_PROXY", raising=False)
    if kind == "agent":
        model = InternAgentModel(
            AgentSettings(_env_file=None), api_key=SecretStr("test-only")
        )
    else:
        model = InternProvider(Settings(_env_file=None, intern_api_key="test-only"))
    try:
        assert model._client._client.trust_env is False
        assert model._client.max_retries == 0
        context = model._client._client._transport._pool._ssl_context
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
    finally:
        model._client.close()


@pytest.mark.parametrize("settings_type", [AgentSettings, Settings])
def test_proxy_opt_in_and_retry_limits(settings_type, monkeypatch):
    monkeypatch.setenv("INTERN_USE_SYSTEM_PROXY", "true")
    settings = settings_type(_env_file=None)
    assert settings.intern_use_system_proxy is True
    field = (
        "agent_model_max_attempts"
        if settings_type is AgentSettings
        else "llm_max_attempts"
    )
    assert getattr(settings, field) == 2
    for value in (0, 4):
        with pytest.raises(ValidationError):
            settings_type(_env_file=None, **{field: value})


@pytest.mark.parametrize("kind", ["agent", "structured"])
@pytest.mark.parametrize("timeout", [False, True])
def test_connection_failure_retries_same_completion_only(kind, timeout, monkeypatch):
    transport = importlib.import_module("materials_screening.llm.intern_transport")
    monkeypatch.setattr(transport.time, "sleep", lambda _: None)
    create = Mock(side_effect=[connection_error(timeout), response()])
    call_entry(kind, create)
    assert create.call_count == 2
    first, second = create.call_args_list
    for field in ("model", "messages", "stream", "max_tokens", "extra_body"):
        assert first.kwargs.get(field) == second.kwargs.get(field)


@pytest.mark.parametrize("kind", ["agent", "structured"])
def test_persistent_failure_is_bounded_and_preserves_cause(kind, monkeypatch):
    transport = importlib.import_module("materials_screening.llm.intern_transport")
    monkeypatch.setattr(transport.time, "sleep", lambda _: None)
    root = ssl.SSLEOFError(8, "EOF")
    error = connection_error()
    error.__cause__ = root
    create = Mock(side_effect=error)
    with pytest.raises((AgentModelError, APIConnectionError)) as caught:
        call_entry(kind, create)
    assert create.call_count == 2
    if kind == "agent":
        assert "SSLEOFError" in str(caught.value)
        assert "attempts=2" in str(caught.value)
    else:
        assert caught.value.__cause__ is root


@pytest.mark.parametrize("kind", ["agent", "structured"])
@pytest.mark.parametrize(
    "error_type,status",
    [
        (AuthenticationError, 401),
        (BadRequestError, 400),
        (RateLimitError, 429),
        (InternalServerError, 500),
    ],
)
def test_non_connection_errors_are_not_retried(kind, error_type, status):
    res = httpx.Response(status, request=httpx.Request("POST", "https://example.test"))
    create = Mock(side_effect=error_type("test status error", response=res, body=None))
    with pytest.raises(AgentModelError if kind == "agent" else error_type):
        call_entry(kind, create)
    assert create.call_count == 1


@pytest.mark.parametrize("sdk_retries", [0, 2])
def test_sdk_retries_are_not_stacked_with_application_retries(monkeypatch, sdk_retries):
    transport = importlib.import_module("materials_screening.llm.intern_transport")
    monkeypatch.setattr(transport.time, "sleep", lambda _: None)
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            raise httpx.RemoteProtocolError("Server disconnected")
        return httpx.Response(
            200,
            json={
                "id": "probe",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    with OpenAI(
        api_key="test-only",
        base_url="https://example.test/v1",
        max_retries=sdk_retries,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as client:
        model = InternAgentModel(
            AgentSettings(_env_file=None), client=client, api_key=SecretStr("test-only")
        )
        result = model._create_completion(
            {"model": "test", "messages": [{"role": "user", "content": "probe"}]}
        )
        assert result.id == "probe"
    assert count == 2


def test_retry_logging_contains_types_not_secret_exception_text(monkeypatch, caplog):
    transport = importlib.import_module("materials_screening.llm.intern_transport")
    monkeypatch.setattr(transport.time, "sleep", lambda _: None)
    error = connection_error()
    error.__cause__ = RuntimeError("private-request-content secret-test-value")
    call_entry("agent", Mock(side_effect=[error, response()]))
    assert "secret-test-value" not in caplog.text
    assert "private-request-content" not in caplog.text
    assert "APIConnectionError" in caplog.text


def test_direct_mode_preserves_custom_certificate_trust(monkeypatch):
    transport = importlib.import_module("materials_screening.llm.intern_transport")
    context = ssl.create_default_context()
    factory = Mock(return_value=context)
    monkeypatch.setattr(httpx, "create_ssl_context", factory)
    with transport.create_intern_client(
        api_key="test-only",
        base_url="https://example.test/v1",
        timeout_seconds=120,
        use_system_proxy=False,
    ) as client:
        assert client._client.trust_env is False
        assert client._client._transport._pool._ssl_context is context
    factory.assert_called_once_with(trust_env=True)


@pytest.mark.parametrize("kind", ["agent", "structured"])
@pytest.mark.parametrize("attempts", [1, 3])
def test_configured_attempt_cap(kind, attempts, monkeypatch):
    transport = importlib.import_module("materials_screening.llm.intern_transport")
    sleep = Mock()
    monkeypatch.setattr(transport.time, "sleep", sleep)
    create = Mock(side_effect=connection_error())
    with pytest.raises((AgentModelError, APIConnectionError)):
        call_entry(kind, create, attempts=attempts)
    assert create.call_count == attempts
    assert sleep.call_count == attempts - 1


@pytest.mark.parametrize("kind", ["agent", "structured"])
def test_explicit_proxy_opt_in_is_applied(kind, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    monkeypatch.setenv("INTERN_USE_SYSTEM_PROXY", "true")
    if kind == "agent":
        model = InternAgentModel(
            AgentSettings(_env_file=None), api_key=SecretStr("test-only")
        )
    else:
        model = InternProvider(Settings(_env_file=None, intern_api_key="test-only"))
    try:
        assert model._client._client.trust_env is True
        assert model._client.max_retries == 0
    finally:
        model._client.close()


def test_invalid_structured_output_does_not_retry():
    from materials_screening.llm.errors import LLMStructuredOutputError

    bad = response()
    bad.choices[0].message.content = "not json"
    create = Mock(return_value=bad)
    with pytest.raises(LLMStructuredOutputError):
        call_entry("structured", create)
    assert create.call_count == 1


def test_timeout_is_enforced_and_request_not_mutated():
    transport = importlib.import_module("materials_screening.llm.intern_transport")
    create = Mock(return_value=response())
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    kwargs = {"model": "test", "messages": [], "timeout": 999}
    transport.create_completion(client, kwargs, max_attempts=1, timeout_seconds=12)
    assert create.call_args.kwargs["timeout"] == 12
    assert kwargs["timeout"] == 999
