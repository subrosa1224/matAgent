"""Intern-only transport policy; never retry tools or whole Agent turns."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from openai import APIConnectionError, OpenAI

_LOG = logging.getLogger(__name__)


def create_intern_client(
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: float,
    use_system_proxy: bool = False,
) -> OpenAI:
    # Ignore proxy discovery by default, but retain explicitly configured CA trust.
    # Certificate and hostname verification remain enabled in either mode.
    transport = httpx.Client(
        trust_env=use_system_proxy,
        verify=httpx.create_ssl_context(trust_env=True),
        timeout=timeout_seconds,
    )
    try:
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=transport,
            timeout=timeout_seconds,
            max_retries=0,
        )
    except Exception:
        transport.close()
        raise


def connection_failure_detail(exc: APIConnectionError) -> str:
    """Safe persisted diagnostics: types/count only, not request or exception text."""
    causes: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(causes) < 10:
        seen.add(id(current))
        causes.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    attempts = getattr(exc, "_intern_connection_attempts", 1)
    return f"attempts={attempts}; cause={' -> '.join(causes)}"


def create_completion(
    client: Any,
    kwargs: dict[str, Any],
    *,
    max_attempts: int,
    timeout_seconds: float,
) -> Any:
    """Repeat only a completion after connection/timeout failure, at most 3 times.

    The timeout applies to each HTTP phase/attempt, not an absolute turn deadline.
    HTTP status and output-validation errors are deliberately not retried.
    """
    if not 1 <= max_attempts <= 3:
        raise ValueError("Intern max_attempts must be between 1 and 3")
    if isinstance(client, OpenAI) and client.max_retries != 0:
        client = client.with_options(max_retries=0)
    request = {**kwargs, "timeout": timeout_seconds}
    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(**request)
        except APIConnectionError as exc:
            # APITimeoutError is a subclass of APIConnectionError.
            exc._intern_connection_attempts = attempt
            _LOG.warning(
                "Intern transport failure (%s); attempt=%d/%d; retry=%s",
                connection_failure_detail(exc),
                attempt,
                max_attempts,
                attempt < max_attempts,
            )
            if attempt == max_attempts:
                raise
            time.sleep(0.5 * attempt)
    raise AssertionError("unreachable")
