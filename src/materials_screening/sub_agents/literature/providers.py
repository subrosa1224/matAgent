"""Bounded clients and normalization for supported literature providers."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from .models import LiteratureSearchInput, PaperProvenance, PaperRecord

JsonObject = dict[str, Any]
JsonTransport = Callable[[str, Mapping[str, str], Mapping[str, str]], JsonObject]


class LiteratureProviderError(RuntimeError):
    """Safe provider failure with a stable category."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class LiteratureProvider(Protocol):
    name: str

    def search(self, request: LiteratureSearchInput) -> tuple[PaperRecord, ...]: ...


class OfflineLiteratureProvider:
    """Zero-network provider used by the CLI's explicit mock mode."""

    def __init__(self, name: str) -> None:
        self.name = name

    def search(self, request: LiteratureSearchInput) -> tuple[PaperRecord, ...]:
        return ()


def normalize_doi(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized or None


def reconstruct_abstract(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    positioned: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and position >= 0:
                positioned.append((position, word))
    if not positioned:
        return None
    positioned.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned)


class UrllibJsonTransport:
    """Small JSON transport with bounded retries and no redirect customization."""

    def __init__(self, *, timeout_seconds: float = 15.0, max_retries: int = 2) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def __call__(
        self, base_url: str, params: Mapping[str, str], headers: Mapping[str, str]
    ) -> JsonObject:
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(url, headers=dict(headers))
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:  # noqa: S310
                    payload = response.read(10_000_001)
                if len(payload) > 10_000_000:
                    raise LiteratureProviderError(
                        "PROVIDER_RESPONSE_TOO_LARGE", "provider response exceeds 10 MB"
                    )
                decoded = json.loads(payload)
                if not isinstance(decoded, dict):
                    raise LiteratureProviderError(
                        "PROVIDER_INVALID_RESPONSE",
                        "provider response is not an object",
                    )
                return decoded
            except urllib.error.HTTPError as exc:
                if exc.code in {401, 403}:
                    raise LiteratureProviderError(
                        "PROVIDER_AUTH", "literature provider rejected authentication"
                    ) from exc
                if exc.code == 429 and attempt < self.max_retries:
                    retry_after = exc.headers.get("Retry-After")
                    try:
                        delay = (
                            float(retry_after) if retry_after else 1.1 * (attempt + 1)
                        )
                    except ValueError:
                        delay = 1.1 * (attempt + 1)
                    time.sleep(min(max(delay, 1.1), 10.0))
                    continue
                if 500 <= exc.code < 600 and attempt < self.max_retries:
                    time.sleep(0.2 * (2**attempt))
                    continue
                code = (
                    "PROVIDER_RATE_LIMIT" if exc.code == 429 else "PROVIDER_UNAVAILABLE"
                )
                raise LiteratureProviderError(
                    code, f"provider HTTP error {exc.code}"
                ) from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                if attempt < self.max_retries:
                    time.sleep(0.2 * (2**attempt))
                    continue
                raise LiteratureProviderError(
                    "PROVIDER_UNAVAILABLE", "literature provider is unavailable"
                ) from exc
            except json.JSONDecodeError as exc:
                raise LiteratureProviderError(
                    "PROVIDER_INVALID_RESPONSE", "provider returned invalid JSON"
                ) from exc
        raise LiteratureProviderError("PROVIDER_UNAVAILABLE", "provider is unavailable")


class OpenAlexProvider:
    name = "openalex"
    base_url = "https://api.openalex.org/works"

    def __init__(
        self,
        *,
        transport: JsonTransport | None = None,
        mailto: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._transport = transport or UrllibJsonTransport()
        self._mailto = mailto
        self._api_key = (api_key or os.getenv("OPENALEX_API_KEY", "")).strip() or None

    def search(self, request: LiteratureSearchInput) -> tuple[PaperRecord, ...]:
        params = {
            "search": _search_text(request),
            "per-page": str(request.max_papers),
            "select": (
                "id,doi,title,publication_year,authorships,primary_location,"
                "abstract_inverted_index,cited_by_count,open_access"
            ),
        }
        filters = _year_filters(request)
        if filters:
            params["filter"] = ",".join(filters)
        if self._mailto:
            params["mailto"] = self._mailto
        headers = {
            "Accept": "application/json",
            "User-Agent": "materials-screening-core/0.1",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = self._transport(
            self.base_url,
            params,
            headers,
        )
        records = payload.get("results")
        if not isinstance(records, list):
            raise LiteratureProviderError(
                "PROVIDER_INVALID_RESPONSE", "OpenAlex results are missing"
            )
        return tuple(
            paper
            for raw in records[: request.max_papers]
            if (paper := _openalex_paper(raw))
        )


class SemanticScholarProvider:
    name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(
        self,
        *,
        transport: JsonTransport | None = None,
        api_key: str | None = None,
        min_request_interval_seconds: float = 1.05,
    ) -> None:
        self._transport = transport or UrllibJsonTransport()
        self._api_key = api_key
        self._min_request_interval_seconds = max(0.0, min_request_interval_seconds)
        self._last_request_at: float | None = None

    def search(self, request: LiteratureSearchInput) -> tuple[PaperRecord, ...]:
        params = {
            "query": _search_text(request),
            "limit": str(request.max_papers),
            "fields": (
                "paperId,title,abstract,year,authors,venue,citationCount,"
                "externalIds,openAccessPdf,url"
            ),
        }
        if request.year_from is not None or request.year_to is not None:
            lower = request.year_from or 1900
            upper = request.year_to or 2100
            params["year"] = f"{lower}-{upper}"
        headers = {
            "Accept": "application/json",
            "User-Agent": "materials-screening-core/0.1",
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        if self._last_request_at is not None:
            remaining = self._min_request_interval_seconds - (
                time.monotonic() - self._last_request_at
            )
            if remaining > 0:
                time.sleep(remaining)
        payload = self._transport(self.base_url, params, headers)
        self._last_request_at = time.monotonic()
        records = payload.get("data")
        if not isinstance(records, list):
            code = str(payload.get("code", "")).strip()
            message = str(payload.get("message", "")).casefold()
            if code == "429" or "too many requests" in message:
                raise LiteratureProviderError(
                    "PROVIDER_RATE_LIMIT", "Semantic Scholar rate limit reached"
                )
            if code in {"401", "403"}:
                raise LiteratureProviderError(
                    "PROVIDER_AUTH", "Semantic Scholar rejected authentication"
                )
            if payload.get("total") == 0 and records is None:
                return ()
            raise LiteratureProviderError(
                "PROVIDER_INVALID_RESPONSE", "Semantic Scholar data are missing"
            )
        return tuple(
            paper for raw in records[: request.max_papers] if (paper := _s2_paper(raw))
        )


def _search_text(request: LiteratureSearchInput) -> str:
    return " ".join((request.topic, *request.material_keywords))[:1500]


def _year_filters(request: LiteratureSearchInput) -> list[str]:
    filters: list[str] = []
    if request.year_from is not None:
        filters.append(f"from_publication_date:{request.year_from}-01-01")
    if request.year_to is not None:
        filters.append(f"to_publication_date:{request.year_to}-12-31")
    return filters


def _raw_hash(raw: JsonObject) -> str:
    encoded = json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _paper_id(doi: str | None, provider: str, provider_id: str) -> str:
    identity = f"doi:{doi}" if doi else f"{provider}:{provider_id}"
    return f"paper-{hashlib.sha256(identity.encode()).hexdigest()[:24]}"


def _openalex_paper(value: object) -> PaperRecord | None:
    if not isinstance(value, dict):
        return None
    title = value.get("title")
    provider_id = value.get("id")
    if (
        not isinstance(title, str)
        or not title.strip()
        or not isinstance(provider_id, str)
    ):
        return None
    doi = normalize_doi(value.get("doi"))
    authors: list[str] = []
    for authorship in value.get("authorships", []):
        if isinstance(authorship, dict) and isinstance(authorship.get("author"), dict):
            name = authorship["author"].get("display_name")
            if isinstance(name, str) and name:
                authors.append(name)
    location = value.get("primary_location")
    venue = None
    landing = None
    if isinstance(location, dict):
        landing = (
            location.get("landing_page_url")
            if isinstance(location.get("landing_page_url"), str)
            else None
        )
        source = location.get("source")
        if isinstance(source, dict) and isinstance(source.get("display_name"), str):
            venue = source["display_name"]
    oa = value.get("open_access")
    is_oa = (
        oa.get("is_oa")
        if isinstance(oa, dict) and isinstance(oa.get("is_oa"), bool)
        else None
    )
    return PaperRecord(
        paper_id=_paper_id(doi, "openalex", provider_id),
        title=" ".join(title.split()),
        doi=doi,
        year=value.get("publication_year")
        if isinstance(value.get("publication_year"), int)
        else None,
        authors=tuple(authors),
        venue=venue,
        abstract=reconstruct_abstract(value.get("abstract_inverted_index")),
        cited_by_count=value.get("cited_by_count")
        if isinstance(value.get("cited_by_count"), int)
        else None,
        open_access=is_oa,
        landing_page_url=landing,
        provenance=(
            PaperProvenance(
                provider="openalex",
                provider_id=provider_id,
                retrieved_at=datetime.now(UTC),
                raw_record_sha256=_raw_hash(value),
            ),
        ),
    )


def _s2_paper(value: object) -> PaperRecord | None:
    if not isinstance(value, dict):
        return None
    title = value.get("title")
    provider_id = value.get("paperId")
    if (
        not isinstance(title, str)
        or not title.strip()
        or not isinstance(provider_id, str)
    ):
        return None
    external = value.get("externalIds")
    doi = normalize_doi(external.get("DOI") if isinstance(external, dict) else None)
    authors = tuple(
        author["name"]
        for author in value.get("authors", [])
        if isinstance(author, dict) and isinstance(author.get("name"), str)
    )
    open_pdf = value.get("openAccessPdf")
    is_oa = bool(open_pdf.get("url")) if isinstance(open_pdf, dict) else False
    return PaperRecord(
        paper_id=_paper_id(doi, "semantic_scholar", provider_id),
        title=" ".join(title.split()),
        doi=doi,
        year=value.get("year") if isinstance(value.get("year"), int) else None,
        authors=authors,
        venue=value.get("venue")
        if isinstance(value.get("venue"), str) and value.get("venue")
        else None,
        abstract=value.get("abstract")
        if isinstance(value.get("abstract"), str) and value.get("abstract")
        else None,
        cited_by_count=value.get("citationCount")
        if isinstance(value.get("citationCount"), int)
        else None,
        open_access=is_oa,
        landing_page_url=value.get("url")
        if isinstance(value.get("url"), str)
        else None,
        provenance=(
            PaperProvenance(
                provider="semantic_scholar",
                provider_id=provider_id,
                retrieved_at=datetime.now(UTC),
                raw_record_sha256=_raw_hash(value),
            ),
        ),
    )
