"""Unit tests for MaterialsProjectRepository behavior (M3)."""

from __future__ import annotations

import time

import pytest
import requests
from mp_api.client.core.exceptions import MPRestError

from materials_screening.errors import (
    InvalidRequestError,
    RepositoryAuthenticationError,
    RepositoryError,
    RepositoryMappingError,
    RepositoryRateLimitError,
    RepositoryTimeoutError,
)
from materials_screening.models import FloatRange, MaterialRecord, ScreeningRequest
from materials_screening.repositories import materials_project as mp_module
from materials_screening.repositories.materials_project import (
    MaterialsProjectRepository,
)
from materials_screening.repositories.mock import (
    FIXED_TEST_TIME,
    MockMaterialsRepository,
)

MINIMAL_DOC: dict[str, object] = {
    "material_id": "mp-1",
    "formula_pretty": "Fe2O3",
    "elements": ["Fe", "O"],
    "chemsys": "Fe-O",
    "band_gap": 2.0,
    "energy_above_hull": 0.01,
    "formation_energy_per_atom": -1.5,
    "density": 5.2,
    "is_metal": False,
    "is_gap_direct": True,
    "is_stable": True,
    "theoretical": False,
    "deprecated": False,
    "symmetry": {"crystal_system": "Trigonal", "symbol": "R-3c", "number": 167},
    "structure": None,
}


class FakeSummary:
    def __init__(
        self,
        failures: list[Exception] | None = None,
        documents: list[dict[str, object]] | None = None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.failures = list(failures or [])
        self.documents = [dict(document) for document in documents or [MINIMAL_DOC]]
        self.attempts = 0

    def search(self, **kwargs: object) -> list[dict[str, object]]:
        self.calls.append(kwargs)
        self.attempts += 1
        if self.failures:
            raise self.failures.pop(0)
        return [dict(document) for document in self.documents]


class FakeMPRester:
    instances: list[FakeMPRester] = []
    failures: list[Exception] = []
    documents: list[dict[str, object]] = []
    init_error: Exception | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        if FakeMPRester.init_error is not None:
            raise FakeMPRester.init_error
        FakeMPRester.instances.append(self)
        self.kwargs = kwargs
        self.db_version = "v2026.08"
        self.summary = FakeSummary(
            failures=list(FakeMPRester.failures),
            documents=list(FakeMPRester.documents),
        )

    @property
    def materials(self) -> FakeMPRester:
        return self

    def __enter__(self) -> FakeMPRester:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        return None


@pytest.fixture(autouse=True)
def _reset_fake_mp_rester() -> None:
    FakeMPRester.instances.clear()
    FakeMPRester.failures.clear()
    FakeMPRester.documents.clear()
    FakeMPRester.init_error = None
    yield
    FakeMPRester.instances.clear()
    FakeMPRester.failures.clear()
    FakeMPRester.documents.clear()
    FakeMPRester.init_error = None


@pytest.fixture
def fake_mp(monkeypatch: pytest.MonkeyPatch) -> type[FakeMPRester]:
    monkeypatch.setattr(mp_module, "MPRester", FakeMPRester)
    return FakeMPRester


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


class TestSearch:
    def test_timeout_session_applies_default_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_request(
            session: object, method: str, url: str, **kwargs: object
        ) -> object:
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(requests.Session, "request", fake_request)
        session = mp_module._TimeoutSession(12.5)
        session.get("https://example.com")
        assert captured["timeout"] == 12.5

    def test_passes_built_query_and_maps_records(
        self, fake_mp: type[FakeMPRester]
    ) -> None:
        repository = MaterialsProjectRepository(api_key="test-key")
        request = ScreeningRequest(
            required_elements=("Fe",),
            band_gap_ev=FloatRange(min=1.0, max=3.0),
        )
        result = repository.search(request)

        instance = FakeMPRester.instances[-1]
        assert instance.kwargs["api_key"] == "test-key"
        assert instance.kwargs["mute_progress_bars"] is True
        assert "timeout" not in instance.kwargs
        assert instance.kwargs["session"]._timeout_seconds == 30
        call = instance.summary.calls[0]
        assert call["band_gap"] == (1.0, 3.0)
        assert call["elements"] == ["Fe"]
        assert call["num_chunks"] == mp_module._MAX_MP_CHUNKS
        assert "limit" not in call
        assert result.source == "materials_project"
        assert result.database_version == "v2026.08"
        assert len(result.records) == 1
        assert result.records[0].material_id == "mp-1"

    def test_does_not_truncate_to_request_limit(
        self, fake_mp: type[FakeMPRester]
    ) -> None:
        repository = MaterialsProjectRepository(api_key="test-key")
        repository.search(ScreeningRequest(required_elements=("Fe",), limit=3))
        call = FakeMPRester.instances[-1].summary.calls[0]
        assert "limit" not in call

    def test_exclude_only_request_rejected_before_network(
        self, fake_mp: type[FakeMPRester]
    ) -> None:
        repository = MaterialsProjectRepository(api_key="test-key")
        with pytest.raises(InvalidRequestError, match="too broad"):
            repository.search(
                ScreeningRequest(excluded_elements=("Pb",), is_metal=False)
            )
        assert FakeMPRester.instances == []

    def test_empty_request_rejected_before_network(
        self, fake_mp: type[FakeMPRester]
    ) -> None:
        repository = MaterialsProjectRepository(api_key="test-key")
        with pytest.raises(InvalidRequestError, match="too broad"):
            repository.search(ScreeningRequest())
        assert FakeMPRester.instances == []


class TestRetryAndErrors:
    def test_retries_temporary_error_then_succeeds(
        self,
        fake_mp: type[FakeMPRester],
        no_sleep: None,
    ) -> None:
        FakeMPRester.failures = [
            MPRestError(
                "REST query returned with error status code 503 on URL "
                "https://api.materialsproject.org/materials/summary/ "
                "with message:\ntemporarily unavailable"
            ),
            MPRestError(
                "REST query returned with error status code 503 on URL "
                "https://api.materialsproject.org/materials/summary/ "
                "with message:\ntemporarily unavailable"
            ),
        ]
        repository = MaterialsProjectRepository(api_key="test-key")
        result = repository.search(ScreeningRequest(required_elements=("Fe",)))
        assert FakeMPRester.instances[-1].summary.attempts == 3
        assert result.records[0].material_id == "mp-1"

    def test_server_error_exhausted_raises_repository_error(
        self,
        fake_mp: type[FakeMPRester],
        no_sleep: None,
    ) -> None:
        FakeMPRester.failures = [
            MPRestError(
                "REST query returned with error status code 502 on URL "
                "https://api.materialsproject.org/materials/summary/ "
                "with message:\nbad gateway"
            )
        ] * 3
        repository = MaterialsProjectRepository(api_key="test-key")
        with pytest.raises(RepositoryError):
            repository.search(ScreeningRequest(required_elements=("Fe",)))
        assert FakeMPRester.instances[-1].summary.attempts == 3

    def test_timeout_error_raises_repository_timeout_error(
        self,
        fake_mp: type[FakeMPRester],
        no_sleep: None,
    ) -> None:
        FakeMPRester.failures = [
            MPRestError(
                "REST query timed out on URL "
                "https://api.materialsproject.org/materials/summary/. "
                "Try again with a smaller request."
            )
        ] * 3
        repository = MaterialsProjectRepository(api_key="test-key")
        with pytest.raises(RepositoryTimeoutError):
            repository.search(ScreeningRequest(required_elements=("Fe",)))

    def test_requests_timeout_raises_repository_timeout_error(
        self,
        fake_mp: type[FakeMPRester],
        no_sleep: None,
    ) -> None:
        FakeMPRester.failures = [requests.exceptions.ReadTimeout()] * 3
        repository = MaterialsProjectRepository(api_key="test-key")
        with pytest.raises(RepositoryTimeoutError):
            repository.search(ScreeningRequest(required_elements=("Fe",)))

    def test_connection_error_raises_repository_error(
        self,
        fake_mp: type[FakeMPRester],
        no_sleep: None,
    ) -> None:
        FakeMPRester.failures = [requests.exceptions.ConnectionError()] * 3
        repository = MaterialsProjectRepository(api_key="test-key")
        with pytest.raises(RepositoryError):
            repository.search(ScreeningRequest(required_elements=("Fe",)))

    def test_rate_limit_raises_repository_rate_limit_error(
        self,
        fake_mp: type[FakeMPRester],
        no_sleep: None,
    ) -> None:
        FakeMPRester.failures = [
            MPRestError(
                "REST query returned with error status code 429 on URL "
                "https://api.materialsproject.org/materials/summary/ "
                "with message:\nrate limited"
            )
        ] * 3
        repository = MaterialsProjectRepository(api_key="test-key")
        with pytest.raises(RepositoryRateLimitError):
            repository.search(ScreeningRequest(required_elements=("Fe",)))

    def test_authentication_error_is_not_retried(
        self, fake_mp: type[FakeMPRester]
    ) -> None:
        FakeMPRester.failures = [
            MPRestError(
                "REST query returned with error status code 401 on URL "
                "https://api.materialsproject.org/materials/summary/ "
                "with message:\nunauthorized"
            )
        ] * 3
        repository = MaterialsProjectRepository(api_key="test-key")
        with pytest.raises(RepositoryAuthenticationError):
            repository.search(ScreeningRequest(required_elements=("Fe",)))
        assert FakeMPRester.instances[-1].summary.attempts == 1

    def test_bad_request_is_not_retried(self, fake_mp: type[FakeMPRester]) -> None:
        FakeMPRester.failures = [
            MPRestError(
                "The server does not support the request made to "
                "https://api.materialsproject.org/materials/summary/. "
                "This may be due to an outdated mp-api package, or a problem "
                "with the query."
            )
        ] * 3
        repository = MaterialsProjectRepository(api_key="test-key")
        with pytest.raises(InvalidRequestError):
            repository.search(ScreeningRequest(required_elements=("Fe",)))
        assert FakeMPRester.instances[-1].summary.attempts == 1

    def test_init_api_key_error_maps_to_authentication_error(
        self, fake_mp: type[FakeMPRester]
    ) -> None:
        FakeMPRester.init_error = MPRestError(
            "Please obtain a valid API key from https://materialsproject.org/api "
            "and export it as an environment variable `MP_API_KEY`. "
            "Valid API keys are 32 characters."
        )
        repository = MaterialsProjectRepository(api_key="test-key")
        with pytest.raises(RepositoryAuthenticationError):
            repository.search(ScreeningRequest(required_elements=("Fe",)))

    def test_mapping_error_propagates(self, fake_mp: type[FakeMPRester]) -> None:
        FakeMPRester.documents = [
            {
                "material_id": "mp-9",
                "formula_pretty": "Fe2O3",
                "elements": ["Fe", "O"],
                "chemsys": "Fe-O",
                "band_gap": 2.0,
                "energy_above_hull": 0.01,
                "formation_energy_per_atom": -1.5,
                "density": 5.2,
                "is_metal": False,
                "is_gap_direct": True,
                "is_stable": True,
                "theoretical": False,
                "deprecated": False,
                "symmetry": None,
                "structure": "not-a-structure",
            }
        ]
        repository = MaterialsProjectRepository(api_key="test-key")
        with pytest.raises(RepositoryMappingError, match="mp-9"):
            repository.search(ScreeningRequest(required_elements=("Fe",)))

    def test_api_key_not_leaked_in_errors(self, fake_mp: type[FakeMPRester]) -> None:
        FakeMPRester.failures = [
            MPRestError(
                "REST query returned with error status code 503 on URL "
                "https://api.materialsproject.org/materials/summary/ "
                "with message:\ntemporarily unavailable"
            )
        ] * 3
        repository = MaterialsProjectRepository(api_key="super-secret-key")
        with pytest.raises(RepositoryError) as exc_info:
            repository.search(ScreeningRequest(required_elements=("Fe",)))
        assert "super-secret-key" not in str(exc_info.value)

    def test_wrapped_ssl_failure_is_retried_and_mapped_to_connection_error(
        self, fake_mp: type[FakeMPRester]
    ) -> None:
        FakeMPRester.failures = [
            MPRestError(
                "HTTPSConnectionPool(host='api.materialsproject.org', port=443): "
                "Max retries exceeded (Caused by SSLError: "
                "[SSL: UNEXPECTED_EOF_WHILE_READING])"
            )
        ] * 3
        repository = MaterialsProjectRepository(api_key="test-key")

        with pytest.raises(
            RepositoryError,
            match="secure connection failed after retries",
        ):
            repository.search(ScreeningRequest(required_elements=("Fe",)))

        assert len(FakeMPRester.instances[-1].summary.calls) == 3

    def test_healthcheck_true_when_connected(self, fake_mp: type[FakeMPRester]) -> None:
        repository = MaterialsProjectRepository(api_key="test-key")
        assert repository.healthcheck() is True

    def test_healthcheck_false_when_init_fails(
        self, fake_mp: type[FakeMPRester]
    ) -> None:
        FakeMPRester.init_error = MPRestError("boom")
        repository = MaterialsProjectRepository(api_key="test-key")
        assert repository.healthcheck() is False


class TestMockMaterialsRepository:
    def test_search_returns_prebuilt_records(self) -> None:
        record = MaterialRecord(
            source="mock",
            material_id="mp-1",
            formula_pretty="Fe2O3",
            elements=("Fe", "O"),
            band_gap_ev=2.0,
            density_g_cm3=5.2,
        )
        repository = MockMaterialsRepository(records=(record,))
        result = repository.search(ScreeningRequest(excluded_elements=("Pb",)))
        assert result.source == "mock"
        assert result.database_version == "fixture-v1"
        assert result.retrieved_at == FIXED_TEST_TIME
        assert result.records == (record,)
        assert repository.healthcheck() is True

    def test_exact_id_lookup_preserves_requested_fixture_id(self) -> None:
        repository = MockMaterialsRepository(records=())

        record = repository.query_by_material_id("mock-tio2")

        assert record is not None
        assert record.material_id == "mock-tio2"


class TestMaterialLookups:
    def test_formula_lookup_returns_all_entries(
        self, fake_mp: type[FakeMPRester]
    ) -> None:
        repository = MaterialsProjectRepository(api_key="test-key")

        records = repository.query_all_by_formula("TiO2")

        call = FakeMPRester.instances[-1].summary.calls[0]
        assert call["formula"] == "TiO2"
        assert "is_metal" not in call
        assert len(records) == 1

    def test_material_id_lookup_uses_exact_id(
        self, fake_mp: type[FakeMPRester]
    ) -> None:
        repository = MaterialsProjectRepository(api_key="test-key")

        record = repository.query_by_material_id("mp-1")

        call = FakeMPRester.instances[-1].summary.calls[0]
        assert call["material_ids"] == ["mp-1"]
        assert record is not None
        assert record.material_id == "mp-1"
