"""Materials Project repository: query building, mapping, retry and errors (M3)."""

import re
import time
from datetime import UTC, datetime
from typing import Any

import requests
from emmet.core.summary import SummaryDoc
from mp_api.client import MPRester
from mp_api.client.core.exceptions import MPRestError
from pydantic import ValidationError

from materials_screening.errors import (
    InvalidRequestError,
    MaterialsScreeningError,
    RepositoryAuthenticationError,
    RepositoryError,
    RepositoryMappingError,
    RepositoryRateLimitError,
    RepositoryTimeoutError,
)
from materials_screening.models import (
    CrystalSystem,
    FloatRange,
    MaterialRecord,
    PropertyProvenance,
    PropertyValueType,
    ScreeningRequest,
    SymmetryInfo,
)
from materials_screening.repositories.base import RetrievalResult

_RETRY_WAIT_SECONDS = 0.5
_MP_STATUS_CODE_RE = re.compile(r"status code (\d+)")

# Hard bound on live retrieval: at most this many chunks (pages) of 500
# documents are fetched. Broad queries would otherwise paginate the entire
# catalog (tens of thousands of documents) for minutes.
_MAX_MP_CHUNKS = 4

REQUESTED_FIELDS: tuple[str, ...] = (
    "material_id",
    "formula_pretty",
    "elements",
    "chemsys",
    "band_gap",
    "energy_above_hull",
    "formation_energy_per_atom",
    "density",
    "is_metal",
    "is_gap_direct",
    "is_stable",
    "theoretical",
    "deprecated",
    "symmetry",
    "structure",
)

PROVENANCE_PROPERTIES: tuple[str, ...] = (
    "band_gap_ev",
    "energy_above_hull_ev_atom",
    "formation_energy_ev_atom",
    "density_g_cm3",
    "is_metal",
    "is_gap_direct",
    "is_stable",
    "theoretical",
    "symmetry",
    "structure_dict",
)


class _TimeoutSession(requests.Session):
    """requests session that applies a default timeout to every request."""

    def __init__(self, timeout_seconds: float) -> None:
        super().__init__()
        self._timeout_seconds = timeout_seconds

    def request(self, *args: Any, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self._timeout_seconds)
        return super().request(*args, **kwargs)


def range_to_mp_tuple(
    range_: FloatRange, default_min: float, default_max: float
) -> tuple[float, float]:
    """Convert an open-ended FloatRange to a closed MP API tuple."""
    return (
        range_.min if range_.min is not None else default_min,
        range_.max if range_.max is not None else default_max,
    )


def read_field(obj: object, name: str, default: object = None) -> object:
    """Read a field from either a dict or an attribute-style object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _as_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise RepositoryMappingError(f"expected a boolean, got {type(value).__name__}")


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RepositoryMappingError("expected a number, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    raise RepositoryMappingError(f"expected a number, got {type(value).__name__}")


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RepositoryMappingError("expected an integer, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise RepositoryMappingError(f"expected an integer, got {value!r}")
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise RepositoryMappingError(f"expected an integer, got {value!r}") from exc
    raise RepositoryMappingError(f"expected an integer, got {type(value).__name__}")


def _enum_value(value: object) -> str:
    if isinstance(value, str):
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    return str(value)


def _map_symmetry(value: object) -> SymmetryInfo | None:
    if value is None:
        return None
    crystal_system_raw = read_field(value, "crystal_system")
    crystal_system: CrystalSystem | None = None
    if crystal_system_raw is not None:
        crystal_system = CrystalSystem(_enum_value(crystal_system_raw))
    number_raw = read_field(value, "number")
    number = _as_optional_int(number_raw)
    return SymmetryInfo(
        crystal_system=crystal_system,
        symbol=_as_optional_str(read_field(value, "symbol")),
        number=number,
    )


def _map_structure(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        result = as_dict()
        if isinstance(result, dict):
            return result
        raise RepositoryMappingError("structure.as_dict() must return a dict")
    raise RepositoryMappingError(f"unsupported structure type: {type(value).__name__}")


def _build_provenance(
    material_id: str,
    values: dict[str, object],
    database_version: str | None,
    retrieved_at: datetime,
) -> tuple[PropertyProvenance, ...]:
    return tuple(
        PropertyProvenance(
            property_name=name,
            source="materials_project",
            source_material_id=material_id,
            value_type=PropertyValueType.DFT_CALCULATED,
            database_version=database_version,
            retrieved_at=retrieved_at,
        )
        for name in PROVENANCE_PROPERTIES
        if values.get(name) is not None
    )


def map_summary_document(
    document: object,
    database_version: str | None,
    retrieved_at: datetime,
) -> MaterialRecord:
    """Map a SummaryDoc or plain dict to a MaterialRecord with provenance."""
    material_id_raw = read_field(document, "material_id")
    if material_id_raw is None:
        raise RepositoryMappingError("material_id is missing")
    material_id = str(material_id_raw)
    formula_pretty = _as_optional_str(read_field(document, "formula_pretty"))
    if formula_pretty is None:
        raise RepositoryMappingError(
            f"material {material_id!r}: formula_pretty is missing"
        )
    elements_raw = read_field(document, "elements", ())
    if not isinstance(elements_raw, (list, tuple)):
        raise RepositoryMappingError(
            f"material {material_id!r}: elements must be a list or tuple"
        )
    elements = tuple(sorted(str(element) for element in elements_raw))
    try:
        values: dict[str, object] = {
            "source": "materials_project",
            "material_id": material_id,
            "formula_pretty": formula_pretty,
            "elements": elements,
            "chemsys": _as_optional_str(read_field(document, "chemsys")),
            "band_gap_ev": _as_optional_float(read_field(document, "band_gap")),
            "energy_above_hull_ev_atom": _as_optional_float(
                read_field(document, "energy_above_hull")
            ),
            "formation_energy_ev_atom": _as_optional_float(
                read_field(document, "formation_energy_per_atom")
            ),
            "density_g_cm3": _as_optional_float(read_field(document, "density")),
            "is_metal": _as_optional_bool(read_field(document, "is_metal")),
            "is_gap_direct": _as_optional_bool(read_field(document, "is_gap_direct")),
            "is_stable": _as_optional_bool(read_field(document, "is_stable")),
            "theoretical": _as_optional_bool(read_field(document, "theoretical")),
            "deprecated": _as_optional_bool(read_field(document, "deprecated")),
            "symmetry": _map_symmetry(read_field(document, "symmetry")),
            "structure_dict": _map_structure(read_field(document, "structure")),
        }
        record = MaterialRecord.model_validate(values)
    except (RepositoryMappingError, ValueError, ValidationError) as exc:
        raise RepositoryMappingError(
            f"failed to map document {material_id!r}: {exc}"
        ) from exc
    provenance = _build_provenance(material_id, values, database_version, retrieved_at)
    return record.model_copy(update={"provenance": provenance})


def build_mp_query(request: ScreeningRequest) -> dict[str, Any]:
    """Build the SummaryRester.search query for a screening request."""
    query: dict[str, Any] = {
        "deprecated": False,
        "all_fields": False,
        "fields": list(REQUESTED_FIELDS),
        "chunk_size": 500,
    }
    if request.band_gap_ev is not None:
        query["band_gap"] = range_to_mp_tuple(
            request.band_gap_ev, default_min=0.0, default_max=100.0
        )
    if request.energy_above_hull_ev_atom is not None:
        query["energy_above_hull"] = range_to_mp_tuple(
            request.energy_above_hull_ev_atom,
            default_min=0.0,
            default_max=100.0,
        )
    if request.density_g_cm3 is not None:
        query["density"] = range_to_mp_tuple(
            request.density_g_cm3, default_min=0.0, default_max=1000.0
        )
    if request.required_elements:
        query["elements"] = list(request.required_elements)
    if request.excluded_elements:
        query["exclude_elements"] = list(request.excluded_elements)
    if request.chemsys:
        query["chemsys"] = request.chemsys
    if request.formula:
        query["formula"] = request.formula
    if request.material_ids:
        query["material_ids"] = list(request.material_ids)
    if request.num_elements is not None:
        query["num_elements"] = request.num_elements
    if request.crystal_system is not None:
        query["crystal_system"] = request.crystal_system.value
    if request.spacegroup_numbers:
        query["spacegroup_number"] = list(request.spacegroup_numbers)
    if request.is_metal is not None:
        query["is_metal"] = request.is_metal
    if request.is_stable is not None:
        query["is_stable"] = request.is_stable
    if request.theoretical is not None:
        query["theoretical"] = request.theoretical
    return query


def is_broad_request(request: ScreeningRequest) -> bool:
    """True when the request would scan the full catalog in a live query.

    A request is considered broad when it carries no selective constraint.
    Weak filters such as ``excluded_elements``, ``is_metal`` or
    ``theoretical`` still match most of the catalog (e.g. "no Pb
    semiconductor" is just ``is_metal=False`` plus ``exclude_elements``), so
    they do not count as narrowing. Such a query returns tens of thousands of
    documents from the Materials Project API and takes minutes to paginate,
    so it must not run against the live repository.
    """
    return not any(
        (
            request.required_elements,
            request.chemsys,
            request.formula,
            request.material_ids,
            bool(request.spacegroup_numbers),
            request.crystal_system is not None,
            request.band_gap_ev is not None,
            request.energy_above_hull_ev_atom is not None,
            request.density_g_cm3 is not None,
            request.is_stable is True,
        )
    )


def _status_code_from_message(message: str) -> int | None:
    match = _MP_STATUS_CODE_RE.search(message)
    if match is None:
        return None
    return int(match.group(1))


def _is_connection_failure_message(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "sslerror",
            "unexpected_eof",
            "unexpected eof",
            "max retries exceeded",
            "connectionpool",
            "connection aborted",
            "connection reset",
            "failed to establish a new connection",
            "name resolution",
        )
    )


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(
        exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
    ):
        return True
    if not isinstance(exc, MPRestError):
        return False
    message = str(exc)
    if "timed out" in message.lower() or _is_connection_failure_message(message):
        return True
    status = _status_code_from_message(message)
    if status is None:
        return False
    return status == 429 or status >= 500


def _map_repository_error(exc: Exception) -> MaterialsScreeningError:
    if isinstance(exc, requests.exceptions.Timeout):
        return RepositoryTimeoutError(f"Materials Project request timed out: {exc}")
    if isinstance(exc, requests.exceptions.ConnectionError):
        return RepositoryError(f"Materials Project connection failed: {exc}")
    if isinstance(exc, MPRestError):
        message = str(exc)
        lowered = message.lower()
        if "timed out" in lowered:
            return RepositoryTimeoutError(f"Materials Project request timed out: {exc}")
        if _is_connection_failure_message(message):
            return RepositoryError(
                "Materials Project secure connection failed after retries"
            )
        if "api key" in lowered:
            return RepositoryAuthenticationError(
                f"Materials Project API key error: {exc}"
            )
        status = _status_code_from_message(message)
        if status in (401, 403):
            return RepositoryAuthenticationError(
                f"Materials Project authentication failed (HTTP {status})"
            )
        if status == 429:
            return RepositoryRateLimitError(
                "Materials Project rate limit exceeded (HTTP 429)"
            )
        if status is not None and status >= 500:
            return RepositoryError(f"Materials Project server error (HTTP {status})")
        return InvalidRequestError(f"Materials Project rejected the query: {exc}")
    return RepositoryError(f"Materials Project query failed: {exc}")


class MaterialsProjectRepository:
    """Query the Materials Project summary API with bounded retries."""

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 30,
        max_attempts: int = 3,
    ) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts

    def search(self, request: ScreeningRequest) -> RetrievalResult:
        if is_broad_request(request):
            raise InvalidRequestError(
                "Materials Project query is too broad: it would scan the "
                "full catalog. Add at least one narrowing constraint "
                "(elements, chemsys, formula, band gap, density, stability, "
                "crystal system or spacegroup) before querying the live "
                "repository."
            )
        query = build_mp_query(request)
        query["num_chunks"] = _MAX_MP_CHUNKS
        session = _TimeoutSession(self._timeout_seconds)
        try:
            with MPRester(
                api_key=self._api_key,
                session=session,
                mute_progress_bars=True,
            ) as mpr:
                database_version = mpr.db_version or None
                documents = self._search_with_retry(mpr, query)
        except MaterialsScreeningError:
            raise
        except Exception as exc:
            raise _map_repository_error(exc) from exc
        finally:
            session.close()
        retrieved_at = datetime.now(UTC)
        records = tuple(
            map_summary_document(
                document,
                database_version=database_version,
                retrieved_at=retrieved_at,
            )
            for document in documents
        )
        return RetrievalResult(
            source="materials_project",
            database_version=database_version,
            retrieved_at=retrieved_at,
            records=records,
        )

    def _search_with_retry(
        self, mpr: MPRester, query: dict[str, Any]
    ) -> list[SummaryDoc] | list[dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                return list(mpr.materials.summary.search(**query))
            except Exception as exc:
                last_error = exc
                if not _is_retryable_error(exc):
                    raise _map_repository_error(exc) from exc
                if attempt < self._max_attempts - 1:
                    time.sleep(_RETRY_WAIT_SECONDS)
        assert last_error is not None
        raise _map_repository_error(last_error) from last_error

    def query_by_formula(self, formula: str) -> MaterialRecord | None:
        """Look up a single material by its reduced formula via MP API."""
        request = ScreeningRequest(formula=formula, is_metal=None, limit=1)
        result = self.search(request)
        records = result.records
        return records[0] if records else None

    def query_all_by_formula(self, formula: str) -> tuple[MaterialRecord, ...]:
        """Return all non-deprecated MP entries for a reduced formula."""
        request = ScreeningRequest(formula=formula, is_metal=None, limit=100)
        return self.search(request).records

    def query_by_material_id(self, material_id: str) -> MaterialRecord | None:
        """Return the exact MP entry requested by id."""
        request = ScreeningRequest(material_ids=(material_id,), is_metal=None, limit=1)
        records = self.search(request).records
        return records[0] if records else None

    def healthcheck(self) -> bool:
        session = _TimeoutSession(self._timeout_seconds)
        try:
            with MPRester(
                api_key=self._api_key,
                session=session,
                mute_progress_bars=True,
            ) as mpr:
                return bool(mpr.db_version)
        except Exception:
            return False
        finally:
            session.close()
