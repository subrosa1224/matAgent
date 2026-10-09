"""ScreeningService orchestration (M6)."""

import secrets
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from materials_screening import __version__
from materials_screening.errors import ValidationFailedError
from materials_screening.fingerprints import request_fingerprint
from materials_screening.models import (
    RunMetadata,
    ScreeningRequest,
    ScreeningResult,
    ValidationReport,
)
from materials_screening.repositories.base import MaterialsRepository
from materials_screening.services.export_service import ExportResult, ExportService
from materials_screening.services.filter_service import FilterService
from materials_screening.services.ranking_service import RankingService
from materials_screening.services.validation_service import ValidationService

Clock = Callable[[], datetime]


class ScreeningRunOutput(BaseModel):
    """Result of one orchestrated screening run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    result: ScreeningResult
    exports: ExportResult


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)


def _package_version(package_name: str) -> str | None:
    try:
        return package_version(package_name)
    except PackageNotFoundError:
        return None


def _new_run_id(started_at: datetime) -> str:
    return f"run_{started_at.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"


class ScreeningService:
    """Orchestrate repository, filter, ranking, validation and export."""

    def __init__(
        self,
        repository: MaterialsRepository,
        filter_service: FilterService,
        ranking_service: RankingService,
        validation_service: ValidationService,
        export_service: ExportService,
        clock: Clock | None = None,
    ) -> None:
        self._repository = repository
        self._filter_service = filter_service
        self._ranking_service = ranking_service
        self._validation_service = validation_service
        self._export_service = export_service
        self._clock = clock or utc_now

    def run(
        self,
        request: ScreeningRequest,
        output_root: Path,
        *,
        include_cif: bool = True,
    ) -> ScreeningRunOutput:
        started_at = self._clock()
        retrieval = self._repository.search(request)

        filtered, filter_trace = self._filter_service.apply(retrieval.records, request)
        ranked_all = self._ranking_service.rank(filtered, request)
        ranked_limited = ranked_all[: request.limit]
        finished_at = self._clock()

        result = ScreeningResult(
            request=request,
            metadata=RunMetadata(
                run_id=_new_run_id(started_at),
                started_at=started_at,
                finished_at=finished_at,
                source=retrieval.source,
                database_version=retrieval.database_version,
                mp_api_version=_package_version("mp-api"),
                pymatgen_version=_package_version("pymatgen"),
                application_version=(
                    _package_version("materials-screening-core") or __version__
                ),
                query_fingerprint=request_fingerprint(request),
            ),
            retrieved_count=len(retrieval.records),
            passed_filter_count=len(filtered),
            ranked_materials=ranked_limited,
            filter_trace=filter_trace,
            validation=ValidationReport(
                passed=False, errors=(), warnings=(), checked_material_ids=()
            ),
        )

        validation = self._validation_service.validate(result)
        if not validation.passed:
            raise ValidationFailedError(
                f"validation failed with {len(validation.errors)} error(s)"
            )
        result = result.model_copy(update={"validation": validation})

        exports = self._export_service.export(
            result, output_root, include_cif=include_cif
        )
        return ScreeningRunOutput(result=result, exports=exports)
