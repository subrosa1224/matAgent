"""Natural-language facade bridging Planner and stage-1 screening."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from materials_screening.planner.errors import PlannerError
from materials_screening.planner.models import PlannerResult, PlannerStatus
from materials_screening.planner.service import PlannerService
from materials_screening.services.screening_service import (
    ScreeningRunOutput,
    ScreeningService,
)


class NaturalLanguageScreeningResult(BaseModel):
    """Outcome of one natural-language screening request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    planner: PlannerResult
    screening: ScreeningRunOutput | None = None


class NaturalLanguageScreeningFacade:
    """Coordinate PlannerService and the stage-1 ScreeningService.

    The stage-1 core is only invoked for READY plans. NEEDS_CLARIFICATION,
    INVALID and UNSUPPORTED plans stop before any repository call; planner
    errors propagate and never touch the repository.
    """

    def __init__(
        self,
        planner_service: PlannerService,
        screening_service: ScreeningService,
    ) -> None:
        self._planner_service = planner_service
        self._screening_service = screening_service

    def screen(
        self,
        query: str,
        output_root: Path,
        *,
        include_cif: bool = True,
    ) -> NaturalLanguageScreeningResult:
        """Parse the query and screen only when the plan is READY."""
        planner = self._planner_service.parse(query)
        if planner.status is not PlannerStatus.READY:
            return NaturalLanguageScreeningResult(planner=planner)
        return self.screen_ready(planner, output_root, include_cif=include_cif)

    def screen_ready(
        self,
        planner: PlannerResult,
        output_root: Path,
        *,
        include_cif: bool = True,
    ) -> NaturalLanguageScreeningResult:
        """Screen a pre-parsed plan without re-running the planner."""
        if planner.status is not PlannerStatus.READY:
            raise PlannerError("screen_ready requires a READY PlannerResult")
        if planner.request is None:
            raise PlannerError("READY PlannerResult must include a ScreeningRequest")
        screening = self._screening_service.run(
            planner.request,
            output_root,
            include_cif=include_cif,
        )
        return NaturalLanguageScreeningResult(
            planner=planner,
            screening=screening,
        )
