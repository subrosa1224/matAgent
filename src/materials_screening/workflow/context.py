"""Stage 3 workflow runtime context and lightweight protocols (S3-M2)."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from materials_screening.planner.service import PlannerService
from materials_screening.repositories.base import MaterialsRepository
from materials_screening.services.export_service import ExportService
from materials_screening.services.filter_service import FilterService
from materials_screening.services.ranking_service import RankingService
from materials_screening.services.validation_service import ValidationService
from materials_screening.workflow.export_adapter import WorkflowExportAdapter
from materials_screening.workflow.state import ArtifactRef

Clock = Callable[[], datetime]
"""Return the current UTC time; injected for deterministic tests."""


@runtime_checkable
class IdGenerator(Protocol):
    """Generates stable run/thread identifiers."""

    def new_id(self) -> str: ...


@runtime_checkable
class RunArtifactStore(Protocol):
    """Artifact storage contract used by workflow nodes (S3-M3 implements it)."""

    def initialize_run(self, *, run_id: str, metadata: dict[str, Any]) -> None: ...

    def put_json(
        self,
        *,
        run_id: str,
        name: str,
        value: object,
        schema_name: str | None = None,
        schema_version: str | None = None,
        item_count: int | None = None,
    ) -> ArtifactRef: ...

    def get_json(self, ref: ArtifactRef) -> object: ...

    def exists(self, ref: ArtifactRef) -> bool: ...

    def verify(self, ref: ArtifactRef) -> bool: ...


@dataclass(frozen=True)
class WorkflowContext:
    """Immutable runtime dependencies injected into every workflow node.

    The context is delivered through LangGraph's ``Runtime.context`` and is
    never serialized into state, checkpoints or artifacts. Compiling a graph
    must not construct any real Intern or Materials Project client; the
    runner builds this context right before ``invoke``/``stream``.
    """

    planner_service: PlannerService
    materials_repository: MaterialsRepository
    filter_service: FilterService
    ranking_service: RankingService
    validation_service: ValidationService
    export_service: ExportService | WorkflowExportAdapter
    artifact_store: RunArtifactStore
    clock: Clock
    id_generator: IdGenerator
