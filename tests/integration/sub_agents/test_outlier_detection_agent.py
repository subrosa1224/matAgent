"""Integration tests for the Outlier Detection agent (MockAgentModel)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from materials_screening.agent.conversation_store import SqliteConversationStore
from materials_screening.agent.mock_model import WorkflowDrivenMockAgentModel
from materials_screening.agent.runner import MaterialAgentRunner
from materials_screening.agent.settings import AgentSettings
from materials_screening.agent.tool_registry import AgentToolRegistry
from materials_screening.agent_tools.result_reader import FileWorkflowResultReader
from materials_screening.master import SubAgentSpec
from materials_screening.models import MaterialRecord
from materials_screening.repositories.base import MaterialsRepository
from materials_screening.sub_agents.outlier_detection.spec_factory import (
    create_spec,
)
from materials_screening.workflow.context import Clock
from materials_screening.workflow.runner import WorkflowRunner


def _build_spec(
    tmp_path,
    records: tuple[MaterialRecord, ...],
) -> SubAgentSpec:
    """Build a SubAgentSpec wired to a mock repository with *records*."""
    repo = Mock(spec=MaterialsRepository)
    repo.query_all_by_formula.side_effect = lambda f: tuple(
        r for r in records if r.formula_pretty == f
    )
    repo.query_by_material_id.side_effect = lambda material_id: next(
        (r for r in records if r.material_id == material_id), None
    )
    repo.healthcheck.return_value = True

    reader = FileWorkflowResultReader(tmp_path)

    workflow_runner = Mock(spec=WorkflowRunner)

    return create_spec(
        result_reader=reader,
        repository=repo,
        workflow_runner=workflow_runner,
    )


def _rec(
    material_id: str = "mp-1",
    formula: str = "TiO2",
    band_gap: float | None = 2.0,
    density: float | None = 4.0,
    hull: float | None = 0.05,
) -> MaterialRecord:
    return MaterialRecord(
        source="mock",
        material_id=material_id,
        formula_pretty=formula,
        elements=("Ti", "O"),
        band_gap_ev=band_gap,
        density_g_cm3=density,
        energy_above_hull_ev_atom=hull,
    )


# ── Tests ───────────────────────────────────────────────────────────────────


class TestOutlierDetectionAgentIntegration:
    def test_spec_is_valid(self, tmp_path) -> None:
        """Smoke test: spec builds without errors."""
        records: tuple[MaterialRecord, ...] = (
            _rec("mp-1", "TiO2"),
            _rec("mp-2", "BaTiO3"),
            _rec("mp-3", "SrTiO3"),
        )
        spec = _build_spec(tmp_path, records)
        assert spec.name == "outlier_detection"
        assert spec.delegate_function_name == "delegate_to_outlier_detection"
        assert len(spec.tool_definitions) == 1
        tool_names = {td.name for td in spec.tool_definitions}
        assert tool_names == {"run_outlier_detection"}

    def test_runner_factory_creates_runner(self, tmp_path) -> None:
        """Smoke test: runner factory creates a MaterialAgentRunner."""
        records: tuple[MaterialRecord, ...] = (
            _rec("mp-1", "TiO2"),
            _rec("mp-2", "BaTiO3"),
            _rec("mp-3", "SrTiO3"),
        )
        spec = _build_spec(tmp_path, records)
        runner = spec.runner_factory()
        assert isinstance(runner, MaterialAgentRunner)
        assert runner._settings.agent_system_prompt == spec.system_prompt
        assert runner._settings.agent_max_tool_output_bytes == 262_144
        assert runner._settings.agent_max_input_bytes == 524_288

    def test_system_prompt_no_api_keys(self, tmp_path) -> None:
        """System prompt must not contain any API key pattern."""
        records: tuple[MaterialRecord, ...] = (_rec(),)
        spec = _build_spec(tmp_path, records)
        prompt = spec.system_prompt
        assert "sk-" not in prompt
        assert "api_key" not in prompt.lower()
        assert "API_KEY" not in prompt

    def test_system_prompt_matches_final_draft_contract(self, tmp_path) -> None:
        """Prompt must request the same final fields the runner validates."""
        spec = _build_spec(tmp_path, (_rec(),))
        prompt = spec.system_prompt
        assert "referenced_material_ids" in prompt
        assert "active_workflow_thread_id" in prompt
        assert "analysed_material_ids" not in prompt

    def test_tool_definitions_have_required_fields(self, tmp_path) -> None:
        """All tool definitions must have name, description, parameters."""
        records: tuple[MaterialRecord, ...] = (_rec(),)
        spec = _build_spec(tmp_path, records)
        for td in spec.tool_definitions:
            assert td.name
            assert td.description
            assert td.parameters["type"] == "object"
            assert td.parameters["additionalProperties"] is False
