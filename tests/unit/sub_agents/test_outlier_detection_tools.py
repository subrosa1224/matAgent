"""Unit tests for Outlier Detection tools — mock Resolver + Service + Ledger."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from materials_screening.agent.context import AgentToolContext
from materials_screening.agent.ledger import ToolExecutionLedger
from materials_screening.agent.tool_base import ToolSideEffect
from materials_screening.models import MaterialRecord
from materials_screening.sub_agents.outlier_detection.models import (
    MaterialSetReference,
    MultivariateOutlierReport,
    PropertyOutlierReport,
    RunOutlierDetectionInput,
)
from materials_screening.sub_agents.outlier_detection.tools import (
    DetectMultivariateOutliersTool,
    DetectPropertyOutliersTool,
    RunOutlierDetectionTool,
)
from materials_screening.workflow.context import Clock


def _rec(material_id: str = "mp-1", formula: str = "TiO2") -> MaterialRecord:
    return MaterialRecord(
        source="mock",
        material_id=material_id,
        formula_pretty=formula,
        elements=("Ti", "O"),
        band_gap_ev=2.0,
        density_g_cm3=4.0,
        energy_above_hull_ev_atom=0.05,
    )


class _FakeIdGenerator:
    def __init__(self) -> None:
        self._n = 0

    def new_id(self) -> str:
        self._n += 1
        return f"evt_{self._n}"


def _context() -> AgentToolContext:
    return AgentToolContext(
        workflow_runner=Mock(),
        workflow_result_reader=Mock(),
        clock=Mock(spec=Clock),
        id_generator=_FakeIdGenerator(),
        ledger=ToolExecutionLedger(),
        call_id="call-1",
        user_turn_id="turn-1",
        conversation_id="conv-1",
    )


class TestRunOutlierDetectionTool:
    def test_query_is_parsed_and_executed(self) -> None:
        resolver = Mock()
        resolver.resolve_with_warnings.return_value = (
            [_rec("mp-1", "TiO2"), _rec("mp-2", "BaTiO3")],
            (),
        )
        tool = RunOutlierDetectionTool(resolver)
        ctx = _context()

        report = tool.execute(
            RunOutlierDetectionInput(
                query="Compare TiO2 and BaTiO3 band gaps and find outliers"
            ),
            ctx,
        )

        assert isinstance(report, PropertyOutlierReport)
        assert report.property_name == "band_gap_ev"
        assert report.evidence_id == "evt_1"
        assert ctx.ledger.executed_tool_names() == ("run_outlier_detection",)
        source = resolver.resolve_with_warnings.call_args.args[0]
        assert source.material_formulas == ("TiO2", "BaTiO3")
        assert source.material_ids is None

    def test_material_ids_take_precedence_over_formulas(self) -> None:
        resolver = Mock()
        resolver.resolve_with_warnings.return_value = (
            [_rec("mp-1", "TiO2"), _rec("mp-2", "BaTiO3")],
            (),
        )
        tool = RunOutlierDetectionTool(resolver)

        tool.execute(
            RunOutlierDetectionInput(
                query="Compare TiO2 mp-1 and BaTiO3 mp-2 band gaps"
            ),
            _context(),
        )

        source = resolver.resolve_with_warnings.call_args.args[0]
        assert source.material_ids == ("mp-1", "mp-2")
        assert source.material_formulas is None

    def test_csv_path_is_routed_to_data_file(self) -> None:
        resolver = Mock()
        resolver.resolve_with_warnings.return_value = (
            [
                _rec("user-1", "TiO2"),
                _rec("user-2", "BaTiO3"),
                _rec("user-3", "SiO2"),
            ],
            (),
        )
        tool = RunOutlierDetectionTool(resolver)

        report = tool.execute(
            RunOutlierDetectionInput(
                query=(
                    "分析 data/outlier/properties.csv 的带隙、密度和凸包能，"
                    "找出离群材料"
                )
            ),
            _context(),
        )

        assert isinstance(report, MultivariateOutlierReport)
        source = resolver.resolve_with_warnings.call_args.args[0]
        assert source.data_file == "data/outlier/properties.csv"
        assert source.material_ids is None
        assert source.material_formulas is None

    def test_windows_csv_path_is_supported(self) -> None:
        resolver = Mock()
        resolver.resolve_with_warnings.return_value = (
            [_rec("user-1"), _rec("user-2"), _rec("user-3")],
            (),
        )

        RunOutlierDetectionTool(resolver).execute(
            RunOutlierDetectionInput(
                query=r"analyse data\outlier\properties.csv band gap and density"
            ),
            _context(),
        )

        source = resolver.resolve_with_warnings.call_args.args[0]
        assert source.data_file == r"data\outlier\properties.csv"

    def test_workflow_thread_is_routed_as_material_batch(self) -> None:
        resolver = Mock()
        resolver.resolve_with_warnings.return_value = (
            [_rec("mp-1", "TiO2"), _rec("mp-2", "SiO2")],
            (),
        )

        RunOutlierDetectionTool(resolver).execute(
            RunOutlierDetectionInput(
                query=(
                    "分析工作流 thread oxide-demo 的带隙，找出离群材料"
                )
            ),
            _context(),
        )

        source = resolver.resolve_with_warnings.call_args.args[0]
        assert source.workflow_thread_ids == ("oxide-demo",)
        assert source.material_ids is None
        assert source.material_formulas is None

    def test_uuid_is_recognized_as_workflow_thread(self) -> None:
        resolver = Mock()
        resolver.resolve_with_warnings.return_value = (
            [_rec("mp-1", "TiO2"), _rec("mp-2", "SiO2")],
            (),
        )
        thread_id = "057c019e-8f5a-4d87-9172-a8326d194d3f"

        RunOutlierDetectionTool(resolver).execute(
            RunOutlierDetectionInput(
                query=f"分析 {thread_id} 这批结果的密度离群值"
            ),
            _context(),
        )

        source = resolver.resolve_with_warnings.call_args.args[0]
        assert source.workflow_thread_ids == (thread_id,)

    def test_query_requires_two_formulas(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            RunOutlierDetectionTool(Mock()).execute(
                RunOutlierDetectionInput(query="Find the band-gap outlier TiO2"),
                _context(),
            )

    def test_unspecified_property_defaults_to_band_gap(self) -> None:
        resolver = Mock()
        resolver.resolve_with_warnings.return_value = (
            [_rec("mp-1", "TiO2"), _rec("mp-2", "SiO2")],
            (),
        )

        report = RunOutlierDetectionTool(resolver).execute(
            RunOutlierDetectionInput(
                query="分析工作流 thread batch-1，找这些材料的离群材料"
            ),
            _context(),
        )

        assert isinstance(report, PropertyOutlierReport)
        assert report.property_name == "band_gap_ev"
        assert any("defaulted to band_gap_ev" in item for item in report.warnings)


# ── DetectPropertyOutliersTool ──────────────────────────────────────────────


class TestDetectPropertyOutliersTool:
    def test_successful_execution(self) -> None:
        resolver = Mock()
        resolver.resolve_with_warnings.return_value = ([
            _rec("mp-1", "A"),
            _rec("mp-2", "B"),
            _rec("mp-3", "C"),
        ], ())
        tool = DetectPropertyOutliersTool(resolver)
        ctx = _context()

        ref = MaterialSetReference(workflow_thread_ids=("run-1",))
        report = tool.execute(
            arguments=tool.input_model(source=ref, property="band_gap_ev"),
            context=ctx,
        )
        assert isinstance(report, PropertyOutlierReport)
        assert report.property_name == "band_gap_ev"
        assert report.evidence_id == "evt_1"

    def test_invalid_property_raises(self) -> None:
        tool = DetectPropertyOutliersTool(Mock())
        ref = MaterialSetReference(workflow_thread_ids=("run-1",))
        with pytest.raises(ValueError, match="not in allowed set"):
            tool.execute(
                arguments=tool.input_model(source=ref, property="invalid_prop"),
                context=_context(),
            )

    def test_evidence_recorded_in_ledger(self) -> None:
        resolver = Mock()
        resolver.resolve_with_warnings.return_value = ([
            _rec("mp-1", "A"),
            _rec("mp-2", "B"),
        ], ())
        tool = DetectPropertyOutliersTool(resolver)
        ctx = _context()
        ref = MaterialSetReference(workflow_thread_ids=("run-1",))

        report = tool.execute(
            arguments=tool.input_model(source=ref, property="band_gap_ev"),
            context=ctx,
        )
        assert ctx.ledger.evidence_ids() == (report.evidence_id,)
        assert ctx.ledger.executed_tool_names() == ("detect_property_outliers",)

    def test_tool_metadata(self) -> None:
        tool = DetectPropertyOutliersTool(Mock())
        assert tool.name == "detect_property_outliers"
        assert tool.side_effect == ToolSideEffect.READ_ONLY
        assert len(tool.description) > 0

    def test_resolver_called_with_correct_reference(self) -> None:
        resolver = Mock()
        resolver.resolve_with_warnings.return_value = ([
            _rec("mp-1", "TiO2"),
            _rec("mp-2", "BaTiO3"),
        ], ())
        tool = DetectPropertyOutliersTool(resolver)
        ref = MaterialSetReference(material_formulas=("TiO2", "BaTiO3"))

        tool.execute(
            arguments=tool.input_model(
                source=ref, property="formation_energy_ev_atom"
            ),
            context=_context(),
        )
        resolver.resolve_with_warnings.assert_called_once()
        called_ref = resolver.resolve_with_warnings.call_args[0][0]
        assert called_ref.material_formulas == ("TiO2", "BaTiO3")

    def test_resolver_warning_is_included_in_report(self) -> None:
        resolver = Mock()
        resolver.resolve_with_warnings.return_value = (
            [_rec("mp-1", "TiO2"), _rec("mp-2", "BaTiO3")],
            ("No material found for formula 'Unknown'; excluded from analysis",),
        )
        report = DetectPropertyOutliersTool(resolver).execute(
            arguments=DetectPropertyOutliersTool.input_model(
                source=MaterialSetReference(
                    material_formulas=("TiO2", "BaTiO3", "Unknown")
                ),
                property="band_gap_ev",
            ),
            context=_context(),
        )
        assert report.warnings[0].startswith("No material found")


# ── DetectMultivariateOutliersTool ──────────────────────────────────────────


class TestDetectMultivariateOutliersTool:
    def test_successful_execution(self) -> None:
        resolver = Mock()
        resolver.resolve_with_warnings.return_value = ([
            MaterialRecord(
                source="mock", material_id="mp-1", formula_pretty="A",
                elements=("X",), band_gap_ev=2.0, density_g_cm3=4.0,
            ),
            MaterialRecord(
                source="mock", material_id="mp-2", formula_pretty="B",
                elements=("X",), band_gap_ev=2.5, density_g_cm3=3.5,
            ),
            MaterialRecord(
                source="mock", material_id="mp-3", formula_pretty="C",
                elements=("X",), band_gap_ev=1.5, density_g_cm3=4.5,
            ),
            MaterialRecord(
                source="mock", material_id="mp-4", formula_pretty="D",
                elements=("X",), band_gap_ev=2.3, density_g_cm3=3.7,
            ),
        ], ())
        tool = DetectMultivariateOutliersTool(resolver)
        ctx = _context()

        ref = MaterialSetReference(workflow_thread_ids=("run-1",))
        report = tool.execute(
            arguments=tool.input_model(
                source=ref,
                properties=("band_gap_ev", "density_g_cm3"),
                method="mahalanobis",
            ),
            context=ctx,
        )
        assert isinstance(report, MultivariateOutlierReport)
        assert report.method == "mahalanobis"
        assert report.evidence_id == "evt_1"

    def test_invalid_property_raises(self) -> None:
        tool = DetectMultivariateOutliersTool(Mock())
        ref = MaterialSetReference(workflow_thread_ids=("run-1",))
        with pytest.raises(ValueError, match="not in allowed set"):
            tool.execute(
                arguments=tool.input_model(
                    source=ref,
                    properties=("band_gap_ev", "invalid_prop"),
                ),
                context=_context(),
            )

    def test_evidence_recorded_in_ledger(self) -> None:
        resolver = Mock()
        resolver.resolve_with_warnings.return_value = ([
            MaterialRecord(
                source="mock", material_id="mp-1", formula_pretty="A",
                elements=("X",), band_gap_ev=2.0, density_g_cm3=4.0,
            ),
            MaterialRecord(
                source="mock", material_id="mp-2", formula_pretty="B",
                elements=("X",), band_gap_ev=2.5, density_g_cm3=3.5,
            ),
            MaterialRecord(
                source="mock", material_id="mp-3", formula_pretty="C",
                elements=("X",), band_gap_ev=1.5, density_g_cm3=4.5,
            ),
        ], ())
        tool = DetectMultivariateOutliersTool(resolver)
        ctx = _context()
        ref = MaterialSetReference(workflow_thread_ids=("run-1",))

        report = tool.execute(
            arguments=tool.input_model(
                source=ref,
                properties=("band_gap_ev", "density_g_cm3"),
            ),
            context=ctx,
        )
        assert ctx.ledger.evidence_ids() == (report.evidence_id,)
        assert ctx.ledger.executed_tool_names() == (
            "detect_multivariate_outliers",
        )

    def test_tool_metadata(self) -> None:
        tool = DetectMultivariateOutliersTool(Mock())
        assert tool.name == "detect_multivariate_outliers"
        assert tool.side_effect == ToolSideEffect.READ_ONLY
        assert len(tool.description) > 0

    def test_isolation_forest_method(self) -> None:
        resolver = Mock()
        resolver.resolve_with_warnings.return_value = ([
            _rec("mp-1", "A"),
            _rec("mp-2", "B"),
            _rec("mp-3", "C"),
            _rec("mp-4", "D"),
        ], ())
        tool = DetectMultivariateOutliersTool(resolver)
        ref = MaterialSetReference(workflow_thread_ids=("run-1",))

        report = tool.execute(
            arguments=tool.input_model(
                source=ref,
                properties=("band_gap_ev", "density_g_cm3"),
                method="isolation_forest",
            ),
            context=_context(),
        )
        assert report.method == "isolation_forest"
