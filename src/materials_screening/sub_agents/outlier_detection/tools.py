"""Outlier Detection sub-agent tools — 2 whitelisted tools.

Both follow the AgentTool Protocol and use MaterialSetResolver → Service.
"""

from __future__ import annotations

import re

from materials_screening.agent.context import AgentToolContext
from materials_screening.agent.tool_base import ToolSideEffect
from materials_screening.services.material_set_resolver import MaterialSetResolver
from materials_screening.services.outlier_detection_service import (
    detect_multivariate_outliers,
    detect_property_outliers,
)
from materials_screening.sub_agents.outlier_detection.models import (
    ALLOWED_PROPERTIES,
    DetectMultivariateOutliersInput,
    DetectPropertyOutliersInput,
    MaterialSetReference,
    MultivariateOutlierReport,
    PropertyOutlierReport,
    RunOutlierDetectionInput,
)

_FORMULA_PATTERN = re.compile(r"\b(?:[A-Z][a-z]?\d*){1,8}\b")
_MATERIAL_ID_PATTERN = re.compile(r"\bmp-[A-Za-z0-9]+\b", re.IGNORECASE)
_DATA_FILE_PATTERN = re.compile(
    r"data[\\/][^\s\"'<>|]*?\.(?:csv|json)\b",
    re.IGNORECASE,
)
_WORKFLOW_THREAD_PATTERN = re.compile(
    r"(?:workflow[ _-]?thread|thread[ _-]?id|thread|"
    r"工作流(?:线程)?(?!\s*thread))"
    r"\s*(?:为|是|[:：=#])?\s*([A-Za-z0-9][A-Za-z0-9._-]{0,253})",
    re.IGNORECASE,
)
_UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_PROPERTY_MARKERS = {
    "band_gap_ev": ("band gap", "band_gap", "带隙", "禁带"),
    "density_g_cm3": ("density", "密度"),
    "formation_energy_ev_atom": ("formation energy", "formation_energy", "形成能"),
    "energy_above_hull_ev_atom": ("energy above hull", "above hull", "hull", "凸包"),
}


class RunOutlierDetectionTool:
    """Single simple entry point, matching the screening agent call pattern."""

    name = "run_outlier_detection"
    description = (
        "Run outlier detection for materials named in a natural-language query. "
        "The query must name at least two formulas and the property or properties "
        "to analyse, for example: 'compare TiO2, ZnO and SiO2 band gaps and find "
        "outliers'. Without material ids every database entry/polymorph for each "
        "formula is included; when mp-... ids are present only those exact entries "
        "are included. The tool resolves materials and chooses univariate or "
        "multivariate detection safely."
    )
    input_model = RunOutlierDetectionInput
    output_model = PropertyOutlierReport | MultivariateOutlierReport
    side_effect = ToolSideEffect.READ_ONLY

    def __init__(self, resolver: MaterialSetResolver) -> None:
        self._resolver = resolver

    def execute(
        self,
        arguments: RunOutlierDetectionInput,
        context: AgentToolContext,
    ) -> PropertyOutlierReport | MultivariateOutlierReport:
        query = arguments.query
        lowered = query.lower()
        data_files = tuple(dict.fromkeys(_DATA_FILE_PATTERN.findall(query)))
        if len(data_files) > 1:
            raise ValueError("outlier detection accepts only one data file")
        workflow_thread_ids = tuple(
            dict.fromkeys(
                [
                    *_WORKFLOW_THREAD_PATTERN.findall(query),
                    *_UUID_PATTERN.findall(query),
                ]
            )
        )
        material_ids = tuple(
            dict.fromkeys(
                match.lower() for match in _MATERIAL_ID_PATTERN.findall(query)
            )
        )
        formulas = tuple(dict.fromkeys(_FORMULA_PATTERN.findall(query)))
        references = material_ids or formulas
        if not data_files and not workflow_thread_ids and len(references) < 2:
            raise ValueError(
                "outlier detection requires at least two material ids or formulas"
            )
        properties = [
            property_name
            for property_name, markers in _PROPERTY_MARKERS.items()
            if any(marker in lowered for marker in markers)
        ]
        request_warnings: tuple[str, ...] = ()
        if not properties:
            properties = ["band_gap_ev"]
            request_warnings = (
                "No supported property was named; defaulted to band_gap_ev.",
            )
        records, resolver_warnings = self._resolver.resolve_with_warnings(
            MaterialSetReference(
                data_file=data_files[0] if data_files else None,
                workflow_thread_ids=(
                    workflow_thread_ids
                    if workflow_thread_ids and not data_files
                    else None
                ),
                material_ids=(
                    material_ids
                    if material_ids and not data_files and not workflow_thread_ids
                    else None
                ),
                material_formulas=(
                    formulas
                    if not material_ids and not data_files and not workflow_thread_ids
                    else None
                ),
            )
        )
        evidence_id = context.id_generator.new_id()
        if len(properties) == 1:
            report = detect_property_outliers(
                records=records,
                property_name=properties[0],
                # Interactive comparisons commonly contain only a handful of
                # materials; use a more sensitive exploratory cutoff and keep
                # the service's small-sample warning in the report.
                method="zscore",
                threshold=1.5,
            )
        else:
            report = detect_multivariate_outliers(
                records=records,
                properties=tuple(properties),
                method="mahalanobis",
            )
        report = report.model_copy(
            update={
                "evidence_id": evidence_id,
                "warnings": (
                    *request_warnings,
                    *resolver_warnings,
                    *report.warnings,
                ),
            }
        )
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=evidence_id,
            result_json=report.model_dump_json(),
            side_effect=self.side_effect,
        )
        return report


class DetectPropertyOutliersTool:
    """Detect property outliers using Z-score or IQR method."""

    name = "detect_property_outliers"
    description = (
        "Detect outlier materials for a single property using Z-score "
        "(|z| > threshold) or IQR (value outside Q1 - threshold×IQR to "
        "Q3 + threshold×IQR). Supported properties: band_gap_ev, "
        "formation_energy_ev_atom, energy_above_hull_ev_atom, density_g_cm3. "
        "Materials can be specified by workflow_thread_ids, material_formulas, or "
        "a CSV/JSON data file. "
        "Returns distribution statistics and per-material outlier records "
        "with z_score, is_outlier flag, and direction (high/low)."
    )
    input_model = DetectPropertyOutliersInput
    output_model = PropertyOutlierReport
    side_effect = ToolSideEffect.READ_ONLY

    def __init__(self, resolver: MaterialSetResolver) -> None:
        self._resolver = resolver

    def execute(
        self,
        arguments: DetectPropertyOutliersInput,
        context: AgentToolContext,
    ) -> PropertyOutlierReport:
        evidence_id = context.id_generator.new_id()

        if arguments.property not in ALLOWED_PROPERTIES:
            raise ValueError(
                f"Property {arguments.property!r} not in allowed set: "
                f"{sorted(ALLOWED_PROPERTIES)}"
            )

        records, resolver_warnings = self._resolver.resolve_with_warnings(
            arguments.source
        )
        report = detect_property_outliers(
            records=records,
            property_name=arguments.property,
            method=arguments.method,
            threshold=arguments.threshold,
        )
        report = report.model_copy(
            update={
                "evidence_id": evidence_id,
                "warnings": (*resolver_warnings, *report.warnings),
            }
        )
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=evidence_id,
            result_json=report.model_dump_json(),
            side_effect=self.side_effect,
        )
        return report


class DetectMultivariateOutliersTool:
    """Detect multivariate outliers using Mahalanobis distance or Isolation Forest."""

    name = "detect_multivariate_outliers"
    description = (
        "Detect multivariate outlier materials using Mahalanobis distance "
        "(chi-squared test, p < 0.05) or Isolation Forest (deterministic, "
        "fixed seed). Requires 2–4 properties from the whitelist: "
        "band_gap_ev, formation_energy_ev_atom, energy_above_hull_ev_atom, "
        "density_g_cm3. Materials can be specified by workflow_thread_ids, "
        "material_formulas, or a CSV/JSON data file. "
        "Returns per-material anomaly scores, abnormal property lists, and "
        "human-readable outlier reasons."
    )
    input_model = DetectMultivariateOutliersInput
    output_model = MultivariateOutlierReport
    side_effect = ToolSideEffect.READ_ONLY

    def __init__(self, resolver: MaterialSetResolver) -> None:
        self._resolver = resolver

    def execute(
        self,
        arguments: DetectMultivariateOutliersInput,
        context: AgentToolContext,
    ) -> MultivariateOutlierReport:
        evidence_id = context.id_generator.new_id()

        for prop in arguments.properties:
            if prop not in ALLOWED_PROPERTIES:
                raise ValueError(
                    f"Property {prop!r} not in allowed set: "
                    f"{sorted(ALLOWED_PROPERTIES)}"
                )

        records, resolver_warnings = self._resolver.resolve_with_warnings(
            arguments.source
        )
        report = detect_multivariate_outliers(
            records=records,
            properties=arguments.properties,
            method=arguments.method,
        )
        report = report.model_copy(
            update={
                "evidence_id": evidence_id,
                "warnings": (*resolver_warnings, *report.warnings),
            }
        )
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=evidence_id,
            result_json=report.model_dump_json(),
            side_effect=self.side_effect,
        )
        return report
