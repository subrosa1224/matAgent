"""Final candidate validation before export (M5)."""

from materials_screening.models import ScreeningResult, ValidationReport
from materials_screening.repositories.materials_project import PROVENANCE_PROPERTIES
from materials_screening.services.filter_service import FilterService
from materials_screening.services.ranking_service import RankingService


class ValidationService:
    """Re-validate final ranked results without modifying them."""

    def __init__(self) -> None:
        self._filter_service = FilterService()
        self._ranking_service = RankingService()

    def validate(self, result: ScreeningResult) -> ValidationReport:
        """Return a report with classified errors and warnings."""
        errors: list[str] = []
        warnings: list[str] = []
        request = result.request
        ranked = result.ranked_materials
        records = tuple(item.record for item in ranked)

        passed_records, trace = self._filter_service.apply(records, request)
        if len(passed_records) != len(ranked):
            for rejection in trace.rejections:
                errors.append(
                    f"candidate {rejection.material_id} violates hard "
                    f"constraints: {', '.join(rejection.reasons)}"
                )

        seen_ids: set[str] = set()
        for item in ranked:
            material_id = item.record.material_id
            if material_id in seen_ids:
                errors.append(f"duplicate material id in ranked results: {material_id}")
            seen_ids.add(material_id)

        for index, item in enumerate(ranked, start=1):
            material_id = item.record.material_id
            if item.rank != index:
                errors.append(
                    f"rank sequence broken at position {index}: "
                    f"expected {index}, got {item.rank}"
                )
            if not (0.0 <= item.total_score <= 1.0):
                errors.append(
                    f"total_score out of range for {material_id}: {item.total_score}"
                )
            expected_total, expected_breakdown = self._ranking_service.score(
                item.record, request
            )
            if item.total_score != expected_total:
                errors.append(f"total_score mismatch for {material_id}")
            if item.score_breakdown != expected_breakdown:
                errors.append(f"score breakdown mismatch for {material_id}")

        for index in range(1, len(ranked)):
            previous = ranked[index - 1]
            current = ranked[index]
            if previous.total_score < current.total_score:
                errors.append(
                    f"total_score not non-increasing between rank "
                    f"{previous.rank} and {current.rank}"
                )

        for item in ranked:
            record = item.record
            present_names = {entry.property_name for entry in record.provenance}
            for name in PROVENANCE_PROPERTIES:
                if getattr(record, name) is not None and name not in present_names:
                    errors.append(
                        f"missing provenance for {name} on {record.material_id}"
                    )

        if len(ranked) > request.limit:
            errors.append(f"ranked count {len(ranked)} exceeds limit {request.limit}")

        for item in ranked:
            record = item.record
            if record.structure_dict is None:
                warnings.append(
                    f"no structure for {record.material_id}; CIF export unavailable"
                )
            if record.formation_energy_ev_atom is None:
                warnings.append(f"missing formation energy for {record.material_id}")
            if record.density_g_cm3 is None:
                warnings.append(f"missing density for {record.material_id}")
            if record.is_gap_direct is not True:
                warnings.append(f"non-direct band gap for {record.material_id}")
            if record.theoretical is True:
                warnings.append(f"theoretical material: {record.material_id}")
        if result.metadata.database_version is None:
            warnings.append("database version unavailable")

        return ValidationReport(
            passed=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
            checked_material_ids=tuple(item.record.material_id for item in ranked),
        )
