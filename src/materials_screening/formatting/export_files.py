"""Shared export file builders used by stage 1 and the workflow adapter.

The public ``ExportService`` keeps its behavior; these pure builders are
reused by the workflow exporter so output formats stay identical.
"""

import csv
import io
import json
import os
from pathlib import Path
from typing import Any

from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter

from materials_screening.errors import ExportError
from materials_screening.models import RankedMaterial, ScreeningResult

_CIF_ALLOWED_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
)

CSV_COLUMNS: tuple[str, ...] = (
    "rank",
    "material_id",
    "formula_pretty",
    "elements",
    "chemsys",
    "band_gap_ev",
    "energy_above_hull_ev_atom",
    "formation_energy_ev_atom",
    "density_g_cm3",
    "crystal_system",
    "spacegroup_symbol",
    "spacegroup_number",
    "is_metal",
    "is_gap_direct",
    "is_stable",
    "theoretical",
    "total_score",
    "stability_score",
    "band_gap_match_score",
    "completeness_score",
    "direct_gap_score",
    "source",
    "database_version",
)


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text atomically via a same-directory temp file plus replace."""
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        with temp_path.open("w", encoding=encoding, newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def json_text(value: Any) -> str:
    """Serialize JSON-compatible data with deterministic key ordering."""
    return json.dumps(value, indent=2, ensure_ascii=False)


def fmt(value: object) -> str:
    return "" if value is None else str(value)


def build_provenance_document(result: ScreeningResult) -> dict[str, object]:
    materials: list[dict[str, object]] = []
    for item in result.ranked_materials:
        record = item.record
        materials.append(
            {
                "material_id": record.material_id,
                "source": record.source,
                "database_version": result.metadata.database_version,
                "provenance": [
                    entry.model_dump(mode="json") for entry in record.provenance
                ],
            }
        )
    return {"materials": materials}


def csv_row(item: RankedMaterial, database_version: str | None) -> list[str]:
    record = item.record
    symmetry = record.symmetry
    crystal_system = ""
    if symmetry is not None and symmetry.crystal_system is not None:
        crystal_system = symmetry.crystal_system.value
    spacegroup_symbol = symmetry.symbol if symmetry is not None else None
    spacegroup_number = (
        str(symmetry.number)
        if symmetry is not None and symmetry.number is not None
        else ""
    )
    return [
        str(item.rank),
        record.material_id,
        record.formula_pretty,
        " ".join(record.elements),
        fmt(record.chemsys),
        fmt(record.band_gap_ev),
        fmt(record.energy_above_hull_ev_atom),
        fmt(record.formation_energy_ev_atom),
        fmt(record.density_g_cm3),
        crystal_system,
        fmt(spacegroup_symbol),
        spacegroup_number,
        fmt(record.is_metal),
        fmt(record.is_gap_direct),
        fmt(record.is_stable),
        fmt(record.theoretical),
        str(item.total_score),
        str(item.score_breakdown.stability),
        str(item.score_breakdown.band_gap_match),
        str(item.score_breakdown.completeness),
        str(item.score_breakdown.direct_gap),
        record.source,
        database_version or "",
    ]


def build_csv(result: ScreeningResult) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(CSV_COLUMNS)
    database_version = result.metadata.database_version
    for item in result.ranked_materials:
        writer.writerow(csv_row(item, database_version))
    return buffer.getvalue()


def safe_cif_name(material_id: str) -> str:
    cleaned = "".join(ch for ch in material_id if ch in _CIF_ALLOWED_CHARS)
    if not cleaned:
        raise ExportError(f"material_id {material_id!r} cannot form a CIF filename")
    return cleaned


def write_cif(structure_dict: dict[str, Any], path: Path) -> None:
    try:
        structure = Structure.from_dict(structure_dict)
        cif_text = str(CifWriter(structure))
    except Exception as exc:
        raise ExportError(f"failed to convert structure to CIF: {exc}") from exc
    atomic_write_text(path, cif_text)


def write_cif_files(
    result: ScreeningResult,
    cif_dir: Path,
    warnings: list[str],
) -> list[Path]:
    """Write CIF files for structured candidates; missing structure warns."""
    cif_dir.mkdir(parents=True, exist_ok=True)
    cif_root = cif_dir.resolve()
    files: list[Path] = []
    used_names: set[str] = set()
    for item in result.ranked_materials:
        record = item.record
        if record.structure_dict is None:
            warnings.append(f"no structure for {record.material_id}; CIF skipped")
            continue
        cleaned = safe_cif_name(record.material_id)
        if cleaned in used_names:
            raise ExportError(f"CIF filename collision for {record.material_id!r}")
        used_names.add(cleaned)
        cif_path = cif_dir / f"{cleaned}.cif"
        if cif_root not in cif_path.resolve().parents:
            raise ExportError(f"unsafe CIF path for {record.material_id!r}")
        write_cif(record.structure_dict, cif_path)
        files.append(cif_path)
    return files
