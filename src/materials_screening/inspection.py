"""Inspect installed mp-api signatures and fields.

Offline by default: no Materials Project request is made unless MP_API_KEY is set.
"""

import inspect
import os
from importlib.metadata import version

from emmet.core.summary import SummaryDoc
from emmet.core.symmetry import CrystalSystem
from mp_api.client import MPRester
from mp_api.client.routes.materials.summary import SummaryRester

QUERY_PARAMS_TO_CHECK: tuple[str, ...] = (
    "band_gap",
    "chemsys",
    "crystal_system",
    "density",
    "deprecated",
    "elements",
    "energy_above_hull",
    "exclude_elements",
    "formula",
    "is_metal",
    "is_stable",
    "spacegroup_number",
    "theoretical",
    "all_fields",
    "fields",
    "chunk_size",
)

SUMMARY_FIELDS_TO_CHECK: tuple[str, ...] = (
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


def print_versions() -> None:
    """Print installed package versions."""
    print("mp-api:", version("mp-api"))
    print("pymatgen:", version("pymatgen"))
    print("emmet-core:", version("emmet-core"))


def print_search_signature() -> None:
    """Print the SummaryRester.search signature without creating a client."""
    print("SummaryRester.search signature:")
    print(inspect.signature(SummaryRester.search))


def print_query_params() -> None:
    """Check that required query parameters exist in the search signature."""
    parameters = inspect.signature(SummaryRester.search).parameters
    for name in QUERY_PARAMS_TO_CHECK:
        status = "present" if name in parameters else "MISSING"
        print(f"query param {name}: {status}")


def print_summary_fields() -> None:
    """Check required fields against SummaryDoc and the available_fields whitelist."""
    model_fields = set(SummaryDoc.model_fields)
    available_fields = set(
        SummaryRester.document_model.model_json_schema()["properties"]
    )
    for name in SUMMARY_FIELDS_TO_CHECK:
        status = (
            "present"
            if name in model_fields and name in available_fields
            else "MISSING"
        )
        print(f"SummaryDoc field {name}: {status}")


def print_crystal_system_values() -> None:
    """Print the crystal system values accepted by mp-api."""
    print("emmet CrystalSystem values:", [crystal.value for crystal in CrystalSystem])


def print_offline_available_fields() -> None:
    """Print available fields computed offline from the document model."""
    properties = SummaryRester.document_model.model_json_schema()["properties"]
    print("available_fields count (offline):", len(properties))


def print_remote_checks() -> None:
    """Print database version and live available fields when a key is present."""
    api_key = os.getenv("MP_API_KEY", "").strip()
    if not api_key:
        print("MP_API_KEY not set; skipping remote MPRester checks.")
        return
    with MPRester(api_key=api_key) as mpr:
        print("database version:", mpr.db_version)
        print("live available_fields:", mpr.materials.summary.available_fields)


def main() -> None:
    """Run all inspection steps."""
    print_versions()
    print_search_signature()
    print_query_params()
    print_summary_fields()
    print_crystal_system_values()
    print_offline_available_fields()
    print_remote_checks()
