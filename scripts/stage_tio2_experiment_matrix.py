"""Stage the complete TiO2 Table 1/2 experiment matrix and abstract claim."""

from __future__ import annotations

import hashlib
import os
import sys

from dotenv import load_dotenv

from materials_screening.sub_agents.literature.claim_assessment import (
    assess_claim_against_comparison,
)
from materials_screening.sub_agents.literature.comparison import calculate_comparison
from materials_screening.sub_agents.literature.matrix import (
    validate_claim_evidence,
    validate_matrix_evidence,
)
from materials_screening.sub_agents.literature.models import (
    ExperimentalGroup,
    ExperimentalMeasurement,
    PaperClaim,
)
from materials_screening.sub_agents.literature.pgvector_store import (
    PgVectorLiteratureStore,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord

DOCUMENT_ID = "doc-3250ce3cd68ac48d95b866ae"


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _find_chunk(store: PgVectorLiteratureStore, phrase: str) -> ChunkRecord:
    for chunk in store.get_document_chunks(DOCUMENT_ID):
        if phrase.lower() in chunk.text.lower():
            return chunk
    raise RuntimeError(f"missing evidence: {phrase}")


def _quote(chunk: ChunkRecord, start: str, end: str) -> str:
    lower = chunk.text.lower()
    first = lower.find(start.lower())
    last = lower.find(end.lower(), first)
    if first < 0 or last < 0:
        raise RuntimeError(f"missing quote anchors in {chunk.chunk_id}")
    return chunk.text[first : last + len(end)]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    database_url = os.getenv("LITERATURE_DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("LITERATURE_DATABASE_URL is required")
    store = PgVectorLiteratureStore(database_url)
    table1_chunk = _find_chunk(store, "Table 1. Optical bandgap")
    table2_chunk = _find_chunk(store, "Nb 0 mol %")
    abstract_chunk = _find_chunk(store, "The enhancement in electron")
    table1_quote = _quote(table1_chunk, "Table 1.", "0.46\n0.37\n0.41\n0.64")
    table2_quote = _quote(table2_chunk, "Table 2.", "4.4\n0.36\n56\n0.87")
    dopings = ("0 mol %", "1.5 mol %", "3.0 mol %", "5.0 mol %")
    groups = tuple(
        ExperimentalGroup(
            group_id=_id("group", DOCUMENT_ID, doping),
            document_id=DOCUMENT_ID,
            label=f"Nb {doping}",
            role="control" if doping == "0 mol %" else "treatment",
            material="Nb-doped TiO2/TCNQ photoelectrode",
            variables={"Nb doping": doping},
            conditions={"illumination": "AM 1.5 G, 100 mW/cm2"},
            source_quote=table2_quote,
            chunk_id=table2_chunk.chunk_id,
            page_from=table2_chunk.page_from,
            page_to=table2_chunk.page_to,
            source_text_sha256=table2_chunk.text_sha256,
            llm_extracted=False,
            extraction_method="table_parser",
        )
        for doping in dopings
    )
    table1 = {
        "0 mol %": (("optical bandgap", "3.06", "eV"), ("ECBM-EF", "0.46", "eV")),
        "1.5 mol %": (("optical bandgap", "3.07", "eV"), ("ECBM-EF", "0.37", "eV")),
        "3.0 mol %": (("optical bandgap", "3.11", "eV"), ("ECBM-EF", "0.41", "eV")),
        "5.0 mol %": (("optical bandgap", "3.14", "eV"), ("ECBM-EF", "0.64", "eV")),
    }
    table2 = {
        "0 mol %": (
            ("Jsc", "4.5", "mA/cm2"),
            ("Voc", "0.40", "V"),
            ("FF", "63", "%"),
            ("photoconversion efficiency", "1.1", "%"),
        ),
        "1.5 mol %": (
            ("Jsc", "5.5", "mA/cm2"),
            ("Voc", "0.41", "V"),
            ("FF", "59", "%"),
            ("photoconversion efficiency", "1.3", "%"),
        ),
        "3.0 mol %": (
            ("Jsc", "5.7", "mA/cm2"),
            ("Voc", "0.38", "V"),
            ("FF", "57", "%"),
            ("photoconversion efficiency", "1.2", "%"),
        ),
        "5.0 mol %": (
            ("Jsc", "4.4", "mA/cm2"),
            ("Voc", "0.36", "V"),
            ("FF", "56", "%"),
            ("photoconversion efficiency", "0.87", "%"),
        ),
    }
    group_by_doping = dict(zip(dopings, groups, strict=True))
    measurements: list[ExperimentalMeasurement] = []
    for doping, values in (*table1.items(), *table2.items()):
        chunk = (
            table1_chunk
            if doping in table1 and values is table1[doping]
            else table2_chunk
        )
        quote = table1_quote if chunk is table1_chunk else table2_quote
        for metric, value, unit in values:
            group = group_by_doping[doping]
            measurements.append(
                ExperimentalMeasurement(
                    measurement_id=_id("measurement", group.group_id, metric, value),
                    group_id=group.group_id,
                    document_id=DOCUMENT_ID,
                    metric=metric,
                    value_text=value,
                    numeric_value=float(value),
                    unit=unit,
                    source_quote=quote,
                    chunk_id=chunk.chunk_id,
                    page_from=chunk.page_from,
                    page_to=chunk.page_to,
                    source_text_sha256=chunk.text_sha256,
                    llm_extracted=False,
                    extraction_method="table_parser",
                )
            )
    validate_matrix_evidence(
        store,
        document_id=DOCUMENT_ID,
        groups=groups,
        measurements=measurements,
    )
    store.save_experiment_matrix(groups, measurements)
    by_group_metric = {(row.group_id, row.metric): row for row in measurements}
    baseline = group_by_doping["0 mol %"]
    comparisons = []
    for target in groups[1:]:
        for metric in (
            "optical bandgap",
            "ECBM-EF",
            "Jsc",
            "Voc",
            "FF",
            "photoconversion efficiency",
        ):
            comparisons.append(
                calculate_comparison(
                    by_group_metric[(baseline.group_id, metric)],
                    by_group_metric[(target.group_id, metric)],
                )
            )
    store.save_comparisons(comparisons)
    claim_text = _quote(
        abstract_chunk,
        "The enhancement",
        "undoped TiO2",
    )
    claim = PaperClaim(
        claim_id=_id("claim", DOCUMENT_ID, claim_text),
        document_id=DOCUMENT_ID,
        claim_text=claim_text,
        source_section="abstract",
        source_quote=claim_text,
        chunk_id=abstract_chunk.chunk_id,
        page_from=abstract_chunk.page_from,
        page_to=abstract_chunk.page_to,
        source_text_sha256=abstract_chunk.text_sha256,
    )
    validate_claim_evidence(store, claim)
    efficiency_comparison = next(
        row
        for row in comparisons
        if row.target_group_id == group_by_doping["1.5 mol %"].group_id
        and row.metric == "photoconversion efficiency"
    )
    link = assess_claim_against_comparison(claim, efficiency_comparison)
    store.save_claims_and_links((claim,), (link,))
    print(
        f"groups={len(groups)} measurements={len(measurements)} "
        f"comparisons={len(comparisons)} claims=1 links=1"
    )


if __name__ == "__main__":
    main()
