"""Stage the reviewed main-text matrix for the 2025 CaP bioceramic paper."""

from __future__ import annotations

import hashlib
import os
import sys

from dotenv import load_dotenv

from materials_screening.sub_agents.literature.matrix import (
    validate_claim_evidence,
    validate_matrix_evidence,
)
from materials_screening.sub_agents.literature.models import (
    ClaimEvidenceLink,
    ExperimentalComparison,
    ExperimentalGroup,
    ExperimentalMeasurement,
    PaperClaim,
)
from materials_screening.sub_agents.literature.pgvector_store import (
    PgVectorLiteratureStore,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord

DOCUMENT_ID = "doc-6be19aada0b95fe176f931ee"
GEOMETRIES = ("triangular", "diamond", "square", "polyhedral")
POROSITIES = ("50 %", "55 %", "60 %", "65 %", "70 %", "75 %")


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _find_chunk(store: PgVectorLiteratureStore, phrase: str) -> ChunkRecord:
    for chunk in store.get_document_chunks(DOCUMENT_ID):
        if phrase.casefold() in chunk.text.casefold():
            return chunk
    raise RuntimeError(f"missing evidence phrase: {phrase}")


def _quote(chunk: ChunkRecord, start: str, end: str) -> str:
    normalized = chunk.text.casefold()
    first = normalized.find(start.casefold())
    last = normalized.find(end.casefold(), first)
    if first < 0 or last < 0:
        raise RuntimeError(f"missing quote anchors in {chunk.chunk_id}")
    return chunk.text[first : last + len(end)]


def _group(
    *,
    label: str,
    role: str,
    material: str,
    variables: dict[str, str],
    conditions: dict[str, str],
    chunk: ChunkRecord,
    quote: str,
) -> ExperimentalGroup:
    return ExperimentalGroup.model_validate(
        {
            "group_id": _id("group", DOCUMENT_ID, label),
            "document_id": DOCUMENT_ID,
            "label": label,
            "role": role,
            "material": material,
            "variables": variables,
            "conditions": conditions,
            "source_quote": quote,
            "chunk_id": chunk.chunk_id,
            "page_from": chunk.page_from,
            "page_to": chunk.page_to,
            "source_text_sha256": chunk.text_sha256,
            "llm_extracted": False,
            "extraction_method": "manual",
        }
    )


def _measurement(
    group: ExperimentalGroup,
    metric: str,
    value_text: str,
    *,
    chunk: ChunkRecord,
    unit: str | None = None,
    numeric_value: float | None = None,
    uncertainty_text: str | None = None,
) -> ExperimentalMeasurement:
    return ExperimentalMeasurement(
        measurement_id=_id(
            "measurement", group.group_id, metric, value_text, chunk.chunk_id
        ),
        group_id=group.group_id,
        document_id=DOCUMENT_ID,
        metric=metric,
        value_text=value_text,
        numeric_value=numeric_value,
        unit=unit,
        uncertainty_text=uncertainty_text,
        source_quote=chunk.text,
        chunk_id=chunk.chunk_id,
        page_from=chunk.page_from,
        page_to=chunk.page_to,
        source_text_sha256=chunk.text_sha256,
        llm_extracted=False,
        extraction_method="manual",
    )


def _reported_comparison(
    baseline: ExperimentalMeasurement,
    target: ExperimentalMeasurement,
    *,
    direction: str,
    reported_text: str,
) -> ExperimentalComparison:
    return ExperimentalComparison.model_validate(
        {
            "comparison_id": _id(
                "comparison",
                baseline.measurement_id,
                target.measurement_id,
                reported_text,
            ),
            "document_id": DOCUMENT_ID,
            "baseline_group_id": baseline.group_id,
            "target_group_id": target.group_id,
            "metric": target.metric,
            "baseline_measurement_id": baseline.measurement_id,
            "target_measurement_id": target.measurement_id,
            "direction": direction,
            "provenance_type": "reported",
            "reported_text": reported_text,
        }
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    database_url = os.getenv("LITERATURE_DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("LITERATURE_DATABASE_URL is required")
    store = PgVectorLiteratureStore(database_url)

    abstract_chunk = _find_chunk(store, "Results indicate that bone regenerative")
    design_chunk = _find_chunk(store, "each featuring six porosity levels")
    table_chunk = _find_chunk(store, "Characteristic parameters of 3D printed")
    day14_chunk = _find_chunk(store, "culture, in diamond and polyhedral")
    in_vivo_chunk = _find_chunk(store, "largest amount of new bone")
    nonlinear_chunk = _find_chunk(store, "platform appeared at the position of SSA")
    integrity_chunk = _find_chunk(store, "remained largely intact after 180 days")
    conclusion_chunk = _find_chunk(store, "Experimental screening revealed")

    design_quote = _quote(
        design_chunk,
        "Two types of integrated ceramic scaffolds",
        "as shown in Fig. 1",
    )
    common_conditions = {
        "material": "DLP-printed calcium phosphate ceramic",
        "in vitro platform": "CaP Chip",
        "in vivo platform": "CaP Cyl-scaffold; 180 days",
        "replicates": "at least 3",
    }
    groups = tuple(
        _group(
            label=f"{geometry} {porosity}",
            role="treatment",
            material="calcium phosphate ceramic scaffold",
            variables={
                "pore geometry": geometry,
                "porosity series": "50 % to 75 %",
            },
            conditions={**common_conditions, "selected porosity": porosity},
            chunk=design_chunk,
            quote=design_quote,
        )
        for geometry in GEOMETRIES
        for porosity in POROSITIES
    )
    aggregate = _group(
        label="all 24 scaffold configurations",
        role="reference",
        material="calcium phosphate ceramic scaffold screening cohort",
        variables={"analysis scope": "24 structures"},
        conditions={"analysis": "PCC and XGBoost nonlinear fitting"},
        chunk=design_chunk,
        quote=design_quote,
    )
    groups = (*groups, aggregate)
    group_by_key = {
        (
            group.variables.get("pore geometry"),
            group.conditions.get("selected porosity"),
        ): group
        for group in groups
        if group.role == "treatment"
    }

    measurements: list[ExperimentalMeasurement] = [
        _measurement(
            aggregate, "screened porosity range", "50–75 %", chunk=table_chunk
        ),
        _measurement(
            aggregate,
            "model specific surface area range",
            "7.29–13.38",
            chunk=table_chunk,
            unit="mm2 mm−3",
        ),
        _measurement(
            aggregate,
            "permeability range",
            "1.15–5.67",
            chunk=table_chunk,
            unit="× 10−9 m2",
        ),
        _measurement(
            aggregate,
            "PCC between porosity and BV/TV",
            "0.541",
            chunk=in_vivo_chunk,
            numeric_value=0.541,
            uncertainty_text="P < 0.05",
        ),
        _measurement(
            aggregate,
            "SSA plateau associated with BV/TV",
            "9.66–11.73",
            chunk=nonlinear_chunk,
            unit="mm2 mm−3",
        ),
        _measurement(
            aggregate,
            "SSA plateau for COL-I expression",
            "9.91 and 10.69",
            chunk=nonlinear_chunk,
            unit="mm2 mm−3",
        ),
        _measurement(
            aggregate,
            "SSA plateau for OCN expression",
            "10.49 and 10.69",
            chunk=nonlinear_chunk,
            unit="mm2 mm−3",
        ),
        _measurement(
            aggregate,
            "SSA plateau for OPN expression",
            "9.91 and 10.69",
            chunk=nonlinear_chunk,
            unit="mm2 mm−3",
        ),
        _measurement(
            aggregate,
            "SSA plateau for Runx-2 expression",
            "9.34 and 9.72",
            chunk=nonlinear_chunk,
            unit="mm2 mm−3",
        ),
        _measurement(
            aggregate,
            "SSA at ALP expression peak",
            "9.817",
            chunk=nonlinear_chunk,
            unit="mm2 mm−3",
            numeric_value=9.817,
        ),
        _measurement(
            aggregate,
            "permeability at COL-I/OPN/OCN expression peak",
            "3.74 × 10−9",
            chunk=nonlinear_chunk,
            unit="m2",
            numeric_value=3.74e-9,
        ),
    ]

    def add(
        geometry: str,
        porosity: str,
        metric: str,
        value: str,
        chunk: ChunkRecord,
    ) -> ExperimentalMeasurement:
        row = _measurement(
            group_by_key[(geometry, porosity)], metric, value, chunk=chunk
        )
        measurements.append(row)
        return row

    for geometry, porosity in (
        ("diamond", "60 %"),
        ("polyhedral", "50 %"),
        ("polyhedral", "55 %"),
    ):
        add(
            geometry,
            porosity,
            "BMSC proliferation at day 7",
            "Higher proliferation rate",
            table_chunk,
        )
    for geometry, porosity in (
        ("diamond", "70 %"),
        ("diamond", "75 %"),
        ("polyhedral", "70 %"),
        ("polyhedral", "75 %"),
    ):
        add(
            geometry,
            porosity,
            "BMSC proliferation at day 14",
            "higher than that of others with the same structure",
            day14_chunk,
        )
    for geometry, porosity in (("polyhedral", "65 %"), ("square", "70 %")):
        add(
            geometry,
            porosity,
            "COL-I and OPN expression at day 7",
            "elevated",
            day14_chunk,
        )
    for geometry in GEOMETRIES:
        add(
            geometry,
            "75 %",
            "BV/TV analysis status",
            "not analyzed and calculated",
            day14_chunk,
        )

    poly65 = add(
        "polyhedral",
        "65 %",
        "in vivo osteoinductivity (micro-CT)",
        "excellent osteoinductive properties",
        in_vivo_chunk,
    )
    poly70 = add(
        "polyhedral",
        "70 %",
        "in vivo osteoinductivity (micro-CT)",
        "largest amount of new bone",
        in_vivo_chunk,
    )
    for geometry, porosity in (("triangular", "65 %"), ("square", "70 %")):
        add(
            geometry,
            porosity,
            "in vivo osteoinductivity (micro-CT)",
            "excellent osteoinductive properties",
            in_vivo_chunk,
        )
    add(
        "triangular",
        "70 %",
        "new bone area in histological sections",
        "largest new bone area",
        in_vivo_chunk,
    )

    integrity_pairs: list[tuple[ExperimentalMeasurement, ExperimentalMeasurement]] = []
    for geometry in GEOMETRIES:
        intact = add(
            geometry,
            "70 %",
            "structural integrity after 180 days",
            "remained largely intact",
            integrity_chunk,
        )
        disintegrated = add(
            geometry,
            "75 %",
            "structural integrity after 180 days",
            "extensive degradation",
            in_vivo_chunk,
        )
        integrity_pairs.append((intact, disintegrated))

    validate_matrix_evidence(
        store,
        document_id=DOCUMENT_ID,
        groups=groups,
        measurements=measurements,
    )

    comparisons = [
        _reported_comparison(
            poly65,
            poly70,
            direction="increase",
            reported_text=(
                "The polyhedral structure with 70 % porosity had the largest amount "
                "of new bone."
            ),
        )
    ]
    comparisons.extend(
        _reported_comparison(
            intact,
            disintegrated,
            direction="decrease",
            reported_text=(
                "Structures with 50 %–70 % porosity remained largely intact after "
                "180 days, whereas 75 % porosity showed significant disintegration."
            ),
        )
        for intact, disintegrated in integrity_pairs
    )

    claim_text = _quote(
        abstract_chunk,
        "Results indicate that bone regenerative ability",
        "approximately 65 %–70 % \nporosity",
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
    conclusion_quote = _quote(
        conclusion_chunk,
        "Experimental screening revealed",
        "performed well in new bone \nformation in vivo",
    )
    if not conclusion_quote:
        raise RuntimeError("missing conclusion evidence")
    link = ClaimEvidenceLink(
        link_id=_id("link", claim.claim_id, conclusion_chunk.chunk_id),
        claim_id=claim.claim_id,
        document_id=DOCUMENT_ID,
        evidence_type="text_chunk",
        evidence_id=conclusion_chunk.chunk_id,
        assessment="supported",
        explanation=(
            "The conclusion independently restates the 70 % polyhedral result, the "
            "porosity/SSA finding, and the optimal SSA/permeability values."
        ),
        assessment_method="rule",
    )

    store.save_experiment_matrix(groups, measurements)
    store.save_comparisons(comparisons)
    store.save_claims_and_links((claim,), (link,))
    print(
        f"groups={len(groups)} measurements={len(measurements)} "
        f"comparisons={len(comparisons)} claims=1 links=1"
    )


if __name__ == "__main__":
    main()
