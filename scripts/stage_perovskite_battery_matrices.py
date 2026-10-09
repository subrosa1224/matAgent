"""Stage evidence-bound experiment matrices for the perovskite and battery PDFs."""

from __future__ import annotations

import hashlib
import os
import sys

from dotenv import load_dotenv

from materials_screening.sub_agents.literature.claim_assessment import (
    assess_claim_against_measurement,
)
from materials_screening.sub_agents.literature.comparison import calculate_comparison
from materials_screening.sub_agents.literature.matrix import (
    validate_claim_evidence,
    validate_matrix_evidence,
)
from materials_screening.sub_agents.literature.models import (
    ClaimEvidenceLink,
    ExperimentalGroup,
    ExperimentalMeasurement,
    PaperClaim,
)
from materials_screening.sub_agents.literature.pgvector_store import (
    PgVectorLiteratureStore,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord


def _id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{hashlib.sha256('|'.join(parts).encode()).hexdigest()[:24]}"


def _chunk(
    store: PgVectorLiteratureStore, document_id: str, phrase: str
) -> ChunkRecord:
    for item in store.get_document_chunks(document_id):
        if phrase.lower() in item.text.lower():
            return item
    raise RuntimeError(f"missing evidence phrase: {phrase}")


def _quote(chunk: ChunkRecord, start: str, end: str) -> str:
    lower = chunk.text.lower()
    first = lower.find(start.lower())
    last = lower.find(end.lower(), first)
    if first < 0 or last < 0:
        raise RuntimeError(f"missing quote anchors in {chunk.chunk_id}")
    return chunk.text[first : last + len(end)]


def _group(
    document_id: str,
    label: str,
    role: str,
    material: str,
    variables: dict[str, str],
    conditions: dict[str, str],
    quote: str,
    chunk: ChunkRecord,
) -> ExperimentalGroup:
    return ExperimentalGroup.model_validate(
        {
            "group_id": _id("group", document_id, label),
            "document_id": document_id,
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
    value: str,
    unit: str | None,
    quote: str,
    chunk: ChunkRecord,
) -> ExperimentalMeasurement:
    return ExperimentalMeasurement(
        measurement_id=_id("measurement", group.group_id, metric, value, quote),
        group_id=group.group_id,
        document_id=group.document_id,
        metric=metric,
        value_text=value,
        numeric_value=float(value.replace(",", "")),
        unit=unit,
        source_quote=quote,
        chunk_id=chunk.chunk_id,
        page_from=chunk.page_from,
        page_to=chunk.page_to,
        source_text_sha256=chunk.text_sha256,
        llm_extracted=False,
        extraction_method="manual",
    )


def stage_perovskite(store: PgVectorLiteratureStore) -> tuple[int, int, int, int]:
    document_id = "doc-cb4c833fc0948254786473b9"
    abstract_chunk = _chunk(store, document_id, "maximum power-conversion")
    result_chunk = _chunk(store, document_id, "We achieve the highest PCE")
    lifetime_chunk = _chunk(store, document_id, "We deduced 220 and 350 ns")
    abstract_quote = _quote(
        abstract_chunk, "we fabricate perovskite solar cells", "1.18 V"
    )
    result_quote = _quote(result_chunk, "We achieve the highest PCE", "Fig. 1A")
    lifetime_quote = _quote(lifetime_chunk, "The photoluminescence", "trend in VOC")
    common = {"device": "perovskite solar cell"}
    g105 = _group(
        document_id,
        "PbI2/FAI 1.05",
        "treatment",
        "mixed-cation mixed-halide perovskite",
        {"PbI2/FAI molar ratio": "1.05"},
        common,
        abstract_quote,
        abstract_chunk,
    )
    g100 = _group(
        document_id,
        "PbI2/FAI 1.00",
        "control",
        "mixed-cation mixed-halide perovskite",
        {"PbI2/FAI molar ratio": "1"},
        common,
        lifetime_quote,
        lifetime_chunk,
    )
    g116 = _group(
        document_id,
        "PbI2/FAI 1.16",
        "treatment",
        "mixed-cation mixed-halide perovskite",
        {"PbI2/FAI molar ratio": "1:16"},
        common,
        lifetime_quote,
        lifetime_chunk,
    )
    groups = (g100, g105, g116)
    measurements = (
        _measurement(g105, "PCE", "20.8", "%", abstract_quote, abstract_chunk),
        _measurement(g105, "Jsc", "24.6", "mA cm-2", result_quote, result_chunk),
        _measurement(g105, "Voc", "1.16", "V", result_quote, result_chunk),
        _measurement(g105, "FF", "0.73", None, result_quote, result_chunk),
        _measurement(g105, "excess PbI2", "3", "wt %", abstract_quote, abstract_chunk),
        _measurement(
            g116, "nonradiative lifetime", "350", "ns", lifetime_quote, lifetime_chunk
        ),
        _measurement(
            g100, "nonradiative lifetime", "220", "ns", lifetime_quote, lifetime_chunk
        ),
    )
    validate_matrix_evidence(
        store, document_id=document_id, groups=groups, measurements=measurements
    )
    store.save_experiment_matrix(groups, measurements)
    comparison = calculate_comparison(measurements[6], measurements[5])
    store.save_comparisons((comparison,))
    claim_text = _quote(
        abstract_chunk,
        "we fabricate perovskite solar cells",
        "precursor solution",
    )
    claim = PaperClaim(
        claim_id=_id("claim", document_id, claim_text),
        document_id=document_id,
        claim_text=claim_text,
        source_section="abstract",
        source_quote=claim_text,
        chunk_id=abstract_chunk.chunk_id,
        page_from=1,
        page_to=1,
        source_text_sha256=abstract_chunk.text_sha256,
    )
    validate_claim_evidence(store, claim)
    link = assess_claim_against_measurement(claim, measurements[0])
    store.save_claims_and_links((claim,), (link,))
    return len(groups), len(measurements), 1, 1


def stage_battery(store: PgVectorLiteratureStore) -> tuple[int, int, int, int]:
    document_id = "doc-fa6c9a1bfa870189168c3e29"
    intro_chunk = _chunk(store, document_id, "at least 20 C")
    pouch_chunk = _chunk(store, document_id, "retains 75%")
    flex_chunk = _chunk(store, document_id, "70% capacity retention")
    intro_quote = _quote(intro_chunk, "Here, we report", "at least 20 C")
    pouch_quote = _quote(pouch_chunk, "The cell retains", "after 350 cycles")
    foam_quote = _quote(flex_chunk, "~0.5-mm-thick", "at 2 C")
    bending_quote = _quote(flex_chunk, "Bulk LiCoO2 is brittle", "10,000 cycles")
    g_planar = _group(
        document_id,
        "planar electroplated LiCoO2",
        "treatment",
        "LiCoO2 film on Al foil",
        {"electrodeposition temperature": "~260°C"},
        {},
        intro_quote,
        intro_chunk,
    )
    g_pouch = _group(
        document_id,
        "LiCoO2/Al foil pouch cell",
        "treatment",
        "LiCoO2/Al foil cathode with graphite anode",
        {},
        {"baseline rate": "0.5 C"},
        pouch_quote,
        pouch_chunk,
    )
    g_foam = _group(
        document_id,
        "thick LiCoO2 carbon foam electrode",
        "treatment",
        "LiCoO2/carbon foam electrode",
        {},
        {},
        foam_quote,
        flex_chunk,
    )
    g_flexible = _group(
        document_id,
        "flexible LiCoO2/CNF full cell",
        "treatment",
        "LiCoO2/CNF cathode with graphitized CNF anode",
        {},
        {"bending radius": "~5 mm", "bending angle": "180 degrees"},
        bending_quote,
        flex_chunk,
    )
    groups = (g_planar, g_pouch, g_foam, g_flexible)
    measurements = (
        _measurement(
            g_planar, "maximum discharge rate", "20", "C", intro_quote, intro_chunk
        ),
        _measurement(
            g_pouch, "capacity retention at 10 C", "75", "%", pouch_quote, pouch_chunk
        ),
        _measurement(
            g_pouch, "capacity retention at 20 C", "55", "%", pouch_quote, pouch_chunk
        ),
        _measurement(
            g_pouch,
            "capacity retention after 350 cycles",
            "80",
            "%",
            pouch_quote,
            pouch_chunk,
        ),
        _measurement(
            g_foam,
            "areal capacity at C/5 to C/20",
            "20",
            "mA hour cm-2",
            foam_quote,
            flex_chunk,
        ),
        _measurement(
            g_foam,
            "areal capacity at 2 C",
            "15",
            "mA hour cm-2",
            foam_quote,
            flex_chunk,
        ),
        _measurement(
            g_flexible,
            "capacity retention after 5000 bends",
            "70",
            "%",
            bending_quote,
            flex_chunk,
        ),
        _measurement(
            g_flexible,
            "capacity retention after 10000 bends",
            "36",
            "%",
            bending_quote,
            flex_chunk,
        ),
    )
    validate_matrix_evidence(
        store, document_id=document_id, groups=groups, measurements=measurements
    )
    store.save_experiment_matrix(groups, measurements)
    claim_text = _quote(
        intro_chunk, "Here, we report a general low-temperature", "700° to 1000°C"
    )
    claim = PaperClaim(
        claim_id=_id("claim", document_id, claim_text),
        document_id=document_id,
        claim_text=claim_text,
        source_section="abstract",
        source_quote=claim_text,
        chunk_id=intro_chunk.chunk_id,
        page_from=1,
        page_to=1,
        source_text_sha256=intro_chunk.text_sha256,
    )
    validate_claim_evidence(store, claim)
    link = ClaimEvidenceLink(
        link_id=_id("link", claim.claim_id, intro_chunk.chunk_id),
        claim_id=claim.claim_id,
        document_id=document_id,
        evidence_type="text_chunk",
        evidence_id=intro_chunk.chunk_id,
        assessment="not_verifiable",
        explanation=(
            "The main text states comparability, but a complete matched conventional "
            "control table is not available in the downloaded main PDF."
        ),
        assessment_method="rule",
    )
    store.save_claims_and_links((claim,), (link,))
    return len(groups), len(measurements), 0, 1


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    database_url = os.getenv("LITERATURE_DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("LITERATURE_DATABASE_URL is required")
    store = PgVectorLiteratureStore(database_url)
    print("perovskite", stage_perovskite(store))
    print("battery", stage_battery(store))


if __name__ == "__main__":
    main()
