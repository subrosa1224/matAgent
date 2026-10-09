"""Stage a small evidence-bound review set from the three batch PDFs."""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

from materials_screening.sub_agents.literature.extraction import (
    ExperimentalExtractionService,
)
from materials_screening.sub_agents.literature.models import ExperimentalDataCandidate
from materials_screening.sub_agents.literature.pgvector_store import (
    PgVectorLiteratureStore,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord


def _chunk_containing(
    store: PgVectorLiteratureStore, document_id: str, phrase: str
) -> ChunkRecord:
    for chunk in store.get_document_chunks(document_id):
        if phrase.casefold() in chunk.text.casefold():
            return chunk
    raise RuntimeError(f"evidence phrase not found: {phrase}")


def _quote(chunk: ChunkRecord, start: str, end: str) -> str:
    text = chunk.text
    lower = text.casefold()
    start_index = lower.find(start.casefold())
    end_index = lower.find(end.casefold(), start_index)
    if start_index < 0 or end_index < 0:
        raise RuntimeError(f"quote anchors not found in {chunk.chunk_id}")
    return text[start_index : end_index + len(end)]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    database_url = os.getenv("LITERATURE_DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("LITERATURE_DATABASE_URL is required")
    store = PgVectorLiteratureStore(database_url)
    service = ExperimentalExtractionService(store)
    outputs = []

    perovskite_id = "doc-cb4c833fc0948254786473b9"
    perovskite = _chunk_containing(store, perovskite_id, "maximum power-conversion")
    perovskite_quote = _quote(
        perovskite,
        "we fabricate perovskite solar cells",
        "1.18 V",
    )
    outputs.append(
        service.validate_and_save(
            document_id=perovskite_id,
            candidates=(
                ExperimentalDataCandidate(
                    chunk_id=perovskite.chunk_id,
                    material="mixed-cation mixed-halide perovskite solar cell",
                    variable_name="PbI2/FAI molar ratio",
                    variable_value="1.05",
                    performance_metric="power-conversion efficiency",
                    performance_value="20.8%",
                    conditions="precursor solution",
                    source_quote=perovskite_quote,
                ),
                ExperimentalDataCandidate(
                    chunk_id=perovskite.chunk_id,
                    material="PbI2-enriched perovskite film",
                    variable_name="excess PbI2 content",
                    variable_value="about 3 wt %",
                    performance_metric=(
                        "external electroluminescence quantum efficiency"
                    ),
                    performance_value="about 0.5%",
                    conditions="AM 1.5 sunlight; open-circuit photovoltage 1.18 V",
                    source_quote=perovskite_quote,
                ),
            ),
        )
    )

    battery_id = "doc-fa6c9a1bfa870189168c3e29"
    battery_temp = _chunk_containing(store, battery_id, "at least 20 C")
    battery_temp_quote = _quote(
        battery_temp,
        "Here, we report a general low-temperature",
        "at least 20 C",
    )
    battery_porosity = _chunk_containing(store, battery_id, "coating porosity (~20%)")
    battery_porosity_quote = _quote(
        battery_porosity,
        "The porosity of the coating",
        "at a 5-mm radius",
    )
    outputs.append(
        service.validate_and_save(
            document_id=battery_id,
            candidates=(
                ExperimentalDataCandidate(
                    chunk_id=battery_temp.chunk_id,
                    material="electroplated LiCoO2 film",
                    variable_name="electrodeposition temperature",
                    variable_value="~260°C",
                    performance_metric="high-rate discharge",
                    performance_value="at least 20 C",
                    conditions="~25-mm-thick, ~80% dense LiCoO2 film on Al foil",
                    source_quote=battery_temp_quote,
                ),
                ExperimentalDataCandidate(
                    chunk_id=battery_porosity.chunk_id,
                    material="electroplated LiCoO2 electrode",
                    variable_name="coating porosity",
                    variable_value="~20%",
                    performance_metric="rolling radius",
                    performance_value="5-mm radius",
                    conditions="~70-mm-thick electrode",
                    source_quote=battery_porosity_quote,
                ),
            ),
        )
    )

    titania_id = "doc-3250ce3cd68ac48d95b866ae"
    titania = _chunk_containing(store, titania_id, "Nb 0 mol %")
    titania_quote = _quote(titania, "Table 2.", "Nb 5.0 mol %\n4.4\n0.36\n56\n0.87")
    outputs.append(
        service.validate_and_save(
            document_id=titania_id,
            candidates=(
                ExperimentalDataCandidate(
                    chunk_id=titania.chunk_id,
                    material="Nb-doped TiO2/TCNQ photoelectrode",
                    variable_name="Nb doping",
                    variable_value="1.5 mol %",
                    performance_metric="photoconversion efficiency",
                    performance_value="1.3",
                    conditions="AM 1.5 G, 100 mW/cm2; efficiency unit %",
                    source_quote=titania_quote,
                ),
                ExperimentalDataCandidate(
                    chunk_id=titania.chunk_id,
                    material="Nb-doped TiO2/TCNQ photoelectrode",
                    variable_name="Nb doping",
                    variable_value="5.0 mol %",
                    performance_metric="photoconversion efficiency",
                    performance_value="0.87",
                    conditions="AM 1.5 G, 100 mW/cm2; efficiency unit %",
                    source_quote=titania_quote,
                ),
            ),
        )
    )
    print(
        json.dumps(
            [output.model_dump(mode="json") for output in outputs],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
