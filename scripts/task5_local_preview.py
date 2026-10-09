"""Run evidence-bound paper previews without the pgvector service."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from materials_screening.llm.factory import create_llm_provider
from materials_screening.planner.settings import Settings
from materials_screening.sub_agents.literature.batch import document_id_for_pdf
from materials_screening.sub_agents.literature.preview import (
    PaperPreviewExtractor,
    PaperPreviewStore,
    PreviewBoundaryPolicy,
)
from materials_screening.sub_agents.literature.rag import (
    PyMuPdfParser,
    parse_pdf_chunks,
)

TOPIC = (
    "任务5：研究纯、未负载、未掺杂纳米羟基磷灰石在沉淀法、水热法或溶胶-凝胶法中，"
    "Ca/P比、反应pH、温度、时间和老化条件对成核与晶体生长的影响，以及对物相、"
    "XRD晶粒尺寸、结晶度、TEM/SEM一次颗粒尺寸和形貌的影响。"
)

PAPERS = {
    "s41598-021-91064-y.pdf": (
        "Pure hydroxyapatite synthesis originating from amorphous calcium carbonate"
    ),
    "1-s2.0-S0272884221025645-main.pdf": "Bioinspired nano-HA",
    "J_S_Earl_2006_J._Phys.__Conf._Ser._26_268.pdf": (
        "Hydrothermal synthesis of hydroxyapatite"
    ),
    "msa20110200003_19975477.pdf": (
        "Effect of Hydroxide Ion Concentration on the Morphology of the "
        "Hydroxyapatite Nanorods Synthesized Using Electrophoretic Deposition"
    ),
    "cg801353n.pdf": (
        "Multiform hydroxyapatite nano- and microcrystals by hydrothermal synthesis"
    ),
    (
        "Bioinorganic Chemistry and Applications - 2022 - Szterner - The Synthesis "
        "of Hydroxyapatite by Hydrothermal Process with.pdf"
    ): (
        "The Synthesis of Hydroxyapatite by Hydrothermal Process with Different "
        "Reagent Concentrations"
    ),
}


def main() -> None:
    load_dotenv()
    settings = Settings()
    extraction_model = os.getenv(
        "LITERATURE_EXTRACTION_MODEL", "intern-s1-mini"
    ).strip()
    provider = create_llm_provider(
        settings.model_copy(
            update={
                "intern_model": extraction_model or settings.intern_model,
                "intern_thinking_mode": False,
            }
        )
    )
    extractor = PaperPreviewExtractor(
        provider,
        boundary_policy=PreviewBoundaryPolicy(
            allowed_route_terms=(
                "沉淀法",
                "湿化学沉淀",
                "precipitation",
                "水热法",
                "水热",
                "hydrothermal",
                "溶胶-凝胶",
                "sol-gel",
                "sol–gel",
            ),
            excluded_material_term_groups=(
                ("微量元素", "微量矿物", "trace mineral", "trace element"),
                ("掺杂", "doped", "doping", "substituted hydroxyapatite"),
                ("负载", "loaded hydroxyapatite", "hydroxyapatite loaded with"),
                ("复合材料", "hydroxyapatite composite", "composite hydroxyapatite"),
                ("羟基磷灰石涂层", "hydroxyapatite coating", "coated hydroxyapatite"),
            ),
        ),
    )
    store = PaperPreviewStore(Path("data/literature_previews"))
    parser = PyMuPdfParser()
    source_root = Path(r"D:\Downloads")
    output: list[dict[str, object]] = []

    for index, (name, title) in enumerate(PAPERS.items(), 1):
        path = source_root / name
        document_id = document_id_for_pdf(path)
        chunks = parse_pdf_chunks(
            path,
            parser=parser,
            document_id=document_id,
        )
        print(f"[{index}/{len(PAPERS)}] {name}", flush=True)
        preview = extractor.extract(
            document_id=document_id,
            title=title,
            topic=TOPIC,
            chunks=chunks,
            max_output_tokens=min(settings.llm_max_output_tokens, 2048),
        )
        saved_path = store.save(preview)
        item = preview.model_dump(mode="json")
        item["source_pdf"] = str(path)
        item["preview_path"] = str(saved_path)
        output.append(item)
        print(
            f"  -> {preview.topic_relevance} / {preview.recommendation} / "
            f"page {preview.page_from}",
            flush=True,
        )

    output_dir = Path("outputs/task5_nha_trial")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "paper_previews.json"
    result_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved: {result_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
