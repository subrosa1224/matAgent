"""Run the blind P3 matrix evaluation against approved database gold records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from materials_screening.llm.factory import create_llm_provider
from materials_screening.planner.settings import Settings
from materials_screening.sub_agents.literature.matrix_automation import (
    AutomatedMatrixExtractor,
    PendingMatrixExtraction,
    _consolidate_semantic_groups,
    _extract_factorial_groups,
    _extract_factorial_measurements,
    _extract_series_groups,
    expand_measurement_group_evidence,
    sanitize_measurements,
)
from materials_screening.sub_agents.literature.matrix_evaluation import (
    evaluate_matrix,
    summarize_matrix_evaluations,
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

DOCUMENT_IDS = (
    "doc-3250ce3cd68ac48d95b866ae",
    "doc-cb4c833fc0948254786473b9",
    "doc-6be19aada0b95fe176f931ee",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", action="append", dest="document_ids")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/literature_p3_evaluation.json"),
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("data/literature_p3_candidates"),
    )
    parser.add_argument("--reuse-candidates", action="store_true")
    parser.add_argument(
        "--adjudications",
        type=Path,
        default=Path("data/literature_p3_adjudications.json"),
    )
    args = parser.parse_args()
    load_dotenv()
    settings = Settings()
    database_url = os.environ["LITERATURE_DATABASE_URL"]
    store = PgVectorLiteratureStore(database_url)
    model = os.getenv("LITERATURE_EXTRACTION_MODEL", "intern-s1-mini").strip()
    provider = create_llm_provider(
        settings.model_copy(update={"intern_model": model or settings.intern_model})
    )
    evaluations = []
    adjudications = _load_adjudications(args.adjudications)
    document_ids = tuple(args.document_ids or DOCUMENT_IDS)
    for document_id in document_ids:
        chunks = store.get_document_chunks(document_id)
        candidate_path = args.candidate_dir / f"{document_id}.json"
        if args.reuse_candidates:
            automatic = _load_candidates(candidate_path, document_id, chunks)
        else:
            automatic = AutomatedMatrixExtractor(provider, store).extract(
                document_id=document_id,
                chunks=chunks,
                max_output_tokens=settings.llm_max_output_tokens,
            )
            _save_candidates(candidate_path, automatic)
        gold_groups, gold_measurements, _, _, _ = store.load_matrix(
            document_id, status="approved"
        )
        evaluation = evaluate_matrix(
            automatic,
            document_id=document_id,
            gold_groups=gold_groups,
            gold_measurements=gold_measurements,
            adjudications=adjudications.get(document_id, {}),
        )
        evaluations.append(evaluation)
        print(json.dumps(evaluation.model_dump(mode="json"), ensure_ascii=True))
    summary = summarize_matrix_evaluations(evaluations)
    payload = summary.model_dump(mode="json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=True))
    if not summary.passed:
        raise SystemExit(1)


def _load_adjudications(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, str]] = {}
    for row in payload.get("decisions", []):
        document_id = str(row.get("document_id", "")).strip()
        measurement_id = str(row.get("measurement_id", "")).strip()
        decision = str(row.get("decision", "")).strip()
        if not document_id or not measurement_id:
            raise ValueError(
                "every P3 adjudication requires document_id and measurement_id"
            )
        if decision not in {"approved", "rejected"}:
            raise ValueError(f"invalid P3 adjudication decision: {decision}")
        result.setdefault(document_id, {})[measurement_id] = decision
    return result


def _save_candidates(path: Path, result: PendingMatrixExtraction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "groups": [item.model_dump(mode="json") for item in result.groups],
        "measurements": [item.model_dump(mode="json") for item in result.measurements],
        "comparisons": [item.model_dump(mode="json") for item in result.comparisons],
        "claims": [item.model_dump(mode="json") for item in result.claims],
        "claim_evidence_links": [
            item.model_dump(mode="json") for item in result.claim_evidence_links
        ],
        "warnings": list(result.warnings),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_candidates(
    path: Path, document_id: str, chunks: list[ChunkRecord]
) -> PendingMatrixExtraction:
    payload = json.loads(path.read_text(encoding="utf-8"))
    warnings = [str(item) for item in payload["warnings"]]
    groups = [ExperimentalGroup.model_validate(row) for row in payload["groups"]]
    groups.extend(_extract_series_groups(document_id, chunks, groups))
    groups.extend(_extract_factorial_groups(document_id, chunks, groups))
    generated_measurements = _extract_factorial_measurements(
        document_id, chunks, groups
    )
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    loaded_measurements = expand_measurement_group_evidence(
        tuple(
            ExperimentalMeasurement.model_validate(row)
            for row in payload["measurements"]
        )
        + generated_measurements,
        groups=groups,
        chunks_by_id=chunks_by_id,
    )
    groups, consolidated_measurements = _consolidate_semantic_groups(
        groups,
        loaded_measurements,
    )
    measurements = sanitize_measurements(
        consolidated_measurements,
        groups=groups,
        warnings=warnings,
    )
    return PendingMatrixExtraction(
        groups=tuple(groups),
        measurements=measurements,
        comparisons=tuple(
            ExperimentalComparison.model_validate(row) for row in payload["comparisons"]
        ),
        claims=tuple(PaperClaim.model_validate(row) for row in payload["claims"]),
        claim_evidence_links=tuple(
            ClaimEvidenceLink.model_validate(row)
            for row in payload["claim_evidence_links"]
        ),
        warnings=tuple(warnings),
    )


if __name__ == "__main__":
    main()
