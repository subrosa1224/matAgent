"""Read-only provider replay; never changes production queries or filtering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from materials_screening.sub_agents.literature.models import (
    ExpandedQuery,
    LiteratureSearchInput,
    PaperRecord,
)
from materials_screening.sub_agents.literature.providers import (
    LiteratureProviderError,
    OpenAlexProvider,
    SemanticScholarProvider,
    UrllibJsonTransport,
)
from materials_screening.sub_agents.literature.service import (
    _merge_papers,
    _paper_mentions_formula,
    _relevance,
    _term_in,
)


def describe(
    paper: PaperRecord, request: LiteratureSearchInput, expanded: ExpandedQuery
) -> dict:
    relevance = _relevance(paper, request, expanded)
    text = f"{paper.title} {paper.abstract or ''}".casefold()
    return {
        "paper": paper.model_dump(mode="json"),
        "level": relevance.level,
        "score": relevance.score,
        "matched_terms": relevance.matched_terms,
        "missing_concepts": relevance.missing_concepts,
        "literal_formula_matches": [
            formula
            for formula in expanded.normalized_materials
            if _term_in(formula.casefold(), text)
        ],
        "composition_matches": [
            formula
            for formula in expanded.normalized_materials
            if _paper_mentions_formula(paper, formula)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument(
        "--provider", choices=("all", "openalex", "semantic_scholar"), default="all"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/literature_zero_diagnostic_2026-10-05.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output already exists; choose a new diagnostic path.")
    load_dotenv()
    saved = json.loads(
        Path(
            "data/literature_queries/lit-ac1ebe2234a4b9ccab9877af/result.json"
        ).read_text(encoding="utf-8")
    )
    expanded = ExpandedQuery.model_validate(saved["expanded_query"])
    thread = "agent_sub_literature_" + hashlib.sha256(b"433b").hexdigest()[:24]
    with sqlite3.connect(
        "file:data/literature_checkpoints.sqlite?mode=ro", uri=True
    ) as connection:
        checkpoint = SqliteSaver(connection).get_tuple(
            {"configurable": {"thread_id": thread}}
        )
        assert checkpoint is not None
        calls = [
            item
            for item in checkpoint.checkpoint["channel_values"]["input_items"]
            if item.get("type") == "function_call"
            and item.get("name") == "literature_search"
        ]
        assert len(calls) == 1
        request = LiteratureSearchInput.model_validate_json(calls[0]["arguments"])
    assert request.topic == expanded.original_topic
    transport = UrllibJsonTransport(timeout_seconds=15.0, max_retries=0)
    providers = (
        OpenAlexProvider(transport=transport, mailto=os.getenv("OPENALEX_MAILTO")),
        SemanticScholarProvider(
            transport=transport, api_key=os.getenv("S2_API_KEY") or None
        ),
    )
    if args.provider != "all":
        providers = tuple(
            provider for provider in providers if provider.name == args.provider
        )
    # Two existing, previously unqueried candidates. Not added to production.
    controls = ("DyVO4", "BaCrO4")
    assert set(controls).issubset(expanded.normalized_materials)
    queries = [("original", query) for query in expanded.search_queries] + [
        ("coverage_control", f"{formula} photocatalysis visible light")
        for formula in controls
    ]
    runs = []
    all_original: list[PaperRecord] = []
    for provider in providers:
        for kind, query in queries:
            bounded = request.model_copy(
                update={
                    "topic": query,
                    "material_keywords": (),
                    "max_papers": min(max(request.max_papers * 3, 30), 100),
                }
            )
            try:
                papers = provider.search(bounded)
            except LiteratureProviderError as exc:
                runs.append(
                    {
                        "provider": provider.name,
                        "kind": kind,
                        "query": query,
                        "status": exc.code,
                        "records": [],
                    }
                )
                print(provider.name, query, exc.code, flush=True)
                # Do not keep hitting a blocked provider; remaining queries
                # remain unattempted, not "no matching papers".
                break
            records = [describe(paper, request, expanded) for paper in papers]
            if kind == "original":
                all_original.extend(papers)
            selected = sum(record["level"] is not None for record in records)
            runs.append(
                {
                    "provider": provider.name,
                    "kind": kind,
                    "query": query,
                    "status": "ok",
                    "returned": len(papers),
                    "accepted": selected,
                    "records": records,
                }
            )
            print(
                provider.name,
                query,
                "returned",
                len(papers),
                "accepted",
                selected,
                flush=True,
            )
    merged = _merge_papers(all_original)
    audit = [describe(paper, request, expanded) for paper in merged]
    summary = {
        "replay_source_rows": len(all_original),
        "replay_unique_rows": len(merged),
        "replay_accepted": sum(row["level"] is not None for row in audit),
        "replay_with_abstract": sum(bool(paper.abstract) for paper in merged),
        "replay_literal_formula_hits": sum(
            bool(row["literal_formula_matches"]) for row in audit
        ),
        "replay_composition_hits": sum(
            bool(row["composition_matches"]) for row in audit
        ),
        "replay_rejection_concepts": dict(
            Counter(
                concept
                for row in audit
                if row["level"] is None
                for concept in row["missing_concepts"]
            )
        ),
    }
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "fresh provider replay, not recovered historical raw records",
        "credential_configuration": "production environment variable names",
        "historical_query_id": saved["query_id"],
        "request": request.model_dump(mode="json"),
        "expanded_query": expanded.model_dump(mode="json"),
        "planned_queries": queries,
        "summary": summary,
        "runs": runs,
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("Diagnostic saved:", args.output, flush=True)


if __name__ == "__main__":
    main()
