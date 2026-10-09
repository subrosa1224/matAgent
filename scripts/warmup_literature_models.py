"""Download, load, and smoke-test LiteratureAgent embedding models."""

from __future__ import annotations

import json
import math
import os

from dotenv import load_dotenv

from materials_screening.sub_agents.literature.rag import (
    BgeM3EmbeddingProvider,
    FlagEmbeddingReranker,
)


def main() -> None:
    load_dotenv()
    device = os.getenv("LITERATURE_MODEL_DEVICE", "cpu")
    print("loading bge-m3...", flush=True)
    embeddings = BgeM3EmbeddingProvider(
        revision=os.getenv("LITERATURE_EMBEDDING_REVISION") or None,
        device=device,
    )
    vector = embeddings.embed_query("titanium dioxide photocatalysis")
    print(f"embedding_dimensions={len(vector)}", flush=True)

    print("loading reranker...", flush=True)
    reranker = FlagEmbeddingReranker(device=device)
    scores = reranker.rerank(
        "titanium dioxide photocatalysis",
        (
            "TiO2 photocatalytic hydrogen production",
            "unrelated steel mechanics",
        ),
    )
    result = {
        "dimensions": len(vector),
        "finite": all(math.isfinite(value) for value in vector),
        "rerank_scores": scores,
        "relevant_ranked_first": scores[0] > scores[1],
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["dimensions"] != 1024 or not result["finite"]:
        raise SystemExit("bge-m3 smoke test failed")
    if len(scores) != 2 or not result["relevant_ranked_first"]:
        raise SystemExit("reranker smoke test failed")


if __name__ == "__main__":
    main()
