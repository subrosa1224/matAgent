# Crystalline Inorganic Benchmark V1

This directory contains the frozen RA-0 benchmark. It is intentionally separate
from production query stores and must not be rewritten by application code.

## Assets

- `database_snapshots/`: a compact 300-record Materials Project oxide slice and
  immutable source metadata;
- `screening_cases/`: 15 deterministic filter/ranking task definitions;
- `expected/`: independently generated reference results and the pre-RA-1 baseline;
- `identity_cases/`: 20 exact-ID, polymorph, and chemical-system boundary pairs;
- `literature_evidence/index.json`: hashes and review references for five approved seed dossiers;
- `literature_evidence/candidates.json`: three task-specific open-full-text leads that remain pending local ingestion and domain review;
- `end_to_end_tasks/`: three project-level task drafts.

## Trust status

The database rows retain their Materials Project database version and source query
IDs. The deterministic expectations are silver labels pending a second-person
check. Ten polymorph labels and all project-level answer keys require domain review.
The benchmark therefore remains `draft_pending_domain_review` and is not yet a
released scientific gold standard.

## Rebuild and validate

```powershell
uv run python scripts/build_ra0_benchmark.py
uv run python scripts/evaluate_ra0_current_baseline.py
uv run pytest tests/unit/research/test_ra0_benchmark.py -q
```

The builder uses only existing local snapshots. It does not call an external API.
Running it twice must produce the same benchmark asset hashes.
