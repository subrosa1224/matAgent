"""PostgreSQL/pgvector implementation of the LiteratureAgent vector store."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from .models import (
    ClaimEvidenceLink,
    ExperimentalComparison,
    ExperimentalDataRow,
    ExperimentalFactReview,
    ExperimentalGroup,
    ExperimentalMeasurement,
    KnowledgeEdge,
    LiteratureDocumentMetadata,
    MatrixReviewSummary,
    PaperClaim,
)
from .rag import ChunkRecord, RagConfigurationError, VectorHit

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS literature_documents (
    document_id text PRIMARY KEY,
    file_sha256 text NOT NULL UNIQUE,
    file_name text NOT NULL,
    paper_id text,
    embedding_model text NOT NULL,
    title text,
    doi text,
    publication_year integer,
    created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE literature_documents ADD COLUMN IF NOT EXISTS title text;
ALTER TABLE literature_documents ADD COLUMN IF NOT EXISTS doi text;
ALTER TABLE literature_documents ADD COLUMN IF NOT EXISTS publication_year integer;
CREATE TABLE IF NOT EXISTS literature_chunks (
    chunk_id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES literature_documents(document_id),
    paper_id text,
    page_from integer NOT NULL CHECK (page_from >= 1),
    page_to integer NOT NULL CHECK (page_to >= page_from),
    text_content text NOT NULL,
    text_sha256 text NOT NULL,
    embedding vector(1024) NOT NULL
);
CREATE INDEX IF NOT EXISTS literature_chunks_embedding_hnsw
ON literature_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS literature_chunks_paper_id
ON literature_chunks (paper_id);
CREATE TABLE IF NOT EXISTS literature_experimental_facts (
    fact_id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES literature_documents(document_id),
    paper_id text,
    chunk_id text NOT NULL REFERENCES literature_chunks(chunk_id),
    page_from integer NOT NULL CHECK (page_from >= 1),
    page_to integer NOT NULL CHECK (page_to >= page_from),
    material text NOT NULL,
    variable_name text NOT NULL,
    variable_value text NOT NULL,
    performance_metric text NOT NULL,
    performance_value text NOT NULL,
    conditions text,
    source_quote text NOT NULL,
    source_text_sha256 text NOT NULL,
    llm_extracted boolean NOT NULL DEFAULT true,
    extraction_method text NOT NULL DEFAULT 'llm'
        CHECK (extraction_method IN ('llm', 'table_parser', 'manual')),
    review_status text NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected')),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS literature_facts_document_id
ON literature_experimental_facts (document_id);
CREATE INDEX IF NOT EXISTS literature_facts_review_status
ON literature_experimental_facts (review_status);
CREATE TABLE IF NOT EXISTS literature_extraction_reviews (
    review_id text PRIMARY KEY,
    fact_id text NOT NULL REFERENCES literature_experimental_facts(fact_id),
    previous_status text NOT NULL
        CHECK (previous_status IN ('pending', 'approved', 'rejected')),
    decision text NOT NULL CHECK (decision IN ('approved', 'rejected')),
    reviewer text NOT NULL,
    reason text,
    reviewed_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS literature_reviews_fact_id
ON literature_extraction_reviews (fact_id);
CREATE TABLE IF NOT EXISTS literature_knowledge_edges (
    edge_id text PRIMARY KEY,
    fact_id text NOT NULL UNIQUE REFERENCES literature_experimental_facts(fact_id),
    document_id text NOT NULL REFERENCES literature_documents(document_id),
    paper_id text,
    subject text NOT NULL,
    predicate text NOT NULL,
    object_value text NOT NULL,
    variable_name text NOT NULL,
    variable_value text NOT NULL,
    performance_metric text NOT NULL,
    performance_value text NOT NULL,
    conditions text,
    chunk_id text NOT NULL REFERENCES literature_chunks(chunk_id),
    page_from integer NOT NULL CHECK (page_from >= 1),
    page_to integer NOT NULL CHECK (page_to >= page_from),
    source_quote text NOT NULL,
    source_text_sha256 text NOT NULL,
    review_status text NOT NULL DEFAULT 'approved'
        CHECK (review_status = 'approved'),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS literature_edges_document_id
ON literature_knowledge_edges (document_id);
CREATE TABLE IF NOT EXISTS literature_experimental_groups (
    group_id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES literature_documents(document_id),
    label text NOT NULL,
    role text NOT NULL CHECK (role IN ('control', 'treatment', 'reference', 'unknown')),
    material text NOT NULL,
    variables jsonb NOT NULL DEFAULT '{}'::jsonb,
    conditions jsonb NOT NULL DEFAULT '{}'::jsonb,
    source_quote text NOT NULL,
    chunk_id text NOT NULL REFERENCES literature_chunks(chunk_id),
    page_from integer NOT NULL CHECK (page_from >= 1),
    page_to integer NOT NULL CHECK (page_to >= page_from),
    source_text_sha256 text NOT NULL,
    llm_extracted boolean NOT NULL DEFAULT true,
    extraction_method text NOT NULL DEFAULT 'llm'
        CHECK (extraction_method IN ('llm', 'table_parser', 'manual')),
    review_status text NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected')),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS literature_groups_document_id
ON literature_experimental_groups (document_id);
CREATE TABLE IF NOT EXISTS literature_measurements (
    measurement_id text PRIMARY KEY,
    group_id text NOT NULL REFERENCES literature_experimental_groups(group_id),
    document_id text NOT NULL REFERENCES literature_documents(document_id),
    metric text NOT NULL,
    value_text text NOT NULL,
    numeric_value double precision,
    unit text,
    uncertainty_text text,
    sample_size integer CHECK (sample_size IS NULL OR sample_size >= 1),
    source_quote text NOT NULL,
    chunk_id text NOT NULL REFERENCES literature_chunks(chunk_id),
    page_from integer NOT NULL CHECK (page_from >= 1),
    page_to integer NOT NULL CHECK (page_to >= page_from),
    source_text_sha256 text NOT NULL,
    llm_extracted boolean NOT NULL DEFAULT true,
    review_status text NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected')),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (group_id, metric, value_text, chunk_id)
);
CREATE INDEX IF NOT EXISTS literature_measurements_group_id
ON literature_measurements (group_id);
CREATE TABLE IF NOT EXISTS literature_comparisons (
    comparison_id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES literature_documents(document_id),
    baseline_group_id text NOT NULL REFERENCES literature_experimental_groups(group_id),
    target_group_id text NOT NULL REFERENCES literature_experimental_groups(group_id),
    metric text NOT NULL,
    baseline_measurement_id text REFERENCES literature_measurements(measurement_id),
    target_measurement_id text REFERENCES literature_measurements(measurement_id),
    baseline_value double precision,
    target_value double precision,
    absolute_change double precision,
    relative_change_percent double precision,
    direction text NOT NULL
        CHECK (direction IN ('increase', 'decrease', 'unchanged', 'not_computable')),
    provenance_type text NOT NULL CHECK (provenance_type IN ('reported', 'calculated')),
    reported_text text,
    unit text,
    review_status text NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected')),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS literature_comparisons_document_id
ON literature_comparisons (document_id);
CREATE TABLE IF NOT EXISTS literature_paper_claims (
    claim_id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES literature_documents(document_id),
    claim_text text NOT NULL,
    source_section text NOT NULL
        CHECK (source_section IN ('abstract', 'conclusion', 'results', 'discussion')),
    source_quote text NOT NULL,
    chunk_id text NOT NULL REFERENCES literature_chunks(chunk_id),
    page_from integer NOT NULL CHECK (page_from >= 1),
    page_to integer NOT NULL CHECK (page_to >= page_from),
    source_text_sha256 text NOT NULL,
    llm_extracted boolean NOT NULL DEFAULT true,
    review_status text NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected')),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS literature_claims_document_id
ON literature_paper_claims (document_id);
CREATE TABLE IF NOT EXISTS literature_claim_evidence_links (
    link_id text PRIMARY KEY,
    claim_id text NOT NULL REFERENCES literature_paper_claims(claim_id),
    document_id text NOT NULL REFERENCES literature_documents(document_id),
    evidence_type text NOT NULL
        CHECK (evidence_type IN ('group', 'measurement', 'comparison', 'text_chunk')),
    evidence_id text NOT NULL,
    assessment text NOT NULL CHECK (assessment IN (
        'supported', 'partially_supported', 'unsupported', 'contradicted',
        'not_verifiable'
    )),
    explanation text NOT NULL,
    assessment_method text NOT NULL
        CHECK (assessment_method IN ('rule', 'llm', 'human')),
    review_status text NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected')),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (claim_id, evidence_type, evidence_id, assessment_method)
);
CREATE INDEX IF NOT EXISTS literature_claim_links_claim_id
ON literature_claim_evidence_links (claim_id);
CREATE TABLE IF NOT EXISTS literature_matrix_reviews (
    review_event_id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES literature_documents(document_id),
    entity_type text NOT NULL CHECK (entity_type IN
        ('group', 'measurement', 'comparison', 'claim', 'claim_evidence_link')),
    entity_id text NOT NULL,
    previous_status text NOT NULL
        CHECK (previous_status IN ('pending', 'approved', 'rejected')),
    decision text NOT NULL CHECK (decision IN ('approved', 'rejected')),
    reviewer text NOT NULL,
    reason text,
    reviewed_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS literature_matrix_reviews_document_id
ON literature_matrix_reviews (document_id);
ALTER TABLE literature_experimental_groups
ADD COLUMN IF NOT EXISTS extraction_method text NOT NULL DEFAULT 'llm';
ALTER TABLE literature_measurements
ADD COLUMN IF NOT EXISTS extraction_method text NOT NULL DEFAULT 'llm';
"""


class PgVectorLiteratureStore:
    """Transactional vector storage; schema migration remains an explicit call."""

    def __init__(self, database_url: str, *, dimensions: int = 1024) -> None:
        if not database_url.strip():
            raise ValueError("database URL must not be blank")
        if dimensions != 1024:
            raise ValueError(
                "the LiteratureAgent pgvector schema requires 1024 dimensions"
            )
        self._database_url = database_url
        self._dimensions = dimensions
        self._host_override = (
            "127.0.0.1" if urlsplit(database_url).hostname == "localhost" else None
        )

    def migrate(self) -> None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(SCHEMA_SQL)
            connection.commit()

    def document_exists(self, document_id: str) -> bool:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM literature_documents WHERE document_id = %s",
                (document_id,),
            )
            return cursor.fetchone() is not None

    def set_document_metadata(
        self,
        *,
        document_id: str,
        title: str,
        doi: str | None,
        year: int | None,
        paper_id: str | None = None,
    ) -> LiteratureDocumentMetadata:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE literature_documents
                SET title = %s, doi = %s, publication_year = %s,
                    paper_id = COALESCE(%s, paper_id)
                WHERE document_id = %s
                RETURNING document_id, file_name, paper_id, title, doi,
                          publication_year
                """,
                (title, doi, year, paper_id, document_id),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(f"literature document not found: {document_id}")
            connection.commit()
        return LiteratureDocumentMetadata.model_validate(
            dict(
                zip(
                    ("document_id", "file_name", "paper_id", "title", "doi", "year"),
                    row,
                    strict=True,
                )
            )
        )

    def get_document_metadata(
        self, document_id: str
    ) -> LiteratureDocumentMetadata | None:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT document_id, file_name, paper_id, title, doi,
                       publication_year
                FROM literature_documents WHERE document_id = %s
                """,
                (document_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return LiteratureDocumentMetadata.model_validate(
            dict(
                zip(
                    ("document_id", "file_name", "paper_id", "title", "doi", "year"),
                    row,
                    strict=True,
                )
            )
        )

    def upsert(
        self,
        *,
        document_id: str,
        file_sha256: str,
        file_name: str,
        paper_id: str | None,
        embedding_model: str,
        chunks: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunk and embedding counts differ")
        with self._connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO literature_documents
                            (document_id, file_sha256, file_name, paper_id,
                             embedding_model)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (document_id) DO NOTHING
                        """,
                        (
                            document_id,
                            file_sha256,
                            file_name,
                            paper_id,
                            embedding_model,
                        ),
                    )
                    cursor.executemany(
                        """
                        INSERT INTO literature_chunks
                            (chunk_id, document_id, paper_id, page_from, page_to,
                             text_content, text_sha256, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (chunk_id) DO NOTHING
                        """,
                        [
                            (
                                chunk.chunk_id,
                                chunk.document_id,
                                chunk.paper_id,
                                chunk.page_from,
                                chunk.page_to,
                                chunk.text,
                                chunk.text_sha256,
                                _vector_literal(vector, self._dimensions),
                            )
                            for chunk, vector in zip(chunks, embeddings, strict=True)
                        ],
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        paper_ids: Sequence[str],
        limit: int,
    ) -> list[VectorHit]:
        vector = _vector_literal(query_embedding, self._dimensions)
        where = "WHERE paper_id = ANY(%s)" if paper_ids else ""
        parameters: tuple[Any, ...]
        if paper_ids:
            parameters = (vector, list(paper_ids), vector, limit)
        else:
            parameters = (vector, vector, limit)
        query = f"""
            SELECT chunk_id, document_id, paper_id, page_from, page_to,
                   text_content, text_sha256, embedding <=> %s::vector AS distance
            FROM literature_chunks
            {where}
            ORDER BY embedding <=> %s::vector, chunk_id
            LIMIT %s
        """
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
        return [
            VectorHit(
                chunk=ChunkRecord(
                    chunk_id=str(row[0]),
                    document_id=str(row[1]),
                    paper_id=str(row[2]) if row[2] is not None else None,
                    page_from=int(row[3]),
                    page_to=int(row[4]),
                    text=str(row[5]),
                    text_sha256=str(row[6]),
                ),
                distance=float(row[7]),
            )
            for row in rows
        ]

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[ChunkRecord]:
        if not chunk_ids:
            return []
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id, document_id, paper_id, page_from, page_to,
                       text_content, text_sha256
                FROM literature_chunks
                WHERE chunk_id = ANY(%s)
                ORDER BY chunk_id
                """,
                (list(chunk_ids),),
            )
            rows = cursor.fetchall()
        return [
            ChunkRecord(
                chunk_id=str(row[0]),
                document_id=str(row[1]),
                paper_id=str(row[2]) if row[2] is not None else None,
                page_from=int(row[3]),
                page_to=int(row[4]),
                text=str(row[5]),
                text_sha256=str(row[6]),
            )
            for row in rows
        ]

    def get_document_chunks(self, document_id: str) -> list[ChunkRecord]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id, document_id, paper_id, page_from, page_to,
                       text_content, text_sha256
                FROM literature_chunks
                WHERE document_id = %s
                ORDER BY page_from, chunk_id
                """,
                (document_id,),
            )
            rows = cursor.fetchall()
        return [
            ChunkRecord(
                chunk_id=str(row[0]),
                document_id=str(row[1]),
                paper_id=str(row[2]) if row[2] is not None else None,
                page_from=int(row[3]),
                page_to=int(row[4]),
                text=str(row[5]),
                text_sha256=str(row[6]),
            )
            for row in rows
        ]

    def save_experimental_facts(self, rows: Sequence[ExperimentalDataRow]) -> None:
        if not rows:
            return
        with self._connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.executemany(
                        """
                        INSERT INTO literature_experimental_facts
                            (fact_id, document_id, paper_id, chunk_id, page_from,
                             page_to, material, variable_name, variable_value,
                             performance_metric, performance_value, conditions,
                             source_quote, source_text_sha256, llm_extracted,
                             review_status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (fact_id) DO NOTHING
                        """,
                        [
                            (
                                row.fact_id,
                                row.document_id,
                                row.paper_id,
                                row.chunk_id,
                                row.page_from,
                                row.page_to,
                                row.material,
                                row.variable_name,
                                row.variable_value,
                                row.performance_metric,
                                row.performance_value,
                                row.conditions,
                                row.source_quote,
                                row.source_text_sha256,
                                row.llm_extracted,
                                row.review_status,
                            )
                            for row in rows
                        ],
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def save_experiment_matrix(
        self,
        groups: Sequence[ExperimentalGroup],
        measurements: Sequence[ExperimentalMeasurement],
    ) -> None:
        if not groups and not measurements:
            return
        with self._connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.executemany(
                        """
                        INSERT INTO literature_experimental_groups
                            (group_id, document_id, label, role, material,
                             variables, conditions, source_quote, chunk_id,
                             page_from, page_to, source_text_sha256,
                             llm_extracted, extraction_method, review_status)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                                %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (group_id) DO NOTHING
                        """,
                        [
                            (
                                row.group_id,
                                row.document_id,
                                row.label,
                                row.role,
                                row.material,
                                json.dumps(row.variables, ensure_ascii=False),
                                json.dumps(row.conditions, ensure_ascii=False),
                                row.source_quote,
                                row.chunk_id,
                                row.page_from,
                                row.page_to,
                                row.source_text_sha256,
                                row.llm_extracted,
                                row.extraction_method,
                                row.review_status,
                            )
                            for row in groups
                        ],
                    )
                    cursor.executemany(
                        """
                        INSERT INTO literature_measurements
                            (measurement_id, group_id, document_id, metric,
                             value_text, numeric_value, unit, uncertainty_text,
                             sample_size, source_quote, chunk_id, page_from,
                             page_to, source_text_sha256, llm_extracted,
                             extraction_method, review_status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (measurement_id) DO NOTHING
                        """,
                        [
                            (
                                row.measurement_id,
                                row.group_id,
                                row.document_id,
                                row.metric,
                                row.value_text,
                                row.numeric_value,
                                row.unit,
                                row.uncertainty_text,
                                row.sample_size,
                                row.source_quote,
                                row.chunk_id,
                                row.page_from,
                                row.page_to,
                                row.source_text_sha256,
                                row.llm_extracted,
                                row.extraction_method,
                                row.review_status,
                            )
                            for row in measurements
                        ],
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def save_comparisons(self, rows: Sequence[ExperimentalComparison]) -> None:
        if not rows:
            return
        with self._connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.executemany(
                        """
                        INSERT INTO literature_comparisons
                            (comparison_id, document_id, baseline_group_id,
                             target_group_id, metric, baseline_measurement_id,
                             target_measurement_id, baseline_value, target_value,
                             absolute_change, relative_change_percent, direction,
                             provenance_type, reported_text, unit, review_status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (comparison_id) DO NOTHING
                        """,
                        [
                            (
                                row.comparison_id,
                                row.document_id,
                                row.baseline_group_id,
                                row.target_group_id,
                                row.metric,
                                row.baseline_measurement_id,
                                row.target_measurement_id,
                                row.baseline_value,
                                row.target_value,
                                row.absolute_change,
                                row.relative_change_percent,
                                row.direction,
                                row.provenance_type,
                                row.reported_text,
                                row.unit,
                                row.review_status,
                            )
                            for row in rows
                        ],
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def save_claims_and_links(
        self,
        claims: Sequence[PaperClaim],
        links: Sequence[ClaimEvidenceLink],
    ) -> None:
        if not claims and not links:
            return
        with self._connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.executemany(
                        """
                        INSERT INTO literature_paper_claims
                            (claim_id, document_id, claim_text, source_section,
                             source_quote, chunk_id, page_from, page_to,
                             source_text_sha256, llm_extracted, review_status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (claim_id) DO NOTHING
                        """,
                        [
                            (
                                row.claim_id,
                                row.document_id,
                                row.claim_text,
                                row.source_section,
                                row.source_quote,
                                row.chunk_id,
                                row.page_from,
                                row.page_to,
                                row.source_text_sha256,
                                row.llm_extracted,
                                row.review_status,
                            )
                            for row in claims
                        ],
                    )
                    cursor.executemany(
                        """
                        INSERT INTO literature_claim_evidence_links
                            (link_id, claim_id, document_id, evidence_type,
                             evidence_id, assessment, explanation,
                             assessment_method, review_status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (link_id) DO UPDATE SET
                            assessment = EXCLUDED.assessment,
                            explanation = EXCLUDED.explanation,
                            assessment_method = EXCLUDED.assessment_method,
                            review_status = 'pending'
                        """,
                        [
                            (
                                row.link_id,
                                row.claim_id,
                                row.document_id,
                                row.evidence_type,
                                row.evidence_id,
                                row.assessment,
                                row.explanation,
                                row.assessment_method,
                                row.review_status,
                            )
                            for row in links
                        ],
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def matrix_counts(self, document_id: str) -> dict[str, int]:
        tables = {
            "groups": "literature_experimental_groups",
            "measurements": "literature_measurements",
            "comparisons": "literature_comparisons",
            "claims": "literature_paper_claims",
            "claim_evidence_links": "literature_claim_evidence_links",
        }
        counts: dict[str, int] = {}
        with self._connection() as connection, connection.cursor() as cursor:
            for label, table in tables.items():
                cursor.execute(
                    f"SELECT count(*) FROM {table} WHERE document_id = %s",
                    (document_id,),
                )
                counts[label] = int(cursor.fetchone()[0])
            cursor.execute(
                """
                SELECT count(*) FROM literature_measurements
                WHERE document_id = %s AND review_status = 'pending'
                """,
                (document_id,),
            )
            counts["pending_measurements"] = int(cursor.fetchone()[0])
            cursor.execute(
                """
                SELECT count(*) FROM literature_claim_evidence_links
                WHERE document_id = %s
                  AND assessment IN (
                      'supported', 'partially_supported', 'contradicted'
                  )
                """,
                (document_id,),
            )
            counts["verified_claim_checks"] = int(cursor.fetchone()[0])
        return counts

    def load_matrix(
        self, document_id: str, *, status: str = "approved"
    ) -> tuple[
        list[ExperimentalGroup],
        list[ExperimentalMeasurement],
        list[ExperimentalComparison],
        list[PaperClaim],
        list[ClaimEvidenceLink],
    ]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT group_id, document_id, label, role, material, variables,
                       conditions, source_quote, chunk_id, page_from, page_to,
                       source_text_sha256, llm_extracted, extraction_method,
                       review_status
                FROM literature_experimental_groups
                WHERE document_id = %s AND review_status = %s
                ORDER BY created_at, group_id
                """,
                (document_id, status),
            )
            groups = [
                ExperimentalGroup.model_validate(
                    dict(
                        zip(
                            (
                                "group_id",
                                "document_id",
                                "label",
                                "role",
                                "material",
                                "variables",
                                "conditions",
                                "source_quote",
                                "chunk_id",
                                "page_from",
                                "page_to",
                                "source_text_sha256",
                                "llm_extracted",
                                "extraction_method",
                                "review_status",
                            ),
                            row,
                            strict=True,
                        )
                    )
                )
                for row in cursor.fetchall()
            ]
            cursor.execute(
                """
                SELECT measurement_id, group_id, document_id, metric, value_text,
                       numeric_value, unit, uncertainty_text, sample_size,
                       source_quote, chunk_id, page_from, page_to, source_text_sha256,
                       llm_extracted, extraction_method, review_status
                FROM literature_measurements
                WHERE document_id = %s AND review_status = %s
                ORDER BY created_at, measurement_id
                """,
                (document_id, status),
            )
            measurements = [
                ExperimentalMeasurement.model_validate(
                    dict(
                        zip(
                            (
                                "measurement_id",
                                "group_id",
                                "document_id",
                                "metric",
                                "value_text",
                                "numeric_value",
                                "unit",
                                "uncertainty_text",
                                "sample_size",
                                "source_quote",
                                "chunk_id",
                                "page_from",
                                "page_to",
                                "source_text_sha256",
                                "llm_extracted",
                                "extraction_method",
                                "review_status",
                            ),
                            row,
                            strict=True,
                        )
                    )
                )
                for row in cursor.fetchall()
            ]
            cursor.execute(
                """
                SELECT comparison_id, document_id, baseline_group_id,
                       target_group_id, metric, baseline_measurement_id,
                       target_measurement_id, baseline_value, target_value,
                       absolute_change, relative_change_percent, direction,
                       provenance_type, reported_text, unit, review_status
                FROM literature_comparisons
                WHERE document_id = %s AND review_status = %s
                ORDER BY created_at, comparison_id
                """,
                (document_id, status),
            )
            comparisons = [
                ExperimentalComparison.model_validate(
                    dict(
                        zip(
                            (
                                "comparison_id",
                                "document_id",
                                "baseline_group_id",
                                "target_group_id",
                                "metric",
                                "baseline_measurement_id",
                                "target_measurement_id",
                                "baseline_value",
                                "target_value",
                                "absolute_change",
                                "relative_change_percent",
                                "direction",
                                "provenance_type",
                                "reported_text",
                                "unit",
                                "review_status",
                            ),
                            row,
                            strict=True,
                        )
                    )
                )
                for row in cursor.fetchall()
            ]
            cursor.execute(
                """
                SELECT claim_id, document_id, claim_text, source_section,
                       source_quote, chunk_id, page_from, page_to,
                       source_text_sha256, llm_extracted, review_status
                FROM literature_paper_claims
                WHERE document_id = %s AND review_status = %s
                ORDER BY created_at, claim_id
                """,
                (document_id, status),
            )
            claims = [
                PaperClaim.model_validate(
                    dict(
                        zip(
                            (
                                "claim_id",
                                "document_id",
                                "claim_text",
                                "source_section",
                                "source_quote",
                                "chunk_id",
                                "page_from",
                                "page_to",
                                "source_text_sha256",
                                "llm_extracted",
                                "review_status",
                            ),
                            row,
                            strict=True,
                        )
                    )
                )
                for row in cursor.fetchall()
            ]
            cursor.execute(
                """
                SELECT link_id, claim_id, document_id, evidence_type, evidence_id,
                       assessment, explanation, assessment_method, review_status
                FROM literature_claim_evidence_links
                WHERE document_id = %s AND review_status = %s
                ORDER BY created_at, link_id
                """,
                (document_id, status),
            )
            links = [
                ClaimEvidenceLink.model_validate(
                    dict(
                        zip(
                            (
                                "link_id",
                                "claim_id",
                                "document_id",
                                "evidence_type",
                                "evidence_id",
                                "assessment",
                                "explanation",
                                "assessment_method",
                                "review_status",
                            ),
                            row,
                            strict=True,
                        )
                    )
                )
                for row in cursor.fetchall()
            ]
        return groups, measurements, comparisons, claims, links

    def load_approved_matrix(
        self, document_id: str
    ) -> tuple[
        list[ExperimentalGroup],
        list[ExperimentalMeasurement],
        list[ExperimentalComparison],
        list[PaperClaim],
        list[ClaimEvidenceLink],
    ]:
        return self.load_matrix(document_id, status="approved")

    def review_document_matrix(
        self,
        *,
        document_id: str,
        decision: str,
        reviewer: str,
        reason: str | None = None,
    ) -> MatrixReviewSummary:
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValueError("reviewer must not be blank")
        entities = {
            "group": ("literature_experimental_groups", "group_id"),
            "measurement": ("literature_measurements", "measurement_id"),
            "comparison": ("literature_comparisons", "comparison_id"),
            "claim": ("literature_paper_claims", "claim_id"),
            "claim_evidence_link": (
                "literature_claim_evidence_links",
                "link_id",
            ),
        }
        event_ids: list[str] = []
        counts: dict[str, int] = {}
        reviewed_at: Any = None
        with self._connection() as connection:
            try:
                with connection.cursor() as cursor:
                    for entity_type, (table, id_column) in entities.items():
                        cursor.execute(
                            f"""
                            SELECT {id_column}, review_status FROM {table}
                            WHERE document_id = %s FOR UPDATE
                            """,
                            (document_id,),
                        )
                        rows = cursor.fetchall()
                        counts[entity_type] = len(rows)
                        for entity_id, previous_status in rows:
                            event_id = f"matrix-review-{uuid4().hex[:24]}"
                            cursor.execute(
                                """
                                INSERT INTO literature_matrix_reviews
                                    (review_event_id, document_id, entity_type,
                                     entity_id, previous_status, decision,
                                     reviewer, reason)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                RETURNING reviewed_at
                                """,
                                (
                                    event_id,
                                    document_id,
                                    entity_type,
                                    entity_id,
                                    previous_status,
                                    decision,
                                    reviewer,
                                    reason,
                                ),
                            )
                            reviewed_at = cursor.fetchone()[0]
                            event_ids.append(event_id)
                        cursor.execute(
                            f"UPDATE {table} SET review_status = %s "
                            "WHERE document_id = %s",
                            (decision, document_id),
                        )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        if not event_ids or reviewed_at is None:
            raise ValueError("document matrix has no reviewable records")
        return MatrixReviewSummary.model_validate(
            {
                "document_id": document_id,
                "decision": decision,
                "reviewer": reviewer,
                "reason": reason,
                "reviewed_counts": counts,
                "review_event_ids": event_ids,
                "reviewed_at": reviewed_at,
            }
        )

    def list_experimental_facts(
        self,
        *,
        status: str | None = None,
        document_id: str | None = None,
    ) -> list[ExperimentalDataRow]:
        clauses: list[str] = []
        parameters: list[str] = []
        if status is not None:
            clauses.append("review_status = %s")
            parameters.append(status)
        if document_id is not None:
            clauses.append("document_id = %s")
            parameters.append(document_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT fact_id, document_id, paper_id, chunk_id, page_from,
                       page_to, material, variable_name, variable_value,
                       performance_metric, performance_value, conditions,
                       source_quote, source_text_sha256, llm_extracted,
                       review_status
                FROM literature_experimental_facts
                {where}
                ORDER BY created_at, fact_id
                """,
                tuple(parameters),
            )
            rows = cursor.fetchall()
        return [_experimental_fact_from_row(row) for row in rows]

    def review_experimental_fact(
        self,
        *,
        fact_id: str,
        decision: str,
        reviewer: str,
        reason: str | None = None,
    ) -> ExperimentalFactReview:
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValueError("reviewer must not be blank")
        review_id = f"review-{uuid4().hex[:24]}"
        with self._connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT review_status
                        FROM literature_experimental_facts
                        WHERE fact_id = %s
                        FOR UPDATE
                        """,
                        (fact_id,),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise KeyError(f"experimental fact not found: {fact_id}")
                    previous_status = str(row[0])
                    cursor.execute(
                        """
                        INSERT INTO literature_extraction_reviews
                            (review_id, fact_id, previous_status, decision,
                             reviewer, reason)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING reviewed_at
                        """,
                        (
                            review_id,
                            fact_id,
                            previous_status,
                            decision,
                            reviewer,
                            reason,
                        ),
                    )
                    reviewed_at = cursor.fetchone()[0]
                    cursor.execute(
                        """
                        UPDATE literature_experimental_facts
                        SET review_status = %s
                        WHERE fact_id = %s
                        """,
                        (decision, fact_id),
                    )
                    if decision == "rejected":
                        cursor.execute(
                            "DELETE FROM literature_knowledge_edges WHERE fact_id = %s",
                            (fact_id,),
                        )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return ExperimentalFactReview.model_validate(
            {
                "review_id": review_id,
                "fact_id": fact_id,
                "previous_status": previous_status,
                "decision": decision,
                "reviewer": reviewer,
                "reason": reason,
                "reviewed_at": reviewed_at,
            }
        )

    def build_knowledge_edges(
        self, *, document_id: str | None = None
    ) -> list[KnowledgeEdge]:
        document_clause = "AND document_id = %s" if document_id else ""
        parameters: tuple[str, ...] = (document_id,) if document_id else ()
        with self._connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"""
                        SELECT fact_id, document_id, paper_id, material,
                               variable_name, variable_value, performance_metric,
                               performance_value, conditions, chunk_id, page_from,
                               page_to, source_quote, source_text_sha256
                        FROM literature_experimental_facts
                        WHERE review_status = 'approved' {document_clause}
                        ORDER BY fact_id
                        """,
                        parameters,
                    )
                    facts = cursor.fetchall()
                    for fact in facts:
                        edge_id = (
                            "edge-"
                            + hashlib.sha256(str(fact[0]).encode()).hexdigest()[:24]
                        )
                        object_value = f"{fact[4]}={fact[5]} -> {fact[6]}={fact[7]}"
                        cursor.execute(
                            """
                            INSERT INTO literature_knowledge_edges
                                (edge_id, fact_id, document_id, paper_id, subject,
                                 predicate, object_value, variable_name,
                                 variable_value, performance_metric,
                                 performance_value, conditions, chunk_id,
                                 page_from, page_to, source_quote,
                                 source_text_sha256)
                            VALUES (%s, %s, %s, %s, %s,
                                    'has_parameter_performance_relation', %s,
                                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (fact_id) DO UPDATE SET
                                subject = EXCLUDED.subject,
                                object_value = EXCLUDED.object_value,
                                variable_name = EXCLUDED.variable_name,
                                variable_value = EXCLUDED.variable_value,
                                performance_metric = EXCLUDED.performance_metric,
                                performance_value = EXCLUDED.performance_value,
                                conditions = EXCLUDED.conditions,
                                source_quote = EXCLUDED.source_quote,
                                source_text_sha256 = EXCLUDED.source_text_sha256
                            """,
                            (
                                edge_id,
                                *fact[:4],
                                object_value,
                                *fact[4:],
                            ),
                        )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.list_knowledge_edges(document_id=document_id)

    def list_knowledge_edges(
        self, *, document_id: str | None = None
    ) -> list[KnowledgeEdge]:
        where = "WHERE document_id = %s" if document_id else ""
        parameters: tuple[str, ...] = (document_id,) if document_id else ()
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT edge_id, fact_id, document_id, paper_id, subject,
                       predicate, object_value, variable_name, variable_value,
                       performance_metric, performance_value, conditions,
                       chunk_id, page_from, page_to, source_quote,
                       source_text_sha256, review_status, created_at
                FROM literature_knowledge_edges
                {where}
                ORDER BY created_at, edge_id
                """,
                parameters,
            )
            rows = cursor.fetchall()
        return [
            KnowledgeEdge.model_validate(
                {
                    "edge_id": row[0],
                    "fact_id": row[1],
                    "document_id": row[2],
                    "paper_id": row[3],
                    "subject": row[4],
                    "predicate": row[5],
                    "object": row[6],
                    "variable_name": row[7],
                    "variable_value": row[8],
                    "performance_metric": row[9],
                    "performance_value": row[10],
                    "conditions": row[11],
                    "chunk_id": row[12],
                    "page_from": row[13],
                    "page_to": row[14],
                    "source_quote": row[15],
                    "source_text_sha256": row[16],
                    "review_status": row[17],
                    "created_at": row[18],
                }
            )
            for row in rows
        ]

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        try:
            psycopg = importlib.import_module("psycopg")
        except ImportError as exc:
            raise RagConfigurationError(
                "psycopg is not installed; install the literature extra"
            ) from exc
        connection_kwargs: dict[str, Any] = {"connect_timeout": 5}
        if self._host_override is not None:
            connection_kwargs["host"] = self._host_override
        connection = psycopg.connect(self._database_url, **connection_kwargs)
        try:
            yield connection
        finally:
            connection.close()


def _vector_literal(vector: Sequence[float], dimensions: int) -> str:
    if len(vector) != dimensions:
        raise ValueError("vector dimension mismatch")
    return "[" + ",".join(format(float(value), ".9g") for value in vector) + "]"


def _experimental_fact_from_row(row: Sequence[Any]) -> ExperimentalDataRow:
    return ExperimentalDataRow.model_validate(
        {
            "fact_id": row[0],
            "document_id": row[1],
            "paper_id": row[2],
            "chunk_id": row[3],
            "page_from": row[4],
            "page_to": row[5],
            "material": row[6],
            "variable_name": row[7],
            "variable_value": row[8],
            "performance_metric": row[9],
            "performance_value": row[10],
            "conditions": row[11],
            "source_quote": row[12],
            "source_text_sha256": row[13],
            "llm_extracted": row[14],
            "review_status": row[15],
        }
    )
