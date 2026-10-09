"""Conservative local checks for attribution and multi-sample text tables.

These checks do not infer missing experimental conditions or digitize figures.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence

from .models import ExperimentalGroup, ExperimentalMeasurement
from .rag import ChunkRecord


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("−", "-")
    return " ".join(value.split()).casefold()


def is_reference_evidence(
    quote: str, chunk: ChunkRecord, chunks: Sequence[ChunkRecord]
) -> bool:
    """Locate reference sections across ordered chunks, including continuation pages.

    Use section boundaries, not page numbers or scientific content keywords.
    Text before a same-chunk heading remains eligible as the paper's own work.
    """
    active = False
    heading = re.compile(
        r"(?im)^\s*(?:\d+[.)]?\s+)?(?:references|bibliography|参考文献)\s*$"
    )
    end = re.compile(
        r"(?im)^\s*(?:appendix\b[^\n]*|supplementary (?:material|information)[^\n]*)$"
    )
    for current in sorted(
        (item for item in chunks if item.document_id == chunk.document_id),
        key=lambda item: (item.page_from, item.page_to),
    ):
        boundaries = sorted(
            [
                (match.start(), match.end(), True)
                for match in heading.finditer(current.text)
            ]
            + [
                (match.start(), match.end(), False)
                for match in end.finditer(current.text)
            ]
        )
        start = 0
        for lower, upper, reference in (
            *boundaries,
            (len(current.text), len(current.text), None),
        ):
            if current.chunk_id == chunk.chunk_id and active:
                segment = normalized(current.text[start:lower])
                excerpt = normalized(quote)
                if excerpt and excerpt in segment:
                    return True
                # A long quote spanning the heading must not make citations eligible.
                if segment and segment in excerpt:
                    return True
            start = upper
            if reference is not None:
                active = reference
        if current.chunk_id == chunk.chunk_id:
            return False
    return False


def is_prior_work(quote: str, context: str) -> bool:
    """Include preceding attribution, rather than classifying a bare 'Here'."""
    text, excerpt = normalized(context), normalized(quote)
    position = text.find(excerpt)
    if position < 0:
        return False
    window = text[max(0, position - 450) : position + len(excerpt)]
    own = list(
        re.finditer(
            r"\bin (?:this|the present) (?:work|study)\b|\bherein\b|"
            r"\b(?:experimental|materials and methods)\b|本文|本研究",
            window,
        )
    )
    if own:
        window = window[own[-1].end() :]
    return bool(
        re.search(
            r"\bet\s+al\.?\s*(?:\[\d[^]]*\])?[^.!?]{0,150}"
            r"\b(?:report\w*|prepar\w*|used|stud\w*|synthesi\w*)\b|"
            r"\b(?:previous|prior|earlier) (?:work|study|studies|research)\b|"
            r"(?:前人|先前|已有)研究",
            window,
        )
    )


def sample_table_rows(text: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Read only row-major, line-separated tables with an explicit sample header."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if normalized(line)
            in {
                "sample name",
                "sample",
                "catalyst",
                "material",
                "device",
                "compound",
            }
        ),
        None,
    )
    if start is None:
        return ()
    rows: list[tuple[str, tuple[str, ...]]] = []
    index = start + 1
    while index + 1 < len(lines):
        label = lines[index]
        if not _numeric_cell(label) and _numeric_cell(lines[index + 1]):
            cells: list[str] = []
            cursor = index + 1
            while cursor < len(lines) and _numeric_cell(lines[cursor]):
                cells.append(lines[cursor])
                cursor += 1
            # Do not mistake R2/header text or an end-of-table section for a sample.
            if not re.match(r"^(?:R\s*2|R²|\d+\.|Table|Figure|Fig\.)", label):
                rows.append((label, tuple(cells)))
            index = cursor
        else:
            index += 1
    if len(rows) < 2 or len({len(cells) for _, cells in rows}) != 1:
        return ()
    return tuple(rows)


def _numeric_cell(value: str) -> bool:
    return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:\s+\d+)?\s*%?", value))


def measurement_binding_context(quote: str, context: str) -> str:
    """Keep table context for row quotes, not for unrelated surrounding prose.

    Only bypass a table after locating its contiguous row-major boundary. An
    unsupported table or an unlocated excerpt still uses the conservative check.
    """
    position = context.find(quote)
    if position < 0:
        return context
    lines: list[tuple[str, int, int]] = []
    offset = 0
    for line in context.splitlines(keepends=True):
        if line.strip():
            lines.append((line.strip(), offset, offset + len(line)))
        offset += len(line)
    spans: list[tuple[int, int]] = []
    for index, (line, lower, _) in enumerate(lines):
        if normalized(line) not in {
            "sample name",
            "sample",
            "catalyst",
            "material",
            "device",
            "compound",
        }:
            continue
        cursor = index + 1
        while cursor + 1 < len(lines) and not (
            not _numeric_cell(lines[cursor][0]) and _numeric_cell(lines[cursor + 1][0])
        ):
            cursor += 1
        widths: list[int] = []
        while cursor + 1 < len(lines) and (
            not _numeric_cell(lines[cursor][0]) and _numeric_cell(lines[cursor + 1][0])
        ):
            cursor += 1
            count = 0
            while cursor < len(lines) and _numeric_cell(lines[cursor][0]):
                count += 1
                cursor += 1
            widths.append(count)
        if len(widths) < 2 or len(set(widths)) != 1:
            return context
        spans.append((lower, lines[cursor - 1][2]))
    if not spans or any(
        position < upper and position + len(quote) > lower for lower, upper in spans
    ):
        return context
    return quote


def _sample_name(value: str) -> str:
    text = normalized(value)
    text = re.sub(r"\b(?:pure|pristine|sample|composite)\b", "", text)
    return re.sub(r"\s+", "", text)


def percent_table_column(text: str) -> tuple[str, int] | None:
    rows = sample_table_rows(text)
    if not rows:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    start = next(
        i
        for i, line in enumerate(lines)
        if normalized(line)
        in {
            "sample name",
            "sample",
            "catalyst",
            "material",
            "device",
            "compound",
        }
    )
    headers = lines[start + 1 : lines.index(rows[0][0])]
    if len(headers) != len(rows[0][1]):
        return None
    indexes = [i for i, header in enumerate(headers) if re.search(r"\(%\)", header)]
    if len(indexes) != 1:
        return None
    index = indexes[0]
    return re.sub(r"\s*\(%\)\s*", "", headers[index]).strip(), index


def measurement_sample_bound(
    *,
    label: str,
    value: str,
    unit: str | None,
    metric: str,
    quote: str,
) -> bool:
    """Reject identifiable table/prose swaps; leave unrelated formats unchanged."""
    rows = sample_table_rows(quote)
    if not rows and any(
        normalized(line)
        in {"sample name", "sample", "catalyst", "material", "device", "compound"}
        for line in quote.splitlines()
    ):
        # An identifiable but unsupported table must not become page-level evidence.
        return False
    if rows:
        matching = [
            cells for name, cells in rows if _sample_name(name) == _sample_name(label)
        ]
        if len(matching) != 1:
            return False
        cells = matching[0]
        column = percent_table_column(quote)
        if unit == "%" and column is not None:
            header_metric, column_index = column
            meaningful = set(re.findall(r"[a-z]{3,}", normalized(header_metric)))
            aliases = {"降解": "degradation", "孔隙": "porosity", "效率": "efficiency"}
            metric_text = normalized(metric)
            for term, alias in aliases.items():
                metric_text = metric_text.replace(term, alias)
            if not meaningful.intersection(re.findall(r"[a-z]{3,}", metric_text)):
                return False
            # A percent value must be in a percent column, not K or R².
            return normalized(value).rstrip(" %") == normalized(
                cells[column_index]
            ).rstrip(" %")
        # Non-percent columns require an unambiguous named header, not any number.
        return False
    if re.search(r"\bSample\s+[A-Z]\b", label):
        clauses = re.split(r";|(?<=[.!?])\s+", quote)
        matching = [
            clause for clause in clauses if normalized(label) in normalized(clause)
        ]
        if matching:
            return any(
                re.search(rf"(?<![\d.]){re.escape(value)}(?![\d.])", clause)
                for clause in matching
            )
    return True


def extract_percent_sample_tables(
    document_id: str,
    chunks: Sequence[ChunkRecord],
) -> tuple[list[ExperimentalGroup], list[ExperimentalMeasurement]]:
    """Recover unambiguous percent-column rows; no paper/formula-specific rules."""
    groups: list[ExperimentalGroup] = []
    measurements: list[ExperimentalMeasurement] = []
    for chunk in chunks:
        column = percent_table_column(chunk.text)
        if (
            column is None
            or is_prior_work(chunk.text, chunk.text)
            or is_reference_evidence(chunk.text, chunk, chunks)
        ):
            continue
        metric, index = column
        for label, cells in sample_table_rows(chunk.text):
            cell = cells[index]
            value = cell.rstrip(" %").strip()
            if not re.fullmatch(r"\d+(?:\.\d+)?", value):
                continue
            digest = hashlib.sha256(
                f"{document_id}|{chunk.chunk_id}|{label}".encode()
            ).hexdigest()[:24]
            group_id = f"group-{digest}"
            provenance = dict(
                document_id=document_id,
                source_quote=chunk.text,
                chunk_id=chunk.chunk_id,
                page_from=chunk.page_from,
                page_to=chunk.page_to,
                source_text_sha256=chunk.text_sha256,
                extraction_method="table_parser",
                llm_extracted=False,
                review_status="pending",
            )
            groups.append(
                ExperimentalGroup(
                    group_id=group_id,
                    label=label,
                    material=label,
                    role="unknown",
                    **provenance,
                )
            )
            measurement_digest = hashlib.sha256(
                f"{digest}|{metric}|{cell}".encode()
            ).hexdigest()[:24]
            measurements.append(
                ExperimentalMeasurement(
                    measurement_id=f"measurement-{measurement_digest}",
                    group_id=group_id,
                    metric=metric,
                    value_text=value,
                    numeric_value=float(value),
                    unit="%",
                    **provenance,
                )
            )
    return groups, measurements
