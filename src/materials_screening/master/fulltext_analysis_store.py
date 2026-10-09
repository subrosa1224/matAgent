"""Append-only analysis and readable-report storage with bounded hashed replay."""

import hashlib
from pathlib import Path

from .fulltext_analysis_contracts import (
    FulltextAnalysisRecord,
    FulltextAnalysisReference,
)


class FulltextAnalysisStore:
    def __init__(self, root: Path, *, max_bytes=16 * 1024 * 1024):
        self.root, self.max_bytes = root.resolve(), max_bytes

    def _path(self, ref, suffix):
        ref = FulltextAnalysisReference.model_validate(ref.model_dump())
        path = (self.root / f"{ref.record_id}.{suffix}").resolve()
        if path.parent != self.root:
            raise ValueError("Invalid analysis checkpoint path")
        return path

    def _read(self, path):
        with path.open("rb") as stream:
            content = stream.read(self.max_bytes + 1)
        if len(content) > self.max_bytes:
            raise ValueError("Oversized analysis checkpoint")
        return content

    def save(self, record):
        record = FulltextAnalysisRecord.model_validate(record.model_dump())
        content = record.model_dump_json(indent=2).encode()
        if len(content) > self.max_bytes:
            raise ValueError("Oversized analysis checkpoint")
        ref = FulltextAnalysisReference(
            record_id=record.record_id,
            content_sha256=hashlib.sha256(content).hexdigest(),
        )
        self.root.mkdir(parents=True, exist_ok=True)
        items = [("json", content)]
        if record.status == "complete":
            if not record.report_markdown or record.scope is None:
                raise ValueError("Complete analysis requires a scope and report")
            items.append(("md", record.report_markdown.encode()))
        for suffix, payload in items:
            path = self._path(ref, suffix)
            try:
                with path.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
            except FileExistsError:
                if self._read(path) != payload:
                    raise ValueError(
                        "Refusing to overwrite immutable analysis"
                    ) from None
        return ref

    def load(self, ref):
        content = self._read(self._path(ref, "json"))
        if hashlib.sha256(content).hexdigest() != ref.content_sha256:
            raise ValueError("Analysis checkpoint content differs from reference")
        record = FulltextAnalysisRecord.model_validate_json(content)
        if record.record_id != ref.record_id:
            raise ValueError("Analysis checkpoint identity mismatch")
        if (
            record.status == "complete"
            and self._read(self._path(ref, "md")) != record.report_markdown.encode()
        ):
            raise ValueError("Readable report differs from checkpoint")
        return record
