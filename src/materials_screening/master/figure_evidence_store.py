"""Append-only, bounded local evidence. No canonical dataset writes."""

import hashlib
import struct
from pathlib import Path

from .figure_evidence_contracts import (
    FigureBatchReference,
    FigureEvidenceBatch,
    ImageReference,
)


def batch_reference(batch):
    batch = FigureEvidenceBatch.model_validate(batch.model_dump())
    return FigureBatchReference(
        record_id=batch.record_id,
        content_sha256=hashlib.sha256(
            batch.model_dump_json(indent=2).encode()
        ).hexdigest(),
    )


class FigureEvidenceStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def _path(self, name):
        path = (self.root / name).resolve()
        if path.parent != self.root:
            raise ValueError("Evidence path outside controlled storage")
        return path

    @staticmethod
    def _read(path, maximum):
        with path.open("rb") as stream:
            content = stream.read(maximum + 1)
        if len(content) > maximum:
            raise ValueError("Evidence byte budget exceeded")
        return content

    def _write(self, path, content, maximum):
        if len(content) > maximum:
            raise ValueError("Evidence byte budget exceeded")
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(content)
                stream.flush()
        except FileExistsError:
            if self._read(path, maximum) != content:
                raise ValueError("Refusing to overwrite immutable evidence") from None

    def save(self, batch):
        batch = FigureEvidenceBatch.model_validate(batch.model_dump())
        content = batch.model_dump_json(indent=2).encode()
        ref = batch_reference(batch)
        self._write(self._path(ref.record_id + ".json"), content, 2 * 1024 * 1024)
        return ref

    def load(self, ref):
        ref = FigureBatchReference.model_validate(ref.model_dump())
        content = self._read(self._path(ref.record_id + ".json"), 2 * 1024 * 1024)
        if hashlib.sha256(content).hexdigest() != ref.content_sha256:
            raise ValueError("Figure batch hash differs")
        batch = FigureEvidenceBatch.model_validate_json(content)
        if batch.record_id != ref.record_id:
            raise ValueError("Figure batch identity differs")
        return batch

    def find_revision(self, record_id):
        """Read one deterministic revision, never scan unrelated task history."""
        checked = FigureBatchReference(record_id=record_id, content_sha256="0" * 64)
        path = self._path(checked.record_id + ".json")
        if not path.exists():
            return None
        content = self._read(path, 2 * 1024 * 1024)
        batch = FigureEvidenceBatch.model_validate_json(content)
        if (
            batch.record_id != record_id
            or batch.model_dump_json(indent=2).encode() != content
        ):
            raise ValueError("Interrupted figure revision integrity differs")
        return self.load(batch_reference(batch))

    def save_image(self, content):
        if (
            len(content) < 24
            or content[:8] != b"\x89PNG\r\n\x1a\n"
            or content[12:16] != b"IHDR"
        ):
            raise ValueError("Only rendered PNG evidence is accepted")
        width, height = struct.unpack(">II", content[16:24])
        ref = ImageReference(
            sha256=hashlib.sha256(content).hexdigest(), width=width, height=height
        )
        self._write(self._path(ref.sha256 + ".png"), content, 10 * 1024 * 1024)
        return ref

    def image_path(self, ref):
        ref = ImageReference.model_validate(ref.model_dump())
        path = self._path(ref.sha256 + ".png")
        content = self._read(path, 10 * 1024 * 1024)
        if hashlib.sha256(content).hexdigest() != ref.sha256:
            raise ValueError("Figure image hash differs")
        if len(content) < 24 or struct.unpack(">II", content[16:24]) != (
            ref.width,
            ref.height,
        ):
            raise ValueError("Figure image dimensions differ")
        return path
