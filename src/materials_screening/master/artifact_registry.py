"""Private-path artifact registry for the unified multi-agent application."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from materials_screening.data_analysis.parser import TabularDataParser
from materials_screening.master.application_contracts import MultiAgentArtifact


class ArtifactRegistry:
    """Register uploaded files while exposing only stable opaque identifiers."""

    def __init__(
        self,
        root: Path,
        *,
        max_pdf_bytes: int = 100 * 1024 * 1024,
        max_data_bytes: int = 50 * 1024 * 1024,
        dataset_store: Any | None = None,
    ) -> None:
        self._root = root.resolve()
        self._max_pdf_bytes = max_pdf_bytes
        self._data_parser = TabularDataParser(max_file_bytes=max_data_bytes)
        self._dataset_store = dataset_store
        self._lock = threading.Lock()

    @property
    def dataset_store(self) -> Any | None:
        """Return the application-owned store used for tabular registration."""

        return self._dataset_store

    def register_pdf(
        self, source: str | Path | Any, *, conversation_id: str | None = None
    ) -> MultiAgentArtifact:
        if conversation_id is not None:
            _validate_conversation_id(conversation_id)
        raw_source = source if isinstance(source, (str, Path)) else source.name
        path = Path(raw_source).resolve()
        if path.suffix.casefold() != ".pdf" or not path.is_file():
            raise ValueError("只支持PDF文件。")
        size = path.stat().st_size
        if size > self._max_pdf_bytes:
            raise ValueError("单个PDF不能超过100 MB。")
        digest = _sha256_pdf(path)
        artifact_id = f"artifact-pdf-{digest[:24]}"
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._")
        safe_stem = safe_stem or "paper"
        target_root = self._root / "pdf"
        target = (target_root / f"{safe_stem[:80]}-{digest[:10]}.pdf").resolve()
        if target.parent != target_root:
            raise ValueError("PDF文件名无效。")
        artifact = MultiAgentArtifact(
            artifact_id=artifact_id,
            owner_agent="literature",
            artifact_type="pdf_document",
            domain_id=f"pdf-{digest[:24]}",
            display_name=path.name[:512],
            metadata={"size_bytes": size, "sha256": digest},
        )
        with self._lock:
            # A stable artifact keeps its first metadata and path. Registering
            # identical content in another conversation grants only that upload.
            try:
                existing, existing_path = self._load_record(artifact_id)
            except KeyError:
                existing = None
            if existing is not None:
                self._validate_path(existing, existing_path)
                if conversation_id is not None:
                    self._grant_pdf(artifact_id, conversation_id)
                return existing
            target_root.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    with target.open("xb") as destination, path.open("rb") as origin:
                        shutil.copyfileobj(origin, destination)
                except FileExistsError:
                    # Another registry may have registered identical content.
                    # Never overwrite its file; verify it below instead.
                    pass
            self._validate_path(artifact, target)
            self._persist_record(artifact, target)
            if conversation_id is not None:
                self._grant_pdf(artifact_id, conversation_id)
        return artifact

    def register_data_file(self, source: str | Path | Any) -> MultiAgentArtifact:
        """Register one bounded CSV/JSON/XLSX file as a data artifact."""

        raw_source = source if isinstance(source, (str, Path)) else source.name
        path = Path(raw_source).resolve()
        frame = self._data_parser.parse(path)
        digest = _sha256_file(path)
        artifact_id = f"artifact-data-{digest[:24]}"
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._")
        safe_stem = safe_stem or "dataset"
        suffix = path.suffix.casefold()
        target_root = (self._root / "data").resolve()
        target = (target_root / f"{safe_stem[:80]}-{digest[:10]}{suffix}").resolve()
        if target.parent != target_root:
            raise ValueError("数据文件名无效。")
        with self._lock:
            target_root.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(path, target)
        domain_id = f"dataset-{digest[:24]}"
        if self._dataset_store is not None:
            reference = self._dataset_store.register_file(
                target,
                source_artifact_id=artifact_id,
                display_name=path.name,
            )
            domain_id = reference.dataset_id
        artifact = MultiAgentArtifact(
            artifact_id=artifact_id,
            owner_agent="data_analysis",
            artifact_type="tabular_dataset",
            domain_id=domain_id,
            display_name=path.name[:512],
            metadata={
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "row_count": len(frame.index),
                "column_count": len(frame.columns),
                "format": suffix.removeprefix("."),
            },
        )
        with self._lock:
            self._validate_path(artifact, target)
            self._persist_record(artifact, target)
        return artifact

    def resolve_path(self, artifact_id: str) -> Path:
        with self._lock:
            artifact, stored_path = self._load_record(artifact_id)
        return self._validate_path(artifact, stored_path)

    def resolve_pdf(self, artifact_id: str, *, conversation_id: str) -> Path:
        """Resolve only a PDF explicitly uploaded for this conversation."""

        _validate_conversation_id(conversation_id)
        with self._lock:
            artifact, stored_path = self._load_record(artifact_id)
            if artifact.artifact_type != "pdf_document":
                raise ValueError("当前附件不是PDF。")
            with self._catalog_connection() as connection:
                grant = connection.execute(
                    "SELECT 1 FROM pdf_grants "
                    "WHERE artifact_id=? AND conversation_id=?",
                    (artifact_id, conversation_id),
                ).fetchone()
            if grant is None:
                raise ValueError("当前会话没有该PDF的上传授权。")
        return self._validate_path(artifact, stored_path)

    def _validate_path(self, artifact: MultiAgentArtifact, stored_path: Path) -> Path:
        path = stored_path.resolve()
        owner_root = (
            (self._root / "pdf").resolve()
            if artifact.artifact_type == "pdf_document"
            else (self._root / "data").resolve()
            if artifact.artifact_type == "tabular_dataset"
            else None
        )
        if (
            owner_root is None
            or owner_root.parent != self._root
            or path.parent != owner_root
            or not path.is_file()
        ):
            raise ValueError("附件存储路径无效。")
        digest = artifact.metadata.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
            raise ValueError("附件内容校验信息无效。")
        size = artifact.metadata.get("size_bytes")
        if type(size) is not int or path.stat().st_size != size:
            raise ValueError("附件内容已改变，拒绝复用。")
        if artifact.artifact_type == "pdf_document":
            if (
                artifact.owner_agent != "literature"
                or artifact.artifact_id != f"artifact-pdf-{digest[:24]}"
                or artifact.domain_id != f"pdf-{digest[:24]}"
                or size > self._max_pdf_bytes
            ):
                raise ValueError("PDF附件内容或身份无效。")
            current_digest = _sha256_pdf(path)
        else:
            current_digest = _sha256_file(path)
        if current_digest != digest:
            raise ValueError("附件内容已改变，拒绝复用。")
        return path

    def get(self, artifact_id: str) -> MultiAgentArtifact:
        with self._lock:
            return self._load_record(artifact_id)[0]

    @contextmanager
    def _catalog_connection(self) -> Iterator[sqlite3.Connection]:
        self._root.mkdir(parents=True, exist_ok=True)
        catalog = self._root / "artifacts.sqlite"
        if catalog.resolve().parent != self._root:
            raise ValueError("附件目录存储路径无效。")
        connection = None
        try:
            connection = sqlite3.connect(catalog, timeout=5)
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS artifacts ("
                    "artifact_id TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                    "relative_path TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS pdf_grants ("
                    "artifact_id TEXT NOT NULL, conversation_id TEXT NOT NULL, "
                    "PRIMARY KEY (artifact_id, conversation_id))"
                )
                yield connection
        except sqlite3.Error as exc:
            raise ValueError("附件目录无法安全读取或保存。") from exc
        finally:
            if connection is not None:
                connection.close()

    def _persist_record(self, artifact: MultiAgentArtifact, path: Path) -> None:
        relative = path.relative_to(self._root).as_posix()
        with self._catalog_connection() as connection:
            connection.execute(
                "INSERT INTO artifacts (artifact_id,payload,relative_path) "
                "VALUES (?,?,?) ON CONFLICT (artifact_id) DO NOTHING",
                (artifact.artifact_id, artifact.model_dump_json(), relative),
            )

    def _load_record(self, artifact_id: str) -> tuple[MultiAgentArtifact, Path]:
        if re.fullmatch(r"artifact-(?:pdf|data)-[a-f0-9]{24}", artifact_id) is None:
            raise KeyError(f"unknown artifact: {artifact_id!r}")
        with self._catalog_connection() as connection:
            row = connection.execute(
                "SELECT payload,relative_path FROM artifacts WHERE artifact_id=?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown artifact: {artifact_id!r}")
        try:
            artifact = MultiAgentArtifact.model_validate(json.loads(row[0]))
            relative = Path(row[1])
        except (ValueError, TypeError) as exc:
            raise ValueError("附件目录记录无效。") from exc
        if artifact.artifact_id != artifact_id or relative.is_absolute():
            raise ValueError("附件目录记录或路径无效。")
        return artifact, self._root / relative

    def _grant_pdf(self, artifact_id: str, conversation_id: str) -> None:
        with self._catalog_connection() as connection:
            connection.execute(
                "INSERT INTO pdf_grants VALUES (?,?) ON CONFLICT DO NOTHING",
                (artifact_id, conversation_id),
            )

    def resolve_owner_reference(
        self, artifact_ids: list[str], message: str
    ) -> str | None:
        """Resolve explicit attachment references without domain topic routing."""

        if not artifact_ids:
            return None
        owners = {self.get(artifact_id).owner_agent for artifact_id in artifact_ids}
        if re.search(
            r"第\s*\d+\s*篇|这些(?:论文|文献|PDF)|上传的(?:论文|文献|PDF)|"
            r"(?:预览|深度分析|综合|总结|对比).*(?:论文|文献|PDF)",
            message,
            flags=re.IGNORECASE,
        ):
            return owners.pop() if len(owners) == 1 else None
        if re.search(
            r"这些数据|上传的(?:数据|CSV|JSON)|(?:这个|该|刚才的)(?:表格|数据集|数据)|"
            r"(?:检查|分析|清洗|统计|绘图).*(?:CSV|JSON|表格|数据集)",
            message,
            flags=re.IGNORECASE,
        ):
            return owners.pop() if len(owners) == 1 else None
        return None


def _sha256_pdf(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError("文件不是有效PDF。")
        handle.seek(0)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_conversation_id(value: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,253}", value) is None:
        raise ValueError("会话标识无效。")
