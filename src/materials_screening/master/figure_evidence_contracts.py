"""Text-only figure evidence; never a measurement or experimental condition."""

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .fulltext_snapshots import SnapshotReference

SHA = r"^[a-f0-9]{64}$"
Kind = Literal["axis_label", "axis_ticks", "legend_text", "other_text"]


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class PageRegion(StrictRecord):
    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(gt=0)
    y1: float = Field(gt=0)

    @model_validator(mode="after")
    def positive_area(self):
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("Region must have positive area")
        return self


class ImageReference(StrictRecord):
    sha256: str = Field(pattern=SHA)
    width: int = Field(gt=0, le=10000, strict=True)
    height: int = Field(gt=0, le=10000, strict=True)

    @model_validator(mode="after")
    def bounded_pixels(self):
        if self.width * self.height > 4_000_000:
            raise ValueError("Image pixel budget exceeded")
        return self


class OcrWord(StrictRecord):
    text: str = Field(max_length=1000)
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class OcrResult(StrictRecord):
    status: Literal["ok", "unavailable", "failed", "cancelled"]
    engine: str = Field(default="none", max_length=160)
    language: str | None = Field(default=None, max_length=80)
    installed_languages: tuple[str, ...] = Field(default=(), max_length=100)
    raw_text: str = Field(default="", max_length=10000)
    words: tuple[OcrWord, ...] = Field(default=(), max_length=10000)
    error_code: str | None = Field(default=None, max_length=160)

    @model_validator(mode="after")
    def no_failed_text(self):
        if self.status != "ok" and (self.raw_text or self.words):
            raise ValueError("Failed OCR cannot contain successful text")
        return self


class FigureBatchReference(StrictRecord):
    record_id: str = Field(pattern=r"^figure-batch-[a-f0-9]{32}$")
    content_sha256: str = Field(pattern=SHA)


class FigurePageView(StrictRecord):
    document_id: str = Field(pattern=r"^doc-[a-f0-9]{24}$")
    pdf_sha256: str = Field(pattern=SHA)
    snapshot_ref: SnapshotReference
    page: int = Field(ge=1, strict=True)
    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)
    original_rotation: Literal[0, 90, 180, 270]
    render_rotation: Literal[0] = 0
    dpi: Literal[150] = 150
    renderer_version: str = Field(max_length=160)
    image: ImageReference

    @model_validator(mode="after")
    def bound(self):
        if (
            self.document_id != "doc-" + self.pdf_sha256[:24]
            or self.snapshot_ref.document_id != self.document_id
        ):
            raise ValueError("Source page identity differs")
        return self


class FigureCandidate(StrictRecord):
    candidate_id: str = Field(
        default_factory=lambda: "figure-candidate-" + uuid4().hex,
        pattern=r"^figure-candidate-[a-f0-9]{32}$",
    )
    document_id: str = Field(pattern=r"^doc-[a-f0-9]{24}$")
    artifact_id: str = Field(pattern=r"^artifact-pdf-[a-f0-9]{24}$")
    pdf_sha256: str = Field(pattern=SHA)
    snapshot_ref: SnapshotReference
    page: int = Field(ge=1, strict=True)
    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)
    original_rotation: Literal[0, 90, 180, 270]
    render_rotation: Literal[0] = 0
    region: PageRegion
    figure_label: str = Field(min_length=1, max_length=160)
    human_selected: Literal[True] = True
    kind: Kind
    page_image: ImageReference
    crop_image: ImageReference
    page_dpi: Literal[150] = 150
    crop_dpi: Literal[300] = 300
    renderer_version: str = Field(max_length=160)
    ocr: OcrResult
    reviewed_text: str | None = Field(default=None, max_length=10000)
    text_review_status: Literal["pending", "text_verified", "skipped"] = "pending"
    condition_binding_status: Literal["unbound"] = "unbound"
    manual_transcription: bool = False
    reviewer: str | None = Field(default=None, max_length=254)
    reviewed_at: datetime | None = None
    superseded_by: str | None = Field(
        default=None, pattern=r"^figure-candidate-[a-f0-9]{32}$"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def coherent(self):
        if (
            self.document_id != "doc-" + self.pdf_sha256[:24]
            or self.artifact_id != "artifact-pdf-" + self.pdf_sha256[:24]
            or self.snapshot_ref.document_id != self.document_id
        ):
            raise ValueError("Figure source identity differs")
        if self.region.x1 > self.page_width or self.region.y1 > self.page_height:
            raise ValueError("Region outside source page")
        if self.text_review_status == "text_verified" and (
            not self.reviewed_text
            or not self.reviewed_text.strip()
            or self.reviewer is None
            or self.reviewed_at is None
        ):
            raise ValueError("Text verification requires an explicit reviewer and text")
        if self.text_review_status == "pending" and (
            self.reviewed_text is not None
            or self.reviewer is not None
            or self.reviewed_at is not None
        ):
            raise ValueError("Pending evidence cannot contain a review")
        for word in self.ocr.words:
            if (
                word.x + word.width > self.crop_image.width
                or word.y + word.height > self.crop_image.height
            ):
                raise ValueError("OCR word outside image")
        return self


class FigureEvidenceBatch(StrictRecord):
    record_id: str = Field(
        default_factory=lambda: "figure-batch-" + uuid4().hex,
        pattern=r"^figure-batch-[a-f0-9]{32}$",
    )
    parent_ref: FigureBatchReference | None = None
    conversation_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,253}$")
    task_id: str = Field(pattern=r"^task-fulltext-[a-f0-9]{32}$")
    original_question_sha256: str = Field(pattern=SHA)
    policy: Literal["figure-evidence-review-v1"] = "figure-evidence-review-v1"
    selected_snapshots: dict[str, SnapshotReference]
    candidates: tuple[FigureCandidate, ...] = Field(default=(), max_length=100)
    page_views: tuple[FigurePageView, ...] = Field(default=(), max_length=20)
    applied_operations: dict[str, str] = Field(default_factory=dict)
    revision_number: int = Field(default=0, ge=0, le=100)
    status: Literal["open", "closed"] = "open"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def bound_candidates(self):
        ids = {candidate.candidate_id for candidate in self.candidates}
        if len(ids) != len(self.candidates) or len(self.applied_operations) > 100:
            raise ValueError("Duplicate candidates or operation budget exceeded")
        if len([c for c in self.candidates if c.superseded_by is None]) > 20:
            raise ValueError("Active figure budget exceeded")
        if any(key != ref.document_id for key, ref in self.selected_snapshots.items()):
            raise ValueError("Snapshot map identity differs")
        for candidate in self.candidates:
            if (
                self.selected_snapshots.get(candidate.document_id)
                != candidate.snapshot_ref
            ):
                raise ValueError("Candidate outside batch source scope")
            if (
                candidate.superseded_by is not None
                and candidate.superseded_by not in ids
            ):
                raise ValueError("Missing replacement candidate")
            if candidate.reviewer not in {None, self.conversation_id}:
                raise ValueError("Figure reviewer outside current conversation")
        if len({(v.document_id, v.page) for v in self.page_views}) != len(
            self.page_views
        ):
            raise ValueError("Duplicate source page views")
        if any(
            self.selected_snapshots.get(v.document_id) != v.snapshot_ref
            for v in self.page_views
        ):
            raise ValueError("Source page outside batch")
        if self.status == "closed" and any(
            c.text_review_status == "pending" for c in self.candidates
        ):
            raise ValueError("Closed batch has pending evidence")
        return self
