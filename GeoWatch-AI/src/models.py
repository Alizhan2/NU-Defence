from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


ReviewStatus = Literal["confirmed", "rejected", "needs_review"]
CaseStatus = Literal["open", "in_review", "closed"]
CasePriority = Literal["low", "normal", "high"]


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class ImageMetadata(BaseModel):
    filename: str
    format: str
    width: int
    height: int
    mode: str = "RGB"
    sha256: str
    crs: str | None = None
    bounds: list[float] | None = None
    transform: list[float] | None = None


class ReviewRecord(BaseModel):
    detection_id: str
    status: ReviewStatus
    comment: str = Field(default="", max_length=2000)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReviewEvent(BaseModel):
    analysis_id: str
    image_id: str
    detection_id: str
    status: ReviewStatus
    comment: str = Field(default="", max_length=2000)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Detection(BaseModel):
    id: str
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox
    processing_ms: float = Field(ge=0, default=0)
    model_version: str
    review: ReviewRecord | None = None


class AnalysisResult(BaseModel):
    analysis_id: str
    image_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: ImageMetadata
    detections: list[Detection]
    model_version: str
    confidence_threshold: float
    inference_ms: float
    tile_count: int
    limitations: list[str] = Field(default_factory=list)


class ModelStatus(BaseModel):
    state: Literal["ready", "missing", "load_error"]
    model_version: str
    message_ru: str


class Finding(BaseModel):
    detection_id: str
    description_ru: str
    evidence: str
    risk_note: str


class AnalystSummary(BaseModel):
    summary_ru: str
    findings: list[Finding]
    limitations_ru: list[str]


class AnalystResponse(BaseModel):
    available: bool
    message_ru: str
    summary: AnalystSummary | None = None


class ReviewRequest(BaseModel):
    analysis_id: str
    detection_id: str
    status: ReviewStatus
    comment: str = Field(default="", max_length=2000)


class CaseRecord(BaseModel):
    case_id: str = Field(default_factory=lambda: uuid4().hex)
    title: str = Field(min_length=1, max_length=160)
    note: str = Field(default="", max_length=4000)
    priority: CasePriority = "normal"
    status: CaseStatus = "open"
    analysis_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CaseCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    note: str = Field(default="", max_length=4000)
    priority: CasePriority = "normal"


class CaseUpdateRequest(BaseModel):
    status: CaseStatus


class SummaryRequest(BaseModel):
    image_id: str


class EvaluationMetrics(BaseModel):
    precision: float
    recall: float
    f1_score: float
    map50: float | None = None
    classification_accuracy: float
    false_positives: int
    average_processing_ms: float
    test_images: int
    model_version: str
    confidence_threshold: float
