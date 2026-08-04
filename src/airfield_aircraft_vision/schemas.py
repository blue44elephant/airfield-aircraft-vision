"""Versioned inference-result contracts shared across service boundaries."""

from __future__ import annotations

from enum import StrEnum
from math import isclose
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from .capability import CapabilityTier


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeoJSONGeometry(StrictModel):
    type: Literal["Polygon", "MultiPolygon"]
    coordinates: list[Any]


class OrientedBox(StrictModel):
    center_x: float
    center_y: float
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    angle_degrees_clockwise_from_north: float = Field(ge=0, lt=360)
    coordinate_space: Literal["image_pixels", "projected_meters"] = "image_pixels"


class ImageQualityAssessment(StrictModel):
    status: Literal["acceptable", "poor", "unusable", "unknown"]
    cloud_fraction: float | None = Field(default=None, ge=0, le=1)
    haze_severity: Literal["none", "low", "medium", "high", "unknown"] = "unknown"
    shadow_fraction: float | None = Field(default=None, ge=0, le=1)
    obstruction_fraction: float | None = Field(default=None, ge=0, le=1)
    usable_fraction: float | None = Field(default=None, ge=0, le=1)
    notes: list[str] = Field(default_factory=list)


class DetectionStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Detection(StrictModel):
    detection_id: str
    status: DetectionStatus
    oriented_box: OrientedBox | None = None
    geometry: GeoJSONGeometry | None = None
    predicted_class: str | None = None
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0, le=1)
    uncertainty_flags: list[str] = Field(default_factory=list)
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def validate_detection(self) -> "Detection":
        if self.status == DetectionStatus.ACCEPTED:
            if not self.predicted_class:
                raise ValueError("accepted detections require predicted_class")
            if not self.class_probabilities:
                raise ValueError("accepted detections require a class probability distribution")
            if self.predicted_class not in self.class_probabilities:
                raise ValueError("predicted_class must appear in class_probabilities")
        elif not self.rejection_reason:
            raise ValueError("rejected detections require rejection_reason")
        if self.class_probabilities:
            if any(value < 0 or value > 1 for value in self.class_probabilities.values()):
                raise ValueError("class probabilities must be in [0, 1]")
            if not isclose(sum(self.class_probabilities.values()), 1.0, abs_tol=1e-6):
                raise ValueError("class probabilities must sum to 1")
        return self


class CountRange(StrictModel):
    minimum: int = Field(ge=0)
    maximum: int = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    method: str

    @model_validator(mode="after")
    def ordered(self) -> "CountRange":
        if self.maximum < self.minimum:
            raise ValueError("count range maximum must be >= minimum")
        return self


class TemporalComparison(StrictModel):
    baseline_scene_id: str
    baseline_acquisition_timestamp: AwareDatetime
    change_status: Literal["increase", "decrease", "no_material_change", "indeterminate"]
    accepted_count_delta: int | None = None
    occupancy_change: dict[str, Any] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class InferenceResult(StrictModel):
    contract_version: Literal["1.0"] = "1.0"
    aoi: GeoJSONGeometry
    source_scene_id: str
    sensor_id: str
    acquisition_timestamp: AwareDatetime
    native_resolution_m: float = Field(gt=0)
    effective_resolution_m: float | None = Field(default=None, gt=0)
    selected_capability_tier: CapabilityTier
    image_quality: ImageQualityAssessment
    detections: list[Detection] = Field(default_factory=list)
    total_detection_count: int = Field(ge=0, description="Total accepted aircraft detections")
    total_candidate_detections: int = Field(ge=0)
    known_classified_detections: int = Field(ge=0)
    unknown_detections: int = Field(ge=0)
    rejected_detections: int = Field(ge=0)
    per_class_totals: dict[str, int]
    count_range: CountRange | None = None
    unavailable_features: list[str] = Field(default_factory=list)
    unavailable_feature_reasons: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    temporal_comparison: TemporalComparison | None = None

    @model_validator(mode="after")
    def reconcile_counts(self) -> "InferenceResult":
        if any(count < 0 for count in self.per_class_totals.values()):
            raise ValueError("per-class totals cannot be negative")
        unknown_total = self.per_class_totals.get("unknown_aircraft", 0)
        known_total = sum(
            count for label, count in self.per_class_totals.items() if label != "unknown_aircraft"
        )
        if unknown_total != self.unknown_detections:
            raise ValueError("unknown_aircraft total must equal unknown_detections")
        if known_total != self.known_classified_detections:
            raise ValueError("known per-class totals must equal known_classified_detections")
        if known_total + unknown_total != self.total_detection_count:
            raise ValueError("known plus unknown detections must equal total_detection_count")
        if self.total_detection_count + self.rejected_detections != self.total_candidate_detections:
            raise ValueError("accepted plus rejected detections must equal total_candidate_detections")

        accepted = [item for item in self.detections if item.status == DetectionStatus.ACCEPTED]
        rejected = [item for item in self.detections if item.status == DetectionStatus.REJECTED]
        if self.detections:
            if len(accepted) != self.total_detection_count or len(rejected) != self.rejected_detections:
                raise ValueError("detection records must reconcile with accepted and rejected totals")
            observed: dict[str, int] = {}
            for item in accepted:
                assert item.predicted_class is not None
                observed[item.predicted_class] = observed.get(item.predicted_class, 0) + 1
            if observed != {label: count for label, count in self.per_class_totals.items() if count}:
                raise ValueError("accepted detection classes must reconcile with per_class_totals")
        missing_reasons = set(self.unavailable_features) - set(self.unavailable_feature_reasons)
        if missing_reasons:
            raise ValueError(f"unavailable features lack reasons: {sorted(missing_reasons)}")
        return self
