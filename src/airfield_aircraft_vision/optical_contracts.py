"""AW-01 experimental optical contracts; VISION-0 contracts remain compatible."""

from __future__ import annotations
from datetime import timezone
from math import isfinite
from typing import Literal
from pydantic import Field, AwareDatetime, model_validator
from .schemas import StrictModel, Detection, InferenceResult


class PixelBox(StrictModel):
    xyxy: tuple[float, float, float, float]
    score: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def valid_box(self):
        x1, y1, x2, y2 = self.xyxy
        if not all(isfinite(v) for v in self.xyxy) or x2 <= x1 or y2 <= y1:
            raise ValueError("box must have finite, ordered pixel edges")
        return self


class RasterSource(StrictModel):
    scene_id: str
    sensor_id: str
    sensor_domain: Literal["optical", "sar", "unknown"] = "optical"
    acquisition_timestamp: AwareDatetime | None = None
    timestamp_precision: Literal["second", "minute", "date", "unknown"] = "unknown"
    acquisition_time_raw: str | None = None
    timestamp_timezone_basis: Literal["provider_utc", "assumed_utc", "unknown"] = (
        "unknown"
    )
    native_gsd_m: float = Field(gt=0, allow_inf_nan=False)
    effective_resolution_m: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    panchromatic_gsd_m: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    pan_sharpened_gsd_m: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    pixel_spacing_m: tuple[float, float] | None = None
    quality: Literal["acceptable", "poor", "unusable", "unknown"] = "unknown"
    airfield_group: str | None = None
    capture_group: str | None = None
    source_uri: str | None = None
    license: str | None = None
    attribution: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    crs: str | None = None
    transform: tuple[float, float, float, float, float, float] | None = None
    metadata_sha256: str | None = None
    sidecar_sha256: str | None = None
    provenance_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def time_precision(self):
        if (
            self.timestamp_precision in ("second", "minute")
            and self.acquisition_timestamp is None
        ):
            raise ValueError("sensing time precision requires timestamp")
        if (
            self.acquisition_timestamp is not None
            and self.timestamp_precision == "unknown"
        ):
            raise ValueError("timestamp requires declared precision")
        if self.acquisition_timestamp is not None:
            self.acquisition_timestamp = self.acquisition_timestamp.astimezone(
                timezone.utc
            )
        if self.pixel_spacing_m and not all(
            isfinite(v) and v > 0 for v in self.pixel_spacing_m
        ):
            raise ValueError("source grid spacing must be finite and positive")
        if self.transform and not all(isfinite(v) for v in self.transform):
            raise ValueError("source affine transform must be finite")
        return self


class Sample(StrictModel):
    image_id: str
    path: str
    split: Literal["train", "validation", "test"]
    source: RasterSource
    boxes: list[PixelBox] = Field(default_factory=list)
    ignored_boxes: list[PixelBox] = Field(default_factory=list)


class DatasetManifest(StrictModel):
    version: Literal["aw01-1"] = "aw01-1"
    samples: list[Sample]
    preparation: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def no_leakage(self):
        seen: dict[tuple[str, str], str] = {}
        ids = set()
        for s in self.samples:
            if s.image_id in ids:
                raise ValueError("duplicate image_id")
            ids.add(s.image_id)
            for kind, value in (
                ("airfield", s.source.airfield_group),
                ("capture", s.source.capture_group),
                ("checksum", s.source.sha256),
            ):
                if not value:
                    raise ValueError(f"dataset requires {kind} grouping")
                key = (kind, value)
                if key in seen and seen[key] != s.split:
                    raise ValueError(f"{kind} leakage across splits: {value}")
                seen[key] = s.split
            for b in s.boxes + s.ignored_boxes:
                x1, y1, x2, y2 = b.xyxy
                if x1 < 0 or y1 < 0 or x2 > s.source.width or y2 > s.source.height:
                    raise ValueError("ground-truth box outside source pixel grid")
        return self


class OpticalResult(StrictModel):
    contract_version: Literal["aw01-1"] = "aw01-1"
    experimental: Literal[True] = True
    source: RasterSource
    model: dict
    preprocessing: dict
    detections: list[Detection]
    total_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    legacy_inference_result: InferenceResult | None = None

    @model_validator(mode="after")
    def counts(self):
        accepted = sum(d.status == "accepted" for d in self.detections)
        if self.total_count != accepted or self.accepted_count != accepted:
            raise ValueError("accepted count does not match detections")
        if self.rejected_count != len(self.detections) - accepted:
            raise ValueError("rejected count does not match detections")
        return self
