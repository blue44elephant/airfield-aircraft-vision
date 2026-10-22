"""Experimental AW-01D scene results; unknown/failed counts are never zero."""

from typing import Literal
from pydantic import Field, model_validator
from .optical_contracts import RasterSource, PixelBox
from .schemas import StrictModel, GeoJSONGeometry


class RobustDetection(StrictModel):
    detection_id: str
    status: Literal["accepted", "uncertain", "rejected"]
    box: PixelBox
    geometry: GeoJSONGeometry | None = None
    calibrated_probability: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None
    uncertainty_flags: list[str] = Field(default_factory=list)
    tile_index: int = Field(ge=0)
    tile_window: tuple[int, int, int, int]
    valid_pixel_fraction: float = Field(ge=0, le=1)


class CoverageSummary(StrictModel):
    total_pixels: int = Field(gt=0)
    inspected_pixels: int = Field(ge=0)
    analyzable_pixels: int = Field(ge=0)
    inspected_fraction: float = Field(ge=0, le=1)
    analyzable_fraction: float = Field(ge=0, le=1)
    # Pixel-coordinate analyst exclusions, not a machine-generated cloud mask.
    excluded_regions: list[tuple[int, int, int, int]] = Field(default_factory=list)
    unavailable_tile_windows: list[tuple[int, int, int, int]] = Field(
        default_factory=list
    )
    unavailable_locations_precision: Literal["coarse_tile_windows_not_exact_masks"] = (
        "coarse_tile_windows_not_exact_masks"
    )
    unavailable_reason: str = (
        "Nodata and analyst exclusions are unavailable; clouds/occlusion are not automatically detected."
    )

    @model_validator(mode="after")
    def reconciled(self):
        if (
            not 0
            <= self.analyzable_pixels
            <= self.inspected_pixels
            <= self.total_pixels
        ):
            raise ValueError("coverage pixel totals do not reconcile")
        for pixels, fraction in (
            (self.inspected_pixels, self.inspected_fraction),
            (self.analyzable_pixels, self.analyzable_fraction),
        ):
            if abs(fraction - pixels / self.total_pixels) > 1e-10:
                raise ValueError("coverage fractions do not reconcile")
        return self


class RobustSceneResult(StrictModel):
    contract_version: Literal["aw01d-1"] = "aw01d-1"
    experimental: Literal[True] = True
    status: Literal["complete", "partial", "cancelled", "failed", "unsuitable"]
    source: RasterSource
    model: dict
    preprocessing: dict
    # Detections observed in analyzable pixels; never a true aircraft inventory.
    observed_count: int | None = Field(default=None, ge=0)
    scene_total: None = None
    accepted_count: int = Field(ge=0)
    uncertain_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    detections: list[RobustDetection] = Field(default_factory=list)
    coverage: CoverageSummary
    completed_tiles: int = Field(ge=0)
    planned_tiles: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    analysis_failure_reason: str | None = None
    requires_analyst_verification: Literal[True] = True
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def counts(self):
        for state in ("accepted", "uncertain", "rejected"):
            if getattr(self, state + "_count") != sum(
                d.status == state for d in self.detections
            ):
                raise ValueError("detection count mismatch")
        if self.status in ("failed", "unsuitable", "cancelled"):
            if self.observed_count is not None:
                raise ValueError("unsuccessful analysis cannot report a count")
        elif self.observed_count != self.accepted_count:
            raise ValueError("observed count does not reconcile")
        if self.completed_tiles > self.planned_tiles:
            raise ValueError("tile count mismatch")
        if self.status == "complete" and self.coverage.analyzable_fraction != 1:
            raise ValueError("complete analysis requires full analyzable coverage")
        if self.status == "partial" and self.coverage.analyzable_pixels == 0:
            raise ValueError("partial analysis requires analyzable pixels")
        return self
