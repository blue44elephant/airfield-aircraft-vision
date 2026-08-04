"""Provider-neutral metadata-first imagery access protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Literal, Protocol


GeoJSON = dict[str, Any]


@dataclass(frozen=True)
class SceneSearchRequest:
    aoi: GeoJSON
    start_time: datetime
    end_time: datetime
    sensor_domains: tuple[Literal["optical", "sar"], ...] = ("optical",)
    max_cloud_fraction: float | None = None


@dataclass(frozen=True)
class SceneMetadataRecord:
    provider_id: str
    scene_id: str
    sensor_id: str
    sensor_domain: Literal["optical", "sar"]
    acquisition_time: datetime
    footprint: GeoJSON
    native_gsd_m: float | None
    effective_spatial_resolution_m: float | None = None
    bands: tuple[str, ...] = ()
    cloud_cover_fraction: float | None = None
    licence_id: str | None = None
    licence_url: str | None = None
    preview_url: str | None = None
    download_eligible: bool = False
    authentication_required: bool = False
    coverage_mode: Literal["archive_only", "recurring", "unknown"] = "unknown"
    acquisition_mode: Literal["archive", "tasking", "user_supplied", "unknown"] = "unknown"
    estimated_download_size_bytes: int | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


class ImageryProvider(Protocol):
    """Adapter boundary; implementations own auth and provider-specific semantics."""

    provider_id: str

    def search(self, request: SceneSearchRequest) -> AsyncIterator[SceneMetadataRecord]:
        """Yield metadata records without downloading imagery."""
        ...

    async def get_metadata(self, scene_id: str) -> SceneMetadataRecord:
        """Refresh and normalize one scene's metadata."""
        ...

    async def download(self, scene_id: str, destination: str) -> str:
        """Download an eligible scene and return its local path."""
        ...
