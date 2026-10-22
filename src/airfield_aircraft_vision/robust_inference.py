"""Bounded native-pixel scene inference and an explicit experimental transport.

The AW-01C inference function is unchanged. Default IoU suppression is compatible
with that path. Optional boundary containment is a reviewable heuristic, never a
claim that overlapping aircraft or severe occlusions have been resolved.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path
from time import monotonic

from .capability import CapabilityTier, SceneMetadata, route_scene
from .geospatial_export import pixel_polygon
from .optical_contracts import PixelBox, RasterSource
from .postprocessing import iou
from .preprocessing import NativeRaster, tile_windows, sha256_file
from .robust_contracts import CoverageSummary, RobustDetection, RobustSceneResult


class _PixelCoverage:
    """Packed, exact union of native-pixel masks, bounded to one bit per pixel."""

    def __init__(self, width, height):
        import numpy as np

        self.width = width
        self.bits = np.zeros((height, ceil(width / 8)), dtype=np.uint8)
        self.count = 0

    def add(self, x, y, valid):
        import numpy as np

        h, w = valid.shape
        first, last = x // 8, ceil((x + w) / 8)
        old = np.unpackbits(self.bits[y : y + h, first:last], axis=1)
        selected = old[:, x % 8 : x % 8 + w]
        self.count += int(np.count_nonzero(valid & ~selected.astype(bool)))
        selected |= valid.astype(np.uint8)
        self.bits[y : y + h, first:last] = np.packbits(old, axis=1)


@dataclass
class _Candidate:
    box: PixelBox
    tile_index: int
    window: tuple[int, int, int, int]
    valid_fraction: float
    status: str
    reason: str | None
    flags: list[str]


def _boundary_fragment(small, large, width, height, threshold):
    """Require containment AND a smaller box clipped by an internal tile edge."""
    if small.tile_index == large.tile_index:
        return False
    sx1, sy1, sx2, sy2 = small.box.xyxy
    lx1, ly1, lx2, ly2 = large.box.xyxy
    small_area = (sx2 - sx1) * (sy2 - sy1)
    large_area = (lx2 - lx1) * (ly2 - ly1)
    intersection = max(0, min(sx2, lx2) - max(sx1, lx1)) * max(
        0, min(sy2, ly2) - max(sy1, ly1)
    )
    if large_area < small_area * 1.2 or intersection / small_area < threshold:
        return False
    x, y, w, h = small.window
    return any(
        (
            x > 0 and sx1 <= x + 1 and lx1 < x - 1,
            y > 0 and sy1 <= y + 1 and ly1 < y - 1,
            x + w < width and sx2 >= x + w - 1 and lx2 > x + w + 1,
            y + h < height and sy2 >= y + h - 1 and ly2 > y + h + 1,
        )
    )


def _deduplicate(candidates, source, nms_iou, mode, containment):
    eligible = sorted(
        (c for c in candidates if c.status != "rejected"),
        key=lambda c: (-c.box.score, c.box.xyxy, c.tile_index),
    )
    kept = []
    area = lambda c: (c.box.xyxy[2] - c.box.xyxy[0]) * (c.box.xyxy[3] - c.box.xyxy[1])
    for candidate in eligible:
        if mode == "boundary_containment":
            matches = [
                k
                for k in kept
                if _boundary_fragment(
                    candidate, k, source.width, source.height, containment
                )
                or _boundary_fragment(
                    k, candidate, source.width, source.height, containment
                )
            ]
            if matches:
                # Compare containment before IoU: a high-scoring truncated view
                # must not suppress a lower-scoring full view of the same object.
                retained = max(
                    [candidate] + matches, key=lambda c: (area(c), c.box.score)
                )
                retained.flags.append("heuristic_containment_requires_review")
                for fragment in [candidate] + matches:
                    if fragment is retained:
                        continue
                    fragment.status, fragment.reason = (
                        "rejected",
                        "cross_tile_boundary_containment_duplicate",
                    )
                    fragment.flags.append("heuristic_containment_requires_review")
                    if fragment in kept:
                        kept.remove(fragment)
                if retained is not candidate:
                    continue
        duplicate = next(
            (k for k in kept if iou(candidate.box.xyxy, k.box.xyxy) > nms_iou), None
        )
        if duplicate:
            candidate.status, candidate.reason = "rejected", "duplicate_after_scene_nms"
            continue
        kept.append(candidate)
    return candidates


def infer_robust_scene(
    path: Path,
    source: RasterSource,
    detector,
    *,
    confidence=None,
    uncertain_floor=None,
    tile_size=512,
    overlap=128,
    nms_iou=0.4,
    dedup_mode="iou",
    containment=0.9,
    min_valid_fraction=0.8,
    max_tiles=4096,
    max_pixels=100_000_000,
    max_seconds=1800,
    cancelled=None,
    progress=None,
    excluded_regions=(),
    allow_unknown_quality=False,
    validated_domain=None,
    expected_mask_sha256=None,
):
    """Return counts only for successful, suitable analysis.

    Cancellation is checked between tile calls; it cannot interrupt one detector
    invocation. Resource bounds reject oversized requests before opening pixels.
    Exclusions are original-grid integer xyxy rectangles supplied by an analyst.
    Confidence is a model score, not a calibrated probability. The uncertain band
    is floor <= score < accepted threshold and requires explicit analyst review.
    External GDAL masks require an explicit expected SHA-256; undeclared PAM
    sidecars are rejected because they can change nodata interpretation.
    """
    import numpy as np

    start = monotonic()
    manifest = dict(detector.manifest)
    confidence = float(
        manifest.get("selected_threshold", 0.5) if confidence is None else confidence
    )
    uncertain_floor = float(0.2 if uncertain_floor is None else uncertain_floor)
    if not 0 <= uncertain_floor <= confidence <= 1:
        raise ValueError("0 <= uncertain_floor <= confidence <= 1 required")
    if (
        not 0 <= nms_iou <= 1
        or not 0 < containment <= 1
        or not 0 <= min_valid_fraction <= 1
    ):
        raise ValueError("invalid suppression or valid-pixel fraction")
    if dedup_mode not in ("iou", "boundary_containment"):
        raise ValueError("unknown duplicate-suppression mode")
    if (
        type(max_pixels) is not int
        or type(max_tiles) is not int
        or max_pixels <= 0
        or max_tiles <= 0
        or not isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError(
            "finite positive runtime and integer pixel/tile bounds required"
        )
    # Do not materialize an unrestricted list of all tile windows.
    list(tile_windows(1, 1, tile_size, overlap))  # validate configuration
    origins = lambda n: (
        1 if n <= tile_size else ceil((n - tile_size) / (tile_size - overlap)) + 1
    )
    planned = origins(source.width) * origins(source.height)
    domain = dict(validated_domain) if validated_domain is not None else None
    if domain:
        expected_hash = domain.get("checkpoint_sha256")
        if expected_hash and expected_hash != manifest.get("checkpoint_sha256"):
            raise ValueError("validated-domain profile checkpoint hash mismatch")
        limits = domain.get("governing_resolution_range_m")
        if limits is not None and (
            len(limits) != 2
            or not all(isfinite(v) for v in limits)
            or not 0 < limits[0] <= limits[1]
        ):
            raise ValueError("invalid validated-domain resolution range")
    regions = []
    for region in excluded_regions:
        if len(region) != 4 or any(type(v) is not int for v in region):
            raise ValueError("exclusions require integer source-pixel xyxy rectangles")
        x1, y1, x2, y2 = region
        if not 0 <= x1 < x2 <= source.width or not 0 <= y1 < y2 <= source.height:
            raise ValueError("exclusion outside source grid")
        regions.append(tuple(region))
    preprocessing = {
        "profile": "native-rgb-v1",
        "resize": False,
        "reprojection": False,
        "tile_size": tile_size,
        "overlap": overlap,
        "confidence": confidence,
        "uncertain_floor": uncertain_floor,
        "nms_iou": nms_iou,
        "dedup_mode": dedup_mode,
        "boundary_containment": (
            containment if dedup_mode == "boundary_containment" else None
        ),
        "min_valid_fraction": min_valid_fraction,
        "max_tiles": max_tiles,
        "max_pixels": max_pixels,
        "max_seconds": max_seconds,
        "validated_domain": domain,
    }
    warnings = [
        "Experimental visible-aircraft detections; observed count is not aircraft inventory.",
        "All counts require analyst verification; aircraft type and heading are unavailable.",
        "Scores are uncalibrated; the uncertain band is not a statistical confidence interval.",
        "Nodata/analyst exclusions are masked; clouds, shadows and occlusion are not detected automatically.",
    ]
    domain_flags = []
    if domain is None:
        warnings.append(
            "No validated sensor/resolution domain profile supplied; model physical input bounds are not validation evidence."
        )
    if source.timestamp_timezone_basis != "provider_utc":
        warnings.append(
            "Acquisition UTC basis is assumed or unknown; preserve raw time and precision."
        )
    if source.crs is None or source.transform is None:
        warnings.append(
            "Georeferencing unavailable; pixel detections cannot be exported as geographic polygons."
        )
    if dedup_mode == "boundary_containment":
        warnings.append(
            "Boundary-containment suppression is heuristic; inspect overlapping or nested aircraft."
        )

    completed = 0
    candidates = []
    unavailable_tiles = []
    inspected = analyzable = None
    status, failure = "unsuitable", None

    def finish():
        nonlocal candidates
        candidates = _deduplicate(candidates, source, nms_iou, dedup_mode, containment)
        records = []
        for i, c in enumerate(candidates):
            flags = list(dict.fromkeys(c.flags))
            try:
                geometry = pixel_polygon(source, c.box.xyxy)
                if geometry:
                    import math

                    if not all(
                        math.isfinite(v)
                        for ring in geometry.coordinates
                        for point in ring
                        for v in point
                    ):
                        raise ValueError("nonfinite geographic detection polygon")
            except Exception:
                geometry = None
                flags.append("geographic_export_unavailable")
                warning = "Geographic transformation failed; affected detections retain source-pixel coordinates only."
                if warning not in warnings:
                    warnings.append(warning)
            records.append(
                RobustDetection(
                    detection_id=f"d-{i+1}",
                    status=c.status,
                    box=c.box,
                    geometry=geometry,
                    reason=c.reason,
                    uncertainty_flags=flags,
                    tile_index=c.tile_index,
                    tile_window=c.window,
                    valid_pixel_fraction=c.valid_fraction,
                )
            )
        accepted = sum(d.status == "accepted" for d in records)
        total = source.width * source.height
        inspected_count = inspected.count if inspected is not None else 0
        analyzable_count = analyzable.count if analyzable is not None else 0
        return RobustSceneResult(
            status=status,
            source=source,
            model=manifest,
            preprocessing=preprocessing,
            observed_count=accepted if status in ("complete", "partial") else None,
            accepted_count=accepted,
            uncertain_count=sum(d.status == "uncertain" for d in records),
            rejected_count=sum(d.status == "rejected" for d in records),
            detections=records,
            coverage=CoverageSummary(
                total_pixels=total,
                inspected_pixels=inspected_count,
                analyzable_pixels=analyzable_count,
                inspected_fraction=inspected_count / total,
                analyzable_fraction=analyzable_count / total,
                excluded_regions=regions,
                unavailable_tile_windows=unavailable_tiles,
            ),
            completed_tiles=completed,
            planned_tiles=planned,
            elapsed_seconds=max(0, monotonic() - start),
            analysis_failure_reason=failure,
            warnings=warnings,
        )

    decision = route_scene(
        SceneMetadata(
            sensor_domain=source.sensor_domain,
            native_gsd_m=source.native_gsd_m,
            effective_spatial_resolution_m=source.effective_resolution_m,
            panchromatic_gsd_m=source.panchromatic_gsd_m,
            pan_sharpened_gsd_m=source.pan_sharpened_gsd_m,
            image_quality=source.quality,
        )
    )
    warnings.extend(decision.warnings)
    if domain:
        sensors = domain.get("sensor_ids")
        if sensors is not None and source.sensor_id not in sensors:
            domain_flags.append("unvalidated_sensor_domain")
            warnings.append(
                "Source sensor is outside the supplied validation domain; observed counts require independent validation."
            )
        limits = domain.get("governing_resolution_range_m")
        if (
            limits is not None
            and decision.governing_resolution_m is not None
            and not limits[0] <= decision.governing_resolution_m <= limits[1]
        ):
            domain_flags.append("unvalidated_resolution_domain")
            warnings.append(
                "Source physical resolution is outside the supplied validation domain despite passing experimental input bounds."
            )
    bounds = manifest.get("experimental_input_bounds_m", [0.2, 1.0])
    if decision.tier != CapabilityTier.A:
        failure = "resolution_or_quality_gate: " + decision.reason
        return finish()
    if source.quality == "unknown" and not allow_unknown_quality:
        failure = "unknown_quality_requires_explicit_review"
        return finish()
    if not bounds[0] <= decision.governing_resolution_m <= bounds[1]:
        failure = "outside_experimental_model_physical_resolution_bounds"
        return finish()
    if source.width * source.height > max_pixels or planned > max_tiles:
        failure = "request_exceeds_explicit_scene_resource_bounds"
        return finish()
    if manifest.get("candidate_score_floor", 0.005) > uncertain_floor:
        warnings.append(
            "Detector candidate floor exceeds requested uncertainty floor; lower-score candidates are unavailable."
        )
    try:
        mask_path = Path(str(path) + ".msk")
        sidecar_path = Path(str(path) + ".aux.xml")
        if sidecar_path.is_file() and not source.sidecar_sha256:
            raise ValueError(
                "undeclared_aux_xml_sidecar_requires_source_sha256_binding"
            )
        if expected_mask_sha256 is not None and (
            not isinstance(expected_mask_sha256, str)
            or len(expected_mask_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_mask_sha256)
        ):
            raise ValueError("invalid_expected_external_mask_sha256")
        if mask_path.is_file():
            if expected_mask_sha256 is None:
                raise ValueError("external_mask_requires_explicit_sha256_binding")
            actual_mask_sha256 = sha256_file(mask_path)
            if actual_mask_sha256 != expected_mask_sha256:
                raise ValueError("external_mask_sha256_mismatch")
            preprocessing["external_mask"] = {
                "filename": mask_path.name,
                "sha256": actual_mask_sha256,
            }
        elif expected_mask_sha256 is not None:
            raise ValueError("declared_external_mask_is_missing")
        inspected = _PixelCoverage(source.width, source.height)
        analyzable = _PixelCoverage(source.width, source.height)
        with NativeRaster(path, source) as raster:
            for index, window in enumerate(
                tile_windows(source.width, source.height, tile_size, overlap)
            ):
                if (
                    cancelled is not None and cancelled()
                ) or monotonic() - start > max_seconds:
                    status, failure = "cancelled", "cancelled_or_time_budget_exceeded"
                    return finish()
                x, y, w, h = window
                pixels, native_valid = raster.read(window, tile_size)
                valid = native_valid.copy()
                for ex1, ey1, ex2, ey2 in regions:
                    left, top = max(0, ex1 - x), max(0, ey1 - y)
                    right, bottom = min(w, ex2 - x), min(h, ey2 - y)
                    if right > left and bottom > top:
                        valid[top:bottom, left:right] = False
                pixels[:, :h, :w] = np.where(valid[None], pixels[:, :h, :w], 0)
                if not valid.all():
                    unavailable_tiles.append(window)
                output = detector.predict(pixels) if valid.any() else []
                tile_candidates = []
                for raw in output:
                    box = (
                        raw
                        if isinstance(raw, PixelBox)
                        else PixelBox.model_validate(raw)
                    )
                    x1, y1, x2, y2 = box.xyxy
                    # Clip before checking finite area; no padded pixel receives coordinates.
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    left, top, right, bottom = (
                        max(0, x1),
                        max(0, y1),
                        min(w, x2),
                        min(h, y2),
                    )
                    if right <= left or bottom <= top:
                        continue
                    flags = list(domain_flags) + [
                        "experimental_uncalibrated_score",
                        "axis_aligned_box_no_heading",
                    ]
                    center_valid = (
                        0 <= cx < w and 0 <= cy < h and valid[int(cy), int(cx)]
                    )
                    roi = valid[
                        int(np.floor(top)) : int(np.ceil(bottom)),
                        int(np.floor(left)) : int(np.ceil(right)),
                    ]
                    fraction = float(np.count_nonzero(roi) / roi.size)
                    state, reason = "accepted", None
                    if not center_valid:
                        state, reason = (
                            "rejected",
                            "center_in_padding_nodata_or_exclusion",
                        )
                    elif box.score < uncertain_floor:
                        state, reason = "rejected", "below_uncertainty_floor"
                    elif box.score < confidence:
                        state, reason = "uncertain", "below_accepted_score_threshold"
                    if fraction < min_valid_fraction and state != "rejected":
                        state, reason = "uncertain", "insufficient_valid_box_pixels"
                    if fraction < 1:
                        flags.append("box_intersects_unavailable_pixels")
                    scene_box = PixelBox(
                        xyxy=(left + x, top + y, right + x, bottom + y), score=box.score
                    )
                    if (
                        left + x == 0
                        or top + y == 0
                        or right + x == source.width
                        or bottom + y == source.height
                    ):
                        flags.append("touches_source_edge_partial_aircraft_possible")
                    if (
                        (left == 0 and x > 0)
                        or (top == 0 and y > 0)
                        or (right == w and x + w < source.width)
                        or (bottom == h and y + h < source.height)
                    ):
                        flags.append("touches_internal_tile_edge_fragment_possible")
                    tile_candidates.append(
                        _Candidate(
                            scene_box, index, window, fraction, state, reason, flags
                        )
                    )
                # Failed tiles never contribute candidates or inspected coverage.
                candidates.extend(tile_candidates)
                inspected.add(x, y, np.ones_like(valid, dtype=bool))
                analyzable.add(x, y, valid)
                completed += 1
                if progress is not None:
                    progress(completed, planned)
        if analyzable.count == 0:
            status, failure = "unsuitable", "no_analyzable_source_pixels"
        else:
            status = (
                "complete"
                if analyzable.count == source.width * source.height
                else "partial"
            )
    except Exception as exc:
        status, failure = "failed", f"{type(exc).__name__}: {str(exc)[:500]}"
    return finish()


def scene_feature_collection(result: RobustSceneResult):
    """WGS84 polygons plus exact pixel boxes; timestamps retain precision/basis."""
    metadata = result.model_dump(mode="json", exclude={"detections"})
    features = []
    for record in result.detections:
        features.append(
            {
                "type": "Feature",
                "id": record.detection_id,
                "geometry": (
                    record.geometry.model_dump(mode="json") if record.geometry else None
                ),
                "properties": record.model_dump(mode="json", exclude={"geometry"}),
            }
        )
    return {
        "type": "FeatureCollection",
        "metadata": metadata,
        "features": features,
    }
