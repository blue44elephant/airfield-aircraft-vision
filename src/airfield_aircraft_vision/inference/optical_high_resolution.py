"""AW-01 experimental aircraft counting on the untouched source pixel grid."""

from pathlib import Path
from ..capability import SceneMetadata, route_scene, CapabilityTier
from ..optical_contracts import OpticalResult, PixelBox, RasterSource
from ..schemas import Detection, OrientedBox, ImageQualityAssessment, InferenceResult
from ..preprocessing import NativeRaster, tile_windows
from ..postprocessing import suppress
from ..geospatial_export import pixel_polygon


def infer_scene(
    path: Path,
    source: RasterSource,
    detector,
    *,
    tile_size=512,
    overlap=128,
    confidence=0.5,
    nms_iou=0.4,
    allow_unknown_quality=False,
):
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0,1]")
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
    if decision.tier != CapabilityTier.A:
        raise ValueError("AW-01 requires Tier A optical imagery: " + decision.reason)
    if source.quality == "unknown" and not allow_unknown_quality:
        raise ValueError(
            "unknown quality requires explicit experimental review/override"
        )
    bounds = detector.manifest.get("experimental_input_bounds_m", [0.2, 1.0])
    if not bounds[0] <= decision.governing_resolution_m <= bounds[1]:
        raise ValueError("scene outside experimental model physical-resolution bounds")
    candidates = []
    rejected = []
    invalid_tiles = 0
    valid_tiles = 0
    partial_nodata = False
    with NativeRaster(path, source) as raster:
        for window in tile_windows(source.width, source.height, tile_size, overlap):
            x, y, w, h = window
            pixels, valid = raster.read(window, tile_size)
            if not valid.any():
                invalid_tiles += 1
                continue
            valid_tiles += 1
            partial_nodata = partial_nodata or not valid.all()
            for box in detector.predict(pixels):
                x1, y1, x2, y2 = box.xyxy
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                # Never count padded or nodata-only candidates.
                if not (0 <= cx < w and 0 <= cy < h) or not valid[int(cy), int(cx)]:
                    continue
                b = PixelBox(
                    xyxy=(
                        max(0, x1) + x,
                        max(0, y1) + y,
                        min(w, x2) + x,
                        min(h, y2) + y,
                    ),
                    score=box.score,
                )
                if b.score < confidence:
                    rejected.append((b, "below_confidence_threshold"))
                else:
                    candidates.append(b)
    if not valid_tiles:
        raise ValueError("no valid source pixels; an empty count would be misleading")
    accepted, duplicates = suppress(candidates, nms_iou)
    rejected.extend((b, "duplicate_after_scene_nms") for b in duplicates)
    records = []
    for b, reason in [(b, None) for b in accepted] + rejected:
        x1, y1, x2, y2 = b.xyxy
        flags = [
            "experimental_uncalibrated_confidence",
            "axis_aligned_box_orientation_not_estimated",
        ]
        if x1 == 0 or y1 == 0 or x2 == source.width or y2 == source.height:
            flags.append("touches_source_edge")
        records.append(
            Detection(
                detection_id=f"d-{len(records)+1}",
                status="rejected" if reason else "accepted",
                oriented_box=OrientedBox(
                    center_x=(x1 + x2) / 2,
                    center_y=(y1 + y2) / 2,
                    width=x2 - x1,
                    height=y2 - y1,
                    angle_degrees_clockwise_from_north=0,
                ),
                geometry=pixel_polygon(source, b.xyxy),
                predicted_class=None if reason else "unknown_aircraft",
                class_probabilities={} if reason else {"unknown_aircraft": 1.0},
                confidence=b.score,
                uncertainty_flags=flags,
                rejection_reason=reason,
            )
        )
    warnings = list(decision.warnings) + [
        "Experimental generic-aircraft detector; confidence is not calibrated probability of correctness.",
        "Axis-aligned rectangles do not estimate aircraft heading or type.",
        "Counts describe visible detections, not true inventory; clouds, occlusion and duplicate fragments remain limitations.",
        "unknown_aircraft probability 1 is a conditional taxonomy assignment, not detection certainty.",
    ]
    if source.timestamp_timezone_basis != "provider_utc":
        warnings.append(
            "Source timestamp UTC basis is assumed or unknown; exact UTC transport export withheld."
        )
    if partial_nodata:
        warnings.append("Source contains nodata; count covers valid pixels only.")
    if invalid_tiles:
        warnings.append(f"{invalid_tiles} entirely invalid tiles skipped")
    aoi = pixel_polygon(source, (0, 0, source.width, source.height))
    legacy = None
    if (
        aoi
        and source.acquisition_timestamp
        and source.timestamp_precision in ("minute", "second")
        and source.timestamp_timezone_basis == "provider_utc"
    ):
        legacy = InferenceResult(
            aoi=aoi,
            source_scene_id=source.scene_id,
            sensor_id=source.sensor_id,
            acquisition_timestamp=source.acquisition_timestamp,
            native_resolution_m=source.native_gsd_m,
            effective_resolution_m=source.effective_resolution_m,
            selected_capability_tier=decision.tier,
            image_quality=ImageQualityAssessment(status=source.quality),
            detections=records,
            total_detection_count=len(accepted),
            total_candidate_detections=len(records),
            known_classified_detections=0,
            unknown_detections=len(accepted),
            rejected_detections=len(rejected),
            per_class_totals={"unknown_aircraft": len(accepted)},
            warnings=warnings,
            unavailable_features=["aircraft_type", "aircraft_heading"],
            unavailable_feature_reasons={
                "aircraft_type": "AW-01 is single-class detection",
                "aircraft_heading": "AW-01 uses axis-aligned boxes",
            },
        )
    return OpticalResult(
        source=source,
        model=detector.manifest,
        preprocessing={
            "profile": "native-rgb-v1",
            "tile_size": tile_size,
            "overlap": overlap,
            "resize": False,
            "reprojection": False,
            "confidence": confidence,
            "nms_iou": nms_iou,
            "candidate_floor": detector.manifest.get("candidate_score_floor", 0.05),
            "padding": "zeros; candidates in padding excluded",
        },
        detections=records,
        total_count=len(accepted),
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        warnings=warnings,
        legacy_inference_result=legacy,
    )
