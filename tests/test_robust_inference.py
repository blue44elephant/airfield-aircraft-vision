"""Deterministic AW-01D native-grid and safe scene-result regressions."""

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from airfield_aircraft_vision.optical_contracts import PixelBox, RasterSource
from airfield_aircraft_vision.preprocessing import sha256_file
from airfield_aircraft_vision.robust_contracts import CoverageSummary, RobustSceneResult
from airfield_aircraft_vision.robust_inference import (
    _PixelCoverage,
    infer_robust_scene,
    scene_feature_collection,
)


@pytest.fixture
def scene(tmp_path):
    np = pytest.importorskip("numpy")
    rio = pytest.importorskip("rasterio")

    def create(width=768, height=512, mask=None, rotated=False):
        path = tmp_path / f"scene-{width}-{height}-{rotated}.tif"
        affine = rio.Affine(
            0.5, 0.1 if rotated else 0, 500000, 0.05 if rotated else 0, -0.5, 2000000
        )
        pixels = np.zeros((3, height, width), dtype=np.uint8)
        pixels[0] = np.arange(width, dtype=np.uint8)[None, :]
        pixels[1] = np.arange(height, dtype=np.uint8)[:, None]
        pixels[2] = 127
        with rio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=3,
            dtype="uint8",
            crs="EPSG:32643",
            transform=affine,
        ) as raster:
            raster.write(pixels)
            if mask is not None:
                raster.write_mask(mask.astype(np.uint8) * 255)
        source = RasterSource(
            scene_id="synthetic-native",
            sensor_id="synthetic",
            native_gsd_m=0.5,
            quality="acceptable",
            sha256=sha256_file(path),
            width=width,
            height=height,
            crs="EPSG:32643",
            transform=tuple(affine)[:6],
            airfield_group="unseen",
            capture_group="new",
            acquisition_timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            timestamp_precision="date",
            acquisition_time_raw="2020-01-01",
            timestamp_timezone_basis="unknown",
            license="synthetic",
            attribution="unit test",
        )
        return path, source, pixels

    return create


class Detector:
    manifest = {
        "architecture": "deterministic-test",
        "checkpoint_sha256": "a" * 64,
        "selected_threshold": 0.7,
        "candidate_score_floor": 0.005,
        "experimental_input_bounds_m": [0.2, 1],
    }

    def __init__(self, responses):
        self.responses = iter(responses)
        self.inputs = []

    def predict(self, pixels):
        self.inputs.append(pixels.copy())
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def b(coords=(300, 100, 340, 140), score=0.9):
    return PixelBox(xyxy=coords, score=score)


def test_exact_source_coordinates_overlap_suppression_and_no_resampling(scene):
    np = pytest.importorskip("numpy")
    path, source, pixels = scene()
    before = sha256_file(path)
    detector = Detector([[b()], [b((44, 100, 84, 140), 0.8)]])
    result = infer_robust_scene(path, source, detector)
    assert result.status == "complete" and result.observed_count == 1
    assert result.accepted_count == 1 and result.rejected_count == 1
    assert result.coverage.analyzable_pixels == 768 * 512
    assert (
        result.coverage.inspected_fraction == result.coverage.analyzable_fraction == 1
    )
    assert result.detections[0].box.xyxy == (300, 100, 340, 140)
    np.testing.assert_array_equal(detector.inputs[0], pixels[:, :, :512])
    np.testing.assert_array_equal(detector.inputs[1], pixels[:, :, 256:])
    assert (
        result.preprocessing["resize"] is result.preprocessing["reprojection"] is False
    )
    assert sha256_file(path) == before


def test_valid_empty_source_reports_observed_zero_not_inventory(scene):
    path, source, _ = scene()
    result = infer_robust_scene(path, source, Detector([[], []]))
    assert result.observed_count == 0 and result.status == "complete"
    assert result.requires_analyst_verification
    assert result.scene_total is None


def test_uncertain_scores_are_not_accepted_or_discarded(scene):
    path, source, _ = scene(width=512)
    result = infer_robust_scene(
        path,
        source,
        Detector(
            [
                [
                    b((20, 20, 40, 40), 0.7),
                    b((60, 20, 80, 40), 0.4),
                    b((100, 20, 120, 40), 0.1),
                ]
            ]
        ),
    )
    assert result.observed_count == 1
    assert result.uncertain_count == result.rejected_count == 1
    assert [d.status for d in result.detections] == [
        "accepted",
        "uncertain",
        "rejected",
    ]
    assert result.preprocessing["uncertain_floor"] == 0.2


def test_clipped_internal_fragment_can_be_suppressed_even_with_higher_score(scene):
    path, source, _ = scene()
    outputs = [[b((490, 100, 512, 140), 0.95)], [b((234, 100, 314, 140), 0.8)]]
    baseline = infer_robust_scene(path, source, Detector(outputs))
    assert baseline.observed_count == 2  # low IoU fails ordinary NMS
    safer = infer_robust_scene(
        path, source, Detector(outputs), dedup_mode="boundary_containment"
    )
    assert safer.observed_count == 1
    retained = next(d for d in safer.detections if d.status == "accepted")
    assert retained.box.xyxy == (490, 100, 570, 140)
    assert any(
        d.reason == "cross_tile_boundary_containment_duplicate"
        for d in safer.detections
    )
    assert "heuristic_containment_requires_review" in retained.uncertainty_flags


def test_containment_does_not_suppress_same_tile_nested_or_close_aircraft(scene):
    path, source, _ = scene(width=512)
    result = infer_robust_scene(
        path,
        source,
        Detector(
            [
                [
                    b((100, 100, 200, 200)),
                    b((120, 120, 130, 130), 0.85),
                    b((201, 100, 240, 140), 0.8),
                ]
            ]
        ),
        dedup_mode="boundary_containment",
    )
    assert result.observed_count == 3


def test_containment_without_internal_boundary_evidence_is_not_forced(scene):
    path, source, _ = scene()
    result = infer_robust_scene(
        path,
        source,
        Detector([[b((300, 100, 400, 200))], [b((60, 120, 70, 130), 0.85)]]),
        dedup_mode="boundary_containment",
    )
    assert result.observed_count == 2


def test_nodata_and_exclusions_report_exact_partial_coverage(scene):
    np = pytest.importorskip("numpy")
    mask = np.ones((512, 768), dtype=bool)
    mask[:100, :100] = False
    path, source, _ = scene(mask=mask)
    detector = Detector([[b((20, 20, 40, 40)), b((150, 150, 180, 180))], []])
    result = infer_robust_scene(
        path, source, detector, excluded_regions=[(100, 100, 200, 200)]
    )
    assert result.status == "partial" and result.observed_count == 0
    assert result.coverage.analyzable_pixels == 768 * 512 - 20_000
    assert result.coverage.inspected_pixels == 768 * 512
    assert all(d.status == "rejected" for d in result.detections)
    assert not detector.inputs[0][:, :100, :100].any()
    assert not detector.inputs[0][:, 100:200, 100:200].any()


def test_candidate_intersecting_missing_pixels_is_uncertain(scene):
    np = pytest.importorskip("numpy")
    mask = np.ones((512, 512), dtype=bool)
    mask[100:120, 100:160] = False
    path, source, _ = scene(width=512, mask=mask)
    # Center is valid, but only two thirds of the box is valid.
    result = infer_robust_scene(path, source, Detector([[b((100, 100, 160, 160))]]))
    assert result.uncertain_count == 1 and result.observed_count == 0
    assert result.detections[0].reason == "insufficient_valid_box_pixels"


def test_all_nodata_is_not_zero_aircraft(scene):
    np = pytest.importorskip("numpy")
    path, source, _ = scene(mask=np.zeros((512, 768), dtype=bool))
    detector = Detector([])
    result = infer_robust_scene(path, source, detector)
    assert result.status == "unsuitable" and result.observed_count is None
    assert result.analysis_failure_reason == "no_analyzable_source_pixels"
    assert len(detector.inputs) == 0


@pytest.mark.parametrize(
    "change",
    [
        {"native_gsd_m": 10},
        {"quality": "poor"},
        {"quality": "unusable"},
        {"sensor_domain": "sar"},
        {"native_gsd_m": 0.1},
        {"quality": "unknown"},
    ],
)
def test_unsuitable_source_is_not_counted_and_model_not_called(scene, change):
    path, source, _ = scene()
    detector = Detector([])
    result = infer_robust_scene(path, source.model_copy(update=change), detector)
    assert result.status == "unsuitable" and result.observed_count is None
    assert len(detector.inputs) == 0


def test_unknown_quality_only_runs_after_explicit_override(scene):
    path, source, _ = scene(width=512)
    result = infer_robust_scene(
        path,
        source.model_copy(update={"quality": "unknown"}),
        Detector([[]]),
        allow_unknown_quality=True,
    )
    assert result.observed_count == 0
    assert any("quality" in warning.lower() for warning in result.warnings)


def test_model_failure_preserves_failure_not_false_zero(scene):
    path, source, _ = scene()
    result = infer_robust_scene(
        path, source, Detector([[b()], RuntimeError("synthetic failure")])
    )
    assert result.status == "failed" and result.observed_count is None
    assert result.accepted_count == 1  # diagnostic only, not a successful scene count
    assert result.completed_tiles == 1 and result.coverage.inspected_pixels == 512 * 512
    assert "synthetic failure" in result.analysis_failure_reason


def test_cancellation_null_count_preserves_progress_and_diagnostics(scene):
    path, source, _ = scene()
    progress = []
    result = infer_robust_scene(
        path,
        source,
        Detector([[b()]]),
        progress=lambda n, total: progress.append((n, total)),
        cancelled=lambda: bool(progress),
    )
    assert progress == [(1, 2)]
    assert result.status == "cancelled" and result.observed_count is None
    assert result.accepted_count == 1 and result.completed_tiles == 1


@pytest.mark.parametrize("kwargs", [{"max_pixels": 10}, {"max_tiles": 1}])
def test_bounds_reject_before_open_or_inference(scene, kwargs):
    path, source, _ = scene()
    result = infer_robust_scene(
        Path("deliberately-missing"), source, Detector([]), **kwargs
    )
    assert result.status == "unsuitable" and result.observed_count is None
    assert (
        result.analysis_failure_reason
        == "request_exceeds_explicit_scene_resource_bounds"
    )


def test_checksum_failure_is_explicit_not_zero(scene):
    path, source, _ = scene()
    result = infer_robust_scene(
        path, source.model_copy(update={"sha256": "f" * 64}), Detector([])
    )
    assert result.status == "failed" and result.observed_count is None
    assert "checksum" in result.analysis_failure_reason


def test_padding_is_not_counted(scene):
    path, source, pixels = scene(width=80, height=64)
    detector = Detector([[b((100, 100, 120, 120)), b((20, 20, 30, 30))]])
    result = infer_robust_scene(path, source, detector)
    assert result.observed_count == 1
    assert not detector.inputs[0][:, 64:, :].any()
    assert not detector.inputs[0][:, :, 80:].any()
    assert result.coverage.analyzable_pixels == 80 * 64


def test_rotated_affine_exports_original_grid_without_inventing_time(scene):
    rio = pytest.importorskip("rasterio")
    from rasterio.warp import transform

    path, source, _ = scene(width=512, rotated=True)
    result = infer_robust_scene(path, source, Detector([[b()]]))
    exported = scene_feature_collection(result)
    json.dumps(exported, allow_nan=False)
    meta = exported["metadata"]
    assert meta["model"]["checkpoint_sha256"] == "a" * 64
    assert meta["source"]["timestamp_precision"] == "date"
    assert meta["source"]["timestamp_timezone_basis"] == "unknown"
    assert meta["source"]["acquisition_time_raw"] == "2020-01-01"
    feature = exported["features"][0]
    assert feature["properties"]["box"]["xyxy"] == [300, 100, 340, 140]
    affine = rio.Affine(*source.transform)
    x, y = affine * (300, 100)
    lon, lat = transform(source.crs, "EPSG:4326", [x], [y])
    assert feature["geometry"]["coordinates"][0][0] == pytest.approx([lon[0], lat[0]])


def test_missing_georef_does_not_invent_polygon(scene):
    path, source, _ = scene(width=512)
    result = infer_robust_scene(
        path,
        source.model_copy(update={"crs": None, "transform": None}),
        Detector([[b()]]),
    )
    assert scene_feature_collection(result)["features"][0]["geometry"] is None


def test_packed_coverage_exact_nonbyte_aligned_overlaps():
    np = pytest.importorskip("numpy")
    coverage = _PixelCoverage(19, 9)
    expected = np.zeros((9, 19), dtype=bool)
    rng = np.random.default_rng(17)
    for x, y, w, h in [(3, 2, 9, 5), (1, 1, 17, 7), (10, 5, 9, 4), (0, 0, 19, 9)]:
        mask = rng.integers(0, 2, size=(h, w)).astype(bool)
        coverage.add(x, y, mask)
        expected[y : y + h, x : x + w] |= mask
        assert coverage.count == int(expected.sum())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"uncertain_floor": 0.8, "confidence": 0.7},
        {"tile_size": 511},
        {"overlap": 512},
        {"max_seconds": 0},
        {"dedup_mode": "unknown"},
        {"excluded_regions": [(0, 0, 9999, 4)]},
        {"excluded_regions": [(0.0, 0, 4, 4)]},
    ],
)
def test_invalid_options_are_rejected(scene, kwargs):
    path, source, _ = scene()
    with pytest.raises(ValueError):
        infer_robust_scene(path, source, Detector([]), **kwargs)


def test_result_contract_rejects_fabricated_zero_after_failure(scene):
    path, source, _ = scene(width=512)
    result = infer_robust_scene(path, source, Detector([RuntimeError("failed")]))
    data = result.model_dump()
    data["observed_count"] = 0
    with pytest.raises(ValidationError, match="cannot report a count"):
        RobustSceneResult.model_validate(data)


def test_coverage_contract_rejects_inconsistent_fractions():
    with pytest.raises(ValidationError, match="fractions"):
        CoverageSummary(
            total_pixels=100,
            inspected_pixels=100,
            analyzable_pixels=50,
            inspected_fraction=1,
            analyzable_fraction=0.7,
        )


def test_containment_precedes_iou_to_preserve_full_view(scene):
    path, source, _ = scene()
    # Fragment IoU=0.75 would otherwise suppress the more complete view.
    result = infer_robust_scene(
        path,
        source,
        Detector([[b((450, 100, 512, 140), 0.95)], [b((194, 100, 274, 140), 0.8)]]),
        dedup_mode="boundary_containment",
    )
    accepted = [d for d in result.detections if d.status == "accepted"]
    assert len(accepted) == 1
    assert accepted[0].box.xyxy == (450, 100, 530, 140)
    assert result.rejected_count == 1


def test_geographic_transform_failure_keeps_pixels_explicitly(scene, monkeypatch):
    import airfield_aircraft_vision.robust_inference as module

    path, source, _ = scene(width=512)

    def broken(*args):
        raise ValueError("synthetic geographic transformation failure")

    monkeypatch.setattr(module, "pixel_polygon", broken)
    result = infer_robust_scene(path, source, Detector([[b()]]))
    assert result.observed_count == 1
    assert result.detections[0].geometry is None
    assert "geographic_export_unavailable" in result.detections[0].uncertainty_flags
    assert any("transformation failed" in warning for warning in result.warnings)


def test_entire_analyst_exclusion_never_reports_zero(scene):
    path, source, _ = scene(width=512)
    result = infer_robust_scene(
        path, source, Detector([]), excluded_regions=[(0, 0, 512, 512)]
    )
    assert result.status == "unsuitable" and result.observed_count is None
    assert result.coverage.analyzable_pixels == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_seconds": float("nan")},
        {"max_seconds": float("inf")},
        {"max_pixels": float("nan")},
        {"max_pixels": 20.5},
        {"max_tiles": True},
    ],
)
def test_nonfinite_or_noninteger_bounds_cannot_disable_limits(scene, kwargs):
    path, source, _ = scene()
    with pytest.raises(ValueError, match="bounds"):
        infer_robust_scene(path, source, Detector([]), **kwargs)


def test_unavailable_locations_are_explicitly_coarse_native_windows(scene):
    import numpy as np

    mask = np.ones((512, 768), dtype=bool)
    mask[100:110, 100:110] = False
    path, source, _ = scene(mask=mask)
    result = infer_robust_scene(path, source, Detector([[], []]))
    assert result.coverage.unavailable_tile_windows == [(0, 0, 512, 512)]
    assert (
        result.coverage.unavailable_locations_precision
        == "coarse_tile_windows_not_exact_masks"
    )
    assert result.coverage.analyzable_pixels == 768 * 512 - 100


def test_unvalidated_sensor_and_resolution_domain_are_explicit(scene):
    path, source, _ = scene(width=512)
    profile = {
        "checkpoint_sha256": "a" * 64,
        "sensor_ids": ["WorldView-3"],
        "governing_resolution_range_m": [0.311, 0.393],
    }
    result = infer_robust_scene(
        path, source, Detector([[b()]]), validated_domain=profile
    )
    assert result.observed_count == 1  # research inference remains possible
    assert "unvalidated_sensor_domain" in result.detections[0].uncertainty_flags
    assert "unvalidated_resolution_domain" in result.detections[0].uncertainty_flags
    assert result.preprocessing["validated_domain"] == profile


def test_wrong_checkpoint_domain_profile_is_rejected(scene):
    path, source, _ = scene(width=512)
    with pytest.raises(ValueError, match="hash mismatch"):
        infer_robust_scene(
            path, source, Detector([]), validated_domain={"checkpoint_sha256": "b" * 64}
        )


def test_no_profile_does_not_imply_model_bounds_are_validation_evidence(scene):
    path, source, _ = scene(width=512)
    result = infer_robust_scene(path, source, Detector([[]]))
    assert any("not validation evidence" in warning for warning in result.warnings)


def test_matching_domain_has_no_out_of_domain_detection_flags(scene):
    path, source, _ = scene(width=512)
    result = infer_robust_scene(
        path,
        source,
        Detector([[b()]]),
        validated_domain={
            "sensor_ids": ["synthetic"],
            "governing_resolution_range_m": [0.4, 0.6],
        },
    )
    assert not any(
        flag.startswith("unvalidated_")
        for flag in result.detections[0].uncertainty_flags
    )


def _write_external_mask(path, mask):
    import rasterio

    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        with rasterio.open(path, "r+") as dataset:
            dataset.write_mask(mask.astype("uint8") * 255)
    mask_path = Path(str(path) + ".msk")
    assert mask_path.is_file()
    return mask_path


def test_external_mask_without_binding_cannot_produce_trusted_zero(scene):
    import numpy as np

    path, source, _ = scene(width=512)
    _write_external_mask(path, np.zeros((512, 512), dtype=bool))
    assert sha256_file(path) == source.sha256
    detector = Detector([])
    result = infer_robust_scene(path, source, detector)
    assert result.status == "failed" and result.observed_count is None
    assert (
        "external_mask_requires_explicit_sha256_binding"
        in result.analysis_failure_reason
    )
    assert not detector.inputs


def test_external_mask_with_wrong_binding_fails_before_prediction(scene):
    import numpy as np

    path, source, _ = scene(width=512)
    _write_external_mask(path, np.ones((512, 512), dtype=bool))
    result = infer_robust_scene(
        path, source, Detector([]), expected_mask_sha256="f" * 64
    )
    assert result.status == "failed" and result.observed_count is None
    assert "external_mask_sha256_mismatch" in result.analysis_failure_reason


def test_bound_external_mask_metadata_and_later_mutation_are_detected(scene):
    import numpy as np

    path, source, _ = scene(width=512)
    mask_path = _write_external_mask(path, np.ones((512, 512), dtype=bool))
    mask_hash = sha256_file(mask_path)
    result = infer_robust_scene(
        path, source, Detector([[]]), expected_mask_sha256=mask_hash
    )
    assert result.status == "complete" and result.observed_count == 0
    assert result.preprocessing["external_mask"] == {
        "filename": mask_path.name,
        "sha256": mask_hash,
    }
    _write_external_mask(path, np.zeros((512, 512), dtype=bool))
    assert sha256_file(path) == source.sha256
    assert sha256_file(mask_path) != mask_hash
    mutated = infer_robust_scene(
        path, source, Detector([]), expected_mask_sha256=mask_hash
    )
    assert mutated.status == "failed" and mutated.observed_count is None
    assert "external_mask_sha256_mismatch" in mutated.analysis_failure_reason


def test_declared_missing_external_mask_cannot_be_silently_ignored(scene):
    path, source, _ = scene(width=512)
    result = infer_robust_scene(
        path, source, Detector([]), expected_mask_sha256="a" * 64
    )
    assert result.status == "failed" and result.observed_count is None
    assert "declared_external_mask_is_missing" in result.analysis_failure_reason


def test_undeclared_pam_sidecar_cannot_change_nodata_silently(scene):
    path, source, _ = scene(width=512)
    Path(str(path) + ".aux.xml").write_text(
        "<PAMDataset><PAMRasterBand band='1'><NoDataValue>0</NoDataValue></PAMRasterBand></PAMDataset>"
    )
    result = infer_robust_scene(path, source, Detector([]))
    assert result.status == "failed" and result.observed_count is None
    assert "undeclared_aux_xml_sidecar" in result.analysis_failure_reason


def test_declared_pam_sidecar_retains_existing_hash_validation(scene):
    path, source, _ = scene(width=512)
    sidecar = Path(str(path) + ".aux.xml")
    sidecar.write_text("<PAMDataset/>")
    declared = source.model_copy(update={"sidecar_sha256": sha256_file(sidecar)})
    result = infer_robust_scene(path, declared, Detector([[]]))
    assert result.status == "complete"
    sidecar.write_text("<PAMDataset><Metadata/></PAMDataset>")
    mutated = infer_robust_scene(path, declared, Detector([]))
    assert mutated.status == "failed" and mutated.observed_count is None
    assert "sidecar checksum" in mutated.analysis_failure_reason


@pytest.mark.parametrize("value", ["bad", "", 17, "x" * 64])
def test_invalid_external_mask_binding_never_runs_model(scene, value):
    path, source, _ = scene(width=512)
    result = infer_robust_scene(path, source, Detector([]), expected_mask_sha256=value)
    assert result.status == "failed" and result.observed_count is None
    assert "invalid_expected_external_mask_sha256" in result.analysis_failure_reason
