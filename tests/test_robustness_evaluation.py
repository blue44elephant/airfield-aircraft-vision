from types import SimpleNamespace
import pytest
import numpy as np
from airfield_aircraft_vision.optical_contracts import PixelBox
from airfield_aircraft_vision.robustness_evaluation import CachedDetector, localization


class Detector:
    manifest = {"selected_threshold": 0.7}

    def __init__(self):
        self.calls = 0

    def predict(self, pixels):
        self.calls += 1
        return [PixelBox(xyxy=(1, 2, 4, 5), score=0.9)]


def test_cache_reuses_exact_rgb_not_distinct_pixel_values():
    detector = Detector()
    cache = CachedDetector(detector)
    p = np.zeros((3, 32, 32), dtype="uint8")
    assert cache.predict(p) == cache.predict(p.copy())
    p[0, 0, 0] = 1
    cache.predict(p)
    assert detector.calls == 2
    assert cache.cache_hits == 1
    assert len(cache.evidence()["exact_inputs"]) == 2


def test_failed_detector_is_not_cached_as_empty():
    class Failed:
        manifest = {}

        def predict(self, pixels):
            raise RuntimeError("backend failed")

    cache = CachedDetector(Failed())
    with pytest.raises(RuntimeError):
        cache.predict(np.zeros((3, 32, 32), dtype="uint8"))
    assert cache.cache == {}
    assert cache.forward_calls == 0


def test_localization_measures_matched_iou_and_misses_separately():
    sample = SimpleNamespace(
        boxes=[PixelBox(xyxy=(0, 0, 10, 10)), PixelBox(xyxy=(30, 30, 40, 40))],
        ignored_boxes=[],
    )
    report = localization([sample], [[PixelBox(xyxy=(0, 0, 10, 10), score=0.9)]], 0.7)
    assert report["mean_matched_iou"] == 1
    assert report["recall_at_iou_075"] == 0.5
    assert report["median_center_error_source_pixels"] == 0


def test_cache_stops_at_forward_budget_without_empty_result():
    d = Detector()
    cache = CachedDetector(d, max_calls=1)
    p = np.zeros((3, 32, 32), dtype="uint8")
    cache.predict(p)
    p[0, 0, 0] = 1
    with pytest.raises(RuntimeError, match="budget"):
        cache.predict(p)
    assert d.calls == 1


def test_cache_backend_failure_is_latched_without_retry():
    class Failed:
        manifest = {}
        calls = 0

        def predict(self, pixels):
            self.calls += 1
            raise RuntimeError("backend failed")

    detector = Failed()
    cache = CachedDetector(detector)
    pixels = np.zeros((3, 32, 32), dtype="uint8")
    with pytest.raises(RuntimeError, match="backend failed"):
        cache.predict(pixels)
    with pytest.raises(RuntimeError, match="source_backend_stopped"):
        cache.predict(pixels)
    assert detector.calls == 1 and cache.forward_attempts == 1
    assert cache.evidence()["stopped_reason"] == "backend_failed:RuntimeError"


def test_expired_cache_never_forwards_and_cannot_be_restarted():
    import time

    detector = Detector()
    cache = CachedDetector(detector, deadline=time.monotonic() - 1)
    pixels = np.zeros((3, 32, 32), dtype="uint8")
    with pytest.raises(RuntimeError, match="runtime_budget"):
        cache.predict(pixels)
    cache.deadline = time.monotonic() + 100
    with pytest.raises(RuntimeError, match="source_backend_stopped"):
        cache.predict(pixels)
    assert detector.calls == 0 and cache.forward_attempts == 0


def test_oversized_evaluation_metadata_never_opens_legacy_or_forwards(monkeypatch):
    from pathlib import Path
    import airfield_aircraft_vision.robustness_evaluation as module
    from airfield_aircraft_vision.optical_contracts import RasterSource

    source = RasterSource(
        scene_id="huge",
        sensor_id="test",
        native_gsd_m=0.5,
        quality="acceptable",
        sha256="a" * 64,
        width=100_000,
        height=100_000,
    )
    monkeypatch.setattr(
        module,
        "infer_scene",
        lambda *a, **k: pytest.fail("legacy must not open oversized source"),
    )
    detector = Detector()
    result = module.evaluate_one(
        SimpleNamespace(source=source), Path("missing"), detector
    )
    assert detector.calls == 0
    assert result["legacy_predictions"] is None
    assert result["safe"].status == "unsuitable"
    assert result["safe"].observed_count is None
    assert result["contained"].observed_count is None
    assert (
        result["contained"].preprocessing["execution"]
        == "not_run_after_primary_failure"
    )


@pytest.fixture
def tiny_native_scene(tmp_path):
    import rasterio
    from airfield_aircraft_vision.optical_contracts import RasterSource
    from airfield_aircraft_vision.preprocessing import sha256_file

    path = tmp_path / "tiny.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=3,
        dtype="uint8",
        width=32,
        height=32,
        crs="EPSG:32643",
        transform=rasterio.Affine(0.5, 0, 500000, 0, -0.5, 2000000),
    ) as ds:
        ds.write(np.ones((3, 32, 32), dtype="uint8") * 127)
    return path, RasterSource(
        scene_id="tiny",
        sensor_id="test",
        native_gsd_m=0.5,
        quality="acceptable",
        sha256=sha256_file(path),
        width=32,
        height=32,
    )


def test_evaluation_failed_backend_is_not_retried_or_reported_as_zero(
    tiny_native_scene,
):
    from airfield_aircraft_vision.robustness_evaluation import evaluate_one

    class Failed:
        manifest = {"selected_threshold": 0.7}
        calls = 0

        def predict(self, pixels):
            self.calls += 1
            raise RuntimeError("detector failed")

    path, source = tiny_native_scene
    detector = Failed()
    result = evaluate_one(SimpleNamespace(source=source), path, detector)
    assert detector.calls == 1
    assert result["legacy_predictions"] is None
    assert result["safe"].status == "failed"
    assert result["safe"].observed_count is result["contained"].observed_count is None
    assert result["cache"]["forward_attempts"] == 1


def test_evaluation_unbound_external_mask_cannot_bypass_preflight(
    tiny_native_scene, monkeypatch
):
    from pathlib import Path
    import airfield_aircraft_vision.robustness_evaluation as module

    path, source = tiny_native_scene
    Path(str(path) + ".msk").write_bytes(b"untrusted mask")
    monkeypatch.setattr(
        module,
        "infer_scene",
        lambda *a, **k: pytest.fail("legacy must not read unbound mask"),
    )
    detector = Detector()
    result = module.evaluate_one(SimpleNamespace(source=source), path, detector)
    assert detector.calls == 0
    assert result["legacy_predictions"] is None
    assert result["safe"].observed_count is None
    assert "external_mask_requires" in result["safe"].analysis_failure_reason


def test_workflow_failures_and_partial_coverage_have_independent_denominators(
    tmp_path, monkeypatch
):
    import airfield_aircraft_vision.robustness_evaluation as module
    import airfield_aircraft_vision.torch_backend as backend
    from airfield_aircraft_vision.optical_contracts import (
        RasterSource,
        Sample,
        DatasetManifest,
    )
    from airfield_aircraft_vision.robust_contracts import RobustDetection
    from airfield_aircraft_vision.aw01d import failed_result

    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"mock frozen checkpoint")
    monkeypatch.setattr(module, "AW01C_SHA256", module.sha256_file(checkpoint))
    fake = SimpleNamespace(
        manifest={
            "training_airfields": ["train"],
            "training_captures": ["train"],
            "training_source_checksums": ["train"],
        }
    )
    monkeypatch.setattr(backend, "TorchDetector", lambda *a, **k: fake)
    samples = []
    for index in range(3):
        src = RasterSource(
            scene_id=f"scene-{index}",
            sensor_id="test",
            native_gsd_m=0.5,
            quality="acceptable",
            sha256="a" * 64,
            width=32,
            height=32,
            airfield_group="unseen",
            capture_group="new",
        )
        samples.append(
            Sample(
                image_id=f"image-{index}",
                path=f"image-{index}.tif",
                split="test",
                source=src,
                boxes=[PixelBox(xyxy=(1, 2, 4, 5))],
            )
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(DatasetManifest(samples=samples).model_dump_json())

    def scene_result(sample, state):
        result = failed_result(sample.source, "synthetic failure")
        if state == "failed":
            return result
        valid = 1024 if state == "complete" else 512
        coverage = result.coverage.model_copy(
            update={
                "inspected_pixels": 1024,
                "inspected_fraction": 1,
                "analyzable_pixels": valid,
                "analyzable_fraction": valid / 1024,
            }
        )
        detection = RobustDetection(
            detection_id="d-1",
            status="accepted",
            box=PixelBox(xyxy=(1, 2, 4, 5), score=0.9),
            tile_index=0,
            tile_window=(0, 0, 32, 32),
            valid_pixel_fraction=1,
        )
        return result.model_copy(
            update={
                "status": state,
                "observed_count": 1,
                "accepted_count": 1,
                "detections": [detection],
                "coverage": coverage,
                "planned_tiles": 1,
                "completed_tiles": 1,
                "analysis_failure_reason": None,
            }
        )

    def evaluate(sample, *a, **k):
        index = int(sample.image_id[-1])
        safe_state = ("complete", "partial", "failed")[index]
        contained_state = ("failed", "partial", "failed")[index]
        return {
            "legacy_predictions": (
                [PixelBox(xyxy=(1, 2, 4, 5), score=0.9)] if index < 2 else None
            ),
            "legacy_error": None if index < 2 else "failed",
            "safe": scene_result(sample, safe_state),
            "contained": scene_result(sample, contained_state),
            "cache": {"unique_forward_calls": 0},
        }

    monkeypatch.setattr(module, "evaluate_one", evaluate)
    report = module.run_evaluation(
        [{"name": "mock", "manifest": str(manifest)}],
        checkpoint,
        tmp_path / "out",
        threads=1,
    )
    cohort = report["cohorts"]["mock"]
    assert cohort["accuracy_denominators"] == {
        "baseline_complete_images": 1,
        "safe_complete_images": 1,
        "containment_complete_images": 0,
        "partial_images_excluded": 1,
        "containment_failed_or_unsuitable": 2,
    }
    assert cohort["metrics"]["boundary_containment"]["metrics"]["count_mae"] is None
    assert cohort["metrics"]["boundary_containment"]["metrics"]["images"] == 0
    assert cohort["metrics"]["safe_workflow"]["metrics"]["images"] == 1
    assert cohort["incomplete_source_baseline_diagnostic"]["metrics"]["images"] == 1
    assert cohort["scenes"][0]["contained_observed_count"] is None
