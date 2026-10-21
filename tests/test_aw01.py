"""Offline AW-01 regressions. No network or production imagery/credentials."""

from pathlib import Path
import json
import pytest
from pydantic import ValidationError
from airfield_aircraft_vision.optical_contracts import (
    PixelBox,
    RasterSource,
    Sample,
    DatasetManifest,
)
from airfield_aircraft_vision.preprocessing import (
    tile_windows,
    sha256_file,
    NativeRaster,
)
from airfield_aircraft_vision.postprocessing import iou, suppress
from airfield_aircraft_vision.evaluation import match_boxes, summarize
from airfield_aircraft_vision.dataset import (
    group_locations,
    resolve_image,
    BoundedDownload,
)
from airfield_aircraft_vision.inference.optical_high_resolution import infer_scene
from airfield_aircraft_vision.workflows import assert_holdout


@pytest.fixture
def raster(tmp_path):
    np = pytest.importorskip("numpy")
    rio = pytest.importorskip("rasterio")
    p = tmp_path / "source.tif"
    data = np.ones((3, 512, 768), dtype="uint8") * 127
    data[:, 100:110, 300:320] = 219
    with rio.open(
        p,
        "w",
        driver="GTiff",
        width=768,
        height=512,
        count=3,
        dtype="uint8",
        crs="EPSG:32643",
        transform=rio.Affine(0.5, 0, 500000, 0, -0.5, 2000000),
    ) as ds:
        ds.write(data)
    s = RasterSource(
        scene_id="scene",
        sensor_id="test",
        native_gsd_m=0.5,
        quality="acceptable",
        sha256=sha256_file(p),
        width=768,
        height=512,
        crs="EPSG:32643",
        transform=(0.5, 0, 500000, 0, -0.5, 2000000),
        airfield_group="a",
        capture_group="c",
    )
    return p, s, data


class Detector:
    manifest = {
        "architecture": "fake-unit-test",
        "experimental_input_bounds_m": [0.2, 1.0],
    }

    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def predict(self, pixels):
        return next(self.outputs)


def test_native_pixels_and_georeferencing_are_unchanged(raster):
    np = pytest.importorskip("numpy")
    p, s, data = raster
    before = p.read_bytes()
    with NativeRaster(p, s) as ds:
        pixels, valid = ds.read((256, 0, 512, 512), 512)
        np.testing.assert_array_equal(pixels, data[:, :, 256:768])
        assert valid.all()
    assert p.read_bytes() == before


def test_tile_overlap_maps_and_deduplicates_scene_boxes(raster):
    p, s, _ = raster
    detector = Detector(
        [
            [PixelBox(xyxy=(300, 100, 320, 110), score=0.9)],
            [PixelBox(xyxy=(44, 100, 64, 110), score=0.8)],
        ]
    )
    result = infer_scene(p, s, detector)
    assert result.total_count == 1 and result.rejected_count == 1
    box = result.detections[0].oriented_box
    assert (box.center_x, box.center_y, box.width, box.height) == (310, 105, 20, 10)
    assert result.detections[0].geometry.type == "Polygon"
    assert (
        result.legacy_inference_result is None
    )  # Unknown sensing time is never invented.
    assert result.preprocessing["resize"] is False


def test_legacy_contract_available_with_real_time(raster):
    p, s, _ = raster
    s = s.model_copy(
        update={
            "acquisition_timestamp": __import__("datetime").datetime(
                2026, 1, 1, tzinfo=__import__("datetime").timezone.utc
            ),
            "timestamp_precision": "minute",
            "timestamp_timezone_basis": "provider_utc",
        }
    )
    result = infer_scene(p, s, Detector([[], []]))
    assert result.legacy_inference_result.total_detection_count == 0


@pytest.mark.parametrize(
    "change",
    [
        {"sensor_domain": "sar"},
        {"native_gsd_m": 10},
        {"quality": "poor"},
        {"quality": "unusable"},
    ],
)
def test_router_blocks_incompatible_inference(raster, change):
    p, s, _ = raster
    with pytest.raises(ValueError, match="Tier A"):
        infer_scene(p, s.model_copy(update=change), Detector([]))


def test_unknown_quality_needs_explicit_override(raster):
    p, s, _ = raster
    s = s.model_copy(update={"quality": "unknown"})
    with pytest.raises(ValueError, match="unknown quality"):
        infer_scene(p, s, Detector([]))
    assert (
        infer_scene(p, s, Detector([[], []]), allow_unknown_quality=True).total_count
        == 0
    )


def test_source_checksum_and_grid_mismatch_are_blocked(raster):
    p, s, _ = raster
    with pytest.raises(ValueError, match="checksum"):
        NativeRaster(p, s.model_copy(update={"sha256": "0" * 64}))
    with pytest.raises(ValueError, match="dimensions"):
        NativeRaster(p, s.model_copy(update={"width": 1}))
    with pytest.raises(ValueError, match="geotransform"):
        NativeRaster(p, s.model_copy(update={"transform": (1, 0, 0, 0, -1, 0)}))


def test_padding_does_not_add_pixels_or_detections(tmp_path):
    np = pytest.importorskip("numpy")
    rio = pytest.importorskip("rasterio")
    p = tmp_path / "small.tif"
    data = np.arange(3 * 64 * 80, dtype=np.uint8).reshape(3, 64, 80)
    with rio.open(
        p,
        "w",
        driver="GTiff",
        width=80,
        height=64,
        count=3,
        dtype="uint8",
        transform=rio.Affine(1, 0, 500, 0, -1, 500),
    ) as ds:
        ds.write(data)
    s = RasterSource(
        scene_id="s",
        sensor_id="test",
        native_gsd_m=0.5,
        quality="acceptable",
        sha256=sha256_file(p),
        width=80,
        height=64,
    )
    with NativeRaster(p, s) as ds:
        pixels, _ = ds.read((0, 0, 80, 64), 512)
    np.testing.assert_array_equal(pixels[:, :64, :80], data)
    assert not pixels[:, 64:, :].any()
    r = infer_scene(p, s, Detector([[PixelBox(xyxy=(100, 100, 120, 120), score=0.9)]]))
    assert r.total_count == 0


@pytest.mark.parametrize("bad", [(0, 0, 0, 3), (5, 1, 2, 4), (0, 0, float("nan"), 5)])
def test_invalid_boxes_rejected(bad):
    with pytest.raises(ValidationError):
        PixelBox(xyxy=bad)


def test_native_windows_cover_full_grid_without_resize():
    assert list(tile_windows(768, 512)) == [(0, 0, 512, 512), (256, 0, 512, 512)]
    assert list(tile_windows(80, 64)) == [(0, 0, 80, 64)]
    with pytest.raises(ValueError):
        list(tile_windows(100, 100, size=511))


def test_duplicate_suppression_and_confidence_order():
    b = PixelBox(xyxy=(0, 0, 10, 10), score=0.5)
    c = b.model_copy(update={"score": 0.9})
    kept, rejected = suppress([b, c])
    assert kept == [c] and rejected == [b]
    assert iou(b.xyxy, c.xyxy) == 1


def test_metrics_one_to_one_and_false_negative():
    truth = [PixelBox(xyxy=(0, 0, 10, 10)), PixelBox(xyxy=(30, 30, 40, 40))]
    predictions = [
        truth[0],
        truth[0].model_copy(update={"score": 0.9}),
        PixelBox(xyxy=(80, 80, 90, 90)),
    ]
    matches = match_boxes(predictions, truth)
    r = {"matches": matches, "truth_count": 2, "predicted_count": 3}
    summary = summarize([r])
    assert (summary["tp"], summary["fp"], summary["fn"]) == (1, 2, 1)
    assert summary["precision"] == pytest.approx(1 / 3)
    assert summary["recall"] == 0.5 and summary["count_mae"] == 1


def test_empty_metrics_do_not_fabricate_performance():
    assert summarize([])["precision"] is None
    assert summarize([])["recall"] is None
    assert summarize([])["count_mae"] is None


def test_ignored_predictions_are_explicit():
    b = PixelBox(xyxy=(0, 0, 10, 10))
    m = match_boxes([b], [], [b])
    assert m["ignored_predictions"] == [0] and not m["false_positives"]


def test_airfield_capture_and_duplicate_leakage_blocked(raster):
    _, source, _ = raster
    a = Sample(image_id="1", path="1.tif", split="train", source=source)
    b = Sample(image_id="2", path="2.tif", split="test", source=source)
    with pytest.raises(ValidationError, match="leakage"):
        DatasetManifest(samples=[a, b])
    for field in ("airfield_group", "capture_group", "sha256"):
        values = {
            "airfield_group": "other",
            "capture_group": "other",
            "sha256": "1" * 64,
        }
        values[field] = getattr(source, field)
        with pytest.raises(ValidationError, match="leakage"):
            DatasetManifest(
                samples=[
                    a,
                    b.model_copy(update={"source": source.model_copy(update=values)}),
                ]
            )


def test_shared_captures_and_airfield_aliases_union():
    rows = [
        {"loc_id": "1", "cat_id": "A", "Country": "US", "Air_Field": "X"},
        {"loc_id": "2", "cat_id": "A", "Country": "US", "Air_Field": "Y"},
        {"loc_id": "3", "cat_id": "B", "Country": "US", "Air_Field": "x"},
    ]
    groups = group_locations(rows)
    assert len(set(groups.values())) == 1


def test_path_escape_blocked(raster, tmp_path):
    _, s, _ = raster
    sample = Sample(image_id="s", path="../secret.tif", split="test", source=s)
    with pytest.raises(ValueError, match="escapes"):
        resolve_image(tmp_path / "manifest.json", sample)


def test_unknown_time_not_promoted_to_exact_time(raster):
    _, s, _ = raster
    assert s.acquisition_timestamp is None and s.timestamp_precision == "unknown"
    with pytest.raises(ValidationError, match="precision"):
        RasterSource.model_validate(dict(s.model_dump(), timestamp_precision="second"))


def test_holdout_checks_all_training_lineage(raster):
    _, s, _ = raster
    sample = Sample(image_id="x", path="x", split="test", source=s)
    model = {
        "training_airfields": ["train"],
        "training_captures": ["train"],
        "training_source_checksums": ["train"],
    }
    assert_holdout([sample], model)
    for attr, key in (
        ("airfield_group", "training_airfields"),
        ("capture_group", "training_captures"),
        ("sha256", "training_source_checksums"),
    ):
        with pytest.raises(ValueError, match="leakage"):
            assert_holdout([sample], dict(model, **{key: [getattr(s, attr)]}))
    with pytest.raises(ValueError, match="lineage"):
        assert_holdout([sample], {})


def test_download_byte_budget_offline(monkeypatch, tmp_path):
    class Response:
        headers = {"Content-Length": "200"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: Response())
    with pytest.raises(ValueError, match="budget"):
        BoundedDownload(100).fetch("https://example.test/file", tmp_path / "f")
    assert not (tmp_path / "f").exists()


def test_cli_help_and_invalid_workflow():
    from airfield_aircraft_vision.cli import main

    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0


def test_real_torchvision_transform_never_resizes():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from airfield_aircraft_vision.torch_backend import build_model

    model = build_model().eval()
    image = torch.rand(3, 64, 96)
    target = {
        "boxes": torch.tensor([[3.0, 4.0, 30.0, 40.0]]),
        "labels": torch.tensor([1]),
    }
    out, targets = model.transform([image], [target])
    assert out.image_sizes == [(64, 96)]
    assert torch.equal(targets[0]["boxes"], target["boxes"])
    restored = (
        out.tensors[0, :, :64, :96]
        * torch.tensor(model.transform.image_std)[:, None, None]
        + torch.tensor(model.transform.image_mean)[:, None, None]
    )
    assert torch.allclose(image, restored, atol=1e-7)


def test_geographic_grid_spacing_converted_without_resampling():
    rio = pytest.importorskip("rasterio")
    from airfield_aircraft_vision.preprocessing import pixel_spacing_m

    x, y = pixel_spacing_m(
        "EPSG:4326", rio.Affine(0.00001, 0, 10, 0, -0.00001, 60), 512, 512
    )
    assert 0.5 < x < 0.6 and 1.0 < y < 1.2


def test_all_nodata_is_failure_not_zero_aircraft(raster):
    np = pytest.importorskip("numpy")
    rio = pytest.importorskip("rasterio")
    p, s, data = raster
    with rio.open(p, "r+") as ds:
        ds.write(np.zeros_like(data))
        ds.nodata = 0
    s = s.model_copy(update={"sha256": sha256_file(p)})
    with pytest.raises(ValueError, match="no valid source pixels"):
        infer_scene(p, s, Detector([]))


def test_sidecar_tampering_is_blocked(raster):
    p, s, _ = raster
    aux = Path(str(p) + ".aux.xml")
    aux.write_text("<PAMDataset/>")
    s = s.model_copy(update={"sidecar_sha256": sha256_file(aux)})
    aux.write_text("<PAMDataset></PAMDataset>")
    with pytest.raises(ValueError, match="sidecar checksum"):
        NativeRaster(p, s)


def test_partition_sampling_stays_bounded_when_official_locations_overlap():
    from airfield_aircraft_vision.dataset import select_images

    indexes = {
        "train": {"images": [{"file_name": f"1_A_tile_{i}.png"} for i in range(5)]},
        "test": {
            "images": [{"file_name": f"1_B_tile_{i}.png"} for i in range(5)]
            + [{"file_name": "2_C_tile_0.png"}]
        },
    }
    selected = select_images(
        indexes, {"1": "a", "2": "b"}, {"a": "train", "b": "test"}, 3, 17
    )
    assert len(selected["train"]["a"]) == 3
    assert "a" not in selected["test"]
    assert len(selected["test"]["b"]) == 1
    assert selected == select_images(
        indexes, {"1": "a", "2": "b"}, {"a": "train", "b": "test"}, 3, 17
    )


def test_assumed_utc_never_silently_exports_verified_transport_time(raster):
    from datetime import datetime, timezone

    p, s, _ = raster
    s = s.model_copy(
        update={
            "acquisition_timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "timestamp_precision": "minute",
            "timestamp_timezone_basis": "assumed_utc",
        }
    )
    result = infer_scene(p, s, Detector([[], []]))
    assert result.source.acquisition_timestamp == s.acquisition_timestamp
    assert result.legacy_inference_result is None
    assert any("UTC" in w for w in result.warnings)


def test_rgb_band_interpretation_is_required(raster):
    rio = pytest.importorskip("rasterio")
    from rasterio.enums import ColorInterp

    p, s, _ = raster
    with rio.open(p, "r+") as ds:
        ds.colorinterp = (
            ColorInterp.gray,
            ColorInterp.undefined,
            ColorInterp.undefined,
        )
    s = s.model_copy(update={"sha256": sha256_file(p)})
    with pytest.raises(ValueError, match="RGB channel"):
        NativeRaster(p, s)


def test_zero_predictions_with_positive_labels_is_measured_failure():
    truth = [PixelBox(xyxy=(0, 0, 10, 10)), PixelBox(xyxy=(20, 20, 30, 30))]
    result = summarize(
        [{"matches": match_boxes([], truth), "truth_count": 2, "predicted_count": 0}]
    )
    assert result["precision"] is None and result["recall"] == 0
    assert result["count_mae"] == 2 and result["fn"] == 2


def test_source_time_is_utc_but_original_string_is_preserved(raster):
    _, source, _ = raster
    data = source.model_dump()
    data.update(
        acquisition_timestamp="2026-01-01T05:30:00+05:30",
        timestamp_precision="second",
        timestamp_timezone_basis="provider_utc",
        acquisition_time_raw="2026-01-01T05:30:00+05:30",
    )
    parsed = RasterSource.model_validate(data)
    assert parsed.acquisition_timestamp.hour == 0
    assert parsed.acquisition_time_raw.endswith("+05:30")


@pytest.mark.parametrize(
    "updates",
    [
        {"pixel_spacing_m": (float("nan"), 0.3)},
        {"transform": (1, 0, float("inf"), 0, -1, 0)},
    ],
)
def test_invalid_spatial_metadata_is_rejected(raster, updates):
    _, source, _ = raster
    with pytest.raises(ValidationError):
        RasterSource.model_validate(dict(source.model_dump(), **updates))
