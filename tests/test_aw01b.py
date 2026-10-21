"""Offline AW-01B geometry, selection, metrics and data-integrity regressions."""

from types import SimpleNamespace
from pathlib import Path
import pytest
from airfield_aircraft_vision.optical_contracts import PixelBox, RasterSource, Sample
from airfield_aircraft_vision.improved_training import augment_native
from airfield_aircraft_vision.experiment_metrics import (
    ap50,
    choose_threshold,
    evaluate_predictions,
    size_bin,
)
from airfield_aircraft_vision.expanded_dataset import (
    geographic_groups,
    choose_assignments,
    audit_spatial_separation,
    haversine_km,
)


def sample(group="a", split="test", x=10, y=50, boxes=None):
    return Sample(
        image_id=group,
        path=group + ".tif",
        split=split,
        source=RasterSource(
            scene_id=group,
            sensor_id="test",
            native_gsd_m=0.5,
            quality="acceptable",
            width=10,
            height=10,
            sha256="a" * 64,
            airfield_group=group,
            capture_group=group,
            crs="EPSG:4326",
            transform=(0.00001, 0, x, 0, -0.00001, y),
        ),
        boxes=boxes or [],
    )


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("horizontal", [False, True])
def test_native_augmentation_transforms_boxes_and_pixels_together(rotation, horizontal):
    np = pytest.importorskip("numpy")
    pixels = np.zeros((3, 4, 6), dtype="uint8")
    pixels[:, 1:3, 2:5] = 127
    transformed, boxes = augment_native(pixels, [(2, 1, 5, 3)], rotation, horizontal)
    ys, xs = np.where(transformed[0] != 0)
    assert boxes == [(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)]
    assert transformed.sum() == pixels.sum()
    assert transformed.flags.c_contiguous
    assert pixels[:, 1:3, 2:5].sum() == 3 * 2 * 3 * 127  # Original is untouched.


def test_ap50_and_f1_have_explicit_false_positives():
    truth = PixelBox(xyxy=(0, 0, 10, 10))
    false = PixelBox(xyxy=(20, 20, 30, 30), score=0.9)
    correct = truth.model_copy(update={"score": 0.6})
    s = sample(boxes=[truth])
    result = evaluate_predictions([s], [[false, correct]], 0.5)
    assert result["metrics"]["ap50"] == pytest.approx(0.5)
    assert result["metrics"]["f1"] == pytest.approx(2 / 3)
    assert result["metrics"]["fp"] == 1


def test_ap50_perfect_empty_and_missed_cases():
    box = PixelBox(xyxy=(0, 0, 10, 10))
    s = sample(boxes=[box])
    assert ap50([s], [[box]]) == 1
    assert ap50([s], [[]]) == 0
    assert ap50([sample()], [[]]) is None
    assert evaluate_predictions([s], [[]], 0.5)["metrics"]["f1"] == 0


def test_threshold_is_selected_on_validation_only():
    truth = PixelBox(xyxy=(0, 0, 10, 10))
    preds = [
        truth.model_copy(update={"score": 0.25}),
        PixelBox(xyxy=(20, 20, 30, 30), score=0.1),
    ]
    s = sample(split="validation", boxes=[truth])
    selected = choose_threshold([s], [preds], split="validation")
    assert selected["selected_threshold"] == 0.25
    assert selected["metrics"]["f1"] == 1
    with pytest.raises(ValueError, match="validation"):
        choose_threshold([s], [preds], split="test")


def test_size_and_weather_counts_remain_separate_and_honest():
    small = PixelBox(xyxy=(0, 0, 10, 10))
    large = PixelBox(xyxy=(0, 0, 100, 100))
    s = sample(boxes=[small, large])
    result = evaluate_predictions([s], [[small]], 0.5, {"a": "Cloud Cover or Haze"})
    assert result["by_bbox_pixel_size"]["small_lt32px"]["count_mae"] == 0
    assert result["by_bbox_pixel_size"]["large_ge96px"]["count_mae"] == 1
    assert result["by_bbox_pixel_size"]["large_ge96px"]["recall"] == 0
    assert result["by_dataset_weather"]["Cloud Cover or Haze"]["fn"] == 1
    assert size_bin(large) == "large_ge96px"


def rows_and_geo():
    rows = [
        {"loc_id": str(i), "cat_id": str(i), "Country": "X", "Air_Field": f"field{i}"}
        for i in (1, 2, 3)
    ]
    geo = {
        "crs": {"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "properties": {"loc_id": i},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[x, 50], [x + 0.001, 50], [x, 50]]],
                },
            }
            for i, x in ((1, 10), (2, 10.05), (3, 20))
        ],
    }
    return rows, geo


def test_nearby_airfield_ids_join_one_geographic_group():
    rows, geo = rows_and_geo()
    groups, centres, edges = geographic_groups(rows, geo)
    assert groups["1"] == groups["2"] and groups["1"] != groups["3"]
    assert edges and haversine_km(centres["1"], centres["2"]) < 10


def test_unknown_geographic_crs_is_not_assumed():
    rows, geo = rows_and_geo()
    geo["crs"] = {}
    with pytest.raises(ValueError, match="CRS"):
        geographic_groups(rows, geo)


def test_cross_split_overlap_and_nearby_tiles_are_blocked():
    with pytest.raises(ValueError, match="overlap"):
        audit_spatial_separation([sample("a", "train"), sample("b", "test")])
    with pytest.raises(ValueError, match="too close"):
        audit_spatial_separation([sample("a", "train"), sample("b", "test", x=10.0002)])
    audit = audit_spatial_separation([sample("a", "train"), sample("b", "test", x=11)])
    assert (
        not audit["overlap"] and audit["minimum_cross_split_envelope_distance_km"] > 1
    )


def test_original_validation_and_test_groups_cannot_enter_expanded_training():
    groups = {str(i): str(i) for i in range(1, 7)}
    rows = [{"loc_id": str(i), "Country": str(i)} for i in range(1, 7)]
    indexes = {
        "train": {"images": [{"file_name": f"{i}_A_tile_1.png"} for i in (1, 2, 4, 5)]},
        "test": {"images": [{"file_name": f"{i}_A_tile_1.png"} for i in (3, 6)]},
    }
    old = SimpleNamespace(
        samples=[
            SimpleNamespace(image_id=f"{i}_A_tile_1.png", split=split)
            for i, split in ((1, "train"), (2, "validation"), (3, "test"))
        ]
    )
    assigned, _ = choose_assignments(
        indexes, groups, rows, old, train_groups=2, validation_groups=1, test_groups=1
    )
    assert (
        assigned["1"] == "train"
        and assigned["2"] == "validation"
        and assigned["6"] == "test"
    )
    assert "3" not in assigned


def test_truncated_download_is_not_installed(monkeypatch, tmp_path):
    from airfield_aircraft_vision.dataset import BoundedDownload
    import io

    class Response(io.BytesIO):
        headers = {"Content-Length": "100"}

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(b"short"))
    with pytest.raises(ValueError, match="length"):
        BoundedDownload(1000).fetch("https://example.test/data", tmp_path / "data")
    assert not (tmp_path / "data").exists()


def test_test_samples_cannot_be_disguised_as_validation():
    truth = PixelBox(xyxy=(0, 0, 10, 10))
    with pytest.raises(ValueError, match="validation"):
        choose_threshold(
            [sample(split="test", boxes=[truth])], [[truth]], split="validation"
        )


def test_prediction_length_mismatch_cannot_hide_missing_images():
    from airfield_aircraft_vision.diagnostics import score_records

    with pytest.raises(ValueError, match="length"):
        score_records([sample()], [], 0.5)


def test_parallel_download_budget_is_reserved_before_read(monkeypatch, tmp_path):
    from airfield_aircraft_vision.dataset import BoundedDownload
    from concurrent.futures import ThreadPoolExecutor
    import io

    class Response(io.BytesIO):
        headers = {"Content-Length": "100"}

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(b"x" * 100))
    downloader = BoundedDownload(200)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(
                downloader.fetch, "https://example.test/data", tmp_path / str(i)
            )
            for i in range(3)
        ]
        success = 0
        for f in futures:
            try:
                f.result()
                success += 1
            except ValueError as e:
                assert "budget" in str(e)
    assert success == 2 and downloader.downloaded == 200 and downloader.remaining == 0


@pytest.mark.parametrize(
    "stored,explicit,expected",
    [(0.25, None, 0.25), (None, None, 0.5), (0.25, 0.7, 0.7)],
)
def test_inference_uses_checkpoint_validation_threshold_with_legacy_fallback(
    monkeypatch, tmp_path, stored, explicit, expected
):
    pytest.importorskip("torch")
    from airfield_aircraft_vision import cli, torch_backend
    from airfield_aircraft_vision.inference import optical_high_resolution as optical

    s = sample()
    source = tmp_path / "source.json"
    source.write_text(s.source.model_dump_json())
    model = SimpleNamespace(
        manifest={} if stored is None else {"selected_threshold": stored}
    )
    monkeypatch.setattr(torch_backend, "TorchDetector", lambda *a, **k: model)
    seen = {}

    def fake_infer(*a, **k):
        seen.update(k)
        return SimpleNamespace(
            total_count=0, model_dump_json=lambda **kw: '{"count":0}'
        )

    monkeypatch.setattr(optical, "infer_scene", fake_infer)
    args = [
        "infer",
        "--image",
        "unused.tif",
        "--source",
        str(source),
        "--checkpoint",
        "unused.pt",
        "--output",
        str(tmp_path / "out.json"),
    ]
    if explicit is not None:
        args += ["--confidence", str(explicit)]
    assert cli.main(args) == 0
    assert seen["confidence"] == expected


def test_aw01b_command_help():
    from airfield_aircraft_vision.aw01b import main

    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0


def test_frozen_dataset_request_mismatch_fails_before_using_cache(tmp_path):
    import json
    from airfield_aircraft_vision.expanded_dataset import prepare_expanded

    original = tmp_path / "original.json"
    original.write_text("{}")
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    (data / "request.json").write_text(json.dumps({"seed": 999}))
    with pytest.raises(ValueError, match="configuration differs"):
        prepare_expanded(original, data)


def test_validation_revision_bounds_fail_before_accessing_data(tmp_path):
    from airfield_aircraft_vision.expanded_dataset import broaden_validation

    with pytest.raises(ValueError, match="bounded"):
        broaden_validation(tmp_path / "missing.json", extra_groups=3)


def test_identical_frozen_manifest_is_read_only_even_with_new_telemetry(
    tmp_path, monkeypatch
):
    from airfield_aircraft_vision.dataset import save_frozen_manifest
    from airfield_aircraft_vision.optical_contracts import DatasetManifest

    path = tmp_path / "manifest.json"
    old = DatasetManifest(samples=[sample()], preparation={"downloaded_bytes": 123})
    path.write_text(old.model_dump_json(indent=2))
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("cached manifest must not be rewritten")

    monkeypatch.setattr(Path, "write_text", forbidden)
    result = save_frozen_manifest(
        path, old.model_copy(update={"preparation": {"downloaded_bytes": 0}})
    )
    assert result.preparation["downloaded_bytes"] == 123
    assert path.read_bytes() == before
