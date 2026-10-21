import hashlib
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import Affine

from airfield_aircraft_vision.negative_crops import (
    derive_geotiff,
    propose_candidates,
)
from airfield_aircraft_vision.optical_contracts import PixelBox, RasterSource, Sample
from airfield_aircraft_vision.preprocessing import sha256_file


def fixture_source(tmp_path):
    path = tmp_path / "parent.tif"
    pixels = np.arange(3 * 512 * 512, dtype=np.uint8).reshape(3, 512, 512)
    affine = Affine(0.4, 0.05, 500000, 0.03, -0.4, 4000000)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=512,
        height=512,
        count=3,
        dtype="uint8",
        crs="EPSG:32631",
        transform=affine,
    ) as ds:
        ds.write(pixels)
        ds.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        ds.update_tags(original="preserved")
    source = RasterSource(
        scene_id="scene",
        sensor_id="WV03",
        native_gsd_m=0.4,
        panchromatic_gsd_m=0.4,
        source_uri="https://example.org/original",
        license="CC-BY-SA-4.0",
        attribution="Original attribution",
        sha256=sha256_file(path),
        width=512,
        height=512,
        crs="EPSG:32631",
        transform=tuple(affine)[:6],
        airfield_group="airfield",
        capture_group="capture",
        acquisition_timestamp="2019-02-07T09:45:00Z",
        timestamp_precision="minute",
        acquisition_time_raw="2/7/19 9:45",
        timestamp_timezone_basis="assumed_utc",
    )
    sample = Sample(
        image_id="source.png",
        path=str(path),
        split="train",
        source=source,
        boxes=[PixelBox(xyxy=(450, 450, 480, 480))],
        ignored_boxes=[PixelBox(xyxy=(10, 450, 20, 470))],
    )
    return path, sample, pixels, affine


def review_for(sample, window):
    return dict(
        decision="empty_no_aircraft",
        reviewer="Synthetic test",
        reviewed_at="2026-10-21T10:00:00+05:30",
        notes="Synthetic fixture only.",
        category="difficult_background",
        parent_sha256=sample.source.sha256,
        window=list(window),
    )


def test_proposals_deterministic_and_include_ignored_halo(tmp_path):
    _, sample, _, _ = fixture_source(tmp_path)
    candidates = propose_candidates([sample])
    assert [c["window"] for c in candidates] == [[0, 0, 256, 256], [256, 0, 256, 256]]
    assert all(c["review_required"] for c in candidates)
    assert candidates == propose_candidates([sample])


def test_no_independent_test_mining(tmp_path):
    path, sample, _, _ = fixture_source(tmp_path)
    sample = sample.model_copy(update={"split": "test"})
    with pytest.raises(ValueError, match="test"):
        propose_candidates([sample])
    with pytest.raises(ValueError, match="test"):
        derive_geotiff(path, sample, (0, 0, 256, 256), tmp_path / "n.tif", review={})


def test_crop_exact_pixels_rotated_grid_and_provenance(tmp_path):
    path, sample, pixels, affine = fixture_source(tmp_path)
    before = path.read_bytes()
    window = (256, 0, 256, 256)
    out = tmp_path / "negative.tif"
    result = derive_geotiff(
        path, sample, window, out, review=review_for(sample, window)
    )
    assert path.read_bytes() == before
    assert not result.boxes and not result.ignored_boxes
    assert result.split == sample.split
    assert result.source.airfield_group == sample.source.airfield_group
    assert result.source.capture_group == sample.source.capture_group
    assert result.source.acquisition_time_raw == sample.source.acquisition_time_raw
    assert result.source.timestamp_timezone_basis == "assumed_utc"
    assert result.source.license == "CC-BY-SA-4.0"
    assert result.source.source_uri == sample.source.source_uri
    assert result.source.sidecar_sha256 is None
    assert result.source.sha256 == sha256_file(out)
    expected = affine * Affine.translation(256, 0)
    assert result.source.transform == tuple(expected)[:6]
    with rasterio.open(out) as ds:
        np.testing.assert_array_equal(ds.read(), pixels[:, :256, 256:])
        assert ds.transform == expected and ds.crs.to_string() == "EPSG:32631"
        assert ds.tags()["original"] == "preserved"
        assert ds.tags()["AW01C_PARENT_SHA256"] == sample.source.sha256
        assert "Visual review=" in " ".join(result.source.provenance_notes)
    with pytest.raises(FileExistsError):
        derive_geotiff(path, sample, window, out, review=review_for(sample, window))


@pytest.mark.parametrize(
    "mutation",
    [
        {"decision": "candidate"},
        {"parent_sha256": "0" * 64},
        {"window": [1, 0, 256, 256]},
        {"reviewer": ""},
        {"reviewed_at": "2026-10-21"},
        {"category": "unknown"},
    ],
)
def test_explicit_source_bound_review_required(tmp_path, mutation):
    path, sample, _, _ = fixture_source(tmp_path)
    window = (0, 0, 256, 256)
    review = review_for(sample, window)
    review.update(mutation)
    with pytest.raises(ValueError):
        derive_geotiff(path, sample, window, tmp_path / "n.tif", review=review)
    assert not (tmp_path / "n.tif").exists()


def test_halo_rejected_even_when_label_outside_window(tmp_path):
    path, sample, _, _ = fixture_source(tmp_path)
    sample = sample.model_copy(update={"boxes": [PixelBox(xyxy=(270, 10, 280, 20))]})
    window = (0, 0, 256, 256)
    with pytest.raises(ValueError, match="exclusion"):
        derive_geotiff(
            path, sample, window, tmp_path / "n.tif", review=review_for(sample, window)
        )


def test_source_checksum_must_still_match_review(tmp_path):
    path, sample, _, _ = fixture_source(tmp_path)
    with path.open("ab") as f:
        f.write(b"changed")
    window = (0, 0, 256, 256)
    with pytest.raises(ValueError, match="checksum"):
        derive_geotiff(
            path, sample, window, tmp_path / "n.tif", review=review_for(sample, window)
        )


@pytest.mark.parametrize(
    "window",
    [(0, 0, 512, 512), (-1, 0, 256, 256), (400, 0, 256, 256), (0.0, 0, 256, 256)],
)
def test_invalid_grid_rejected(tmp_path, window):
    path, sample, _, _ = fixture_source(tmp_path)
    with pytest.raises(ValueError):
        derive_geotiff(
            path, sample, window, tmp_path / "n.tif", review=review_for(sample, window)
        )


def test_nodata_crop_rejected(tmp_path):
    path, sample, _, _ = fixture_source(tmp_path)
    with rasterio.open(path, "r+") as ds:
        mask = np.full((512, 512), 255, np.uint8)
        mask[0, 0] = 0
        ds.write_mask(mask)
    # Mask may be internal or sidecar depending on GDAL; review binds latest source.
    sample.source.sha256 = sha256_file(path)
    window = (0, 0, 256, 256)
    with pytest.raises(ValueError, match="fully valid"):
        derive_geotiff(
            path, sample, window, tmp_path / "n.tif", review=review_for(sample, window)
        )


def make_attach_fixture(tmp_path):
    import json
    from airfield_aircraft_vision.optical_contracts import DatasetManifest

    path, sample, _, _ = fixture_source(tmp_path)
    sample = sample.model_copy(update={"path": path.name})
    manifest = tmp_path / "base.json"
    manifest.write_text(
        DatasetManifest(
            samples=[sample], preparation={"original_protocol": "preserved"}
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    reviews = [
        dict(
            review_for(sample, (0, 0, 256, 256)),
            parent_image_id=sample.image_id,
            halo=32,
        ),
        dict(
            review_for(sample, (256, 0, 256, 256)),
            parent_image_id=sample.image_id,
            halo=32,
        ),
    ]
    approval = tmp_path / "reviews.json"
    approval.write_text(json.dumps({"reviews": reviews}), encoding="utf-8")
    return manifest, approval, reviews


def test_attach_approved_negatives_preserves_base_and_exact_replay(tmp_path):
    from airfield_aircraft_vision.negative_crops import attach_reviewed_negatives

    manifest, approval, reviews = make_attach_fixture(tmp_path)
    original = manifest.read_bytes()
    output = tmp_path / "with-negatives.json"
    result = attach_reviewed_negatives(manifest, approval, output)
    assert len(result.samples) == 3
    assert manifest.read_bytes() == original
    assert result.preparation["original_protocol"] == "preserved"
    assert len(result.preparation["verified_negative_image_ids"]) == 2
    assert result.preparation["negative_review_sha256"] == sha256_file(approval)
    negative_paths = [tmp_path / s.path for s in result.samples[1:]]
    assert all(p.parent.name == "negative-images" for p in negative_paths)
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in [output] + negative_paths
    }
    repeated = attach_reviewed_negatives(manifest, approval, output)
    assert repeated == result
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


def test_attach_prevalidates_all_reviews_before_any_mutation(tmp_path):
    import json
    from airfield_aircraft_vision.negative_crops import attach_reviewed_negatives

    manifest, approval, reviews = make_attach_fixture(tmp_path)
    reviews[1]["parent_sha256"] = "0" * 64
    approval.write_text(json.dumps({"reviews": reviews}), encoding="utf-8")
    output = tmp_path / "with-negatives.json"
    with pytest.raises(ValueError, match="checksum"):
        attach_reviewed_negatives(manifest, approval, output)
    assert not output.exists()
    assert not (tmp_path / "negative-images").exists()


def test_attach_changed_review_or_raster_rejected_without_overwrite(tmp_path):
    import json
    from airfield_aircraft_vision.negative_crops import attach_reviewed_negatives

    manifest, approval, reviews = make_attach_fixture(tmp_path)
    output = tmp_path / "with-negatives.json"
    result = attach_reviewed_negatives(manifest, approval, output)
    before = output.read_bytes()
    reviews[0]["notes"] = "Changed approval."
    approval.write_text(json.dumps({"reviews": reviews}), encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        attach_reviewed_negatives(manifest, approval, output)
    assert output.read_bytes() == before
    reviews[0]["notes"] = "Synthetic fixture only."
    approval.write_text(json.dumps({"reviews": reviews}), encoding="utf-8")
    negative = tmp_path / result.samples[-1].path
    negative.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="raster differs"):
        attach_reviewed_negatives(manifest, approval, output)
    assert negative.read_bytes() == b"tampered"


def test_attach_duplicate_and_test_reviews_rejected_before_writes(tmp_path):
    import json
    from airfield_aircraft_vision.negative_crops import attach_reviewed_negatives
    from airfield_aircraft_vision.dataset import load_manifest

    manifest, approval, reviews = make_attach_fixture(tmp_path)
    approval.write_text(
        json.dumps({"reviews": [reviews[0], reviews[0]]}), encoding="utf-8"
    )
    output = tmp_path / "with-negatives.json"
    with pytest.raises(ValueError, match="duplicate"):
        attach_reviewed_negatives(manifest, approval, output)
    assert not (tmp_path / "negative-images").exists()
    m = load_manifest(manifest)
    m.samples[0].split = "test"
    manifest.write_text(m.model_dump_json(indent=2), encoding="utf-8")
    approval.write_text(json.dumps({"reviews": [reviews[0]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="test"):
        attach_reviewed_negatives(manifest, approval, output)
    assert not (tmp_path / "negative-images").exists()


def test_attach_refuses_base_overwrite_and_missing_derivative(tmp_path):
    from airfield_aircraft_vision.negative_crops import attach_reviewed_negatives

    manifest, approval, _ = make_attach_fixture(tmp_path)
    with pytest.raises(ValueError, match="different"):
        attach_reviewed_negatives(manifest, approval, manifest)
    output = tmp_path / "with-negatives.json"
    result = attach_reviewed_negatives(manifest, approval, output)
    (tmp_path / result.samples[-1].path).unlink()
    with pytest.raises(ValueError, match="missing derivative"):
        attach_reviewed_negatives(manifest, approval, output)
    assert not (tmp_path / result.samples[-1].path).exists()
