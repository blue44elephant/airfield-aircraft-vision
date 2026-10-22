import pytest
from airfield_aircraft_vision.robustness_dataset import (
    choose_fresh_groups,
    select_robustness_tiles,
    geographic_truth,
    window_truth,
)


def test_group_selection_represents_country_weather_and_is_deterministic():
    rows = [
        dict(loc_id=str(i), Country=c, Weather=w)
        for i, (c, w) in enumerate(
            [
                ("USA", "clear"),
                ("USA", "clear"),
                ("France", "snow"),
                ("Germany", "haze"),
            ]
        )
    ]
    groups = {str(i): f"g{i}" for i in range(4)}
    result = choose_fresh_groups({"g0", "g1", "g2", "g3"}, rows, groups, count=3)
    assert {"g2", "g3"} <= set(result)
    assert result == choose_fresh_groups(
        set(reversed(list(groups.values()))), rows, groups, count=3
    )
    with pytest.raises(ValueError):
        choose_fresh_groups(set(), rows, groups, count=1)


def test_tile_strata_include_crowded_small_partial_and_never_duplicate():
    images = [dict(id=i, file_name=f"1_cap{i%2}_tile_{i}.png") for i in range(10)]
    anns = [dict(id=i, image_id=i, bbox=[0, 0, 10 + i, 10 + i]) for i in range(10)]
    anns += [dict(id=20 + i, image_id=9, bbox=[0, 0, 40, 40]) for i in range(8)]
    rich = {i: dict(partialDec=0.01 if i == 8 else 1) for i in range(30)}
    result = select_robustness_tiles(images, anns, rich=rich, count=6)
    ids = [i["id"] for i in result]
    assert len(ids) == len(set(ids)) == 6
    assert {0, 8, 9} <= set(ids)
    assert result == select_robustness_tiles(images, anns, rich=rich, count=6)
    with pytest.raises(ValueError):
        select_robustness_tiles(images, anns, count=0)


def geo_fixture():
    feature = dict(
        type="Feature",
        properties=dict(loc_id=1, cat_id="c", is_plane=1),
        geometry=dict(
            type="Polygon", coordinates=[[[1, 9], [3, 9], [3, 7], [1, 7], [1, 9]]]
        ),
    )
    return dict(
        crs=dict(properties=dict(name="urn:ogc:def:crs:OGC:1.3:CRS84")),
        features=[feature, feature],
    )


def test_geographic_truth_preserves_native_edges_and_deduplicates_physical_label():
    labels = geographic_truth(
        geo_fixture(), 1, "c", "EPSG:4326", (1, 0, 0, 0, -1, 10), 10, 10
    )
    assert len(labels) == 1
    assert labels[0]["xyxy"] == [1, 1, 3, 3]
    assert labels[0]["bbox_visible_fraction"] == 1
    assert len(labels[0]["annotation_id"]) == 64


def test_geographic_truth_rejects_missing_and_invalid_contract():
    with pytest.raises(ValueError, match="no matching"):
        geographic_truth(
            geo_fixture(), 2, "c", "EPSG:4326", (1, 0, 0, 0, -1, 10), 10, 10
        )
    geo = geo_fixture()
    geo["features"][0]["properties"]["is_plane"] = 0
    with pytest.raises(ValueError, match="unexpected"):
        geographic_truth(geo, 1, "c", "EPSG:4326", (1, 0, 0, 0, -1, 10), 10, 10)


def test_window_clips_without_double_identity_or_coordinate_resampling():
    labels = geographic_truth(
        geo_fixture(), 1, "c", "EPSG:4326", (1, 0, 0, 0, -1, 10), 10, 10
    )
    left = window_truth(labels, (0, 0, 2, 5))[0]
    right = window_truth(labels, (2, 0, 3, 5))[0]
    assert left["xyxy"] == [1, 1, 2, 3]
    assert right["xyxy"] == [0, 1, 1, 3]
    assert left["annotation_id"] == right["annotation_id"]
    assert left["bbox_visible_fraction"] == 0.5
    with pytest.raises(ValueError):
        window_truth(labels, (-1, 0, 5, 5))


def _reservation_fixture(tmp_path):
    import csv, json
    from airfield_aircraft_vision.optical_contracts import (
        DatasetManifest,
        Sample,
        RasterSource,
    )

    meta = tmp_path / "metadata"
    meta.mkdir()
    rows = [
        dict(
            image_id=f"{i}_c{i}",
            loc_id=str(i),
            cat_id=f"c{i}",
            Country=f"country{i}",
            Air_Field=f"field{i}",
            Weather="clear",
        )
        for i in range(1, 7)
    ]
    with (meta / "RarePlanes_Public_Metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    images = [
        dict(id=i, file_name=f"{i}_c{i}_tile_1.png", width=512, height=512)
        for i in range(1, 7)
    ]
    anns = [dict(id=i, image_id=i, bbox=[0, 0, 10, 10]) for i in range(1, 7)]
    (meta / "instances_train_aircraft.json").write_text(
        json.dumps(dict(images=images, annotations=anns))
    )
    (meta / "RarePlanes_Train_Coco_Annotations_tiled.json").write_text(
        json.dumps(
            dict(images=images, annotations=[dict(a, partialDec=1) for a in anns])
        )
    )
    geo = geo_fixture()
    geo["features"] = [
        dict(
            type="Feature",
            properties=dict(loc_id=i),
            geometry=dict(
                type="Polygon",
                coordinates=[
                    [[i * 10, 0], [i * 10 + 0.1, 0], [i * 10 + 0.1, 0.1], [i * 10, 0]]
                ],
            ),
        )
        for i in range(1, 7)
    ]
    (meta / "RarePlanes_Public_All_Annotations.geojson").write_text(json.dumps(geo))
    samples = [
        Sample(
            image_id=f"old{i}",
            path=f"old{i}.tif",
            split=s,
            source=RasterSource(
                scene_id=f"old{i}",
                sensor_id="WV03",
                native_gsd_m=0.3,
                sha256=str(i) * 64,
                width=512,
                height=512,
                airfield_group=f"rareplanes-location-{i}",
                capture_group=f"c{i}",
            ),
        )
        for i, s in enumerate(("train", "validation", "test"), 1)
    ]
    prior = tmp_path / "prior.json"
    prior.write_text(DatasetManifest(samples=samples).model_dump_json())
    return meta, prior, tmp_path / "fresh"


def test_reservation_excludes_all_previous_splits_and_is_immutable(tmp_path):
    import json
    from airfield_aircraft_vision.robustness_dataset import reserve_robustness

    meta, prior, root = _reservation_fixture(tmp_path)
    result = reserve_robustness(
        meta, [prior], root, test_groups=1, validation_groups=1, train_groups=1
    )
    assert set(g for values in result["selected_groups"].values() for g in values) == {
        f"rareplanes-location-{i}" for i in (4, 5, 6)
    }
    frozen = (root / "reservation.json").read_bytes()
    assert (
        reserve_robustness(
            meta, [prior], root, test_groups=1, validation_groups=1, train_groups=1
        )
        == result
    )
    assert (root / "reservation.json").read_bytes() == frozen
    with pytest.raises(ValueError, match="frozen"):
        reserve_robustness(
            meta,
            [prior],
            root,
            test_groups=1,
            validation_groups=1,
            train_groups=1,
            seed=9,
        )


def test_materialization_checks_string_prior_paths_before_network(tmp_path):
    from airfield_aircraft_vision.robustness_dataset import (
        reserve_robustness,
        materialize_robustness,
    )

    meta, prior, root = _reservation_fixture(tmp_path)
    reserve_robustness(
        meta, [prior], root, test_groups=1, validation_groups=1, train_groups=1
    )
    prior.write_text("tampered")
    with pytest.raises(ValueError, match="prior manifest changed"):
        materialize_robustness(meta, root)


def test_native_window_export_preserves_pixels_mask_affine_and_original(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.enums import ColorInterp
    from airfield_aircraft_vision.optical_contracts import RasterSource, Sample
    from airfield_aircraft_vision.preprocessing import sha256_file, NativeRaster
    from airfield_aircraft_vision.robustness_dataset import export_native_window

    path = tmp_path / "source.tif"
    pixels = np.arange(300, dtype=np.uint8).reshape(3, 10, 10)
    mask = np.full((10, 10), 255, dtype=np.uint8)
    mask[4, 3] = 0
    affine = from_origin(500000, 4000000, 0.3, 0.3)
    with (
        rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),
        rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=10,
            height=10,
            count=3,
            dtype="uint8",
            crs="EPSG:32632",
            transform=affine,
        ) as ds,
    ):
        ds.write(pixels)
        ds.write_mask(mask)
        ds.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
    source = RasterSource(
        scene_id="source",
        sensor_id="WV03",
        native_gsd_m=0.3,
        sha256=sha256_file(path),
        width=10,
        height=10,
        crs="EPSG:32632",
        transform=tuple(affine)[:6],
        airfield_group="g",
        capture_group="c",
    )
    sample = Sample(image_id="source", path="source.tif", split="test", source=source)
    before = path.read_bytes()
    crop, labels = export_native_window(
        path, sample, [], (2, 3, 4, 5), tmp_path / "crop.tif", root=tmp_path
    )
    assert path.read_bytes() == before
    assert labels == []
    assert not (tmp_path / "crop.tif.msk").exists()
    with NativeRaster(tmp_path / crop.path, crop.source) as native:
        ds = native.dataset
        assert np.array_equal(ds.read(), pixels[:, 3:8, 2:6])
        assert np.array_equal(ds.dataset_mask(), mask[3:8, 2:6])
        assert ds.transform * (0, 0) == affine * (2, 3)
        assert len(ds.files) == 1
    with pytest.raises(ValueError, match="overwrite"):
        export_native_window(
            path, sample, [], (2, 3, 4, 5), tmp_path / "crop.tif", root=tmp_path
        )


def test_polygon_clip_avoids_spurious_corner_and_matches_visible_aircraft_box():
    geo = geo_fixture()
    feature = geo["features"][0]
    feature["geometry"]["coordinates"] = [[[0, 2], [2, 0], [4, 2], [2, 4], [0, 2]]]
    # Diamond bbox overlaps the corner, but the actual polygon does not.
    assert geographic_truth(geo, 1, "c", "EPSG:4326", (1, 0, 0, 0, 1, 0), 1, 1) == []
    labels = geographic_truth(geo, 1, "c", "EPSG:4326", (1, 0, 0, 0, 1, 0), 4, 4)
    assert window_truth(labels, (0, 0, 1, 1)) == []
    fragment = window_truth(labels, (0, 0, 1, 4))[0]
    assert fragment["xyxy"] == [0, 1, 1, 3]
    assert fragment["polygon_visible_fraction"] == pytest.approx(0.125)
    assert fragment["annotation_id"] == labels[0]["annotation_id"]


def test_future_negative_builder_preserves_review_ids_and_conservative_category(
    tmp_path,
):
    from airfield_aircraft_vision.robustness_dataset import (
        build_reviewed_negative_manifest,
    )
    from airfield_aircraft_vision.dataset import load_manifest
    from airfield_aircraft_vision.generalization_metrics import reviewed_negative_ids

    meta, prior, root = _reservation_fixture(tmp_path)
    sample = load_manifest(prior).samples[2]
    record = dict(
        sample=sample.model_dump(mode="json"),
        parent_image_id="parent",
        parent_sha256="f" * 64,
        proposal=dict(window=[0, 0, 512, 512]),
        review=dict(
            decision="empty_no_aircraft",
            category="empty_apron_or_runway",
            crop_sha256=sample.source.sha256,
            parent_sha256="f" * 64,
            window=[0, 0, 512, 512],
        ),
    )
    result = build_reviewed_negative_manifest([record], review_sha256="e" * 64)
    assert result.preparation["verified_negative_image_ids"] == [sample.image_id]
    assert reviewed_negative_ids(result.preparation, result.samples) == (
        set(),
        {sample.image_id},
    )
    assert (
        result.preparation["negative_review_details"][0]["review"]["original_category"]
        == "empty_apron_or_runway"
    )
    record["review"]["crop_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        build_reviewed_negative_manifest([record], review_sha256="e" * 64)
