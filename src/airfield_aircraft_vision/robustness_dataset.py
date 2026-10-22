"""AW-01D frozen, geographically independent robustness cohorts.

Selection uses source metadata/ground truth, never detector predictions. The
evaluation is deliberately stratified; its error rates are not population rates.
Original imagery and all previous manifests are opened read-only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .dataset import (
    ATTRIBUTION,
    BASE,
    BoundedDownload,
    load_manifest,
    location,
    materialize_samples,
    save_frozen_manifest,
    scene_id,
)
from .expanded_dataset import geographic_groups, audit_spatial_separation
from .optical_contracts import DatasetManifest, PixelBox, RasterSource, Sample
from .preprocessing import sha256_file, pixel_spacing_m


def _order(values, seed, key=lambda x: x):
    return sorted(
        values, key=lambda x: hashlib.sha256(f"{seed}:{key(x)}".encode()).hexdigest()
    )


def choose_fresh_groups(available, rows, groups, *, count, seed=71):
    """Maximize country/weather representation; seeded ties, no model outputs."""
    if type(count) is not int or not 1 <= count <= 16:
        raise ValueError("group count must be 1..16")
    attrs = defaultdict(set)
    for row in rows:
        attrs[groups[row["loc_id"]]].update(
            ("country:" + row["Country"], "weather:" + row["Weather"])
        )
    remaining = set(available)
    selected, seen = [], set()
    while len(selected) < count and remaining:
        candidates = _order(remaining, seed)
        chosen = max(candidates, key=lambda g: len(attrs[g] - seen))
        selected.append(chosen)
        seen |= attrs[chosen]
        remaining.remove(chosen)
    if len(selected) != count:
        raise ValueError("not enough geographically unexposed groups")
    return selected


def select_robustness_tiles(images, annotations, *, count=6, seed=71, rich=None):
    """Fixed crowded/small/partial strata plus capture-balanced seeded samples."""
    if type(count) is not int or not 1 <= count <= 16:
        raise ValueError("tile count must be 1..16")
    labels = defaultdict(list)
    for ann in annotations:
        labels[ann["image_id"]].append(ann)
    rich = rich or {}
    ordered = _order(images, seed, key=lambda i: i["file_name"])
    rank = {i["id"]: n for n, i in enumerate(ordered)}
    crowded = sorted(images, key=lambda i: (-len(labels[i["id"]]), rank[i["id"]]))
    small = sorted(
        images,
        key=lambda i: (
            min(
                (a["bbox"][2] * a["bbox"][3] for a in labels[i["id"]]), default=math.inf
            ),
            rank[i["id"]],
        ),
    )
    partial = sorted(
        images,
        key=lambda i: (
            min(
                (rich.get(a["id"], {}).get("partialDec", 1) for a in labels[i["id"]]),
                default=1,
            ),
            rank[i["id"]],
        ),
    )
    keep, seen = [], set()

    def append(image):
        if len(keep) < count and image["id"] not in seen:
            keep.append(image)
            seen.add(image["id"])

    # Selection is reserved before imagery inspection or any model prediction.
    for sequence in (ordered[:1], crowded[:1], small[:1], partial[:1]):
        for image in sequence:
            append(image)
    captures = defaultdict(list)
    for image in ordered:
        captures[scene_id(image["file_name"])].append(image)
    while len(keep) < min(count, len(images)) and any(captures.values()):
        for capture in sorted(captures):
            if captures[capture]:
                append(captures[capture].pop(0))
    return keep


def reserve_robustness(
    metadata_root,
    prior_paths,
    root,
    *,
    seed=71,
    test_groups=8,
    validation_groups=2,
    train_groups=4,
    tiles_per_group=6,
):
    """Write the immutable complete cohort before any new model results exist."""
    metadata_root, root = Path(metadata_root), Path(root)
    paths = [Path(p) for p in prior_paths]
    prior = {str(p.as_posix()): sha256_file(p) for p in paths}
    names = [
        "RarePlanes_Public_Metadata.csv",
        "instances_train_aircraft.json",
        "RarePlanes_Public_All_Annotations.geojson",
        "RarePlanes_Train_Coco_Annotations_tiled.json",
    ]
    hashes = {name: sha256_file(metadata_root / name) for name in names}
    rows = list(
        csv.DictReader(
            (metadata_root / names[0]).read_text(encoding="utf-8-sig").splitlines()
        )
    )
    index = json.loads((metadata_root / names[1]).read_text())
    geo = json.loads((metadata_root / names[2]).read_text())
    rich = json.loads((metadata_root / names[3]).read_text())
    from .generalization_dataset import validate_rich_contract

    validate_rich_contract(index, rich)
    rich_by_id = {a["id"]: a for a in rich["annotations"]}
    groups, centres, edges = geographic_groups(rows, geo)
    exposed_groups, exposed_captures = set(), set()
    for path in paths:
        for sample in load_manifest(path).samples:
            exposed_groups.add(sample.source.airfield_group)
            exposed_captures.add(sample.source.capture_group)
    by_group = defaultdict(list)
    for image in index["images"]:
        g = groups[location(image["file_name"])]
        capture = scene_id(image["file_name"]).split("_", 1)[1]
        if g not in exposed_groups and capture not in exposed_captures:
            by_group[g].append(image)
    requested = {
        "test": test_groups,
        "validation": validation_groups,
        "train": train_groups,
    }
    selected_groups, selected_ids = {}, {}
    remaining = set(by_group)
    for split, count in requested.items():
        chosen = choose_fresh_groups(
            remaining, rows, groups, count=count, seed=seed + len(selected_groups)
        )
        selected_groups[split] = chosen
        remaining -= set(chosen)
        selected_ids[split] = [
            image["file_name"]
            for group in chosen
            for image in select_robustness_tiles(
                by_group[group],
                index["annotations"],
                count=tiles_per_group,
                seed=seed,
                rich=rich_by_id,
            )
        ]
    request = dict(
        version="aw01d-reservation-1",
        created_date="2026-10-22",
        seed=seed,
        tiles_per_group=tiles_per_group,
        metadata_sha256=hashes,
        previous_manifests=prior,
        exposed_group_count=len(exposed_groups),
        remaining_fresh_group_count=len(by_group),
        selected_groups=selected_groups,
        selected_ids=selected_ids,
        selection="Country/weather diversity with seeded ties. Per group: seeded/crowded/small/partial first, then capture-balanced seeded tiles. No model predictions.",
        evaluation_policy="Stratified diagnostic sample, not prevalence estimate. Test labels cannot select a model/threshold. AW01C baseline fixed before evaluation. Corrective experiments use only reserved train/validation.",
        timestamp_policy="Source minute precision; UTC assumed, not verified.",
        official_partition="All reserved groups are official training partition but have never appeared in any prior experiment split.",
        group_metadata={
            group: dict(
                countries=sorted(
                    {r["Country"] for r in rows if groups[r["loc_id"]] == group}
                ),
                weather=sorted(
                    {r["Weather"] for r in rows if groups[r["loc_id"]] == group}
                ),
                airfields=sorted(
                    {r["Air_Field"] for r in rows if groups[r["loc_id"]] == group}
                ),
                capture_groups=sorted(
                    {scene_id(i["file_name"]).split("_", 1)[1] for i in by_group[group]}
                ),
            )
            for chosen in selected_groups.values()
            for group in chosen
        },
    )
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "reservation.json"
    if dest.exists():
        if json.loads(dest.read_text()) != request:
            raise ValueError("frozen robustness reservation differs")
    else:
        with dest.open("x", encoding="utf-8") as stream:
            json.dump(request, stream, indent=2)
    return request


def materialize_robustness(metadata_root, root, *, workers=4, max_mib=128):
    """Fetch only the reserved positive tiles; reuse original parsing contracts."""
    metadata_root, root = Path(metadata_root), Path(root)
    if type(workers) is not int or not 1 <= workers <= 4 or not 16 <= max_mib <= 192:
        raise ValueError("bounded worker/byte settings required")
    reservation_path = root / "reservation.json"
    reservation = json.loads(reservation_path.read_text())
    for name, digest in reservation["metadata_sha256"].items():
        if sha256_file(metadata_root / name) != digest:
            raise ValueError("reserved metadata changed")
    for path, digest in reservation["previous_manifests"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("prior manifest changed")
    cached = root / "manifest.json"
    if cached.exists():
        manifest = load_manifest(cached)
        if manifest.preparation["reservation_sha256"] != sha256_file(reservation_path):
            raise ValueError("cached reservation changed")
        for sample in manifest.samples:
            if sha256_file(root / sample.path) != sample.source.sha256:
                raise ValueError("cached imagery changed")
            if (
                sample.source.sidecar_sha256
                and sha256_file(Path(str(root / sample.path) + ".aux.xml"))
                != sample.source.sidecar_sha256
            ):
                raise ValueError("cached sidecar changed")
        return manifest
    rows = list(
        csv.DictReader(
            (metadata_root / "RarePlanes_Public_Metadata.csv")
            .read_text(encoding="utf-8-sig")
            .splitlines()
        )
    )
    index = json.loads((metadata_root / "instances_train_aircraft.json").read_text())
    geo = json.loads(
        (metadata_root / "RarePlanes_Public_All_Annotations.geojson").read_text()
    )
    groups, _, _ = geographic_groups(rows, geo)
    assigned = {
        g: split
        for split, values in reservation["selected_groups"].items()
        for g in values
    }
    ids = {i for values in reservation["selected_ids"].values() for i in values}
    selected = {"train": defaultdict(list)}
    for image in index["images"]:
        if image["file_name"] in ids:
            selected["train"][groups[location(image["file_name"])]].append(image)
    if sum(map(len, selected["train"].values())) != len(ids):
        raise ValueError("reserved image missing from verified index")
    downloader = BoundedDownload(max_mib * 1024**2)
    requests = []
    for name in sorted(ids):
        if Path(name).name != name:
            raise ValueError("unsafe reserved image name")
        for suffix, bound in (("", 4 * 1024**2), (".aux.xml", 100000)):
            requests.append(
                (
                    BASE + "real/train/PS-RGB_tiled/" + name + suffix,
                    root / "images" / (name + suffix),
                    bound,
                )
            )

    def fetch(args):
        return downloader.fetch(args[0], args[1], max_file=args[2])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, requests))
    samples = materialize_samples(
        root,
        metadata_root / "RarePlanes_Public_Metadata.csv",
        rows,
        {"train": index},
        groups,
        assigned,
        selected,
        downloader,
    )
    if {s.image_id for s in samples} != ids:
        raise ValueError("materialized membership differs from reservation")
    manifest = DatasetManifest(
        samples=samples,
        preparation=dict(
            dataset="RarePlanes AW01D independent robustness",
            license="CC-BY-SA-4.0",
            reservation_sha256=sha256_file(reservation_path),
            metadata_sha256=reservation["metadata_sha256"],
            metadata_root=metadata_root.as_posix(),
            downloaded_bytes=downloader.downloaded,
            spatial_audit=audit_spatial_separation(samples),
            geography_by_scene={r["image_id"]: r["Country"] for r in rows},
            metadata_weather_by_scene={r["image_id"]: r["Weather"] for r in rows},
            limitations=[
                "Annotation-stratified diagnostic sample, not prevalence estimate.",
                "Provided positive-only tiles; empty-apron/full-scene cohorts separate.",
                "Minute timestamp UTC basis assumed, not verified.",
            ],
        ),
    )
    return save_frozen_manifest(cached, manifest)


def _clip_polygon(points, bounds):
    """Clip source pixel polygon against an axis-aligned rectangle."""
    points = [tuple(p) for p in points]
    if points and points[0] == points[-1]:
        points.pop()
    left, top, right, bottom = bounds
    for axis, value, lower in (
        (0, left, True),
        (0, right, False),
        (1, top, True),
        (1, bottom, False),
    ):
        output = []
        if not points:
            break
        previous = points[-1]
        previous_inside = previous[axis] >= value if lower else previous[axis] <= value
        for current in points:
            current_inside = current[axis] >= value if lower else current[axis] <= value
            if current_inside != previous_inside:
                ratio = (value - previous[axis]) / (current[axis] - previous[axis])
                output.append(
                    tuple(
                        previous[k] + ratio * (current[k] - previous[k]) for k in (0, 1)
                    )
                )
            if current_inside:
                output.append(current)
            previous, previous_inside = current, current_inside
        points = output
    return points


def _polygon_area(points):
    if len(points) < 3:
        return 0.0
    return (
        abs(
            sum(
                a[0] * b[1] - b[0] * a[1]
                for a, b in zip(points, points[1:] + points[:1])
            )
        )
        / 2
    )


def geographic_truth(geojson, location_id, capture_id, crs, transform, width, height):
    """Full-location annotation polygons to original raster edges, without resize.

    A physical GT identifier is a hash of source annotation+capture, independent
    of tile/windows. Touching/clipped labels retain visibility metadata.
    """
    from rasterio.warp import transform as reproject_points
    from affine import Affine

    if geojson.get("crs", {}).get("properties", {}).get("name") not in (
        "urn:ogc:def:crs:OGC:1.3:CRS84",
        "urn:ogc:def:crs:EPSG::4326",
    ):
        raise ValueError("verified WGS84 geographic annotation CRS required")
    affine = Affine(*transform)
    inverse = ~affine
    seen, labels = set(), []
    for feature in geojson["features"]:
        props = feature["properties"]
        if str(props["loc_id"]) != str(location_id) or props["cat_id"] != capture_id:
            continue
        if props.get("is_plane") != 1 or feature["geometry"]["type"] != "Polygon":
            raise ValueError("unexpected full-location aircraft annotation")
        raw = json.dumps(feature, sort_keys=True, separators=(",", ":"))
        identity = hashlib.sha256(raw.encode()).hexdigest()
        if identity in seen:
            continue
        seen.add(identity)
        ring = feature["geometry"]["coordinates"][0]
        if len(ring) < 4 or not all(
            len(p) == 2
            and all(math.isfinite(v) for v in p)
            and -180 <= p[0] <= 180
            and -90 <= p[1] <= 90
            for p in ring
        ):
            raise ValueError("invalid geographic annotation polygon")
        xs, ys = reproject_points(
            "EPSG:4326", crs, [p[0] for p in ring], [p[1] for p in ring]
        )
        pixels = [inverse * (x, y) for x, y in zip(xs, ys)]
        left, right = min(p[0] for p in pixels), max(p[0] for p in pixels)
        top, bottom = min(p[1] for p in pixels), max(p[1] for p in pixels)
        clipped_polygon = _clip_polygon(pixels, (0, 0, width, height))
        if _polygon_area(clipped_polygon) <= 1e-12:
            continue
        clipped = (
            min(p[0] for p in clipped_polygon),
            min(p[1] for p in clipped_polygon),
            max(p[0] for p in clipped_polygon),
            max(p[1] for p in clipped_polygon),
        )
        area = (right - left) * (bottom - top)
        visible = ((clipped[2] - clipped[0]) * (clipped[3] - clipped[1])) / area
        labels.append(
            dict(
                annotation_id=identity,
                xyxy=list(clipped),
                unclipped_xyxy=[left, top, right, bottom],
                bbox_visible_fraction=visible,
                polygon_visible_fraction=_polygon_area(clipped_polygon)
                / _polygon_area(pixels),
                source_pixel_polygon=[list(p) for p in pixels],
                pixel_polygon=[list(p) for p in clipped_polygon],
                footprint_geojson=feature["geometry"],
            )
        )
    if not seen:
        raise ValueError("full-location annotation contract has no matching records")
    return labels


def window_truth(labels, window):
    """Clip unique parent labels to one exact native window; retain identity."""
    if len(window) != 4 or any(type(v) is not int for v in window):
        raise ValueError("window requires four integers")
    x, y, width, height = window
    if min(x, y) < 0 or min(width, height) < 1:
        raise ValueError("invalid native window")
    output = []
    for label in labels:
        left, top, right, bottom = label["xyxy"]
        parent_polygon = label.get(
            "pixel_polygon",
            [[left, top], [right, top], [right, bottom], [left, bottom]],
        )
        clipped_polygon = _clip_polygon(parent_polygon, (x, y, x + width, y + height))
        if _polygon_area(clipped_polygon) <= 1e-12:
            continue
        local = [(px - x, py - y) for px, py in clipped_polygon]
        clipped = [
            min(p[0] for p in local),
            min(p[1] for p in local),
            max(p[0] for p in local),
            max(p[1] for p in local),
        ]
        original = label.get("unclipped_xyxy", label["xyxy"])
        original_area = (original[2] - original[0]) * (original[3] - original[1])
        original_polygon = label.get("source_pixel_polygon", parent_polygon)
        output.append(
            dict(
                label,
                xyxy=clipped,
                pixel_polygon=[list(p) for p in local],
                bbox_visible_fraction=(
                    (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
                )
                / original_area,
                polygon_visible_fraction=_polygon_area(local)
                / _polygon_area(original_polygon),
            )
        )

    return output


def source_scene_sample(cog_path, metadata_root, root, *, split, group):
    """Describe an unchanged full native COG and derive complete-location labels."""
    import rasterio
    from rasterio.enums import ColorInterp

    cog_path, metadata_root, root = map(Path, (cog_path, metadata_root, root))
    name = cog_path.stem
    loc, capture_id = name.split("_", 1)
    rows = list(
        csv.DictReader(
            (metadata_root / "RarePlanes_Public_Metadata.csv")
            .read_text(encoding="utf-8-sig")
            .splitlines()
        )
    )
    row = next((r for r in rows if r["image_id"] == name), None)
    if row is None:
        raise ValueError("source scene absent from verified metadata")
    reservation = json.loads((root / "reservation.json").read_text())
    if (
        group not in reservation["selected_groups"][split]
        or capture_id not in reservation["group_metadata"][group]["capture_groups"]
    ):
        raise ValueError("source scene outside frozen split/capture reservation")
    geo = json.loads(
        (metadata_root / "RarePlanes_Public_All_Annotations.geojson").read_text()
    )
    with rasterio.open(cog_path) as ds:
        if (
            ds.crs is None
            or ds.count != 3
            or any(t != "uint8" for t in ds.dtypes)
            or tuple(ds.colorinterp)
            != (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        ):
            raise ValueError("verified native georeferenced uint8 RGB COG required")
        spacing = pixel_spacing_m(ds.crs, ds.transform, ds.width, ds.height)
        labels = geographic_truth(
            geo, loc, capture_id, ds.crs, tuple(ds.transform)[:6], ds.width, ds.height
        )
        source = RasterSource(
            scene_id=name,
            sensor_id=row["sensor"],
            acquisition_timestamp=datetime.strptime(
                row["collection_time_start"], "%m/%d/%y %H:%M"
            ).replace(tzinfo=timezone.utc),
            timestamp_precision="minute",
            acquisition_time_raw=row["collection_time_start"],
            timestamp_timezone_basis="assumed_utc",
            native_gsd_m=float(row["pan_resolution_maximum"]),
            panchromatic_gsd_m=float(row["pan_resolution_maximum"]),
            pan_sharpened_gsd_m=max(spacing),
            pixel_spacing_m=spacing,
            quality="unknown",
            airfield_group=group,
            capture_group=capture_id,
            source_uri=BASE + "real/train/PS-RGB_cog/" + cog_path.name,
            license="CC-BY-SA-4.0",
            attribution=ATTRIBUTION,
            sha256=sha256_file(cog_path),
            width=ds.width,
            height=ds.height,
            crs=ds.crs.to_string(),
            transform=tuple(ds.transform)[:6],
            metadata_sha256=sha256_file(
                metadata_root / "RarePlanes_Public_Metadata.csv"
            ),
            provenance_notes=[
                "Unmodified complete-location native RGB COG; no resize or reprojection.",
                "Geographic full-location annotations mapped through original inverse affine; physical annotation IDs retained in ground-truth sidecar.",
                "Source minute timestamps retained; UTC basis assumed, not verified.",
                f"Country={row['Country']}; Weather={row['Weather']}; airfield={row['Air_Field']}",
            ],
        )
    sample = Sample(
        image_id=cog_path.name,
        path=cog_path.relative_to(root).as_posix(),
        split=split,
        source=source,
        boxes=[PixelBox(xyxy=tuple(label["xyxy"])) for label in labels],
    )
    return sample, labels


def export_native_window(cog_path, source_sample, labels, window, destination, *, root):
    """Create a lossless pixel-exact georeferenced crop; never marks it empty."""
    import numpy as np
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.windows import Window
    from .preprocessing import NativeRaster

    cog_path, destination, root = map(Path, (cog_path, destination, root))
    if len(window) != 4 or any(type(v) is not int for v in window):
        raise ValueError("native window requires four integers")
    x, y, width, height = window
    if (
        min(x, y) < 0
        or min(width, height) < 1
        or x + width > source_sample.source.width
        or y + height > source_sample.source.height
    ):
        raise ValueError("native crop outside original scene")
    if destination.exists():
        raise ValueError("refusing to overwrite native crop")
    cropped_labels = window_truth(labels, window)
    with NativeRaster(cog_path, source_sample.source) as native:
        ds = native.dataset
        win = Window(*window)
        pixels = ds.read(window=win)
        mask = ds.dataset_mask(window=win)
        affine = ds.window_transform(win)
        profile = dict(
            driver="GTiff",
            width=width,
            height=height,
            count=3,
            dtype="uint8",
            crs=ds.crs,
            transform=affine,
            compress="deflate",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        # GDAL creation can replace files; reserve the destination exclusively.
        with destination.open("xb"):
            pass
        try:
            with (
                rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),
                rasterio.open(destination, "w", **profile) as out,
            ):
                out.write(pixels)
                out.write_mask(mask)
                out.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                out.update_tags(
                    AW01D_PARENT_SHA256=source_sample.source.sha256,
                    AW01D_NATIVE_WINDOW=json.dumps(window),
                )
            with rasterio.open(destination) as check:
                if not np.array_equal(check.read(), pixels) or not np.array_equal(
                    check.dataset_mask(), mask
                ):
                    raise ValueError("native crop changed source pixels/mask")
        except BaseException:
            # Keep incomplete artifact for inspection; never delete original data.
            raise
    source = source_sample.source.model_copy(
        update=dict(
            scene_id=source_sample.source.scene_id
            + f"__native_{x}_{y}_{width}_{height}",
            sha256=sha256_file(destination),
            width=width,
            height=height,
            transform=tuple(affine)[:6],
            sidecar_sha256=None,
            pixel_spacing_m=pixel_spacing_m(
                source_sample.source.crs, affine, width, height
            ),
            provenance_notes=source_sample.source.provenance_notes
            + [
                f"Lossless native window {list(window)} of source SHA256 {source_sample.source.sha256}.",
                "Annotation-free is only a candidate: visual review required before claiming absence.",
            ],
        )
    )
    sample = Sample(
        image_id=destination.name,
        path=destination.relative_to(root).as_posix(),
        split=source_sample.split,
        source=source,
        boxes=[PixelBox(xyxy=tuple(label["xyxy"])) for label in cropped_labels],
    )
    return sample, cropped_labels


def propose_empty_scene_windows(
    cog_path, sample, labels, *, size=512, halo=64, maximum=4
):
    """Texture-rich label-free proposals; absence requires independent review."""
    import numpy as np
    from rasterio.windows import Window
    from .preprocessing import NativeRaster

    if (
        type(size) is not int
        or not 128 <= size <= 1024
        or type(halo) is not int
        or halo < 0
        or not 1 <= maximum <= 16
    ):
        raise ValueError("bounded native window/halo/count required")
    proposals = []
    with NativeRaster(Path(cog_path), sample.source) as native:
        ds = native.dataset
        for y in range(0, ds.height - size + 1, size // 2):
            for x in range(0, ds.width - size + 1, size // 2):
                if any(
                    x < b["xyxy"][2] + halo
                    and x + size > b["xyxy"][0] - halo
                    and y < b["xyxy"][3] + halo
                    and y + size > b["xyxy"][1] - halo
                    for b in labels
                ):
                    continue
                win = Window(x, y, size, size)
                if not (ds.dataset_mask(window=win) == 255).all():
                    continue
                pixels = ds.read(window=win)
                # Source texture only, never detector scores or apparent outcomes.
                score = float(np.std(pixels.astype(np.float32), axis=(1, 2)).mean())
                proposals.append(
                    dict(
                        window=[x, y, size, size],
                        texture_std=score,
                        halo=halo,
                        review_required=True,
                    )
                )
    chosen = []
    for item in sorted(proposals, key=lambda p: (-p["texture_std"], p["window"])):
        x, y, w, h = item["window"]
        if any(
            x < q["window"][0] + w
            and x + w > q["window"][0]
            and y < q["window"][1] + h
            and y + h > q["window"][1]
            for q in chosen
        ):
            continue
        chosen.append(item)
        if len(chosen) == maximum:
            break
    return chosen


def crowded_scene_window(sample, labels, *, size=1536):
    """Choose a bounded native multi-tile diagnostic window by GT density alone."""
    if type(size) is not int or not 512 <= size <= 2048:
        raise ValueError("scene window must be 512..2048 native pixels")
    width, height = min(size, sample.source.width), min(size, sample.source.height)
    candidates = []
    for label in labels:
        left, top, right, bottom = label["xyxy"]
        x = max(0, min(sample.source.width - width, int((left + right - width) / 2)))
        y = max(0, min(sample.source.height - height, int((top + bottom - height) / 2)))
        window = (x, y, width, height)
        visible = window_truth(labels, window)
        full = sum(b["bbox_visible_fraction"] >= 0.99 for b in visible)
        candidates.append((full, len(visible), -y, -x, window))
    if not candidates:
        raise ValueError("crowded-scene diagnostic requires ground truth")
    return max(candidates)[-1]


def build_reviewed_negative_manifest(records, *, review_sha256):
    """Future-cohort builder preserving explicit IDs/categories for evaluators.

    Locked AW01D manifests are intentionally unchanged. Runway/apron combined
    reviews map conservatively to background; the original category is retained.
    """
    approved, details = [], []
    for record in records:
        review = record.get("review") or {}
        if review.get("decision") != "empty_no_aircraft":
            continue
        sample = Sample.model_validate(record["sample"])
        if sample.boxes or sample.ignored_boxes:
            raise ValueError("reviewed negative still has aircraft annotations")
        if (
            review.get("crop_sha256") != sample.source.sha256
            or review.get("parent_sha256") != record["parent_sha256"]
            or review.get("window") != record["proposal"]["window"]
        ):
            raise ValueError("review does not match frozen crop/source/window")
        original = review.get("category")
        category = (
            "other_background" if original == "empty_apron_or_runway" else original
        )
        if category not in ("empty_apron", "difficult_background", "other_background"):
            raise ValueError("unsupported reviewed-negative category")
        approved.append(sample)
        details.append(
            dict(
                image_id=sample.image_id,
                parent_image_id=record["parent_image_id"],
                parent_sha256=record["parent_sha256"],
                sha256=sample.source.sha256,
                window=record["proposal"]["window"],
                review=dict(review, category=category, original_category=original),
            )
        )
    return DatasetManifest(
        samples=approved,
        preparation=dict(
            verified_negative_image_ids=[s.image_id for s in approved],
            negative_review_sha256=review_sha256,
            negative_review_details=details,
            negative_scope="Single non-expert visual review of source-pixel backgrounds; not population prevalence or surveyed full empty airfields.",
        ),
    )


FROZEN_MANIFESTS = {
    "positive-manifest.json": "manifest.json",
    "source-scene-manifest.json": "scene-manifest.json",
    "whole-scene-manifest.json": "whole-scene-manifest.json",
    "window-scene-manifest.json": "window-scene-manifest.json",
    "eval-negatives-manifest.json": "eval-negatives-manifest.json",
    "correction-manifest.json": "correction-manifest.json",
    "reviewed-negatives-manifest.json": "reviewed-negatives-manifest.json",
    "empty-candidates-manifest.json": "empty-candidates-manifest.json",
}


def audit_frozen_cohorts(root, frozen_root):
    """No network or writes: verify every frozen raster/sidecar and native grid."""
    from .preprocessing import NativeRaster

    root, frozen_root = Path(root), Path(frozen_root)
    unique = {}
    counts = {}
    for frozen_name, local_name in FROZEN_MANIFESTS.items():
        manifest = load_manifest(frozen_root / frozen_name)
        counts[local_name] = len(manifest.samples)
        for sample in manifest.samples:
            if sample.path in unique and unique[sample.path].source != sample.source:
                raise ValueError("conflicting frozen source metadata")
            unique[sample.path] = sample
    for sample in unique.values():
        path = (root / sample.path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("frozen raster escapes dataset root")
        with NativeRaster(path, sample.source) as native:
            if any(Path(f).suffix == ".msk" for f in native.dataset.files):
                raise ValueError("external mask not permitted by frozen AW01D dataset")
    return dict(
        unique_rasters=len(unique),
        cohorts=counts,
        network_used=False,
        originals_modified=False,
    )


def replay_frozen_cohorts(root, frozen_root, *, allow_download=False, max_mib=128):
    """Rebuild exact frozen sources/windows; network requires an explicit flag.

    No new scene selection or model inference occurs. Every output must match
    its previously frozen checksum; conflicting files are never replaced.
    """
    from .preprocessing import NativeRaster
    from affine import Affine

    root, frozen_root = Path(root), Path(frozen_root)
    if type(max_mib) is not int or not 64 <= max_mib <= 192:
        raise ValueError("bounded replay byte budget must be 64..192 MiB")
    downloader = BoundedDownload(max_mib * 1024**2)
    originals = []
    for filename in ("positive-manifest.json", "source-scene-manifest.json"):
        originals.extend(load_manifest(frozen_root / filename).samples)
    for sample in originals:
        destination = (root / sample.path).resolve()
        if not destination.is_relative_to(root.resolve()):
            raise ValueError("frozen path escapes root")
        url = sample.source.source_uri
        if not url or not url.startswith(BASE + "real/train/"):
            raise ValueError("only verified anonymous RarePlanes sources permitted")
        required = [(destination, url, 32 * 1024**2, sample.source.sha256)]
        if sample.source.sidecar_sha256:
            required.append(
                (
                    Path(str(destination) + ".aux.xml"),
                    url + ".aux.xml",
                    100000,
                    sample.source.sidecar_sha256,
                )
            )
        for path, uri, bound, digest in required:
            if not path.exists():
                if not allow_download:
                    raise ValueError(
                        "missing frozen source; explicit --allow-download required"
                    )
                downloader.fetch(uri, path, max_file=bound)
            if sha256_file(path) != digest:
                raise ValueError("frozen source checksum differs; refusing replacement")
    parents = load_manifest(frozen_root / "source-scene-manifest.json")
    parent_by_scene = {s.source.scene_id: s for s in parents.samples}
    crops = {}
    for filename in ("window-scene-manifest.json", "empty-candidates-manifest.json"):
        for sample in load_manifest(frozen_root / filename).samples:
            crops[sample.path] = sample
    for sample in crops.values():
        destination = root / sample.path
        if not destination.exists():
            parent = next(
                (
                    p
                    for name, p in parent_by_scene.items()
                    if sample.source.scene_id.startswith(name + "__")
                ),
                None,
            )
            if parent is None:
                raise ValueError("frozen native crop lacks original source")
            original = Affine(*parent.source.transform)
            cropped = Affine(*sample.source.transform)
            if any(
                abs(a - b) > 1e-12
                for a, b in zip(
                    (original.a, original.b, original.d, original.e),
                    (cropped.a, cropped.b, cropped.d, cropped.e),
                )
            ):
                raise ValueError("frozen crop changes original pixel grid")
            x, y = (~original) * (cropped.c, cropped.f)
            if abs(x - round(x)) > 1e-5 or abs(y - round(y)) > 1e-5:
                raise ValueError("frozen crop is not aligned to native source pixels")
            export_native_window(
                root / parent.path,
                parent,
                [],
                (round(x), round(y), sample.source.width, sample.source.height),
                destination,
                root=root,
            )
        with NativeRaster(destination, sample.source):
            pass
    for frozen_name, local_name in FROZEN_MANIFESTS.items():
        source = frozen_root / frozen_name
        target = root / local_name
        if target.exists():
            if load_manifest(target) != load_manifest(source):
                raise ValueError("local frozen manifest differs; refusing replacement")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(source.read_bytes())
    for name in ("reservation.json", "scene-reservation.json", "empty-reviews.json"):
        source = frozen_root / name
        if source.exists():
            target = root / name
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError("frozen reservation/review differs")
            if not target.exists():
                with target.open("xb") as stream:
                    stream.write(source.read_bytes())
    result = audit_frozen_cohorts(root, frozen_root)
    result["downloaded_bytes"] = downloader.downloaded
    result["network_used"] = downloader.downloaded > 0
    return result


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="AW01D frozen dataset replay/audit; no model selection or imagery searches."
    )
    parser.add_argument("action", choices=("audit", "replay"))
    parser.add_argument("--root", type=Path, default=Path("data/rareplanes-aw01d"))
    parser.add_argument("--frozen", type=Path, default=Path("docs/aw01d"))
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Explicitly allow only frozen anonymous source files, total capped128MiB.",
    )
    args = parser.parse_args(argv)
    if args.action == "audit" and args.allow_download:
        parser.error("audit is always offline")
    result = (
        audit_frozen_cohorts(args.root, args.frozen)
        if args.action == "audit"
        else replay_frozen_cohorts(
            args.root, args.frozen, allow_download=args.allow_download
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
