"""AW-01B bounded, geographically grouped RarePlanes experiment preparation."""

from __future__ import annotations
import csv, hashlib, json, math, shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .dataset import (
    BASE,
    BoundedDownload,
    group_locations,
    location,
    scene_id,
    materialize_samples,
    load_manifest,
    select_images,
)
from .optical_contracts import DatasetManifest
from .preprocessing import sha256_file
from .geospatial_export import pixel_polygon


def haversine_km(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    v = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(min(1, math.sqrt(v)))


def geographic_groups(rows, geojson, distance_km=10):
    crs = geojson.get("crs", {}).get("properties", {}).get("name", "")
    if crs not in ("urn:ogc:def:crs:OGC:1.3:CRS84", "urn:ogc:def:crs:EPSG::4326"):
        raise ValueError("verified geographic annotation CRS required")
    groups = group_locations(rows)
    points = defaultdict(list)
    for f in geojson["features"]:
        loc = str(int(f["properties"]["loc_id"]))
        if loc not in groups or f["geometry"]["type"] != "Polygon":
            raise ValueError("unsupported geographic annotation contract")
        ring = f["geometry"]["coordinates"][0]
        if not ring:
            raise ValueError("empty geographic annotation")
        x, y = ring[0]
        if not -180 <= x <= 180 or not -90 <= y <= 90:
            raise ValueError("invalid geographic coordinate")
        points[loc].append((x, y))
    if set(groups) - set(points):
        raise ValueError("airfield location lacks geographic evidence")
    # Spherical means preserve longitudes near the date line.
    centres = {}
    for loc, pts in points.items():
        xyz = [
            (
                math.cos(math.radians(y)) * math.cos(math.radians(x)),
                math.cos(math.radians(y)) * math.sin(math.radians(x)),
                math.sin(math.radians(y)),
            )
            for x, y in pts
        ]
        x, y, z = [sum(p[i] for p in xyz) / len(xyz) for i in range(3)]
        centres[loc] = (
            math.degrees(math.atan2(y, x)),
            math.degrees(math.atan2(z, math.hypot(x, y))),
        )
    parent = {g: g for g in groups.values()}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    locs = sorted(centres, key=int)
    edges = []
    for i, a in enumerate(locs):
        for b in locs[i + 1 :]:
            distance = haversine_km(centres[a], centres[b])
            if distance < distance_km:
                aa, bb = find(groups[a]), find(groups[b])
                if aa != bb:
                    parent[max(aa, bb)] = min(aa, bb)
                edges.append(
                    {"location_a": a, "location_b": b, "distance_km": distance}
                )
    return {loc: find(g) for loc, g in groups.items()}, centres, edges


def choose_assignments(
    indexes,
    groups,
    rows,
    original,
    *,
    seed=23,
    train_groups=16,
    validation_groups=4,
    test_groups=4,
):
    def ordered(vals):
        return sorted(
            vals, key=lambda v: hashlib.sha256(f"{seed}:{v}".encode()).hexdigest()
        )

    sets = {
        s: {groups[location(i["file_name"])] for i in d["images"]}
        for s, d in indexes.items()
    }
    old = {
        split: {
            groups[location(s.image_id)] for s in original.samples if s.split == split
        }
        for split in ("train", "validation", "test")
    }
    # Never move a previously exposed validation/test airfield into training.
    val = list(ordered(old["validation"] - old["train"] - old["test"]))
    tr = list(ordered(old["train"] - old["validation"] - old["test"]))
    if len(tr) > train_groups or len(val) > validation_groups:
        raise ValueError("requested split is smaller than preserved original groups")
    available = sets["train"] - set(tr) - set(val) - old["test"]
    val += ordered(available)[: validation_groups - len(val)]
    available -= set(val)
    countries = defaultdict(set)
    for r in rows:
        countries[groups[r["loc_id"]]].add(r["Country"])
    represented = {country for g in tr for country in countries[g]}
    # Greedy country diversity, seeded tie breaks; no independent test labels inspected.
    while len(tr) < train_groups and available:
        choices = ordered(available)
        choice = max(choices, key=lambda g: len(countries[g] - represented))
        tr.append(choice)
        represented |= countries[choice]
        available.remove(choice)
    te = ordered(sets["test"] - sets["train"] - old["test"] - set(val) - set(tr))[
        :test_groups
    ]
    if (len(tr), len(val), len(te)) != (train_groups, validation_groups, test_groups):
        raise ValueError("not enough independent geographic groups")
    assigned = {
        g: s
        for s, values in (("train", tr), ("validation", val), ("test", te))
        for g in values
    }
    return assigned, {
        "old_groups": {k: sorted(v) for k, v in old.items()},
        "official_train_groups": len(sets["train"]),
        "official_test_only_groups": len(sets["test"] - sets["train"]),
    }


def audit_spatial_separation(samples, minimum_km=1.0):
    footprints = []
    for sample in samples:
        p = pixel_polygon(
            sample.source, (0, 0, sample.source.width, sample.source.height)
        )
        if p is None:
            raise ValueError("georeferencing required for leakage audit")
        xy = p.coordinates[0]
        xs, ys = zip(*xy)
        if max(xs) - min(xs) > 180:
            raise ValueError("date-line footprint needs explicit handling")
        footprints.append((sample, (min(xs), min(ys), max(xs), max(ys))))
    closest = None
    for i, (a, aa) in enumerate(footprints):
        for b, bb in footprints[i + 1 :]:
            if a.split == b.split:
                continue
            if max(aa[0], bb[0]) < min(aa[2], bb[2]) and max(aa[1], bb[1]) < min(
                aa[3], bb[3]
            ):
                raise ValueError("cross-split image footprints overlap")
            # Minimum distance between axis-aligned envelopes, conservative for these small source grids.
            ax = min(max((bb[0] + bb[2]) / 2, aa[0]), aa[2])
            ay = min(max((bb[1] + bb[3]) / 2, aa[1]), aa[3])
            bx = min(max(ax, bb[0]), bb[2])
            by = min(max(ay, bb[1]), bb[3])
            distance = haversine_km((ax, ay), (bx, by))
            closest = distance if closest is None else min(closest, distance)
            if distance < minimum_km:
                raise ValueError("cross-split image footprints too close")
    return {"minimum_cross_split_envelope_distance_km": closest, "overlap": False}


def prepare_expanded(
    original_path: Path,
    root: Path,
    *,
    seed=23,
    train_groups=16,
    validation_groups=4,
    test_groups=4,
    train_tiles=12,
    validation_tiles=8,
    test_tiles=8,
    workers=4,
    max_mib=128,
):
    import threading

    if not all(
        1 <= v <= 32
        for v in (
            train_groups,
            validation_groups,
            test_groups,
            train_tiles,
            validation_tiles,
            test_tiles,
        )
    ):
        raise ValueError("groups and tile bounds must be 1..32")
    if not 1 <= workers <= 4 or not 16 <= max_mib <= 256:
        raise ValueError("bounded workers/byte budget required")
    root = Path(root)
    request = {
        "seed": seed,
        "train_groups": train_groups,
        "validation_groups": validation_groups,
        "test_groups": test_groups,
        "train_tiles": train_tiles,
        "validation_tiles": validation_tiles,
        "test_tiles": test_tiles,
        "workers": workers,
        "max_mib": max_mib,
        "original_manifest_sha256": sha256_file(original_path),
    }
    request_path = root / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("frozen dataset request configuration differs")
    if (root / "manifest.json").exists():
        if not request_path.exists():
            raise ValueError("frozen dataset lacks its request configuration")
        cached = load_manifest(root / "manifest.json")
        for name, checksum in cached.preparation["metadata_sha256"].items():
            if sha256_file(root / "metadata" / name) != checksum:
                raise ValueError("cached source metadata checksum mismatch")
        return cached
    root.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

    original = load_manifest(original_path)
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    downloader = BoundedDownload(max_mib * 1024**2)
    names = [
        "LICENSE.txt",
        "RarePlanes_Public_Metadata.csv",
        "instances_train_aircraft.json",
        "instances_test_aircraft.json",
        "RarePlanes_Public_All_Annotations.geojson",
    ]
    for name in names:
        old = original_path.parent / "metadata" / name
        p = meta / name
        if old.is_file() and not p.exists():
            shutil.copyfile(old, p)
        if not p.exists():
            key = name if name == "LICENSE.txt" else "real/metadata_annotations/" + name
            downloader.fetch(BASE + key, p)
    rows = list(
        csv.DictReader(
            (meta / "RarePlanes_Public_Metadata.csv")
            .read_text(encoding="utf-8-sig")
            .splitlines()
        )
    )
    indexes = {
        s: json.loads((meta / f"instances_{s}_aircraft.json").read_text())
        for s in ("train", "test")
    }
    if any(
        d["categories"] != [{"id": 1, "name": "aircraft"}] for d in indexes.values()
    ):
        raise ValueError("unexpected annotation categories")
    groups, centres, edges = geographic_groups(
        rows,
        json.loads((meta / "RarePlanes_Public_All_Annotations.geojson").read_text()),
    )
    assigned, partition = choose_assignments(
        indexes,
        groups,
        rows,
        original,
        seed=seed,
        train_groups=train_groups,
        validation_groups=validation_groups,
        test_groups=test_groups,
    )
    selected = {s: defaultdict(list) for s in indexes}
    # Capture round robin for training; seed-based sampling for all partitions, independent of holdout labels.
    for official, d in indexes.items():
        candidates = defaultdict(list)
        for image in d["images"]:
            group = groups[location(image["file_name"])]
            if group in assigned and (
                (official == "test") == (assigned[group] == "test")
            ):
                candidates[group].append(image)
        for g, images in candidates.items():
            ordered = sorted(
                images,
                key=lambda i: hashlib.sha256(
                    f"{seed}:{i['file_name']}".encode()
                ).hexdigest(),
            )
            if assigned[g] == "train":
                captures = defaultdict(list)
                for image in ordered:
                    captures[scene_id(image["file_name"])].append(image)
                ordered = []
                while any(captures.values()):
                    for capture in sorted(captures):
                        if captures[capture]:
                            ordered.append(captures[capture].pop(0))
            n = {
                "train": train_tiles,
                "validation": validation_tiles,
                "test": test_tiles,
            }[assigned[g]]
            selected[official][g] = ordered[:n]
    protocol = {
        "seed": seed,
        "assigned_groups": assigned,
        "old_groups": partition["old_groups"],
        "selected_ids": {
            s: [i["file_name"] for items in d.values() for i in items]
            for s, d in selected.items()
        },
        "training_candidates": [
            {
                "name": "expanded_frozen_sgd",
                "epochs": 6,
                "lr": 0.005,
                "optimizer": "sgd",
                "backbone": "frozen",
            },
            {
                "name": "expanded_fpn_adamw",
                "epochs": 6,
                "lr": 0.0003,
                "optimizer": "adamw",
                "backbone": "fpn",
            },
        ],
        "augmentation": "native dihedral flips/90-degree rotations; no resize",
        "selection": "validation F1, then lower count MAE, then higher threshold; no test tuning",
        "threshold_grid": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "candidate_floor": 0.005,
        "nms_iou": 0.4,
        "matching_iou": 0.5,
        "network_limit_mib": max_mib,
        "workers": workers,
    }
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("frozen experiment protocol differs")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    requests = []
    for official, selected_groups in selected.items():
        for images in selected_groups.values():
            for image in images:
                name = image["file_name"]
                if Path(name).name != name:
                    raise ValueError("unsafe source filename")
                uri = BASE + f"real/{official}/PS-RGB_tiled/{name}"
                for suffix, bound in (("", 4 * 1024**2), (".aux.xml", 100000)):
                    destination = root / "images" / (name + suffix)
                    old = original_path.parent / "images" / (name + suffix)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if old.exists() and not destination.exists():
                        shutil.copyfile(old, destination)
                    requests.append((uri + suffix, destination, bound))
    print(
        json.dumps(
            {
                "stage": "download",
                "files": len(requests),
                "maximum_mib": max_mib,
                "workers": workers,
            }
        ),
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(downloader.fetch, url, path, max_file=bound)
            for url, path, bound in requests
        ]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 40 == 0:
                print(
                    json.dumps(
                        {
                            "download_files_completed": index,
                            "total_files": len(requests),
                            "downloaded_bytes": downloader.downloaded,
                        }
                    ),
                    flush=True,
                )
    samples = materialize_samples(
        root,
        meta / "RarePlanes_Public_Metadata.csv",
        rows,
        indexes,
        groups,
        assigned,
        selected,
        downloader,
    )
    spatial = audit_spatial_separation(samples)
    weather = {r["image_id"]: r["Weather"] for r in rows}
    manifest = DatasetManifest(
        samples=samples,
        preparation={
            "dataset": "RarePlanes expanded native PS-RGB",
            "seed": seed,
            "license": "CC-BY-SA-4.0",
            "group_rule": "airfield IDs + shared capture/name + geographic centres <10 km; cross-split footprint envelopes >=1 km",
            "geographic_neighbour_edges": edges,
            "geographic_centres": centres,
            "spatial_audit": spatial,
            "original_manifest_sha256": sha256_file(original_path),
            "protocol_sha256": sha256_file(protocol_path),
            "metadata_weather_by_scene": {
                s.source.scene_id: weather[s.source.scene_id] for s in samples
            },
            "metadata_sha256": {name: sha256_file(meta / name) for name in names},
            "partition": partition,
            "downloaded_bytes": downloader.downloaded,
            "limitations": [
                "Positive-only selected tiles; no empty-airfield false-alarm estimate.",
                "Dataset weather is coarse scene metadata, not independently assessed AOI quality.",
                "Geographic annotation centres and selected-image envelopes do not establish full-scene georegistration accuracy.",
                "CSV acquisition-time UTC basis assumed; minute precision retained.",
            ],
        },
    )
    (root / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest


def broaden_validation(
    manifest_path: Path, *, extra_groups=2, tiles_per_group=8, workers=4
):
    """Add unused validation groups; original training and test samples stay immutable."""
    import statistics

    if extra_groups != 2 or not 1 <= tiles_per_group <= 16 or not 1 <= workers <= 4:
        raise ValueError(
            "bounded validation revision requires two groups and <=16 tiles/group"
        )
    root = manifest_path.parent
    output = root / "manifest-balanced.json"
    if output.exists():
        protocol = json.loads((root / "protocol-balanced.json").read_text())
        if protocol["validation_tiles_per_added_group"] != tiles_per_group:
            raise ValueError("frozen validation configuration differs")
        return load_manifest(output)
    manifest = load_manifest(manifest_path)
    protocol = json.loads((root / "protocol.json").read_text())
    rows = list(
        csv.DictReader(
            (root / "metadata/RarePlanes_Public_Metadata.csv")
            .read_text(encoding="utf-8-sig")
            .splitlines()
        )
    )
    indexes = {
        s: json.loads((root / f"metadata/instances_{s}_aircraft.json").read_text())
        for s in ("train", "test")
    }
    groups, _, _ = geographic_groups(
        rows,
        json.loads(
            (root / "metadata/RarePlanes_Public_All_Annotations.geojson").read_text()
        ),
    )
    assigned = dict(protocol["assigned_groups"])
    images = {i["id"]: i for i in indexes["train"]["images"]}
    eligible = (
        {groups[location(i["file_name"])] for i in images.values()}
        - set(assigned)
        - set(protocol["old_groups"]["test"])
    )
    sizes = defaultdict(list)
    for a in indexes["train"]["annotations"]:
        group = groups[location(images[a["image_id"]]["file_name"])]
        if group in eligible:
            sizes[group].append(math.sqrt(a["bbox"][2] * a["bbox"][3]))
    weather_groups = {
        groups[r["loc_id"]] for r in rows if r["Weather"] == "Cloud Cover or Haze"
    }
    chosen = [max(eligible, key=lambda g: (statistics.median(sizes[g]), g))]
    cloudy = (eligible - set(chosen)) & weather_groups
    if not cloudy:
        raise ValueError("no unused cloudy validation group")
    chosen.append(max(cloudy, key=lambda g: (statistics.median(sizes[g]), g)))
    for g in chosen:
        assigned[g] = "validation"
    selected = select_images(
        indexes,
        groups,
        {g: "validation" for g in chosen},
        tiles_per_group,
        protocol["seed"],
    )
    revised = dict(
        protocol,
        assigned_groups=assigned,
        revision="broaden validation before test evaluation",
        added_validation_groups=chosen,
        added_group_selection="largest unused training-group median size; largest unused cloud/haze group",
        validation_tiles_per_added_group=tiles_per_group,
        inherited_training_and_test_samples_unchanged=True,
    )
    (root / "protocol-balanced.json").write_text(
        json.dumps(revised, indent=2), encoding="utf-8"
    )
    downloader = BoundedDownload(16 * 1024**2)
    requests = []
    for official, by_group in selected.items():
        for values in by_group.values():
            for image in values:
                name = image["file_name"]
                uri = BASE + f"real/{official}/PS-RGB_tiled/{name}"
                for suffix, bound in (("", 4 * 1024**2), (".aux.xml", 100000)):
                    requests.append(
                        (uri + suffix, root / "images" / (name + suffix), bound)
                    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed(
            [pool.submit(downloader.fetch, u, p, max_file=b) for u, p, b in requests]
        ):
            future.result()
    added = materialize_samples(
        root,
        root / "metadata/RarePlanes_Public_Metadata.csv",
        rows,
        indexes,
        groups,
        assigned,
        selected,
        downloader,
    )
    all_samples = manifest.samples + added
    weather = {r["image_id"]: r["Weather"] for r in rows}
    preparation = dict(
        manifest.preparation,
        initial_manifest_sha256=sha256_file(manifest_path),
        protocol_sha256=sha256_file(root / "protocol-balanced.json"),
        spatial_audit=audit_spatial_separation(all_samples),
        validation_revision={
            "added_groups": chosen,
            "added_tiles": len(added),
            "downloaded_bytes": downloader.downloaded,
            "reason": "initial validation lacked larger aircraft; test data unused",
        },
        metadata_weather_by_scene={
            s.source.scene_id: weather[s.source.scene_id] for s in all_samples
        },
    )
    revised_manifest = DatasetManifest(samples=all_samples, preparation=preparation)
    for split in ("train", "test"):
        assert [s.model_dump() for s in manifest.samples if s.split == split] == [
            s.model_dump() for s in revised_manifest.samples if s.split == split
        ]
    output.write_text(revised_manifest.model_dump_json(indent=2), encoding="utf-8")
    return revised_manifest
