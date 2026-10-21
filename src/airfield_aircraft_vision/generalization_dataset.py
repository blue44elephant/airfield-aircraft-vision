"""AW-01C isolated expansion; test membership is fixed before experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .dataset import (
    BASE,
    BoundedDownload,
    load_manifest,
    location,
    scene_id,
    materialize_samples,
    save_frozen_manifest,
)
from .expanded_dataset import geographic_groups, audit_spatial_separation
from .optical_contracts import DatasetManifest
from .preprocessing import sha256_file

RICH_NAME = "RarePlanes_Train_Coco_Annotations_tiled.json"
RICH_SHA256 = "d51f4fc97422934c8f3846207d6ec453dfbb69ae0432537731104a8f0e4c1540"


def validate_rich_contract(index, rich):
    """Rich categories differ; join aircraft by exact IDs and boxes."""

    def images(data):
        values = {i["id"]: i["file_name"] for i in data["images"]}
        if len(values) != len(data["images"]):
            raise ValueError("duplicate rich/index image IDs")
        return values

    def annotations(data):
        values = {
            a["id"]: (a["image_id"], tuple(a["bbox"])) for a in data["annotations"]
        }
        if len(values) != len(data["annotations"]):
            raise ValueError("duplicate rich/index annotation IDs")
        return values

    if images(index) != images(rich) or annotations(index) != annotations(rich):
        raise ValueError("rich and simplified annotation contracts differ")
    for ann in rich["annotations"]:
        visible = ann.get("partialDec")
        if (
            isinstance(visible, bool)
            or not isinstance(visible, (int, float))
            or not math.isfinite(visible)
            or not 0 <= visible <= 1
        ):
            raise ValueError("invalid rich visibility fraction")


def rank_training_tiles(images, annotations, *, seed=41):
    """Alternate small/crowded/random training examples, never select holdout by labels."""
    by_id = defaultdict(list)
    for ann in annotations:
        by_id[ann["image_id"]].append(ann["bbox"])

    def hashed(i):
        return hashlib.sha256(f"{seed}:{i['file_name']}".encode()).hexdigest()

    small = sorted(
        images,
        key=lambda i: (
            min((b[2] * b[3] for b in by_id[i["id"]]), default=float("inf")),
            hashed(i),
        ),
    )
    crowded = sorted(images, key=lambda i: (-len(by_id[i["id"]]), hashed(i)))
    random = sorted(images, key=hashed)
    selected, seen = [], set()
    for values in zip(small, crowded, random):
        for image in values:
            if image["id"] not in seen:
                selected.append(image)
                seen.add(image["id"])
    return selected


def choose_training_groups(available, preserved, rows, groups, *, total=32, seed=41):
    if not len(preserved) <= total <= 48:
        raise ValueError("bounded group expansion must preserve original groups")
    countries = defaultdict(set)
    for row in rows:
        countries[groups[row["loc_id"]]].add(row["Country"])
    chosen = set(preserved)
    represented = set().union(*(countries[g] for g in chosen))
    remaining = set(available) - chosen
    while len(chosen) < total and remaining:
        ordered = sorted(
            remaining, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()
        )
        group = max(ordered, key=lambda g: len(countries[g] - represented))
        chosen.add(group)
        represented |= countries[group]
        remaining.remove(group)
    if len(chosen) != total:
        raise ValueError("not enough unused training groups")
    return chosen


def prepare_generalization(
    baseline_path: Path,
    legacy_path: Path,
    root: Path,
    *,
    total_train_groups=32,
    train_tiles=16,
    seed=41,
    max_mib=192,
    workers=4,
):
    """Read an immutable test reservation; cache originals separately; never overwrite."""
    root, baseline_path, legacy_path = map(Path, (root, baseline_path, legacy_path))
    if not 1 <= workers <= 4 or not 16 <= train_tiles <= 32 or not 32 <= max_mib <= 256:
        raise ValueError("bounded preparation required")
    reservation_path = root / "test-reservation.json"
    reservation = json.loads(reservation_path.read_text())
    if reservation["aw01b_manifest_sha256"] != sha256_file(baseline_path):
        raise ValueError("reserved baseline checksum differs")
    if reservation["legacy_manifest_sha256"] != sha256_file(legacy_path):
        raise ValueError("reserved legacy checksum differs")
    request = dict(
        baseline_sha256=sha256_file(baseline_path),
        reservation_sha256=sha256_file(reservation_path),
        total_train_groups=total_train_groups,
        train_tiles=train_tiles,
        seed=seed,
        max_mib=max_mib,
        workers=workers,
    )
    config = root / "request.json"
    if config.exists() and json.loads(config.read_text()) != request:
        raise ValueError("frozen expansion configuration differs")
    if (root / "manifest.json").exists():
        if not config.is_file() or not (root / "protocol.json").is_file():
            raise ValueError("cached expansion lacks frozen request or protocol")
        data = load_manifest(root / "manifest.json")
        if data.preparation["protocol_sha256"] != sha256_file(root / "protocol.json"):
            raise ValueError("cached protocol checksum differs")
        for filename, digest in data.preparation["metadata_sha256"].items():
            if sha256_file(root / "metadata" / filename) != digest:
                raise ValueError("frozen metadata changed")
        return data
    config.write_text(json.dumps(request, indent=2), encoding="utf-8")
    baseline = load_manifest(baseline_path)
    legacy = load_manifest(legacy_path)
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    for filename in baseline.preparation["metadata_sha256"]:
        old = baseline_path.parent / "metadata" / filename
        if old.is_file() and not (meta / old.name).exists():
            shutil.copyfile(old, meta / old.name)
    for filename, digest in baseline.preparation["metadata_sha256"].items():
        if sha256_file(meta / filename) != digest:
            raise ValueError("copied baseline metadata checksum differs")
    rich_path = meta / RICH_NAME
    if not rich_path.is_file():
        BoundedDownload(40 * 1024**2).fetch(
            BASE + "real/metadata_annotations/" + RICH_NAME,
            rich_path,
            max_file=40 * 1024**2,
        )
    if sha256_file(rich_path) != RICH_SHA256:
        raise ValueError("verified rich annotation artifact checksum differs")
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
    validate_rich_contract(indexes["train"], json.loads(rich_path.read_text()))
    groups, centres, edges = geographic_groups(
        rows,
        json.loads((meta / "RarePlanes_Public_All_Annotations.geojson").read_text()),
    )
    train_groups = {
        s.source.airfield_group for s in baseline.samples if s.split == "train"
    }
    val_groups = {
        s.source.airfield_group for s in baseline.samples if s.split == "validation"
    }
    exposed = {
        s.source.airfield_group
        for s in baseline.samples + legacy.samples
        if s.split == "test"
    }
    test_groups = set(reservation["test_groups"])
    official_train = {
        groups[location(i["file_name"])] for i in indexes["train"]["images"]
    }
    if test_groups & (official_train | val_groups | train_groups | exposed):
        raise ValueError("reserved independent test leakage")
    chosen = choose_training_groups(
        official_train - val_groups - exposed - test_groups,
        train_groups,
        rows,
        groups,
        total=total_train_groups,
        seed=seed,
    )
    assigned = {
        **{g: "train" for g in chosen},
        **{g: "validation" for g in val_groups},
        **{g: "test" for g in test_groups},
    }
    old_ids = {s.image_id for s in baseline.samples if s.split != "test"}
    reserved_ids = set(reservation["test_image_ids"])
    selected = {s: defaultdict(list) for s in indexes}
    for official, data in indexes.items():
        candidates = defaultdict(list)
        for image in data["images"]:
            group = groups[location(image["file_name"])]
            if group in assigned and (
                (official == "test") == (assigned[group] == "test")
            ):
                candidates[group].append(image)
        for group, images in candidates.items():
            split = assigned[group]
            if split == "test":
                keep = [i for i in images if i["file_name"] in reserved_ids]
            elif split == "validation":
                keep = [i for i in images if i["file_name"] in old_ids]
            else:
                keep = [i for i in images if i["file_name"] in old_ids]
                ranked = rank_training_tiles(images, data["annotations"], seed=seed)
                captures = defaultdict(list)
                for image in ranked:
                    captures[scene_id(image["file_name"])].append(image)
                ordered = []
                while any(captures.values()):
                    for capture in sorted(captures):
                        if captures[capture]:
                            ordered.append(captures[capture].pop(0))
                seen = {i["id"] for i in keep}
                for image in ordered:
                    if len(keep) >= train_tiles:
                        break
                    if image["id"] not in seen:
                        keep.append(image)
                        seen.add(image["id"])
            selected[official][group] = keep
    protocol = dict(
        request,
        assigned_groups=assigned,
        selection="Training only: alternate small/crowded/seed order with capture round robin; preserve all AW01B train and validation tiles.",
        selected_ids={
            s: [i["file_name"] for v in d.values() for i in v]
            for s, d in selected.items()
        },
        independent_test_policy=reservation["independent_evaluation"],
    )
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("frozen dataset protocol differs")
    if not protocol_path.exists():
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    downloader = BoundedDownload(max_mib * 1024**2)
    requests = []
    for official, items in selected.items():
        for images in items.values():
            for image in images:
                name = image["file_name"]
                if Path(name).name != name:
                    raise ValueError("unsafe image name")
                for suffix, bound in (("", 4 * 1024**2), (".aux.xml", 100000)):
                    destination = root / "images" / (name + suffix)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    for previous in (baseline_path.parent, legacy_path.parent):
                        cached = previous / "images" / (name + suffix)
                        if cached.exists() and not destination.exists():
                            shutil.copyfile(cached, destination)
                    requests.append(
                        (
                            BASE + f"real/{official}/PS-RGB_tiled/{name}" + suffix,
                            destination,
                            bound,
                        )
                    )

    def fetch(request):
        url, destination, bound = request
        return downloader.fetch(url, destination, max_file=bound)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(fetch, requests), 1):
            if index % 80 == 0:
                print(
                    json.dumps(
                        dict(
                            download_files=index,
                            total=len(requests),
                            downloaded_bytes=downloader.downloaded,
                        )
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
    if {s.image_id for s in samples if s.split == "test"} != reserved_ids:
        raise ValueError("independent membership differs from reservation")
    # Validation remains byte-equivalent in analytical source and labels.
    old_validation = {s.image_id: s for s in baseline.samples if s.split != "test"}
    for s in samples:
        if s.image_id in old_validation and s != old_validation[s.image_id]:
            raise ValueError("preserved training/validation record changed")
    spatial = audit_spatial_separation(samples)
    weather = {r["image_id"]: r["Weather"] for r in rows}
    geography = {r["image_id"]: r["Country"] for r in rows}
    manifest = DatasetManifest(
        samples=samples,
        preparation=dict(
            dataset="RarePlanes AW01C native PS-RGB",
            license="CC-BY-SA-4.0",
            original_manifest_sha256=sha256_file(baseline_path),
            independent_reservation_sha256=sha256_file(reservation_path),
            protocol_sha256=sha256_file(protocol_path),
            spatial_audit=spatial,
            metadata_weather_by_scene={
                s.source.scene_id: weather[s.source.scene_id] for s in samples
            },
            geography_by_scene={
                s.source.scene_id: geography[s.source.scene_id] for s in samples
            },
            metadata_sha256={
                f.name: sha256_file(f) for f in meta.iterdir() if f.is_file()
            },
            downloaded_bytes=downloader.downloaded,
            limitations=[
                "Native provided tiles contain aircraft; reviewed negative crops are separate derivatives.",
                "Dataset minute timestamps retain assumed UTC basis.",
                "Small bbox area can represent cropped aircraft fragments, not physical aircraft size.",
            ],
        ),
    )
    save_frozen_manifest(root / "manifest.json", manifest)
    return manifest
