"""Bounded anonymous RarePlanes preparation with airfield/capture holdouts."""

from __future__ import annotations
import csv, hashlib, json, math, random, time, urllib.request, urllib.error, threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from .optical_contracts import DatasetManifest, Sample, RasterSource, PixelBox
from .preprocessing import sha256_file, pixel_spacing_m

BASE = "https://rareplanes-public.s3.us-west-2.amazonaws.com/"
ATTRIBUTION = "J. Shermeyer, T. Hossler, A. Van Etten, D. Hogan, R. Lewis, D. Kim. In-Q-Tel - CosmiQ Works and AI.Reverie. RarePlanes Dataset, June 2020."


class BoundedDownload:
    def __init__(self, max_bytes=128 * 1024 * 1024):
        self.remaining = max_bytes
        self.downloaded = 0
        self._budget_lock = threading.Lock()

    def fetch(self, url, path, max_file=16 * 1024 * 1024):
        path = Path(path)
        if path.is_file():
            if path.stat().st_size > max_file:
                raise ValueError("cached file exceeds bound")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                with urllib.request.urlopen(url, timeout=45) as response:
                    declared = int(response.headers.get("Content-Length", 0))
                    if declared <= 0:
                        raise ValueError(
                            "bounded downloads require a known content length"
                        )
                    with self._budget_lock:
                        if declared > min(max_file, self.remaining):
                            raise ValueError("download exceeds byte budget")
                        self.remaining -= declared
                    blocks = []
                    size = 0
                    while block := response.read(min(256 * 1024, declared - size + 1)):
                        size += len(block)
                        with self._budget_lock:
                            self.downloaded += len(block)
                        if size > declared:
                            raise ValueError("response content length exceeded")
                        blocks.append(block)
                    if size != declared:
                        raise ValueError("truncated response content length")
                # Exclusive writes; never replace original imagery.
                with path.open("xb") as f:
                    f.write(b"".join(blocks))
                return path
            except (urllib.error.URLError, TimeoutError):
                if attempt:
                    raise
                time.sleep(1)
        raise RuntimeError("download failed")


def location(filename):
    return str(int(filename.split("_")[0]))


def scene_id(filename):
    return filename.split("_tile_")[0]


def capture(filename):
    return scene_id(filename).split("_", 1)[1]


def group_locations(rows):
    parent = {r["loc_id"]: r["loc_id"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b, key=int)] = min(a, b, key=int)

    by_capture = {}
    by_name = {}
    for r in rows:
        loc = r["loc_id"]
        for index, key in (
            (by_capture, r["cat_id"]),
            (by_name, (r["Country"].casefold(), r["Air_Field"].strip().casefold())),
        ):
            if key in index:
                join(loc, index[key])
            else:
                index[key] = loc
    return {loc: "rareplanes-location-" + find(loc) for loc in parent}


def select_images(indexes, groups, assigned, tiles_per_group, seed):
    selected = {}
    for official, d in indexes.items():
        images = defaultdict(list)
        for i in d["images"]:
            group = groups[location(i["file_name"])]
            if group in assigned and (
                (official == "test") == (assigned[group] == "test")
            ):
                images[group].append(i)
        selected[official] = {
            g: sorted(
                items,
                key=lambda i: hashlib.sha256(
                    f"{seed}:{i['file_name']}".encode()
                ).hexdigest(),
            )[:tiles_per_group]
            for g, items in images.items()
        }
    return selected


def prepare(
    root: Path,
    *,
    seed=17,
    train_groups=3,
    validation_groups=2,
    test_groups=3,
    tiles_per_group=3,
):
    import rasterio

    root = Path(root)
    meta = root / "metadata"
    downloader = BoundedDownload()
    if not 1 <= tiles_per_group <= 32 or not all(
        1 <= n <= 32 for n in (train_groups, validation_groups, test_groups)
    ):
        raise ValueError("bounded preparation requires 1..32 groups/tiles")
    metadata = downloader.fetch(
        BASE + "real/metadata_annotations/RarePlanes_Public_Metadata.csv",
        meta / "RarePlanes_Public_Metadata.csv",
    )
    downloader.fetch(BASE + "LICENSE.txt", meta / "LICENSE.txt")
    rows = list(csv.DictReader(metadata.read_text(encoding="utf-8-sig").splitlines()))
    by_scene = {r["image_id"]: r for r in rows}
    groups = group_locations(rows)
    indexes = {}
    for split in ("train", "test"):
        p = downloader.fetch(
            BASE + f"real/metadata_annotations/instances_{split}_aircraft.json",
            meta / f"instances_{split}_aircraft.json",
        )
        indexes[split] = json.loads(p.read_text())
        if indexes[split]["categories"] != [{"id": 1, "name": "aircraft"}]:
            raise ValueError("unexpected aircraft category contract")
    group_sets = {
        s: {groups[location(i["file_name"])] for i in d["images"]}
        for s, d in indexes.items()
    }

    def ordered(values):
        return sorted(
            values, key=lambda v: hashlib.sha256(f"{seed}:{v}".encode()).hexdigest()
        )

    tr = ordered(group_sets["train"])
    te = ordered(group_sets["test"] - group_sets["train"])
    if len(tr) < train_groups + validation_groups or len(te) < test_groups:
        raise ValueError("not enough independent airfields for requested split")
    assigned = {g: "train" for g in tr[:train_groups]}
    assigned.update(
        {g: "validation" for g in tr[train_groups : train_groups + validation_groups]}
    )
    assigned.update({g: "test" for g in te[:test_groups]})
    selected = select_images(indexes, groups, assigned, tiles_per_group, seed)
    samples = materialize_samples(
        root, metadata, rows, indexes, groups, assigned, selected, downloader
    )
    manifest = DatasetManifest(
        samples=samples,
        preparation={
            "dataset": "RarePlanes real PS-RGB tiles",
            "seed": seed,
            "tiles_per_group": tiles_per_group,
            "group_rule": "location IDs union shared capture IDs and same country/airfield name",
            "official_train_groups": len(group_sets["train"]),
            "official_test_only_groups": len(group_sets["test"] - group_sets["train"]),
            "official_group_overlap": len(group_sets["train"] & group_sets["test"]),
            "downloaded_bytes": downloader.downloaded,
            "license": "CC-BY-SA-4.0",
            "annotation_sha256": {
                s: sha256_file(meta / f"instances_{s}_aircraft.json") for s in indexes
            },
            "limitations": [
                "Positive-only tiles; no empty-apron false-alarm-rate estimate.",
                "Counts are per tile, not whole-airfield inventory.",
                "Location/lineage grouping enforced; no comprehensive geographic-distance audit.",
                "No truncation visibility fraction in instances_aircraft; edge annotations retained.",
            ],
        },
    )
    return save_frozen_manifest(root / "manifest.json", manifest)


def save_frozen_manifest(path, manifest):
    """Reuse a frozen manifest without writing, even when only telemetry changes."""
    path = Path(path)
    if path.exists():
        old = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if [s.model_dump() for s in old.samples] != [
            s.model_dump() for s in manifest.samples
        ]:
            raise ValueError("refusing to replace an existing frozen dataset manifest")
        return old
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def load_manifest(path: Path):
    return DatasetManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def resolve_image(manifest_path: Path, sample):
    root = Path(manifest_path).resolve().parent
    path = (root / sample.path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("dataset image path escapes manifest root")
    return path


def materialize_samples(
    root, metadata, rows, indexes, groups, assigned, selected, downloader
):
    import rasterio

    by_scene = {r["image_id"]: r for r in rows}
    samples = []
    for official, d in indexes.items():
        annotations = defaultdict(list)
        for a in d["annotations"]:
            annotations[a["image_id"]].append(a)
        for group, chosen in sorted(selected[official].items()):
            for i in chosen:
                name = i["file_name"]
                if Path(name).name != name:
                    raise ValueError("unsafe remote image filename")
                uri = BASE + f"real/{official}/PS-RGB_tiled/{name}"
                path = downloader.fetch(
                    uri, root / "images" / name, max_file=4 * 1024 * 1024
                )
                aux = downloader.fetch(
                    uri + ".aux.xml", Path(str(path) + ".aux.xml"), max_file=100000
                )
                row = by_scene[scene_id(name)]
                with rasterio.open(path) as ds:
                    if ds.crs is None:
                        raise ValueError("verified source georeferencing required")
                    if (ds.width, ds.height) != (i["width"], i["height"]):
                        raise ValueError("COCO dimensions mismatch")
                    spacing_xy = pixel_spacing_m(
                        ds.crs, ds.transform, ds.width, ds.height
                    )
                    spacing = max(spacing_xy)
                    transform = tuple(ds.transform)[:6]
                    crs = ds.crs.to_string()
                time_raw = row["collection_time_start"]
                stamp = datetime.strptime(time_raw, "%m/%d/%y %H:%M").replace(
                    tzinfo=timezone.utc
                )
                pan = float(row["pan_resolution_maximum"])
                source = RasterSource(
                    scene_id=scene_id(name),
                    sensor_id=row["sensor"],
                    acquisition_timestamp=stamp,
                    timestamp_precision="minute",
                    acquisition_time_raw=time_raw,
                    timestamp_timezone_basis="assumed_utc",
                    native_gsd_m=pan,
                    panchromatic_gsd_m=pan,
                    pan_sharpened_gsd_m=spacing,
                    pixel_spacing_m=spacing_xy,
                    quality="unknown",
                    airfield_group=group,
                    capture_group=row["cat_id"],
                    source_uri=uri,
                    license="CC-BY-SA-4.0",
                    attribution=ATTRIBUTION,
                    sha256=sha256_file(path),
                    sidecar_sha256=sha256_file(aux),
                    metadata_sha256=sha256_file(metadata),
                    width=i["width"],
                    height=i["height"],
                    crs=crs,
                    transform=transform,
                    provenance_notes=[
                        "CSV collection times preserved to published minute precision; dataset acquisition times treated as UTC.",
                        "Pan-sharpened RGB spacing and source panchromatic resolution are separate; effective optical resolution unknown.",
                        "Quality not independently assessed; local cloud/occlusion unknown.",
                        f"Country={row['Country']}; official_split={official}; airfield={row['Air_Field']}",
                    ],
                )
                truth = []
                ignored = []
                for a in annotations[i["id"]]:
                    if a["category_id"] != 1:
                        raise ValueError("unexpected category")
                    x, y, w, h = a["bbox"]
                    box = PixelBox(
                        xyxy=(
                            max(0, x),
                            max(0, y),
                            min(i["width"], x + w),
                            min(i["height"], y + h),
                        )
                    )
                    (ignored if a.get("iscrowd") else truth).append(box)
                samples.append(
                    Sample(
                        image_id=name,
                        path="images/" + name,
                        split=assigned[group],
                        source=source,
                        boxes=truth,
                        ignored_boxes=ignored,
                    )
                )
    return samples
