"""AW-01D bounded, native-pixel NAIP evaluation windows; never model selection.

NAIP is aerial imagery, independently sourced from the RarePlanes satellite data.
Anonymous short-lived signing tokens are held only in memory and never reported.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .optical_contracts import RasterSource
from .preprocessing import sha256_file

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/naipeuwest/naip"
LICENSE_URL = (
    "https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip-imagery"
)
LOCATIONS = (
    dict(
        name="canyonlands-moab",
        lon=-109.751,
        lat=38.755,
        item="ut_m_3810910_se_12_060_20210917",
    ),
    dict(
        name="cody", lon=-109.0245, lat=44.5194, item="wy_m_4410932_se_12_060_20220706"
    ),
)


def frozen_json(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"preserved file differs: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def json_request(url, *, cap=512_000, opener=urllib.request.urlopen):
    try:
        with opener(url, timeout=25) as response:
            raw = response.read(cap + 1)
            if len(raw) > cap:
                raise ValueError("bounded metadata response exceeded")
            return json.loads(raw)
    except urllib.error.URLError as exc:
        # HTTPError text may contain a signed URL. Never include it.
        code = getattr(exc, "code", None)
        raise RuntimeError(f"anonymous metadata access failed (HTTP {code})") from None


@dataclass
class RangeBudget:
    max_bytes: int = 16 * 1024**2
    max_requests: int = 64
    used_bytes: int = 0
    requests: int = 0
    ranges: list = field(default_factory=list)

    def reserve(self, size):
        if size <= 0 or self.used_bytes + size > self.max_bytes:
            raise ValueError("bounded HTTP Range byte budget exceeded")
        if self.requests >= self.max_requests:
            raise ValueError("bounded HTTP Range request budget exceeded")
        self.used_bytes += size
        self.requests += 1


class RangeReader(io.RawIOBase):
    """Read-only HTTP range file with an aggregate hard transfer reservation.

    Every request requires exact 206/Content-Range/Content-Length and a stable ETag.
    A range response exceeding the budget is refused before reading its body.
    """

    def __init__(
        self,
        signed_url,
        unsigned_url,
        size,
        etag,
        budget,
        *,
        opener=urllib.request.urlopen,
        block_size=262144,
    ):
        super().__init__()
        if not size > 0 or not etag or block_size < 1:
            raise ValueError("positive remote size, ETag and block size required")
        if urllib.parse.urlsplit(unsigned_url).query:
            raise ValueError("unsigned provenance URL must not contain credentials")
        self._url = signed_url
        self.unsigned_url = unsigned_url
        self.size, self.etag = size, etag
        self.budget, self.opener = budget, opener
        self.block_size, self.position, self.cache = block_size, 0, {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            raise ValueError("invalid seek origin")
        position = offset + (
            self.position
            if whence == io.SEEK_CUR
            else self.size if whence == io.SEEK_END else 0
        )
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def _block(self, offset):
        start = offset // self.block_size * self.block_size
        if start in self.cache:
            return self.cache[start]
        end = min(self.size, start + self.block_size) - 1
        expected = end - start + 1
        self.budget.reserve(expected)
        request = urllib.request.Request(
            self._url,
            headers={"Range": f"bytes={start}-{end}", "If-Match": self.etag},
        )
        try:
            with self.opener(request, timeout=25) as response:
                if response.status != 206:
                    raise ValueError("server did not honor bounded HTTP Range")
                if (
                    response.headers.get("Content-Range")
                    != f"bytes {start}-{end}/{self.size}"
                ):
                    raise ValueError("HTTP Content-Range mismatch")
                if response.headers.get("Content-Length") != str(expected):
                    raise ValueError("HTTP range Content-Length mismatch")
                if response.headers.get("ETag") != self.etag:
                    raise ValueError("remote source ETag changed")
                payload = response.read(expected + 1)
                if len(payload) != expected:
                    raise ValueError("HTTP range body length mismatch")
        except urllib.error.URLError as exc:
            code = getattr(exc, "code", None)
            raise RuntimeError(f"bounded range access failed (HTTP {code})") from None
        self.budget.ranges.append(
            dict(
                source_uri=self.unsigned_url,
                etag=self.etag,
                start=start,
                end=end,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
        self.cache[start] = payload
        return payload

    def read(self, size=-1):
        if self.closed:
            raise ValueError("read from closed range reader")
        stop = (
            self.size
            if size is None or size < 0
            else min(self.size, self.position + size)
        )
        parts = []
        while self.position < stop:
            block = self._block(self.position)
            within = self.position % self.block_size
            take = min(stop - self.position, len(block) - within)
            parts.append(block[within : within + take])
            self.position += take
        return b"".join(parts)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def make_range_opener(
    signed_url, unsigned_url, size, etag, budget, *, transport=urllib.request.urlopen
):
    """Each GDAL handle gets its own cursor; immutable byte blocks are shared."""
    cache = {}

    def opener(path, mode="r"):
        if str(path) != "independent-naip.tif" or mode not in ("r", "rb"):
            raise FileNotFoundError("external auxiliary resources are not permitted")
        reader = RangeReader(
            signed_url, unsigned_url, size, etag, budget, opener=transport
        )
        reader.cache = cache
        return reader

    return opener


def nearest_rareplanes(annotation_path, lon, lat):
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    nearest = None
    for feature in data["features"]:
        coords = feature["geometry"]["coordinates"]
        while isinstance(coords[0], list):
            coords = coords[0]
        x, y = coords[:2]
        angle = (
            math.sin(math.radians(y - lat) / 2) ** 2
            + math.cos(math.radians(y))
            * math.cos(math.radians(lat))
            * math.sin(math.radians(x - lon) / 2) ** 2
        )
        distance = 6371000 * 2 * math.asin(min(1, math.sqrt(angle)))
        if nearest is None or distance < nearest["distance_m"]:
            nearest = dict(
                distance_m=distance,
                location_id=feature["properties"].get("loc_id"),
                location=feature["properties"].get("location"),
                capture_id=feature["properties"].get("cat_id"),
            )
    if nearest is None or nearest["distance_m"] < 100_000:
        raise ValueError(
            "independent location must be >100km from all RarePlanes annotations"
        )
    return nearest


def reserve_locations(annotation_path, evidence):
    collection = json_request(f"{STAC}/collections/naip")
    reservations, items = [], []
    for location in LOCATIONS:
        item = json_request(f"{STAC}/collections/naip/items/{location['item']}")
        props, asset = item["properties"], item["assets"]["image"]
        if item["id"] != location["item"] or item["collection"] != "naip":
            raise ValueError("NAIP item identity mismatch")
        if props.get("gsd") != 0.6:
            raise ValueError("reserved native 0.6m sampling differs")
        if [b.get("common_name") for b in asset["eo:bands"]] != [
            "red",
            "green",
            "blue",
            "nir",
        ]:
            raise ValueError("explicit NAIP RGBIR band contract required")
        unsigned = asset["href"]
        parsed = urllib.parse.urlsplit(unsigned)
        if parsed.netloc != "naipeuwest.blob.core.windows.net" or parsed.query:
            raise ValueError("unexpected NAIP asset host or signed provenance")
        items.append(item)
        reservations.append(
            dict(
                **location,
                width=1024,
                height=1024,
                source_uri=unsigned,
                native_gsd_m=props["gsd"],
                raw_stac_datetime=props.get("datetime"),
                timestamp_precision="date",
                sensing_time_verified=False,
                nearest_rareplanes=nearest_rareplanes(
                    annotation_path, location["lon"], location["lat"]
                ),
            )
        )
    contract = dict(
        purpose="untouched independent AW01D robustness evaluation; no model selection",
        source_domain="aerial optical, not satellite",
        collection_license_raw=collection.get("license"),
        collection_license_links=[
            x for x in collection["links"] if x.get("rel") == "license"
        ],
        producer_license_url=LICENSE_URL,
        producer_license="Public domain per official USDA dataset record",
        licensing_note="Planetary Computer machine license says proprietary; its license link says Public Domain. Preserve this contradiction; producer record explicitly labels NAIP public domain.",
        resolution_note="Native delivered orthorectified RGBIR at 0.6m. RGB derivative selects bands1-3 without resampling; effective optical resolution not inferred.",
        time_note="STAC 16:00Z is retained raw, but no verified sensing hour/minute. Date precision only.",
        max_range_bytes=16 * 1024**2,
        max_range_requests=64,
        rareplanes_annotations_sha256=sha256_file(Path(annotation_path)),
        locations=reservations,
        raw_items=items,
        references=[
            LICENSE_URL,
            f"{STAC}/collections/naip",
            "https://planetarycomputer.microsoft.com/docs/",
            "https://raw.githubusercontent.com/microsoft/planetary-computer-sdk-for-python/main/planetary_computer/sas.py",
            "https://raw.githubusercontent.com/microsoft/planetary-computer-sdk-for-python/main/planetary_computer/settings.py",
            "https://rasterio.readthedocs.io/en/stable/topics/vsi.html#python-file-and-filesystem-openers",
        ],
    )
    frozen_json(Path(evidence) / "independent-source-reservation.json", contract)
    return contract


def native_window_grid(item, reservation):
    """Choose the nearest full native window using metadata only, before reading pixels."""
    from rasterio import Affine
    from rasterio.warp import transform

    props = item["properties"]
    affine = Affine(*props["proj:transform"][:6])
    height, width = props["proj:shape"]
    if min(width, height) < 1024:
        raise ValueError("source is too small for reserved native window")
    x, y = transform(
        "EPSG:4326",
        f"EPSG:{props['proj:epsg']}",
        [reservation["lon"]],
        [reservation["lat"]],
    )
    col, row = ~affine * (x[0], y[0])
    col, row = math.floor(col), math.floor(row)
    desired_x, desired_y = col - 512, row - 512
    native_x = min(max(0, desired_x), width - 1024)
    native_y = min(max(0, desired_y), height - 1024)
    return dict(
        name=reservation["name"],
        item=item["id"],
        window=[native_x, native_y, 1024, 1024],
        target_source_pixel=[col, row],
        origin_adjustment_pixels=[native_x - desired_x, native_y - desired_y],
        selection="metadata-only nearest full native1024 window; no pixel or model inspection",
        resampling=False,
    )


def extract_native(item, reservation, signed_url, output, budget):
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.enums import ColorInterp
    from rasterio.warp import transform
    from rasterio.windows import Window

    unsigned = reservation["source_uri"]
    request = urllib.request.Request(signed_url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            size = int(response.headers["Content-Length"])
            etag = response.headers["ETag"]
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"anonymous source HEAD failed (HTTP {getattr(exc, 'code', None)})"
        ) from None
    opener = make_range_opener(signed_url, unsigned, size, etag, budget)

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open("independent-naip.tif", opener=opener) as source:
            if source.count != 4 or source.dtypes != ("uint8",) * 4:
                raise ValueError("native four-band uint8 NAIP contract differs")
            props = item["properties"]
            if source.crs.to_epsg() != props["proj:epsg"]:
                raise ValueError("STAC/raster CRS mismatch")
            if [source.height, source.width] != props["proj:shape"]:
                raise ValueError("STAC/raster shape mismatch")
            if not np.allclose(
                tuple(source.transform)[:6],
                props["proj:transform"][:6],
                rtol=0,
                atol=1e-7,
            ):
                raise ValueError("STAC/raster affine mismatch")
            if not np.allclose(source.res, (0.6, 0.6), rtol=0, atol=1e-8):
                raise ValueError("native pixel spacing differs")
            window = Window(*reservation["native_window"])
            if (
                window.col_off < 0
                or window.row_off < 0
                or window.col_off + 1024 > source.width
                or window.row_off + 1024 > source.height
            ):
                raise ValueError("reserved native window crosses source edge")
            pixels = source.read((1, 2, 3), window=window)
            valid = source.dataset_mask(window=window)
            affine, crs = source.window_transform(window), source.crs
            if pixels.shape != (3, 1024, 1024) or not np.all(valid == 255):
                raise ValueError("independent window has incomplete imagery coverage")
            parent = dict(
                width=source.width,
                height=source.height,
                crs=crs.to_string(),
                transform=list(source.transform)[:6],
                source_bytes=size,
                source_etag=etag,
            )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    name = reservation["name"]
    raster = root / f"{name}.tif"
    if raster.exists():
        raise FileExistsError(
            "independent native derivative already exists; preserve it"
        )
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            raster,
            "w",
            driver="GTiff",
            width=1024,
            height=1024,
            count=3,
            dtype="uint8",
            crs=crs,
            transform=affine,
            compress="deflate",
        ) as destination:
            destination.write(pixels)
            destination.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
            )
            destination.write_mask(valid)
    with rasterio.open(raster) as check:
        if any(str(path).endswith(".msk") for path in check.files):
            raise ValueError("independent derivative mask must be internal")
        if not np.array_equal(check.read(), pixels) or check.transform != affine:
            raise ValueError("native RGB roundtrip verification failed")
    Image.fromarray(np.moveaxis(pixels, 0, -1)).save(root / f"{name}.png")
    source_contract = RasterSource(
        scene_id=item["id"],
        sensor_id="USDA-NAIP-aerial-RGBIR",
        timestamp_precision="date",
        acquisition_time_raw=item["properties"].get("datetime"),
        native_gsd_m=0.6,
        pixel_spacing_m=(0.6, 0.6),
        quality="unknown",
        airfield_group=f"independent-naip-{name}",
        capture_group=item["id"],
        source_uri=unsigned,
        license="Public Domain",
        attribution="USDA NAIP; hosted/processed by Microsoft Planetary Computer and Esri.",
        sha256=sha256_file(raster),
        width=1024,
        height=1024,
        crs=crs.to_string(),
        transform=tuple(affine)[:6],
        provenance_notes=[
            "Aerial, not satellite; independent-source robustness evaluation only.",
            "Native RGB bands1-3 selected from source RGBIR; NIR omitted, no resize or reprojection.",
            "Raw STAC 16:00Z retained; only acquisition date is verified, no sensing time.",
            "Full parent COG not downloaded; source ETag and per-range checksums preserve fetched-byte provenance.",
            "Imagery quality unknown until visual review; analysts must review uncertainty.",
            "Producer public-domain record overrides no machine-field assumption; conflicting catalog license retained in reservation.",
        ],
    )
    result = dict(
        name=name,
        path=str(raster),
        source=source_contract.model_dump(mode="json"),
        parent=parent,
        window=[int(window.col_off), int(window.row_off), 1024, 1024],
        pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
        fully_valid=True,
        native_roundtrip_identical=True,
    )
    frozen_json(root / f"{name}-source.json", result)
    return result


def verify_cached_retrieval(annotation_path, output, evidence):
    """Replay a frozen retrieval locally, checking bytes, RGB/grid and provenance."""
    import numpy as np
    from .preprocessing import NativeRaster

    evidence, output = Path(evidence), Path(output)
    contract = json.loads(
        (evidence / "independent-source-reservation.json").read_text()
    )
    grids = json.loads((evidence / "independent-source-window-grid.json").read_text())
    report = json.loads((evidence / "independent-source-retrieval.json").read_text())
    if contract["rareplanes_annotations_sha256"] != sha256_file(Path(annotation_path)):
        raise ValueError("independence reference annotations changed")
    if (
        report["range_bytes_reserved"] > report["max_range_bytes"]
        or report["range_requests"] > report["max_range_requests"]
    ):
        raise ValueError("recorded range budget exceeded")
    if (
        len(report["samples"]) != 2
        or len(contract["locations"]) != 2
        or len(grids) != 2
    ):
        raise ValueError("both reserved independent samples are required")
    for record, reservation, grid in zip(
        report["samples"], contract["locations"], grids
    ):
        name = reservation["name"]
        if record["name"] != name or record["window"] != grid["window"]:
            raise ValueError("cached independent window identity differs")
        local = json.loads((output / f"{name}-source.json").read_text())
        if local != record:
            raise ValueError("cached independent source record differs")
        source = RasterSource.model_validate(record["source"])
        if (
            source.source_uri != reservation["source_uri"]
            or source.scene_id != reservation["item"]
        ):
            raise ValueError("cached independent acquisition provenance differs")
        with NativeRaster(output / f"{name}.tif", source) as raster:
            pixels = raster.dataset.read()
            valid = raster.dataset.dataset_mask()
            if not np.all(valid == 255) or any(
                str(p).endswith(".msk") for p in raster.dataset.files
            ):
                raise ValueError(
                    "cached independent imagery coverage or internal mask differs"
                )
        if hashlib.sha256(pixels.tobytes()).hexdigest() != record["pixels_sha256"]:
            raise ValueError("cached native RGB pixels differ")
    return report


def prepare_independent(annotation_path, output, evidence, *, reserve_only=False):
    if (
        not reserve_only
        and (Path(evidence) / "independent-source-retrieval.json").exists()
    ):
        return verify_cached_retrieval(annotation_path, output, evidence)
    if not reserve_only and any(
        (Path(output) / f"{x['name']}.tif").exists() for x in LOCATIONS
    ):
        raise FileExistsError(
            "partial independent retrieval exists; preserve and inspect before retry"
        )
    contract = reserve_locations(annotation_path, evidence)
    if reserve_only:
        return contract
    grids = [
        native_window_grid(item, reservation)
        for item, reservation in zip(contract["raw_items"], contract["locations"])
    ]
    frozen_json(Path(evidence) / "independent-source-window-grid.json", grids)
    # Public anonymous token from the official endpoint; never serialized/logged.
    token = json_request(TOKEN_URL, cap=32768)["token"]
    budget = RangeBudget()
    results = []
    for item, reservation, grid in zip(
        contract["raw_items"], contract["locations"], grids
    ):
        signed_url = reservation["source_uri"] + "?" + token
        reserved_grid = dict(reservation, native_window=grid["window"])
        results.append(extract_native(item, reserved_grid, signed_url, output, budget))
    report = dict(
        source_domain="aerial optical, not satellite",
        samples=results,
        range_bytes_reserved=budget.used_bytes,
        range_requests=budget.requests,
        max_range_bytes=budget.max_bytes,
        max_range_requests=budget.max_requests,
        ranges=budget.ranges,
        imagery_resampled=False,
        no_model_inference=True,
    )
    frozen_json(Path(evidence) / "independent-source-retrieval.json", report)
    return report


def prepare_extra_cody_ga(annotation_path, output, evidence):
    """One pre-inference GA-apron extension, within the original aggregate bounds."""
    from .preprocessing import NativeRaster

    evidence, output = Path(evidence), Path(output)
    initial = verify_cached_retrieval(annotation_path, output, evidence)
    contract = json.loads(
        (evidence / "independent-source-reservation.json").read_text()
    )
    item = next(i for i in contract["raw_items"] if i["id"] == LOCATIONS[1]["item"])
    original = next(r for r in contract["locations"] if r["name"] == "cody")
    reservation = dict(
        original,
        name="cody-ga-apron",
        native_window=[5658, 8394, 1024, 1024],
        selection="additional hangar/GA-apron coverage selected from source imagery before any model predictions; evaluation only",
    )
    frozen_json(evidence / "independent-source-ga-reservation.json", reservation)
    target = evidence / "independent-source-ga-retrieval.json"
    if target.exists():
        previous = json.loads(target.read_text())
        if previous.get("status") != "complete":
            raise ValueError(
                "prior bounded GA-apron attempt incomplete; preserve evidence, do not repeat automatically"
            )
        record = previous["sample"]
        with NativeRaster(
            output / "cody-ga-apron.tif", RasterSource.model_validate(record["source"])
        ):
            pass
        return previous
    if (output / "cody-ga-apron.tif").exists():
        raise FileExistsError("existing independent GA derivative must be preserved")
    budget = RangeBudget(
        max_bytes=min(5 * 1024**2, 16 * 1024**2 - initial["range_bytes_reserved"]),
        max_requests=min(20, 64 - initial["range_requests"]),
    )
    token = json_request(TOKEN_URL, cap=32768)["token"]
    try:
        record = extract_native(
            item, reservation, reservation["source_uri"] + "?" + token, output, budget
        )
    except Exception as exc:
        frozen_json(
            target,
            dict(
                status="incomplete",
                failure_type=type(exc).__name__,
                range_bytes_reserved=budget.used_bytes,
                range_requests=budget.requests,
                max_range_bytes=budget.max_bytes,
                max_range_requests=budget.max_requests,
                ranges=budget.ranges,
                no_model_inference=True,
            ),
        )
        raise RuntimeError(
            "bounded independent GA-apron attempt failed; preserved local evidence, no automatic retry"
        ) from None
    report = dict(
        status="complete",
        sample=record,
        range_bytes_reserved=budget.used_bytes,
        range_requests=budget.requests,
        max_range_bytes=budget.max_bytes,
        max_range_requests=budget.max_requests,
        initial_retrieval_sha256=sha256_file(
            evidence / "independent-source-retrieval.json"
        ),
        completed_aggregate_range_bytes=initial["range_bytes_reserved"]
        + budget.used_bytes,
        completed_aggregate_range_requests=initial["range_requests"] + budget.requests,
        preflight_note="A previous metadata/header-only grid probe occurred before the initial successful retrieval; completed aggregate leaves at least0.75MiB and3requests for that overhead.",
        ranges=budget.ranges,
        no_model_inference=True,
    )
    frozen_json(target, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="AW01D bounded independent NAIP native evaluation windows"
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/aw01d-independent"))
    parser.add_argument("--evidence", type=Path, default=Path("docs/aw01d"))
    parser.add_argument("--reserve-only", action="store_true")
    parser.add_argument("--extra-cody-ga", action="store_true")
    args = parser.parse_args(argv)
    if args.extra_cody_ga:
        if args.reserve_only:
            raise ValueError("extra-cody-ga and reserve-only cannot be combined")
        result = prepare_extra_cody_ga(args.annotations, args.output, args.evidence)
    else:
        result = prepare_independent(
            args.annotations, args.output, args.evidence, reserve_only=args.reserve_only
        )
    print(
        json.dumps(
            {
                "reserved": len(LOCATIONS) + int(args.extra_cody_ga),
                "range_requests": result.get("range_requests"),
                "range_bytes": result.get("range_bytes_reserved"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
