"""Reviewed native background crops; never infer emptiness from absent labels alone.

All RarePlanes supplied PNG tiles contain aircraft. Candidates here are smaller
native-pixel background windows inside positive tiles, never full empty scenes.
Annotation exclusion is a proposal filter, not a substitute for visual review.
"""

from __future__ import annotations

from datetime import datetime
import json
import hashlib
from pathlib import Path

from .optical_contracts import Sample, RasterSource
from .preprocessing import NativeRaster, sha256_file, pixel_spacing_m


def _validated_window(sample, window):
    if sample.split == "test":
        raise ValueError(
            "independent/test imagery is not eligible for background mining"
        )
    if len(window) != 4 or any(type(v) is not int for v in window):
        raise ValueError("window requires four integer native pixel coordinates")
    x, y, width, height = window
    if min(x, y) < 0 or not 1 <= width <= 256 or not 1 <= height <= 256:
        raise ValueError("background crop size must be 1..256 native pixels")
    if x + width > sample.source.width or y + height > sample.source.height:
        raise ValueError("background crop outside source grid")
    return x, y, width, height


def _clear_of_labels(sample, window, halo):
    if type(halo) is not int or halo < 0:
        raise ValueError("halo must be a nonnegative integer")
    x, y, width, height = _validated_window(sample, window)
    for box in sample.boxes + sample.ignored_boxes:
        left, top, right, bottom = box.xyxy
        if (
            x < right + halo
            and x + width > left - halo
            and y < bottom + halo
            and y + height > top - halo
        ):
            return False
    return True


def propose_candidates(samples, crop_size=256, halo=32):
    """Return deterministic nonoverlapping candidates requiring explicit review.

    Pass train/validation samples only. All original labels, including ignored
    fragments, retain an exclusion halo. No raster pixels are accessed here.
    """
    if type(crop_size) is not int or not 32 <= crop_size <= 256:
        raise ValueError("crop_size must be an integer in 32..256")
    if type(halo) is not int or halo < 0:
        raise ValueError("halo must be a nonnegative integer")
    candidates = []
    for sample in sorted(samples, key=lambda s: s.image_id):
        if sample.split == "test":
            raise ValueError("test samples must be excluded before background mining")
        for y in range(0, sample.source.height - crop_size + 1, crop_size):
            for x in range(0, sample.source.width - crop_size + 1, crop_size):
                window = (x, y, crop_size, crop_size)
                if _clear_of_labels(sample, window, halo):
                    candidates.append(
                        {
                            "parent_image_id": sample.image_id,
                            "parent_sha256": sample.source.sha256,
                            "split": sample.split,
                            "window": list(window),
                            "halo": halo,
                            "review_required": True,
                        }
                    )
    return candidates


def _verify_review(sample, window, review):
    if not isinstance(review, dict):
        raise ValueError("explicit visual review evidence required")
    if review.get("decision") != "empty_no_aircraft":
        raise ValueError("only explicitly reviewed aircraft-free crops can be added")
    if review.get("parent_sha256") != sample.source.sha256:
        raise ValueError("visual review is bound to the source checksum")
    if review.get("window") != list(window):
        raise ValueError("visual review is bound to the exact native crop window")
    if review.get("category") not in (
        "empty_apron",
        "difficult_background",
        "other_background",
    ):
        raise ValueError("review requires an explicit background category")
    for field in ("reviewer", "notes", "reviewed_at"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError("review requires reviewer, notes and timestamp")
    try:
        when = datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("review timestamp must be an aware ISO datetime") from exc
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("review timestamp must include timezone")


def _prepare_geotiff(source_path, sample, window, outpath, *, review, halo=32):
    """Save an approved, checksum-bound native crop and return its negative Sample.

    The source raster is opened read-only. Output uses exclusive creation and
    lossless encoding; no resizing, reprojection, padding or sharpening occurs.
    The caller makes the returned path relative to its new manifest root.
    """
    from rasterio.enums import ColorInterp
    from rasterio.io import MemoryFile
    from rasterio.windows import Window

    window = _validated_window(sample, window)
    if not _clear_of_labels(sample, window, halo):
        raise ValueError(
            "background crop intersects an original label or exclusion halo"
        )
    _verify_review(sample, window, review)
    outpath = Path(outpath)
    with NativeRaster(Path(source_path), sample.source) as native:
        ds = native.dataset
        win = Window(*window)
        pixels = ds.read(window=win)
        mask = ds.dataset_mask(window=win)
        if not (mask == 255).all():
            raise ValueError("background crops require fully valid source pixels")
        affine = ds.window_transform(win)
        profile = dict(
            driver="GTiff",
            width=window[2],
            height=window[3],
            count=3,
            dtype="uint8",
            transform=affine,
            compress="deflate",
        )
        if ds.crs:
            profile["crs"] = ds.crs
        with MemoryFile() as mem:
            with mem.open(**profile) as cropped:
                cropped.write(pixels)
                cropped.colorinterp = (
                    ColorInterp.red,
                    ColorInterp.green,
                    ColorInterp.blue,
                )
                cropped.update_tags(
                    **ds.tags(),
                    AW01C_PARENT_SHA256=sample.source.sha256,
                    AW01C_PARENT_IMAGE_ID=sample.image_id,
                    AW01C_NATIVE_WINDOW=json.dumps(window),
                    AW01C_VISUAL_REVIEW=json.dumps(review, sort_keys=True),
                )
            encoded = mem.read()
        spacing = (
            pixel_spacing_m(ds.crs, affine, window[2], window[3])
            if ds.crs
            else sample.source.pixel_spacing_m
        )
        crs = ds.crs.to_string() if ds.crs else None
    notes = list(sample.source.provenance_notes) + [
        "AW-01C derived native background crop, not a full empty-apron scene.",
        f"Parent image={sample.image_id}; SHA256={sample.source.sha256}; window={window}; annotation halo={halo}px.",
        "Visual review=" + json.dumps(review, sort_keys=True),
        "No resize, reprojection, padding, sharpening or source modification.",
    ]
    source_values = sample.source.model_dump()
    source_values.update(
        sha256=hashlib.sha256(encoded).hexdigest(),
        width=window[2],
        height=window[3],
        crs=crs,
        transform=tuple(affine)[:6],
        sidecar_sha256=None,
        pixel_spacing_m=spacing,
        provenance_notes=notes,
    )
    source = RasterSource.model_validate(source_values)
    x, y, width, height = window
    return encoded, Sample(
        image_id=f"{Path(sample.image_id).stem}__background_{x}_{y}_{width}_{height}.tif",
        path=str(outpath),
        split=sample.split,
        source=source,
        boxes=[],
        ignored_boxes=[],
    )


def derive_geotiff(source_path, sample, window, outpath, *, review, halo=32):
    """Save one explicitly reviewed native crop, refusing all overwrites."""
    outpath = Path(outpath)
    if outpath.exists():
        raise FileExistsError("background output already exists")
    encoded, sample = _prepare_geotiff(
        source_path, sample, window, outpath, review=review, halo=halo
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("xb") as output:
        output.write(encoded)
    return sample


def attach_reviewed_negatives(manifest_path, review_json, out_manifest):
    """Attach approved train/validation backgrounds without changing the base.

    All approvals, native source windows, existing derivative bytes and final
    dataset invariants are validated in memory before creating any file.
    Exact replay verifies the stored bytes rather than silently replacing them.
    """
    from .dataset import load_manifest, resolve_image
    from .optical_contracts import DatasetManifest

    manifest_path = Path(manifest_path).resolve()
    review_json = Path(review_json).resolve()
    out_manifest = Path(out_manifest).resolve()
    root = manifest_path.parent
    if out_manifest == manifest_path or out_manifest.parent != root:
        raise ValueError(
            "new negative manifest must be a different file in the base root"
        )
    if review_json.stat().st_size > 1024 * 1024:
        raise ValueError("background review file exceeds bounded metadata size")
    raw_review = review_json.read_bytes()
    review_sha = hashlib.sha256(raw_review).hexdigest()
    data = json.loads(raw_review)
    reviews = data.get("reviews") if isinstance(data, dict) else data
    if not isinstance(reviews, list) or not 1 <= len(reviews) <= 128:
        raise ValueError("background attachment requires 1..128 explicit reviews")
    manifest = load_manifest(manifest_path)
    by_id = {s.image_id: s for s in manifest.samples}
    seen = set()
    prepared = []
    negatives = []
    details = []
    for review in reviews:
        if not isinstance(review, dict):
            raise ValueError("each negative review must be a dictionary")
        parent_id = review.get("parent_image_id")
        if parent_id not in by_id:
            raise ValueError("review parent is absent from the base manifest")
        sample = by_id[parent_id]
        window = _validated_window(sample, review.get("window", []))
        key = (parent_id, window)
        if key in seen:
            raise ValueError("duplicate negative crop review")
        seen.add(key)
        halo = review.get("halo", 32)
        identifier = hashlib.sha256(
            json.dumps([parent_id, sample.source.sha256, window]).encode("utf-8")
        ).hexdigest()[:20]
        relative_path = Path("negative-images") / f"background-{identifier}.tif"
        output_path = (root / relative_path).resolve()
        if not output_path.is_relative_to(root):
            raise ValueError("negative output path escapes the dataset root")
        encoded, negative = _prepare_geotiff(
            resolve_image(manifest_path, sample),
            sample,
            window,
            output_path,
            review=review,
            halo=halo,
        )
        negative = negative.model_copy(update={"path": relative_path.as_posix()})
        if output_path.exists() and output_path.read_bytes() != encoded:
            raise ValueError(
                "existing negative raster differs from approved native crop"
            )
        prepared.append((output_path, encoded))
        negatives.append(negative)
        details.append(
            {
                "image_id": negative.image_id,
                "parent_image_id": parent_id,
                "parent_sha256": sample.source.sha256,
                "path": relative_path.as_posix(),
                "window": list(window),
                "halo": halo,
                "sha256": negative.source.sha256,
                "review": review,
            }
        )
    preparation = dict(manifest.preparation)
    preparation.update(
        verified_negative_image_ids=[s.image_id for s in negatives],
        negative_review_sha256=review_sha,
        negative_base_manifest_sha256=sha256_file(manifest_path),
        negative_review_details=details,
        negative_scope=(
            "Native <=256px reviewed background crops, not independently surveyed "
            "full empty scenes. Test imagery excluded; visual review is not analytical validation."
        ),
    )
    result = DatasetManifest(
        version=manifest.version,
        samples=manifest.samples + negatives,
        preparation=preparation,
    )
    encoded_manifest = result.model_dump_json(indent=2).encode("utf-8")
    if out_manifest.exists():
        if out_manifest.read_bytes() != encoded_manifest:
            raise ValueError("existing negative manifest or review checksum differs")
        if any(not path.is_file() for path, _ in prepared):
            raise ValueError(
                "existing negative manifest has a missing derivative raster"
            )
        return result
    # Validation above is complete before the first filesystem mutation.
    for path, encoded in prepared:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                output.write(encoded)
    with out_manifest.open("xb") as output:
        output.write(encoded_manifest)
    return result


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Attach reviewed native background crops"
    )
    parser.add_argument("manifest")
    parser.add_argument("reviews")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    result = attach_reviewed_negatives(args.manifest, args.reviews, args.output)
    print(
        json.dumps(
            {
                "manifest": args.output,
                "negative_images": len(
                    result.preparation["verified_negative_image_ids"]
                ),
                "total_images": len(result.samples),
                "review_sha256": result.preparation["negative_review_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
