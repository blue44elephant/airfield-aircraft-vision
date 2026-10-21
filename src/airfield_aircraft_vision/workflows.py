"""Reproducible CPU/GPU training and isolated-airfield evaluation workflows."""

from __future__ import annotations
import json, random, platform
from pathlib import Path
from .dataset import load_manifest, resolve_image
from .preprocessing import NativeRaster, sha256_file
from .optical_contracts import PixelBox
from .evaluation import match_boxes, summarize
from .inference.optical_high_resolution import infer_scene


def train(
    manifest_path: Path,
    initial_weights: Path,
    output: Path,
    *,
    epochs=3,
    seed=17,
    device="cpu",
    threads=4,
    lr=0.005,
):
    import torch, torchvision
    from .torch_backend import initialize_from_coco, ARCHITECTURE

    if epochs < 1 or epochs > 1000 or threads < 1 or lr <= 0:
        raise ValueError("invalid training bounds")
    if output.exists():
        raise ValueError("refusing to overwrite a checkpoint; choose a fresh output")
    manifest = load_manifest(manifest_path)
    samples = [s for s in manifest.samples if s.split == "train"]
    if not samples:
        raise ValueError("no training samples")
    # Large scenes require explicit training crop preparation; never implicitly downsample them.
    if any(s.source.width > 512 or s.source.height > 512 for s in samples):
        raise ValueError("training expects native crops no larger than 512x512")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model = initialize_from_coco(initial_weights).to(device)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005,
    )
    history = []
    for epoch in range(epochs):
        model.train()
        model.backbone.eval()
        order = list(samples)
        random.Random(seed + epoch).shuffle(order)
        losses = []
        for sample in order:
            # Reuse the operational capability gate; training must not bypass sensor/quality/GSD rules.
            from .capability import route_scene, SceneMetadata, CapabilityTier

            s = sample.source
            decision = route_scene(
                SceneMetadata(
                    sensor_domain=s.sensor_domain,
                    native_gsd_m=s.native_gsd_m,
                    panchromatic_gsd_m=s.panchromatic_gsd_m,
                    pan_sharpened_gsd_m=s.pan_sharpened_gsd_m,
                    effective_spatial_resolution_m=s.effective_resolution_m,
                    image_quality=s.quality,
                )
            )
            if decision.tier != CapabilityTier.A:
                raise ValueError("training requires Tier A optical source")
            with NativeRaster(resolve_image(manifest_path, sample), s) as raster:
                pixels, _ = raster.read((0, 0, s.width, s.height), 512)
            image = torch.from_numpy(pixels).float().to(device) / 255
            boxes = torch.tensor(
                [b.xyxy for b in sample.boxes], dtype=torch.float32, device=device
            ).reshape(-1, 4)
            target = {
                "boxes": boxes,
                "labels": torch.ones(len(boxes), dtype=torch.int64, device=device),
            }
            values = model([image], [target])
            loss = sum(values.values())
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {"epoch": epoch + 1, "mean_training_loss": sum(losses) / len(losses)}
        )
        print(json.dumps(history[-1]), flush=True)
    info = {
        "architecture": ARCHITECTURE,
        "preprocessing": "native-rgb-v1",
        "experimental": True,
        "experimental_input_bounds_m": [0.2, 1.0],
        "classes": ["background", "aircraft"],
        "training_manifest_sha256": sha256_file(manifest_path),
        "initial_weights_sha256": sha256_file(initial_weights),
        "training_airfields": sorted({s.source.airfield_group for s in samples}),
        "training_captures": sorted({s.source.capture_group for s in samples}),
        "training_source_checksums": sorted({s.source.sha256 for s in samples}),
        "epochs": epochs,
        "seed": seed,
        "lr": lr,
        "optimizer": "SGD momentum=0.9 weight_decay=0.0005",
        "backbone": "frozen including normalization statistics",
        "checkpoint_selection": "fixed final epoch; no test tuning",
        "history": history,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "limitations": [
            "Small experimental training sample; no deployment validation",
            "COCO pretrained generic-airplane initialization",
            "Confidence uncalibrated; input bounds are policy, not validated performance range",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "manifest": info}, output)
    output.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def assert_holdout(samples, model):
    for s in samples:
        for attr, key in (
            ("airfield_group", "training_airfields"),
            ("capture_group", "training_captures"),
            ("sha256", "training_source_checksums"),
        ):
            if not model.get(key):
                raise ValueError("checkpoint lacks auditable training lineage")
            if getattr(s.source, attr) in model[key]:
                raise ValueError(f"evaluation leakage: {attr}")


def draw_example(path, result, sample, record, destination):
    from PIL import Image, ImageDraw

    with Image.open(path) as image:
        image = image.convert(
            "RGB"
        )  # preserve dimensions; annotate a separate derivative
        draw = ImageDraw.Draw(image)
        for i, b in enumerate(sample.boxes):
            draw.rectangle(b.xyxy, outline="cyan", width=2)
            if i in record["matches"]["false_negatives"]:
                draw.text(b.xyxy[:2], "FN", fill="cyan")
        accepted = [d for d in result.detections if d.status == "accepted"]
        for i, d in enumerate(accepted):
            b = d.oriented_box
            coords = (
                b.center_x - b.width / 2,
                b.center_y - b.height / 2,
                b.center_x + b.width / 2,
                b.center_y + b.height / 2,
            )
            color = "red" if i in record["matches"]["false_positives"] else "lime"
            draw.rectangle(coords, outline=color, width=2)
            draw.text(
                (coords[0], max(0, coords[1] - 12)), f"{d.confidence:.2f}", fill=color
            )
        image.save(destination)


def evaluate(
    manifest_path: Path,
    checkpoint: Path,
    outdir: Path,
    *,
    split="test",
    confidence=0.5,
    device="cpu",
    threads=4,
):
    import torch
    from .torch_backend import TorchDetector

    if split not in ("validation", "test"):
        raise ValueError("evaluation requires validation or test split")
    manifest = load_manifest(manifest_path)
    samples = [s for s in manifest.samples if s.split == split]
    if not samples:
        raise ValueError("no evaluation samples")
    torch.set_num_threads(threads)
    detector = TorchDetector(checkpoint, device)
    assert_holdout(samples, detector.manifest)
    if outdir.exists():
        raise ValueError("choose a fresh evaluation output directory")
    outdir.mkdir(parents=True)
    records = []
    for sample in samples:
        result = infer_scene(
            resolve_image(manifest_path, sample),
            sample.source,
            detector,
            confidence=confidence,
            allow_unknown_quality=True,
        )
        predicted = []
        for d in result.detections:
            if d.status == "accepted":
                b = d.oriented_box
                predicted.append(
                    PixelBox(
                        xyxy=(
                            b.center_x - b.width / 2,
                            b.center_y - b.height / 2,
                            b.center_x + b.width / 2,
                            b.center_y + b.height / 2,
                        ),
                        score=d.confidence,
                    )
                )
        matches = match_boxes(predicted, sample.boxes, sample.ignored_boxes)
        record = {
            "image_id": sample.image_id,
            "airfield": sample.source.airfield_group,
            "truth_count": len(sample.boxes),
            "predicted_count": len(predicted) - len(matches["ignored_predictions"]),
            "matches": matches,
            "source_sha256": sample.source.sha256,
            "acquisition_timestamp": (
                sample.source.acquisition_timestamp.isoformat()
                if sample.source.acquisition_timestamp
                else None
            ),
            "timestamp_precision": sample.source.timestamp_precision,
            "timestamp_timezone_basis": sample.source.timestamp_timezone_basis,
        }
        records.append(record)
        (outdir / (sample.image_id + ".json")).write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        draw_example(
            resolve_image(manifest_path, sample),
            result,
            sample,
            record,
            outdir / (sample.image_id + ".png"),
        )
    report = {
        "status": "measured_experimental",
        "split": split,
        "confidence_threshold": confidence,
        "matching_iou": 0.5,
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "independent_airfields": sorted({s.source.airfield_group for s in samples}),
        "metrics": summarize(records),
        "records": records,
        "model": detector.manifest,
        "limitations": manifest.preparation.get("limitations", [])
        + [
            "Micro precision/recall at fixed threshold and IoU=0.5; not COCO mAP.",
            "Counts/errors are per native tile; do not sum as whole-airfield inventory.",
            "Qualitative FP/FN overlays are candidate examples; require human review.",
            "Unknown local quality explicitly allowed for this experiment.",
        ],
    }
    (outdir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (outdir / "ATTRIBUTION.txt").write_text(
        "RarePlanes: J. Shermeyer, T. Hossler, A. Van Etten, D. Hogan, R. Lewis, D. Kim. In-Q-Tel - CosmiQ Works and AI.Reverie, June 2020.\n"
        "https://www.iqt.org/library/the-rareplanes-dataset\nCC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/\n"
        "Changes: original pixels retained; separate overlays added (cyan=GT/FN, green=matched detection, red=unmatched detection).\n"
        "These image derivatives are provided under CC BY-SA 4.0; no endorsement or warranty.\n",
        encoding="utf-8",
    )
    return report
