"""AW-01B native-grid augmentation and bounded validation-selected retraining."""

from __future__ import annotations
import json, random, time, platform
from pathlib import Path
from .dataset import load_manifest, resolve_image
from .preprocessing import NativeRaster, sha256_file
from .optical_contracts import PixelBox
from .diagnostics import prediction_boxes
from .inference.optical_high_resolution import infer_scene
from .workflows import assert_holdout
from .experiment_metrics import choose_threshold, evaluate_predictions


def augment_native(pixels, boxes, rotation=0, horizontal=False):
    import numpy as np

    out = pixels
    transformed = [tuple(b) for b in boxes]
    height, width = out.shape[1:]
    for _ in range(rotation % 4):
        out = np.rot90(out, 1, axes=(1, 2))
        transformed = [
            (y1, width - x2, y2, width - x1) for x1, y1, x2, y2 in transformed
        ]
        height, width = width, height
    if horizontal:
        out = out[:, :, ::-1]
        transformed = [
            (width - x2, y1, width - x1, y2) for x1, y1, x2, y2 in transformed
        ]
    return np.ascontiguousarray(out), transformed


def collect_predictions(manifest_path, detector, split, *, allow_training=False):
    manifest = load_manifest(manifest_path)
    samples = [s for s in manifest.samples if s.split == split]
    if split == "train" and not allow_training:
        raise ValueError("training diagnostics require explicit access")
    if split != "train":
        assert_holdout(samples, detector.manifest)
    predictions = []
    for sample in samples:
        result = infer_scene(
            resolve_image(manifest_path, sample),
            sample.source,
            detector,
            confidence=0.005,
            allow_unknown_quality=True,
        )
        predictions.append(prediction_boxes(result))
    return manifest, samples, predictions


def train_candidate(
    manifest_path: Path,
    initial_weights: Path,
    outdir: Path,
    *,
    name,
    epochs=6,
    lr=0.005,
    optimizer_name="sgd",
    backbone="frozen",
    seed=23,
    batch_size=2,
    threads=4,
):
    import torch, torchvision
    from .torch_backend import initialize_from_coco, TorchDetector, ARCHITECTURE

    if outdir.exists():
        raise ValueError("fresh candidate output directory required")
    if (
        epochs < 1
        or epochs > 30
        or batch_size not in (1, 2, 4)
        or backbone not in ("frozen", "fpn")
        or optimizer_name not in ("sgd", "adamw")
    ):
        raise ValueError("unsupported bounded training configuration")
    outdir.mkdir(parents=True)
    manifest = load_manifest(manifest_path)
    samples = [s for s in manifest.samples if s.split == "train"]
    val = [s for s in manifest.samples if s.split == "validation"]
    if not samples or not val:
        raise ValueError("training and validation samples required")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model = initialize_from_coco(initial_weights)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    if backbone == "fpn":
        for p in model.backbone.fpn.parameters():
            p.requires_grad_(True)
    model.roi_heads.score_thresh = 0.005
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = (
        torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=0.0005)
        if optimizer_name == "sgd"
        else torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0001)
    )
    info = {
        "architecture": ARCHITECTURE,
        "preprocessing": "native-rgb-v1",
        "candidate_score_floor": 0.005,
        "experimental": True,
        "experimental_input_bounds_m": [0.2, 1.0],
        "classes": ["background", "aircraft"],
        "training_manifest_sha256": sha256_file(manifest_path),
        "initial_weights_sha256": sha256_file(initial_weights),
        "training_airfields": sorted({s.source.airfield_group for s in samples}),
        "training_captures": sorted({s.source.capture_group for s in samples}),
        "training_source_checksums": sorted({s.source.sha256 for s in samples}),
        "validation_airfields": sorted({s.source.airfield_group for s in val}),
        "configuration": {
            "name": name,
            "epochs": epochs,
            "lr": lr,
            "optimizer": optimizer_name,
            "backbone": backbone,
            "batch_size": batch_size,
            "seed": seed,
            "threads": threads,
            "augmentation": "D4 native flips/90-degree rotations",
        },
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "limitations": [
            "Experimental; confidence uncalibrated",
            "Positive-only dataset; not inventory validation",
        ],
    }
    cached = {}
    from .capability import SceneMetadata, route_scene, CapabilityTier

    for s in samples:
        if s.source.width > 512 or s.source.height > 512:
            raise ValueError("prepare native training crops <=512")
        decision = route_scene(
            SceneMetadata(
                sensor_domain=s.source.sensor_domain,
                native_gsd_m=s.source.native_gsd_m,
                panchromatic_gsd_m=s.source.panchromatic_gsd_m,
                pan_sharpened_gsd_m=s.source.pan_sharpened_gsd_m,
                effective_spatial_resolution_m=s.source.effective_resolution_m,
                image_quality=s.source.quality,
            )
        )
        if decision.tier != CapabilityTier.A:
            raise ValueError("training requires Tier A optical imagery")
        with NativeRaster(resolve_image(manifest_path, s), s.source) as raster:
            pixels, valid = raster.read((0, 0, s.source.width, s.source.height), 512)
        if not valid.any():
            raise ValueError("all-nodata training input")
        cached[s.image_id] = pixels
    detector = object.__new__(TorchDetector)
    detector.device = torch.device("cpu")
    detector.manifest = info
    detector.model = model
    assert_holdout(val, info)
    history = []
    best_rank = None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        model.backbone.eval()
        order = list(samples)
        rng = random.Random(seed + epoch)
        rng.shuffle(order)
        losses = []
        for first in range(0, len(order), batch_size):
            images = []
            targets = []
            for sample in order[first : first + batch_size]:
                pixels, boxes = augment_native(
                    cached[sample.image_id],
                    [b.xyxy for b in sample.boxes],
                    rng.randrange(4),
                    bool(rng.randrange(2)),
                )
                images.append(torch.from_numpy(pixels).float() / 255.0)
                targets.append(
                    {
                        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(
                            -1, 4
                        ),
                        "labels": torch.ones(len(boxes), dtype=torch.int64),
                    }
                )
            values = model(images, targets)
            loss = sum(values.values())
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        _, val, preds = collect_predictions(manifest_path, detector, "validation")
        selection = choose_threshold(val, preds, split="validation")
        evaluation = evaluate_predictions(
            val,
            preds,
            selection["selected_threshold"],
            manifest.preparation.get("metadata_weather_by_scene"),
        )
        row = {
            "epoch": epoch,
            "mean_training_loss": sum(losses) / len(losses),
            "elapsed_seconds": time.perf_counter() - start,
            "validation": selection,
            "validation_ap50": evaluation["metrics"]["ap50"],
        }
        history.append(row)
        rank = (
            selection["metrics"]["f1"] or 0,
            -selection["metrics"]["count_mae"],
            selection["selected_threshold"],
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            model_info = dict(
                info,
                selected_epoch=epoch,
                selected_threshold=selection["selected_threshold"],
                selected_validation_metrics=evaluation["metrics"],
                selection_split="validation",
            )
            torch.save(
                {"state_dict": model.state_dict(), "manifest": model_info},
                outdir / "best.pt",
            )
            (outdir / "best-validation.json").write_text(
                json.dumps(evaluation, indent=2), encoding="utf-8"
            )
        (outdir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "candidate": name,
                    "epoch": epoch,
                    "loss": row["mean_training_loss"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "threshold": selection["selected_threshold"],
                    "validation": evaluation["metrics"],
                }
            ),
            flush=True,
        )
    return history
