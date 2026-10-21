"""AW-01B bounded, cached-data diagnosis before broader training."""

from __future__ import annotations
import json, time, random
from pathlib import Path
from .dataset import load_manifest, resolve_image, location
from .preprocessing import NativeRaster, sha256_file
from .optical_contracts import PixelBox
from .evaluation import match_boxes, summarize


def prediction_boxes(result):
    boxes = []
    for d in result.detections:
        if d.status == "accepted":
            b = d.oriented_box
            boxes.append(
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
    return boxes


def score_records(samples, predictions, threshold):
    if len(samples) != len(predictions):
        raise ValueError("sample/prediction length mismatch")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    records = []
    for s, boxes in zip(samples, predictions):
        filtered = [b for b in boxes if b.score >= threshold]
        m = match_boxes(filtered, s.boxes, s.ignored_boxes)
        records.append(
            {
                "image_id": s.image_id,
                "truth_count": len(s.boxes),
                "predicted_count": len(filtered) - len(m["ignored_predictions"]),
                "matches": m,
            }
        )
    return records


def diagnose(
    manifest_path: Path, checkpoint: Path, outdir: Path, *, steps=80, threads=4
):
    import torch, numpy as np
    from PIL import Image
    from .torch_backend import TorchDetector

    if outdir.exists():
        raise ValueError("fresh diagnosis directory required")
    outdir.mkdir(parents=True)
    torch.set_num_threads(threads)
    torch.manual_seed(17)
    random.seed(17)
    torch.use_deterministic_algorithms(True)
    manifest = load_manifest(manifest_path)
    train = [s for s in manifest.samples if s.split == "train"]
    val = [s for s in manifest.samples if s.split == "validation"]
    detector = TorchDetector(checkpoint)
    detector.model.roi_heads.score_thresh = 0.005
    detector.manifest["candidate_score_floor"] = 0.005
    tensors = {}
    audit = []
    for s in train + val:
        p = resolve_image(manifest_path, s)
        with NativeRaster(p, s.source) as raster:
            pixels, valid = raster.read((0, 0, s.source.width, s.source.height), 512)
        with Image.open(p) as image:
            original = np.array(image.convert("RGB")).transpose(2, 0, 1)
        assert np.array_equal(
            pixels[:, : s.source.height, : s.source.width][:, valid], original[:, valid]
        )
        image = torch.from_numpy(pixels).float() / 255.0
        tensors[s.image_id] = image
        with torch.no_grad():
            batch, targets = detector.model.transform(
                [image], [{"boxes": torch.tensor([b.xyxy for b in s.boxes])}]
            )
            expected = (
                image - torch.tensor(detector.model.transform.image_mean)[:, None, None]
            ) / torch.tensor(detector.model.transform.image_std)[:, None, None]
            assert torch.allclose(batch.tensors[0], expected)
            assert torch.equal(
                targets[0]["boxes"], torch.tensor([b.xyxy for b in s.boxes])
            )
        official = "train" if s.split != "test" else "test"
        index = json.loads(
            (
                manifest_path.parent
                / "metadata"
                / f"instances_{official}_aircraft.json"
            ).read_text()
        )
        i = next(i for i in index["images"] if i["file_name"] == s.image_id)
        parsed = []
        for a in index["annotations"]:
            if a["image_id"] == i["id"] and not a.get("iscrowd"):
                x, y, w, h = a["bbox"]
                parsed.append(
                    (
                        max(0, x),
                        max(0, y),
                        min(i["width"], x + w),
                        min(i["height"], y + h),
                    )
                )
        assert len(parsed) == len(s.boxes)
        assert np.allclose(parsed, [b.xyxy for b in s.boxes])
        audit.append(
            {
                "image_id": s.image_id,
                "rgb_and_native_grid_equal": True,
                "boxes_match_coco": True,
                "normalization_once_and_boxes_unchanged": True,
            }
        )
    with torch.no_grad():
        batch, _ = detector.model.transform([tensors[train[0].image_id]])
        features = detector.model.backbone(batch.tensors)
    feature_report = {
        key: {"shape": list(value.shape), "stride_x": 512 / value.shape[-1]}
        for key, value in features.items()
    }
    predictions = {}
    for split, samples in (("train", train), ("validation", val)):
        preds = [
            detector.predict((tensors[s.image_id] * 255).byte().numpy())
            for s in samples
        ]
        predictions[split] = preds
    thresholds = [0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5]
    scores = {
        split: {
            str(t): summarize(score_records(samples, predictions[split], t))
            for t in thresholds
        }
        for split, samples in (("train", train), ("validation", val))
    }
    # One sparse and one crowded known training image. No validation/test updates.
    chosen = [
        next(s for s in train if len(s.boxes) == 1),
        next(s for s in train if 5 <= len(s.boxes) <= 10),
    ]
    model = detector.model
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )
    history = []
    start = time.perf_counter()
    for step in range(steps + 1):
        if step % 20 == 0 or step == steps:
            model.eval()
            preds = [
                detector.predict((tensors[s.image_id] * 255).byte().numpy())
                for s in chosen
            ]
            row = {
                "step": step,
                "elapsed_seconds": time.perf_counter() - start,
                "metrics_at_0.5": summarize(score_records(chosen, preds, 0.5)),
                "metrics_at_0.05": summarize(score_records(chosen, preds, 0.05)),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            m = row["metrics_at_0.5"]
            if step and (m["precision"] or 0) >= 0.9 and (m["recall"] or 0) >= 0.9:
                break
        if step == steps:
            break
        model.train()
        model.backbone.eval()
        s = chosen[step % len(chosen)]
        image = tensors[s.image_id]
        target = {
            "boxes": torch.tensor([b.xyxy for b in s.boxes], dtype=torch.float32),
            "labels": torch.ones(len(s.boxes), dtype=torch.int64),
        }
        loss = sum(model([image], [target]).values())
        if not torch.isfinite(loss):
            raise ValueError("nonfinite overfit loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    report = {
        "original_checkpoint_sha256": sha256_file(checkpoint),
        "manifest_sha256": sha256_file(manifest_path),
        "audit": audit,
        "feature_maps": feature_report,
        "anchor_sizes": [list(v) for v in model.rpn.anchor_generator.sizes],
        "score_floor": 0.005,
        "threshold_diagnosis_train_and_validation_only": scores,
        "overfit": {
            "images": [s.image_id for s in chosen],
            "labels": sum(len(s.boxes) for s in chosen),
            "optimizer": "original SGD 0.005 momentum 0.9; backbone frozen",
            "history": history,
            "limited_to_original_training_images": True,
        },
    }
    (outdir / "diagnosis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
