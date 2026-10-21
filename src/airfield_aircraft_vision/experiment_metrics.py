"""AW-01B validation-only operating-point selection and detection/count strata."""

from __future__ import annotations
import math
from collections import defaultdict
from .diagnostics import score_records
from .evaluation import summarize, match_boxes

THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def ap50(samples, predictions):
    total = sum(len(s.boxes) for s in samples)
    if not total:
        return None
    ranked = []
    for image_index, (s, preds) in enumerate(zip(samples, predictions)):
        matches = match_boxes(preds, s.boxes, s.ignored_boxes)
        tp = {m["prediction"] for m in matches["true_positives"]}
        ignored = set(matches["ignored_predictions"])
        ranked.extend(
            (p.score, image_index, j, j in tp)
            for j, p in enumerate(preds)
            if j not in ignored
        )
    ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
    correct = 0
    points = []
    for n, row in enumerate(ranked, 1):
        correct += row[3]
        points.append((correct / total, correct / n))
    return (
        sum(max((p for r, p in points if r >= i / 100), default=0) for i in range(101))
        / 101
    )


def choose_threshold(samples, predictions, *, split):
    if split != "validation" or any(s.split != "validation" for s in samples):
        raise ValueError("operating-point selection is restricted to validation")
    if not samples or not sum(len(s.boxes) for s in samples):
        raise ValueError("positive validation labels required")
    grid = [
        {"threshold": t, "metrics": summarize(score_records(samples, predictions, t))}
        for t in THRESHOLDS
    ]
    best = max(
        grid,
        key=lambda r: (
            r["metrics"]["f1"] or 0,
            -r["metrics"]["count_mae"],
            r["threshold"],
        ),
    )
    return {
        "selected_threshold": best["threshold"],
        "metrics": best["metrics"],
        "grid": grid,
        "selection": "maximum validation F1, then lower count MAE, then higher threshold",
    }


def size_bin(box):
    x1, y1, x2, y2 = box.xyxy
    area = (x2 - x1) * (y2 - y1)
    return (
        "small_lt32px"
        if area < 32**2
        else ("medium_32to96px" if area < 96**2 else "large_ge96px")
    )


def evaluate_predictions(samples, predictions, threshold, weather_by_scene=None):
    records = score_records(samples, predictions, threshold)
    groups = defaultdict(list)
    weather = defaultdict(list)
    bins = {
        label: {
            "truth": 0,
            "matched_truth": 0,
            "predicted": 0,
            "absolute_count_error": 0,
        }
        for label in ("small_lt32px", "medium_32to96px", "large_ge96px")
    }
    for sample, preds, record in zip(samples, predictions, records):
        record.update(
            airfield=sample.source.airfield_group,
            source_sha256=sample.source.sha256,
            timestamp_precision=sample.source.timestamp_precision,
            timestamp_timezone_basis=sample.source.timestamp_timezone_basis,
        )
        groups[sample.source.airfield_group].append(record)
        category = (weather_by_scene or {}).get(sample.source.scene_id, "unknown")
        weather[category].append(record)
        chosen = [p for p in preds if p.score >= threshold]
        matched = {m["truth"] for m in record["matches"]["true_positives"]}
        ignored = set(record["matches"]["ignored_predictions"])
        for label, stats in bins.items():
            truth = sum(size_bin(b) == label for b in sample.boxes)
            predicted = sum(
                size_bin(b) == label for i, b in enumerate(chosen) if i not in ignored
            )
            stats["truth"] += truth
            stats["predicted"] += predicted
            stats["matched_truth"] += sum(
                size_bin(b) == label and i in matched
                for i, b in enumerate(sample.boxes)
            )
            stats["absolute_count_error"] += abs(predicted - truth)
    for stats in bins.values():
        stats["recall"] = (
            stats["matched_truth"] / stats["truth"] if stats["truth"] else None
        )
        stats["count_mae"] = (
            stats["absolute_count_error"] / len(samples) if samples else None
        )
    return {
        "metrics": dict(summarize(records), ap50=ap50(samples, predictions)),
        "per_airfield": {g: summarize(rows) for g, rows in groups.items()},
        "by_dataset_weather": {g: summarize(rows) for g, rows in weather.items()},
        "by_bbox_pixel_size": bins,
        "records": records,
        "strata_notes": [
            "Size bins use source-pixel bounding-box area, not aircraft type.",
            "Predicted-size count errors can include localization crossing a size-bin boundary.",
            "Dataset weather is coarse metadata; actual AOI quality remains unknown.",
            "AP50 is 101-point interpolated AP at IoU=0.5 with candidate score floor 0.005.",
        ],
    }
