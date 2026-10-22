"""Validation-only score reliability; no test-based threshold or model selection."""

from __future__ import annotations
import math
from collections import defaultdict
from .evaluation import match_boxes

TEMPERATURES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def scaled_score(score, temperature=1.0):
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("finite confidence in [0,1] required")
    if not math.isfinite(temperature) or not 0.1 <= temperature <= 10:
        raise ValueError("bounded positive temperature required")
    p = min(1 - 1e-8, max(1e-8, score))
    logit = math.log(p / (1 - p)) / temperature
    return min(1 - 1e-12, max(1e-12, 1 / (1 + math.exp(-logit))))


def candidate_outcomes(samples, predictions, *, split):
    if split != "validation" or any(s.split != "validation" for s in samples):
        raise ValueError("calibration is restricted to validation")
    if len(samples) != len(predictions) or not samples:
        raise ValueError("nonempty aligned samples/predictions required")
    rows = []
    for sample, boxes in zip(samples, predictions):
        if not sample.source.airfield_group:
            raise ValueError("airfield groups required")
        matches = match_boxes(boxes, sample.boxes, sample.ignored_boxes)
        correct = {m["prediction"] for m in matches["true_positives"]}
        ignored = set(matches["ignored_predictions"])
        for index, box in enumerate(boxes):
            if index in ignored:
                continue
            rows.append(
                dict(
                    image_id=sample.image_id,
                    airfield=sample.source.airfield_group,
                    confidence=box.score,
                    correct=int(index in correct),
                    xyxy=list(box.xyxy),
                )
            )
    return rows


def reliability(rows, temperature=1.0, *, bins=10, minimum_score=0.0):
    if type(bins) is not int or not 2 <= bins <= 50:
        raise ValueError("bin count must be 2..50")
    if not math.isfinite(minimum_score) or not 0 <= minimum_score <= 1:
        raise ValueError("minimum score must be [0,1]")
    for row in rows:
        if not math.isfinite(row["confidence"]) or not 0 <= row["confidence"] <= 1:
            raise ValueError("finite confidence in [0,1] required")
    selected = [r for r in rows if r["confidence"] >= minimum_score]
    buckets = defaultdict(list)
    for r in selected:
        if r["correct"] not in (0, 1):
            raise ValueError("binary correctness required")
        p = scaled_score(r["confidence"], temperature)
        buckets[min(bins - 1, int(p * bins))].append((p, r["correct"]))
    n = len(selected)
    summary = []
    for i in range(bins):
        values = buckets[i]
        confidence = sum(p for p, y in values) / len(values) if values else None
        accuracy = sum(y for p, y in values) / len(values) if values else None
        summary.append(
            dict(
                lower=i / bins,
                upper=(i + 1) / bins,
                count=len(values),
                mean_confidence=confidence,
                empirical_precision=accuracy,
            )
        )
    brier = nll = 0.0
    for values in buckets.values():
        for p, y in values:
            brier += (p - y) ** 2
            nll -= y * math.log(p) + (1 - y) * math.log1p(-p)
    return dict(
        candidates=n,
        correct=sum(r["correct"] for r in selected),
        brier=brier / n if n else None,
        nll=nll / n if n else None,
        ece=(
            sum(
                b["count"] * abs(b["mean_confidence"] - b["empirical_precision"])
                for b in summary
                if b["count"]
            )
            / n
            if n
            else None
        ),
        bins=summary,
        minimum_raw_score=minimum_score,
        temperature=temperature,
    )


def choose_temperature(rows):
    if not rows or len({r["correct"] for r in rows}) < 2:
        raise ValueError("both matched and unmatched validation candidates required")
    grid = [dict(temperature=t, metrics=reliability(rows, t)) for t in TEMPERATURES]
    best = min(
        grid, key=lambda r: (r["metrics"]["nll"], abs(math.log(r["temperature"])))
    )
    return best["temperature"], grid


def evaluate_calibration(
    samples, predictions, *, split, checkpoint_sha256, threshold=0.7
):
    if len(checkpoint_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in checkpoint_sha256
    ):
        raise ValueError("checkpoint SHA256 required")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite [0,1]")
    rows = candidate_outcomes(samples, predictions, split=split)
    temperature, grid = choose_temperature(rows)
    groups = sorted({r["airfield"] for r in rows})
    folds = []
    heldout_raw = []
    heldout_scaled = []
    for group in groups:
        train = [r for r in rows if r["airfield"] != group]
        held = [r for r in rows if r["airfield"] == group]
        if not train or len({r["correct"] for r in train}) < 2:
            folds.append(
                dict(airfield=group, status="unavailable_one_class_or_no_training")
            )
            continue
        t, _ = choose_temperature(train)
        folds.append(
            dict(
                airfield=group,
                status="evaluated",
                temperature=t,
                raw=reliability(held),
                scaled=reliability(held, t),
            )
        )
        heldout_raw.extend(held)
        heldout_scaled.extend(
            dict(r, confidence=scaled_score(r["confidence"], t)) for r in held
        )
    # Scaled probabilities are already produced; T=1 preserves them.
    crossval = dict(
        folds=folds,
        raw=reliability(heldout_raw),
        scaled=reliability(heldout_scaled),
        interpretation="Leave-one-airfield-out validation diagnostic; not independent test calibration.",
    )
    sensors = sorted({s.source.sensor_id for s in samples})
    gsds = [s.source.native_gsd_m for s in samples]
    return dict(
        schema_version="aw01d-calibration-1",
        selection_split="validation",
        checkpoint_sha256=checkpoint_sha256,
        raw_acceptance_threshold=threshold,
        acceptance_rule_changed=False,
        uncertainty_floor=0.2,
        temperature=temperature,
        temperature_grid=grid,
        raw=reliability(rows),
        temperature_scaled=reliability(rows, temperature),
        raw_accepted=reliability(rows, minimum_score=threshold),
        scaled_accepted=reliability(rows, temperature, minimum_score=threshold),
        grouped_crossvalidation=crossval,
        validation_support=dict(
            images=len(samples),
            airfields=groups,
            sensor_ids=sensors,
            native_gsd_bounds_m=[min(gsds), max(gsds)],
        ),
        predictions_event="post-NMS candidate matches one frozen label at IoU>=0.5; ignored regions excluded",
        candidate_floor=0.005,
        nms_iou=0.4,
        tile_size=512,
        overlap=128,
        status="experimental_validation_diagnostic",
        calibration_applied=False,
        adoption_rule="No automatic deployment: report accepted-set and group-held-out reliability separately; raw acceptance remains unchanged",
        limitations=[
            "Correctness is conditional on a proposed, post-NMS candidate and frozen labels.",
            "Scores do not measure missed aircraft, inventory completeness, or unanalysable areas.",
            "One-dimensional score calibration is not spatial/size-aware detector calibration.",
            "Whole-validation fit is optimistic; grouped folds are descriptive and few.",
            "No probability calibration transfer to new sensors, resolutions or scene distributions is established.",
            "Raw .7 acceptance and .2 review floor remain predeclared; independent labels never choose them.",
        ],
    )


def collect_calibration(manifest_path, checkpoint, outdir, *, threads=4):
    import json
    from pathlib import Path
    import torch
    from .torch_backend import TorchDetector
    from .improved_training import collect_predictions
    from .preprocessing import sha256_file

    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("bounded threads required")
    outdir = Path(outdir)
    if outdir.exists():
        raise ValueError("fresh calibration output directory required")
    torch.set_num_threads(threads)
    detector = TorchDetector(Path(checkpoint), candidate_floor=0.005)
    manifest, samples, predictions = collect_predictions(
        Path(manifest_path), detector, "validation"
    )
    report = evaluate_calibration(
        samples,
        predictions,
        split="validation",
        checkpoint_sha256=sha256_file(Path(checkpoint)),
        threshold=detector.manifest["selected_threshold"],
    )
    report["manifest_sha256"] = sha256_file(Path(manifest_path))
    outdir.mkdir(parents=True)
    (outdir / "calibration.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    (outdir / "raw-validation-predictions.json").write_text(
        json.dumps(
            dict(
                image_ids=[s.image_id for s in samples],
                predictions=[[b.model_dump() for b in bs] for bs in predictions],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            dict(
                raw=report["raw_accepted"],
                scaled=report["scaled_accepted"],
                temperature=report["temperature"],
            )
        ),
        flush=True,
    )
    return report
