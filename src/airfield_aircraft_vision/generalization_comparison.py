"""AW-01C frozen validation selection and one-pass independent comparison.

No imagery downloads or model/threshold selection from test records occurs.
Detection examples reuse the exact predictions used by the metric calculation.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .dataset import load_manifest, resolve_image
from .generalization_metrics import evaluate_generalization, reviewed_negative_ids
from .improved_training import collect_predictions
from .preprocessing import sha256_file
from .torch_backend import TorchDetector


def _set_cpu_threads(threads):
    import torch

    torch.set_num_threads(threads)


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _finite_number(value, name, *, lower=0, upper=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < lower
        or (upper is not None and value > upper)
    ):
        raise ValueError(f"invalid {name}")
    return value


def _checkpoint_record(path, manifest_sha256=None):
    detector = TorchDetector(path, candidate_floor=0.005)
    info = detector.manifest
    if info.get("selection_split") != "validation":
        raise ValueError("checkpoint lacks validation-only selection")
    if (
        manifest_sha256 is not None
        and info.get("training_manifest_sha256") != manifest_sha256
    ):
        raise ValueError("candidate training manifest checksum differs")
    digest = sha256_file(path)
    if info.get("checkpoint_sha256") != digest:
        raise ValueError("checkpoint checksum differs from loaded detector")
    threshold = _finite_number(
        info.get("selected_threshold"), "selected threshold", upper=1
    )
    metrics = info.get("selected_validation_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("checkpoint lacks selected validation metrics")
    _finite_number(metrics.get("f1"), "validation F1", upper=1)
    _finite_number(metrics.get("count_mae"), "validation count MAE")
    return {
        "checkpoint": str(path),
        "sha256": digest,
        "threshold": threshold,
        "epoch": info.get("selected_epoch"),
        "validation_metrics": metrics,
        "configuration": info.get("configuration", {}),
        "architecture": info.get("architecture"),
        "training_manifest_sha256": info.get("training_manifest_sha256"),
    }


def _draw_cached_predictions(image_path, sample, predictions, record, output):
    """Draw native pixels; no model call, image resize, or source write."""
    from PIL import Image, ImageDraw

    if sha256_file(image_path) != sample.source.sha256:
        raise ValueError("overlay source checksum changed")
    with Image.open(image_path) as original:
        if original.size != (sample.source.width, sample.source.height):
            raise ValueError("overlay source dimensions differ")
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in sample.boxes:
        draw.rectangle(box.xyxy, outline="#36d7e7", width=1)
    for box in sample.ignored_boxes:
        draw.rectangle(box.xyxy, outline="#999999", width=1)
    true_positive = {
        match["prediction"] for match in record["matches"]["true_positives"]
    }
    ignored = set(record["matches"]["ignored_predictions"])
    for index, box in enumerate(predictions):
        color = (
            "#999999"
            if index in ignored
            else ("#52e077" if index in true_positive else "#ff5757")
        )
        draw.rectangle(box.xyxy, outline=color, width=2)
        x, y, _, _ = box.xyxy
        draw.text((max(0, x), max(0, y - 11)), f"{box.score:.2f}", fill=color)
    image.save(output)


def _verify_collected_order(samples, collected):
    if [s.image_id for s in samples] != [s.image_id for s in collected]:
        raise ValueError("collected prediction sample order differs from manifest")


def _collect_once(path, record, expected_samples):
    if sha256_file(Path(record["checkpoint"])) != record["sha256"]:
        raise ValueError("locked checkpoint changed before inference")
    detector = TorchDetector(Path(record["checkpoint"]), candidate_floor=0.005)
    _, samples, predictions = collect_predictions(path, detector, "test")
    _verify_collected_order(expected_samples, samples)
    if len(predictions) != len(samples):
        raise ValueError("prediction length differs from selected samples")
    return predictions


def compare(
    manifest_path,
    baseline_checkpoint,
    candidate_dirs,
    outdir,
    regression_manifest=None,
    threads=4,
):
    """Freeze selection before test access; evaluate each model once per cohort.

    The AW-01B baseline retains its stored validation threshold. Candidate ranking
    uses stored validation F1, then lower count MAE, then higher threshold.
    This function must only be run after all experiments have concluded.
    """
    manifest_path = Path(manifest_path)
    baseline_checkpoint = Path(baseline_checkpoint)
    outdir = Path(outdir)
    candidate_dirs = [Path(path) for path in candidate_dirs]
    if outdir.exists():
        raise ValueError("fresh comparison directory required; never repeat in place")
    if not candidate_dirs or len(candidate_dirs) > 8:
        raise ValueError("one to eight predeclared candidates required")
    if (
        isinstance(threads, bool)
        or not isinstance(threads, int)
        or not 1 <= threads <= 16
    ):
        raise ValueError("bounded CPU thread count required")
    manifest_digest = sha256_file(manifest_path)
    candidates = [
        _checkpoint_record(folder / "best.pt", manifest_digest)
        for folder in candidate_dirs
    ]
    baseline = _checkpoint_record(baseline_checkpoint)
    selected = max(
        candidates,
        key=lambda c: (
            c["validation_metrics"]["f1"],
            -c["validation_metrics"]["count_mae"],
            c["threshold"],
        ),
    )
    regression_path = Path(regression_manifest) if regression_manifest else None
    if regression_path and regression_path.resolve() == manifest_path.resolve():
        raise ValueError("regression manifest must be a different cohort")
    regression_digest = sha256_file(regression_path) if regression_path else None
    lock = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation",
        "manifest_sha256": manifest_digest,
        "regression_manifest_sha256": regression_digest,
        "candidate_checkpoints": candidates,
        "selected": selected,
        "aw01b_baseline": baseline,
        "selection_rule": "maximum validation F1, lower count MAE, higher threshold",
        "test_policy": (
            "Model, epoch and thresholds frozen before test-manifest parsing or "
            "test predictions. No test selection. Each model inferred once per cohort."
        ),
    }
    outdir.mkdir(parents=True)
    lock_path = outdir / "selection-lock.json"
    _write_json(lock_path, lock)
    lock_digest = sha256_file(lock_path)
    _set_cpu_threads(threads)
    report = {
        "selection": lock,
        "selection_lock_sha256": lock_digest,
        "cohorts": {},
        "limitations": [
            "Small source-correlated cohorts; descriptive results without confidence intervals.",
            "Native source-pixel size bins are not physical aircraft-size or type classes.",
            "Weather metadata is scene-level, not verified tile-level image quality.",
            "Exposed AW-01B data is regression only, not independent confirmatory evidence.",
            "Per-tile counts can overlap geographically and are not an airfield inventory.",
            "Empty-apron false-alarm statistics require separately reviewed negative IDs.",
        ],
    }
    cohorts = [("independent_test", manifest_path, manifest_digest)]
    if regression_path:
        cohorts.append(("aw01b_regression", regression_path, regression_digest))

    for name, path, expected_digest in cohorts:
        if sha256_file(path) != expected_digest:
            raise ValueError("locked evaluation manifest changed")
        data = load_manifest(path)
        samples = [sample for sample in data.samples if sample.split == "test"]
        if not samples:
            raise ValueError("evaluation cohort has no test samples")
        info = data.preparation
        apron_ids, background_ids = reviewed_negative_ids(info, samples)
        kwargs = {
            "weather_by_scene": info.get("metadata_weather_by_scene", {}),
            "geography_by_scene": info.get("geography_by_scene", {}),
            "verified_empty_image_ids": apron_ids,
            "verified_background_image_ids": background_ids,
        }
        selected_predictions = _collect_once(path, selected, samples)
        baseline_predictions = _collect_once(path, baseline, samples)
        models = {
            "zero_detection_baseline": evaluate_generalization(
                samples, [[] for _ in samples], 0.5, **kwargs
            ),
            "aw01b_validation_selected": evaluate_generalization(
                samples, baseline_predictions, baseline["threshold"], **kwargs
            ),
            "aw01c_validation_selected": evaluate_generalization(
                samples, selected_predictions, selected["threshold"], **kwargs
            ),
        }
        source = models["aw01b_validation_selected"]["metrics"]["count_mae"]
        improved = models["aw01c_validation_selected"]["metrics"]["count_mae"]
        report["cohorts"][name] = {
            "role": (
                "confirmatory_independent"
                if name == "independent_test"
                else "previously_exposed_regression"
            ),
            "images": len(samples),
            "airfields": sorted({sample.source.airfield_group for sample in samples}),
            "manifest_sha256": expected_digest,
            "models": models,
            "aw01c_count_mae_reduction_vs_aw01b": (
                (source - improved) / source if source else None
            ),
            "model_collection_calls": {"aw01b": 1, "aw01c": 1},
        }
        _write_json(
            outdir / (name + "-raw-predictions.json"),
            {
                "selection_lock_sha256": lock_digest,
                "image_ids": [sample.image_id for sample in samples],
                "aw01b_predictions": [
                    [box.model_dump() for box in boxes]
                    for boxes in baseline_predictions
                ],
                "aw01c_predictions": [
                    [box.model_dump() for box in boxes]
                    for boxes in selected_predictions
                ],
            },
        )
        if name == "independent_test":
            for label, predictions, threshold in (
                ("aw01b", baseline_predictions, baseline["threshold"]),
                ("aw01c", selected_predictions, selected["threshold"]),
            ):
                model_key = label + "_validation_selected"
                images = outdir / name / label
                images.mkdir(parents=True)
                # Deterministic first 12 IDs, not selected for flattering outcomes.
                indexes = sorted(
                    range(len(samples)), key=lambda i: samples[i].image_id
                )[:12]
                for index in indexes:
                    sample = samples[index]
                    boxes = [
                        box for box in predictions[index] if box.score >= threshold
                    ]
                    record = models[model_key]["records"][index]
                    _draw_cached_predictions(
                        resolve_image(path, sample),
                        sample,
                        boxes,
                        record,
                        images / (Path(sample.image_id).name + ".overlay.png"),
                    )
                _write_json(
                    images / "index.json",
                    {
                        "selection": "first 12 image IDs in lexical order",
                        "images": [
                            {
                                "image_id": samples[i].image_id,
                                "record": models[model_key]["records"][i],
                                "source": samples[i].source.model_dump(mode="json"),
                            }
                            for i in indexes
                        ],
                    },
                )
        _write_json(outdir / (name + "-metrics.json"), report["cohorts"][name])
        print(
            json.dumps(
                {
                    "cohort": name,
                    "models": {key: model["metrics"] for key, model in models.items()},
                }
            ),
            flush=True,
        )
    _write_json(outdir / "comparison.json", report)
    with (outdir / "ATTRIBUTION.txt").open("x", encoding="utf-8") as stream:
        stream.write(
            "RarePlanes Dataset, June 2020. J. Shermeyer, T. Hossler, A. Van Etten, "
            "D. Hogan, R. Lewis, D. Kim. In-Q-Tel / CosmiQ Works and AI.Reverie.\n"
            "https://www.iqt.org/library/the-rareplanes-dataset\n"
            "https://creativecommons.org/licenses/by-sa/4.0/\n"
            "Changes: native-size ground-truth and detection overlays. Derivatives "
            "retain CC BY-SA 4.0. No endorsement or warranty.\n"
            "Cyan=truth, green=matched detection, red=unmatched detection, gray=ignored.\n"
        )
    return report
