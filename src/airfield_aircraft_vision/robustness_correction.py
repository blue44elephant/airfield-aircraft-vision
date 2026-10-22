"""One fixed, validation-only AW-01D correction pilot; no automatic promotion."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from .dataset import load_manifest
from .preprocessing import sha256_file
from .workflows import assert_holdout
from .generalization_training import train_generalization, validate_training_sources
from .improved_training import collect_predictions
from .experiment_metrics import evaluate_predictions

BASELINE_SHA256 = "d2aecd3624648981f35def1fff7bb36789f9635546f4f848c719c48e647eda42"
CONFIGURATION = {
    "architecture": "fasterrcnn_resnet50_fpn",
    "epochs": 1,
    "lr": 0.00005,
    "threads": 4,
    "max_seconds": 90,
    "seed": 91,
}
FIXED_THRESHOLD = 0.7
LINEAGE_KEYS = ("training_airfields", "training_captures", "training_source_checksums")


def validate_pilot_manifest(manifest):
    if any(sample.split == "test" for sample in manifest.samples):
        raise ValueError("correction pilot forbids test samples")
    train = [s for s in manifest.samples if s.split == "train"]
    validation = [s for s in manifest.samples if s.split == "validation"]
    if not 1 <= len(train) <= 64 or not 1 <= len(validation) <= 32:
        raise ValueError("bounded nonempty train/validation sets required")
    if not any(s.boxes for s in train) or not any(s.boxes for s in validation):
        raise ValueError("positive train/validation examples required")
    if any(s.ignored_boxes for s in train):
        raise ValueError("ignored training regions require explicit exclusion")
    if any(s.source.width > 512 or s.source.height > 512 for s in manifest.samples):
        raise ValueError("native correction tiles must be <=512")
    validate_training_sources(manifest.samples)
    return train, validation


def union_training_lineage(parent, child, parent_sha256):
    result = dict(child)
    for key in LINEAGE_KEYS:
        if not parent.get(key) or not child.get(key):
            raise ValueError("both checkpoints require auditable training lineage")
        result[key] = sorted(set(parent[key]) | set(child[key]))
    result["parent_checkpoint_sha256"] = parent_sha256
    result["ancestry_checkpoint_sha256"] = list(
        dict.fromkeys(
            list(parent.get("ancestry_checkpoint_sha256", [])) + [parent_sha256]
        )
    )
    result["ancestry_validation_airfields"] = sorted(
        set(
            parent.get("validation_airfields", [])
            + parent.get("ancestry_validation_airfields", [])
        )
    )
    result["operating_point_policy"] = "predeclared AW01C raw score threshold .7"
    result["training_helper_validation_selection"] = {
        "threshold": child.get("selected_threshold"),
        "metrics": child.get("selected_validation_metrics"),
    }
    result["selected_threshold"] = FIXED_THRESHOLD
    return result


def _fixed_validation(manifest_path, detector):
    _, samples, predictions = collect_predictions(manifest_path, detector, "validation")
    return evaluate_predictions(samples, predictions, FIXED_THRESHOLD)


def run_pilot(manifest_path, checkpoint, outdir):
    import torch
    import psutil
    from .torch_backend import TorchDetector

    manifest_path, checkpoint, outdir = map(Path, (manifest_path, checkpoint, outdir))
    if outdir.exists():
        raise ValueError("fresh isolated pilot directory required")
    if sha256_file(checkpoint) != BASELINE_SHA256:
        raise ValueError("preserved AW01C baseline checksum differs")
    manifest = load_manifest(manifest_path)
    train, validation = validate_pilot_manifest(manifest)
    manifest_sha = sha256_file(manifest_path)
    lock = {
        "baseline_checkpoint_sha256": BASELINE_SHA256,
        "manifest_sha256": manifest_sha,
        "configuration": dict(CONFIGURATION),
        "fixed_raw_threshold": FIXED_THRESHOLD,
        "training_tiles": len(train),
        "validation_tiles": len(validation),
        "test_tiles": 0,
        "automatic_promotion": False,
        "policy": "One predeclared pilot; validation only. No independent evaluation, search or model promotion.",
    }
    outdir.mkdir(parents=True)
    (outdir / "pilot-lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    torch.set_num_threads(CONFIGURATION["threads"])
    process = psutil.Process()
    start = time.perf_counter()
    cpu_start = process.cpu_times()
    baseline = TorchDetector(checkpoint, candidate_floor=0.005)
    parent = dict(baseline.manifest)
    assert_holdout(validation, parent)
    baseline_report = _fixed_validation(manifest_path, baseline)
    (outdir / "baseline-validation.json").write_text(
        json.dumps(baseline_report, indent=2, allow_nan=False), encoding="utf-8"
    )
    del baseline
    gc.collect()
    if (outdir / "STOP").exists():
        raise ValueError("pilot cancelled before training")
    if sha256_file(manifest_path) != manifest_sha:
        raise ValueError("locked correction manifest changed")
    train_generalization(
        manifest_path,
        checkpoint,
        outdir / "training",
        initial_checkpoint=checkpoint,
        **CONFIGURATION,
    )
    raw = torch.load(
        outdir / "training" / "best.pt", map_location="cpu", weights_only=True
    )
    raw["manifest"] = union_training_lineage(parent, raw["manifest"], BASELINE_SHA256)
    raw["manifest"]["aw01d_correction_configuration"] = dict(CONFIGURATION)
    raw["manifest"]["selected_validation_metrics"] = None
    assert_holdout(validation, raw["manifest"])
    candidate_path = outdir / "candidate.pt"
    torch.save(raw, candidate_path)
    del raw
    gc.collect()
    candidate = TorchDetector(candidate_path, candidate_floor=0.005)
    candidate_report = _fixed_validation(manifest_path, candidate)
    del candidate
    gc.collect()
    # Record fixed-threshold metrics in the new artifact only. The original
    # baseline and the raw training-helper output are never rewritten.
    raw = torch.load(candidate_path, map_location="cpu", weights_only=True)
    raw["manifest"]["selected_validation_metrics"] = candidate_report["metrics"]
    torch.save(raw, candidate_path)
    del raw
    if (
        sha256_file(checkpoint) != BASELINE_SHA256
        or sha256_file(manifest_path) != manifest_sha
    ):
        raise ValueError("preserved baseline or locked manifest changed")
    (outdir / "candidate-validation.json").write_text(
        json.dumps(candidate_report, indent=2, allow_nan=False), encoding="utf-8"
    )
    cpu_end = process.cpu_times()
    comparison = {
        "status": "validation_pilot_only",
        "fixed_raw_threshold": FIXED_THRESHOLD,
        "baseline_checkpoint_sha256": BASELINE_SHA256,
        "candidate_checkpoint_sha256": sha256_file(candidate_path),
        "baseline": baseline_report["metrics"],
        "candidate": candidate_report["metrics"],
        "automatic_promotion": False,
        "independent_test_evaluated": False,
        "limitations": [
            "Small targeted validation cohort; no independent generalization conclusion.",
            "90 seconds bounds training updates between steps, excluding baseline/candidate validation and initialization.",
            "Create training/STOP for cooperative cancellation between updates.",
        ],
    }
    resources = {
        "elapsed_seconds": time.perf_counter() - start,
        "cpu_user_seconds": cpu_end.user - cpu_start.user,
        "cpu_system_seconds": cpu_end.system - cpu_start.system,
        "peak_rss_mib": max(
            process.memory_info().rss, getattr(process.memory_info(), "peak_wset", 0)
        )
        / 1024**2,
        "configuration": dict(CONFIGURATION),
        "training_resources": json.loads(
            (outdir / "training" / "resources.json").read_text()
        ),
    }
    (outdir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8"
    )
    (outdir / "resources.json").write_text(
        json.dumps(resources, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(comparison), flush=True)
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run_pilot(args.manifest, args.checkpoint, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
