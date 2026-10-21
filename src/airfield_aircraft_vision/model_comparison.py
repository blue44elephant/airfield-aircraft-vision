"""AW-01B locked validation selection and independent model comparisons."""

from __future__ import annotations
import json
from pathlib import Path
from .torch_backend import TorchDetector
from .dataset import load_manifest, resolve_image
from .preprocessing import sha256_file
from .improved_training import collect_predictions
from .experiment_metrics import choose_threshold, evaluate_predictions
from .inference.optical_high_resolution import infer_scene
from .workflows import draw_example


def compare(
    manifest_path: Path,
    original_manifest: Path,
    original_checkpoint: Path,
    candidate_dirs: list[Path],
    outdir: Path,
    *,
    threads=4,
):
    import torch

    if not candidate_dirs:
        raise ValueError("at least one candidate checkpoint required")
    torch.set_num_threads(threads)
    if outdir.exists():
        raise ValueError("fresh comparison directory required")
    outdir.mkdir(parents=True)
    manifest = load_manifest(manifest_path)
    candidates = []
    for folder in candidate_dirs:
        detector = TorchDetector(folder / "best.pt", candidate_floor=0.005)
        info = detector.manifest
        if info.get("selection_split") != "validation":
            raise ValueError("candidate lacks validation-only selection")
        if info["training_manifest_sha256"] != sha256_file(manifest_path):
            raise ValueError(
                "candidate was not trained/selected against this frozen manifest"
            )
        candidates.append(
            {
                "checkpoint": str(folder / "best.pt"),
                "sha256": info["checkpoint_sha256"],
                "threshold": info["selected_threshold"],
                "epoch": info["selected_epoch"],
                "validation_metrics": info["selected_validation_metrics"],
                "configuration": info["configuration"],
            }
        )
    best = max(
        candidates,
        key=lambda c: (
            c["validation_metrics"]["f1"] or 0,
            -c["validation_metrics"]["count_mae"],
            c["threshold"],
        ),
    )
    del detector
    original_detector = TorchDetector(original_checkpoint, candidate_floor=0.005)
    _, validation, original_val_preds = collect_predictions(
        manifest_path, original_detector, "validation"
    )
    original_selection = choose_threshold(
        validation, original_val_preds, split="validation"
    )
    lock = {
        "selection_split": "validation",
        "candidate_checkpoints": candidates,
        "selected": best,
        "original_checkpoint_sha256": sha256_file(original_checkpoint),
        "original_validation_threshold": original_selection["selected_threshold"],
        "manifest_sha256": sha256_file(manifest_path),
        "original_manifest_sha256": sha256_file(original_manifest),
        "test_policy": "Fresh and legacy test labels evaluated after this lock; never select from test metrics.",
    }
    lock_path = outdir / "selection-lock.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    report = {
        "selection": lock,
        "selection_lock_sha256": sha256_file(lock_path),
        "original_validation_selection": original_selection,
        "candidate_validation": {
            c["checkpoint"]: c["validation_metrics"] for c in candidates
        },
        "cohorts": {},
    }
    for cohort, path in (
        ("fresh_test", manifest_path),
        ("legacy_test", original_manifest),
    ):
        data = load_manifest(path)
        samples = [s for s in data.samples if s.split == "test"]
        weather = data.preparation.get("metadata_weather_by_scene", {})
        if not weather:
            import csv

            rows = list(
                csv.DictReader(
                    (path.parent / "metadata/RarePlanes_Public_Metadata.csv")
                    .read_text(encoding="utf-8-sig")
                    .splitlines()
                )
            )
            weather = {r["image_id"]: r["Weather"] for r in rows}
        _, _, original_preds = collect_predictions(path, original_detector, "test")
        improved = TorchDetector(Path(best["checkpoint"]), candidate_floor=0.005)
        _, _, improved_preds = collect_predictions(path, improved, "test")
        models = {
            "zero_detection_baseline": evaluate_predictions(
                samples, [[] for _ in samples], 0.5, weather
            ),
            "aw01_at_original_0.5": evaluate_predictions(
                samples, original_preds, 0.5, weather
            ),
            "aw01_validation_selected_threshold": evaluate_predictions(
                samples,
                original_preds,
                original_selection["selected_threshold"],
                weather,
            ),
            "aw01b_validation_selected": evaluate_predictions(
                samples, improved_preds, best["threshold"], weather
            ),
        }
        report["cohorts"][cohort] = {
            "images": len(samples),
            "airfields": sorted({s.source.airfield_group for s in samples}),
            "manifest_sha256": sha256_file(path),
            "models": models,
        }
        raw = {
            "image_ids": [s.image_id for s in samples],
            "original_predictions": [
                [p.model_dump() for p in ps] for ps in original_preds
            ],
            "improved_predictions": [
                [p.model_dump() for p in ps] for ps in improved_preds
            ],
        }
        (outdir / (cohort + "-raw-predictions.json")).write_text(
            json.dumps(raw, indent=2), encoding="utf-8"
        )
        images_dir = outdir / cohort
        images_dir.mkdir()
        for sample, record in zip(
            samples, models["aw01b_validation_selected"]["records"]
        ):
            result = infer_scene(
                resolve_image(path, sample),
                sample.source,
                improved,
                confidence=best["threshold"],
                allow_unknown_quality=True,
            )
            (images_dir / (sample.image_id + ".json")).write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
            draw_example(
                resolve_image(path, sample),
                result,
                sample,
                record,
                images_dir / (sample.image_id + ".png"),
            )
        del improved
        print(
            json.dumps(
                {
                    "cohort": cohort,
                    "models": {name: r["metrics"] for name, r in models.items()},
                }
            ),
            flush=True,
        )
    report["limitations"] = [
        "Small positive-only sample; no inventory or empty-scene false-alarm validation.",
        "Legacy test was exposed in AW-01; fresh test is the confirmatory cohort.",
        "Weather strata are scene metadata, not AOI quality measurements.",
        "AP50 uses candidate floor 0.005 and fixed proposal limits; not COCO multi-IoU AP.",
        "Thresholds, epochs and candidates locked on validation before test evaluation.",
    ]
    (outdir / "comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (outdir / "ATTRIBUTION.txt").write_text(
        "RarePlanes Dataset, June 2020. J. Shermeyer, T. Hossler, A. Van Etten, D. Hogan, R. Lewis, D. Kim. In-Q-Tel / CosmiQ Works and AI.Reverie.\n"
        "https://www.iqt.org/library/the-rareplanes-dataset\nhttps://creativecommons.org/licenses/by-sa/4.0/\n"
        "Changes: native-size detection/ground-truth overlays. These derivatives retain CC BY-SA 4.0. No endorsement or warranty.\n",
        encoding="utf-8",
    )
    return report
