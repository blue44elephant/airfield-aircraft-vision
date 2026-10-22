"""AW-01D commands: bounded research inference, calibration and frozen evaluation."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from .optical_contracts import RasterSource
from .robust_contracts import CoverageSummary, RobustSceneResult
from .robust_inference import infer_robust_scene, scene_feature_collection
from .preprocessing import sha256_file


def failed_result(source, reason, model=None):
    return RobustSceneResult(
        status="failed",
        source=source,
        model=model or {},
        preprocessing={"profile": "native-rgb-v1", "source_verified": False},
        observed_count=None,
        accepted_count=0,
        uncertain_count=0,
        rejected_count=0,
        detections=[],
        coverage=CoverageSummary(
            total_pixels=source.width * source.height,
            inspected_pixels=0,
            analyzable_pixels=0,
            inspected_fraction=0,
            analyzable_fraction=0,
        ),
        completed_tiles=0,
        planned_tiles=0,
        elapsed_seconds=0,
        analysis_failure_reason=reason,
        warnings=[
            "No count available; model/input initialization failed. Analyst verification required."
        ],
    )


def infer_safe(
    image, source, checkpoint, *, detector_factory=None, expected_sha256=None, **kwargs
):
    model = {"checkpoint_sha256": None}
    try:
        if expected_sha256:
            actual = sha256_file(Path(checkpoint))
            if actual != expected_sha256:
                return failed_result(source, "checkpoint_sha256_mismatch")
            model["checkpoint_sha256"] = actual
        if detector_factory is None:
            from .torch_backend import TorchDetector

            detector_factory = TorchDetector
        detector = detector_factory(Path(checkpoint), candidate_floor=0.005)
    except Exception as exc:
        return failed_result(source, "model_unavailable:" + type(exc).__name__, model)
    try:
        return infer_robust_scene(Path(image), source, detector, **kwargs)
    except Exception as exc:
        return failed_result(
            source,
            "invalid_request_or_initialization:" + type(exc).__name__,
            dict(getattr(detector, "manifest", model)),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="AW-01D experimental robustness and safe scene counting"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--manifest", type=Path, required=True)
    calibration.add_argument("--checkpoint", type=Path, required=True)
    calibration.add_argument("--out", type=Path, required=True)
    calibration.add_argument("--threads", type=int, default=4)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--cohorts", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--calibration", type=Path)
    evaluate.add_argument("--threads", type=int, default=4)
    infer = commands.add_parser("infer")
    for name in ("image", "source", "checkpoint", "output"):
        infer.add_argument("--" + name, type=Path, required=True)
    infer.add_argument("--geojson", type=Path)
    infer.add_argument("--expected-sha256")
    infer.add_argument("--mask-sha256")
    infer.add_argument(
        "--dedup", choices=["iou", "boundary_containment"], default="iou"
    )
    infer.add_argument("--max-tiles", type=int, default=256)
    infer.add_argument("--max-pixels", type=int, default=25_000_000)
    infer.add_argument("--max-seconds", type=float, default=300)
    infer.add_argument("--threads", type=int, default=4)
    infer.add_argument("--allow-unknown-quality", action="store_true")
    infer.add_argument("--stop-file", type=Path)
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        from .confidence_calibration import collect_calibration

        collect_calibration(
            args.manifest, args.checkpoint, args.out, threads=args.threads
        )
    elif args.command == "evaluate":
        from .robustness_evaluation import run_evaluation

        run_evaluation(
            json.loads(args.cohorts.read_text()),
            args.checkpoint,
            args.out,
            calibration_path=args.calibration,
            threads=args.threads,
        )
    else:
        protected = [
            args.image,
            args.source,
            args.checkpoint,
            Path(str(args.image) + ".msk"),
            Path(str(args.image) + ".aux.xml"),
        ]
        outputs = [args.output] + ([args.geojson] if args.geojson else [])

        def aliases(left, right):
            return left.resolve() == right.resolve() or (
                left.exists() and right.exists() and left.samefile(right)
            )

        if any(aliases(out, original) for out in outputs for original in protected):
            parser.error(
                "output paths must not overwrite source imagery, metadata, masks or checkpoints"
            )
        if len(outputs) == 2 and aliases(*outputs):
            parser.error("JSON and GeoJSON outputs must use distinct paths")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            source = RasterSource.model_validate_json(args.source.read_text())
        except Exception as exc:
            payload = dict(
                contract_version="aw01d-error-1",
                status="failed",
                observed_count=None,
                scene_total=None,
                analysis_failure_reason="invalid_source_metadata:" + type(exc).__name__,
            )
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps(payload))
            return 2
        if not 1 <= args.threads <= 8:
            result = failed_result(source, "invalid_cpu_thread_bound")
        else:
            try:
                import torch

                torch.set_num_threads(args.threads)
                result = infer_safe(
                    args.image,
                    source,
                    args.checkpoint,
                    expected_sha256=args.expected_sha256,
                    expected_mask_sha256=args.mask_sha256,
                    allow_unknown_quality=args.allow_unknown_quality,
                    dedup_mode=args.dedup,
                    max_tiles=args.max_tiles,
                    max_pixels=args.max_pixels,
                    max_seconds=args.max_seconds,
                    cancelled=(
                        (lambda: args.stop_file.exists()) if args.stop_file else None
                    ),
                )
            except Exception as exc:
                result = failed_result(
                    source, "invalid_request_or_initialization:" + type(exc).__name__
                )
        args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        if args.geojson:
            args.geojson.parent.mkdir(parents=True, exist_ok=True)
            args.geojson.write_text(
                json.dumps(scene_feature_collection(result), indent=2), encoding="utf-8"
            )
        print(
            json.dumps(
                dict(
                    output=str(args.output),
                    status=result.status,
                    observed_count=result.observed_count,
                    analyst_verification_required=True,
                )
            )
        )
        return 0 if result.status in ("complete", "partial") else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
