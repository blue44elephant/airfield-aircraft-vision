"""Frozen AW-01D robustness evaluation; tile outputs reused across workflow checks."""

from __future__ import annotations
import hashlib, json, math, statistics, time
from pathlib import Path
from .dataset import load_manifest, resolve_image
from .preprocessing import sha256_file
from .diagnostics import prediction_boxes
from .evaluation import match_boxes
from .experiment_metrics import evaluate_predictions
from .generalization_metrics import evaluate_generalization, reviewed_negative_ids
from .inference.optical_high_resolution import infer_scene
from .robust_inference import infer_robust_scene, scene_feature_collection
from .optical_contracts import PixelBox
from .workflows import assert_holdout

AW01C_SHA256 = "d2aecd3624648981f35def1fff7bb36789f9635546f4f848c719c48e647eda42"


class CachedDetector:
    """Cache exact native RGB tensors, not rescaled scenes or geospatial identities."""

    def __init__(self, detector, *, max_calls=None, deadline=None):
        self.detector = detector
        self.max_calls = max_calls
        self.deadline = deadline
        self.manifest = detector.manifest
        self.cache = {}
        self.forward_calls = 0
        self.cache_hits = 0
        self.forward_attempts = 0
        self.stopped_reason = None

    def predict(self, pixels):
        if self.stopped_reason is not None:
            raise RuntimeError("source_backend_stopped:" + self.stopped_reason)
        if self.deadline is not None and time.monotonic() > self.deadline:
            self.stopped_reason = "source_runtime_budget_exceeded"
            raise RuntimeError(self.stopped_reason)
        key = hashlib.sha256(
            str((pixels.shape, str(pixels.dtype))).encode() + pixels.tobytes()
        ).hexdigest()
        if key not in self.cache:
            if self.max_calls is not None and self.forward_attempts >= self.max_calls:
                self.stopped_reason = "source_forward_call_budget_exceeded"
                raise RuntimeError(self.stopped_reason)
            self.forward_attempts += 1
            try:
                result = list(self.detector.predict(pixels))
                boxes = [
                    b if isinstance(b, PixelBox) else PixelBox.model_validate(b)
                    for b in result
                ]
            except Exception as exc:
                self.stopped_reason = "backend_failed:" + type(exc).__name__
                raise
            self.cache[key] = boxes
            self.forward_calls += 1
        else:
            self.cache_hits += 1
        return list(self.cache[key])

    def evidence(self):
        return dict(
            unique_forward_calls=self.forward_calls,
            forward_attempts=self.forward_attempts,
            cache_hits=self.cache_hits,
            stopped_reason=self.stopped_reason,
            exact_inputs={
                key: [b.model_dump() for b in boxes]
                for key, boxes in self.cache.items()
            },
        )


def localization(samples, predictions, threshold):
    overlaps = []
    centres = []
    strong = truth = 0
    for sample, boxes in zip(samples, predictions):
        chosen = [b for b in boxes if b.score >= threshold]
        matches = match_boxes(chosen, sample.boxes, sample.ignored_boxes)
        strong += len(
            match_boxes(chosen, sample.boxes, sample.ignored_boxes, iou_threshold=0.75)[
                "true_positives"
            ]
        )
        truth += len(sample.boxes)
        for match in matches["true_positives"]:
            overlaps.append(match["iou"])
            pred = chosen[match["prediction"]].xyxy
            gt = sample.boxes[match["truth"]].xyxy
            centres.append(
                math.hypot(
                    (pred[0] + pred[2] - gt[0] - gt[2]) / 2,
                    (pred[1] + pred[3] - gt[1] - gt[3]) / 2,
                )
            )
    return dict(
        mean_matched_iou=statistics.mean(overlaps) if overlaps else None,
        median_matched_iou=statistics.median(overlaps) if overlaps else None,
        median_center_error_source_pixels=(
            statistics.median(centres) if centres else None
        ),
        recall_at_iou_075=strong / truth if truth else None,
        matched_boxes=len(overlaps),
        notes="Conditional on matches at IoU .5; unmatched labels remain false negatives.",
    )


def evaluate_one(
    sample,
    path,
    detector,
    *,
    validated_domain=None,
    max_tiles=256,
    max_pixels=25_000_000,
    max_seconds=300,
    expected_mask_sha256=None,
):
    cache = CachedDetector(
        detector, max_calls=max_tiles, deadline=time.monotonic() + max_seconds
    )
    common = dict(
        allow_unknown_quality=True,
        validated_domain=validated_domain,
        max_tiles=max_tiles,
        max_pixels=max_pixels,
        max_seconds=max_seconds,
        expected_mask_sha256=expected_mask_sha256,
    )
    # This path performs resource, physical-domain and provenance checks before
    # opening pixels or forwarding. Legacy inference must never bypass it.
    safe = infer_robust_scene(path, sample.source, cache, **common)
    if safe.status not in ("complete", "partial"):
        empty_coverage = safe.coverage.model_copy(
            update=dict(
                inspected_pixels=0,
                analyzable_pixels=0,
                inspected_fraction=0,
                analyzable_fraction=0,
                unavailable_tile_windows=[],
            )
        )
        contained = safe.model_copy(
            deep=True,
            update=dict(
                observed_count=None,
                accepted_count=0,
                uncertain_count=0,
                rejected_count=0,
                detections=[],
                completed_tiles=0,
                coverage=empty_coverage,
                preprocessing=dict(
                    safe.preprocessing,
                    dedup_mode="boundary_containment",
                    execution="not_run_after_primary_failure",
                ),
                analysis_failure_reason="not_run_after_primary_failure:" + safe.status,
            ),
        )
        return dict(
            legacy_predictions=None,
            legacy_error="not_run_after_primary_failure:"
            + str(safe.analysis_failure_reason),
            safe=safe,
            contained=contained,
            cache=cache.evidence(),
        )
    legacy = None
    legacy_error = None
    try:
        # Avoid even opening the legacy raster once the shared source budget
        # has expired. Successful forwards are reused from the safe path.
        if cache.stopped_reason is not None or time.monotonic() > cache.deadline:
            raise RuntimeError("legacy_source_runtime_or_backend_budget_exceeded")
        legacy = infer_scene(
            path, sample.source, cache, confidence=0.005, allow_unknown_quality=True
        )
    except Exception as exc:
        legacy_error = type(exc).__name__ + ": " + str(exc)[:500]
    contained = infer_robust_scene(
        path, sample.source, cache, dedup_mode="boundary_containment", **common
    )
    return dict(
        legacy_predictions=prediction_boxes(legacy) if legacy is not None else None,
        legacy_error=legacy_error,
        safe=safe,
        contained=contained,
        cache=cache.evidence(),
    )


def cohort_metrics(
    samples, predictions, threshold, preparation, *, accepted_only=False
):
    apron, background = reviewed_negative_ids(preparation, samples)
    report = evaluate_generalization(
        samples,
        predictions,
        threshold,
        geography_by_scene=preparation.get("geography_by_scene", {}),
        weather_by_scene=preparation.get("metadata_weather_by_scene", {}),
        verified_empty_image_ids=apron,
        verified_background_image_ids=background,
    )
    report["localization"] = localization(samples, predictions, threshold)
    if accepted_only:
        report["metrics"]["ap50"] = None
        report["ap50_unavailable_reason"] = (
            "Accepted-only workflow outputs censor low scores; full-candidate baseline AP50 is reported separately."
        )
    error = [abs(r["predicted_count"] - r["truth_count"]) for r in report["records"]]
    report["per_image_count_error"] = dict(
        median_absolute=statistics.median(error) if error else None,
        maximum_absolute=max(error) if error else None,
    )
    verified = apron | background
    empty_records = [r for r in report["records"] if r["image_id"] in verified]
    approx_area = 0.0
    for sample in samples:
        if sample.image_id in verified and sample.source.pixel_spacing_m:
            approx_area += (
                sample.source.width
                * sample.source.height
                * math.prod(sample.source.pixel_spacing_m)
                / 1_000_000
            )
    fp = sum(r["predicted_count"] for r in empty_records)
    report["reviewed_negative_false_alarms"] = dict(
        images=len(empty_records),
        images_with_false_alarm=sum(r["predicted_count"] > 0 for r in empty_records),
        false_alarm_image_fraction=(
            sum(r["predicted_count"] > 0 for r in empty_records) / len(empty_records)
            if empty_records
            else None
        ),
        false_positive_count=fp,
        mean_false_positives_per_image=(
            fp / len(empty_records) if empty_records else None
        ),
        approximate_source_area_km2=approx_area or None,
        false_positives_per_approx_km2=fp / approx_area if approx_area else None,
        limitations="Reviewed windows from few related captures are not independent full-airfield surveys; area estimate uses local source spacing, not exact geodesic area.",
    )
    return report


def run_evaluation(cohorts, checkpoint, outdir, *, calibration_path=None, threads=4):
    import torch
    from .torch_backend import TorchDetector

    checkpoint, outdir = Path(checkpoint), Path(outdir)
    if outdir.exists():
        raise ValueError("fresh frozen evaluation directory required")
    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("bounded threads required")
    if not 1 <= len(cohorts) <= 8:
        raise ValueError("one to eight explicit cohorts required")
    if sha256_file(checkpoint) != AW01C_SHA256:
        raise ValueError("frozen AW01C checkpoint checksum differs")
    domain = None
    if calibration_path:
        calibration = json.loads(Path(calibration_path).read_text())
        if calibration["checkpoint_sha256"] != AW01C_SHA256:
            raise ValueError("calibration checkpoint differs")
        support = calibration["validation_support"]
        domain = dict(
            sensor_ids=support["sensor_ids"],
            governing_resolution_range_m=support["native_gsd_bounds_m"],
            checkpoint_sha256=AW01C_SHA256,
        )
    lock = dict(
        checkpoint_sha256=AW01C_SHA256,
        checkpoint=str(checkpoint),
        raw_threshold=0.7,
        uncertain_floor=0.2,
        nms_iou=0.4,
        primary_dedup="iou",
        supplementary_dedup="boundary_containment; no independent-label tuning",
        validation_domain=domain,
        cohorts=[],
        policy="No selection using these labels. Prior C test is exposed regression, not this fresh cohort.",
    )
    names = set()
    for cohort in cohorts:
        name = cohort["name"]
        if (
            name in names
            or not name
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name)
        ):
            raise ValueError("unique safe cohort names required")
        names.add(name)
        path = Path(cohort["manifest"])
        lock["cohorts"].append(
            dict(
                name=name,
                manifest=str(path),
                sha256=sha256_file(path),
                role=cohort.get("role", "independent_robustness"),
            )
        )
    outdir.mkdir(parents=True)
    (outdir / "evaluation-lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    torch.set_num_threads(threads)
    detector = TorchDetector(checkpoint, candidate_floor=0.005)
    report = dict(
        lock=lock, lock_sha256=sha256_file(outdir / "evaluation-lock.json"), cohorts={}
    )
    start = time.perf_counter()
    for cohort in lock["cohorts"]:
        manifest_path = Path(cohort["manifest"])
        if sha256_file(manifest_path) != cohort["sha256"]:
            raise ValueError("locked manifest changed")
        data = load_manifest(manifest_path)
        samples = [s for s in data.samples if s.split == "test"]
        if not samples:
            raise ValueError("explicit evaluation cohort has no test samples")
        assert_holdout(samples, detector.manifest)
        folder = outdir / cohort["name"]
        folder.mkdir()
        successful = []
        legacy_predictions = []
        safe_samples = []
        safe_predictions = []
        contained_samples = []
        contained_predictions = []
        incomplete_samples = []
        incomplete_predictions = []
        scenes = []
        raw = []
        for i, sample in enumerate(samples):
            result = evaluate_one(
                sample,
                resolve_image(manifest_path, sample),
                detector,
                validated_domain=domain,
            )
            safe, contained = result["safe"], result["contained"]
            item = dict(
                image_id=sample.image_id,
                source_sha256=sample.source.sha256,
                truth_count=len(sample.boxes),
                legacy_error=result["legacy_error"],
                status=safe.status,
                contained_status=contained.status,
                contained_observed_count=contained.observed_count,
                contained_failure_reason=contained.analysis_failure_reason,
                legacy_observed_count=(
                    sum(b.score >= 0.7 for b in result["legacy_predictions"])
                    if result["legacy_predictions"] is not None
                    else None
                ),
                legacy_localization_diagnostic=(
                    localization([sample], [result["legacy_predictions"]], 0.7)
                    if result["legacy_predictions"] is not None
                    else None
                ),
                observed_count=safe.observed_count,
                scene_total=None,
                uncertain_count=safe.uncertain_count,
                coverage=safe.coverage.model_dump(),
                failure_reason=safe.analysis_failure_reason,
                unique_forward_calls=result["cache"]["unique_forward_calls"],
            )
            scenes.append(item)
            if result["legacy_predictions"] is not None:
                if safe.status == "complete":
                    successful.append(sample)
                    legacy_predictions.append(result["legacy_predictions"])
                elif safe.status == "partial":
                    incomplete_samples.append(sample)
                    incomplete_predictions.append(result["legacy_predictions"])
            if safe.status == "complete":
                safe_samples.append(sample)
                safe_predictions.append(
                    [d.box for d in safe.detections if d.status == "accepted"]
                )
            if contained.status == "complete":
                contained_samples.append(sample)
                contained_predictions.append(
                    [d.box for d in contained.detections if d.status == "accepted"]
                )
            raw.append(
                dict(
                    image_id=sample.image_id,
                    legacy_predictions=(
                        [b.model_dump() for b in result["legacy_predictions"]]
                        if result["legacy_predictions"] is not None
                        else None
                    ),
                    cache=result["cache"],
                )
            )
            stem = f"{i:03d}-" + Path(sample.image_id).name
            (folder / (stem + ".json")).write_text(
                safe.model_dump_json(indent=2), encoding="utf-8"
            )
            (folder / (stem + ".geojson")).write_text(
                json.dumps(scene_feature_collection(safe), indent=2), encoding="utf-8"
            )
            (folder / (stem + ".containment.json")).write_text(
                contained.model_dump_json(indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    dict(
                        cohort=cohort["name"],
                        image=i + 1,
                        total=len(samples),
                        status=safe.status,
                        observed_count=safe.observed_count,
                        truth=len(sample.boxes),
                    )
                ),
                flush=True,
            )
        metrics = dict(
            baseline=cohort_metrics(
                successful, legacy_predictions, 0.7, data.preparation
            ),
            safe_workflow=cohort_metrics(
                safe_samples, safe_predictions, 0, data.preparation, accepted_only=True
            ),
            boundary_containment=cohort_metrics(
                contained_samples,
                contained_predictions,
                0,
                data.preparation,
                accepted_only=True,
            ),
        )
        incomplete_diagnostic = (
            cohort_metrics(
                incomplete_samples, incomplete_predictions, 0.7, data.preparation
            )
            if incomplete_samples
            else None
        )
        summary = dict(
            role=cohort["role"],
            accuracy_denominators=dict(
                baseline_complete_images=len(successful),
                safe_complete_images=len(safe_samples),
                containment_complete_images=len(contained_samples),
                partial_images_excluded=sum(
                    row["status"] == "partial" for row in scenes
                ),
                containment_failed_or_unsuitable=sum(
                    row["contained_status"] not in ("complete", "partial")
                    for row in scenes
                ),
            ),
            incomplete_source_baseline_diagnostic=incomplete_diagnostic,
            incomplete_source_diagnostic_limitation=(
                "Whole-source truth can include unavailable pixels. These diagnostic scores are coverage-confounded and excluded from primary accuracy metrics."
            ),
            images_requested=len(samples),
            scenes=scenes,
            failures_or_unsuitable=sum(
                s["status"] not in ("complete", "partial") for s in scenes
            ),
            metrics=metrics,
            source_sha256=cohort["sha256"],
        )
        report["cohorts"][cohort["name"]] = summary
        (folder / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        (folder / "raw-predictions.json").write_text(
            json.dumps(raw, indent=2, allow_nan=False), encoding="utf-8"
        )
    report["elapsed_seconds"] = time.perf_counter() - start
    report["limitations"] = [
        "Stratified diagnostic cohorts are not prevalence samples.",
        "Failed/unsuitable scenes have null counts; partial-coverage scenes are excluded from primary accuracy denominators because full-source labels can cover unavailable pixels. Each workflow reports its own successful complete-image denominator.",
        "Full scenes, tile crops and negative windows may share captures; metrics are separate and never summed.",
        "Independent NAIP imagery is aerial and manually annotated, not an independent satellite benchmark.",
        "Accepted-only workflow AP50 is intentionally unavailable; candidate-level AP50 belongs to baseline report.",
        "Supplementary containment is a geometry heuristic and is not selected from evaluation scores.",
    ]
    (outdir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report
