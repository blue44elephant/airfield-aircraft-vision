"""Fixed-operating-point AW-01C generalization strata.

This module never selects thresholds or models. Zero annotated aircraft is not
evidence of an empty apron; verified negatives must be supplied explicitly.
"""

from __future__ import annotations

from collections import defaultdict

from .evaluation import summarize
from .experiment_metrics import evaluate_predictions
from .postprocessing import iou


def crowding_bin(truth_count: int) -> str:
    """Predeclared annotation-count bins; not an airport congestion measure."""
    if (
        isinstance(truth_count, bool)
        or not isinstance(truth_count, int)
        or truth_count < 0
    ):
        raise ValueError("truth_count must be a nonnegative integer")
    if truth_count == 0:
        return "zero_labeled"
    if truth_count == 1:
        return "one_labeled"
    return "two_to_five_labeled" if truth_count < 6 else "six_or_more_labeled"


def reviewed_negative_ids(preparation, samples):
    """Return reviewed apron/background ID sets restricted to these samples.

    Categories come from documented manual reviews, never from model predictions
    or the absence of annotations. Unknown categories and incomplete review
    records fail explicitly for relevant verified-negative IDs.
    """
    sample_ids = {sample.image_id for sample in samples}
    relevant = set(preparation.get("verified_negative_image_ids", [])) & sample_ids
    reviews = {}
    for detail in preparation.get("negative_review_details", []):
        key = detail.get("image_id")
        if key not in relevant:
            continue
        if key in reviews:
            raise ValueError("duplicate negative review record")
        category = detail.get("review", {}).get("category")
        if category not in ("empty_apron", "difficult_background", "other_background"):
            raise ValueError("unknown verified-negative review category")
        reviews[key] = category
    if relevant != reviews.keys():
        raise ValueError("verified negatives lack documented review categories")
    return (
        {key for key, category in reviews.items() if category == "empty_apron"},
        {
            key
            for key, category in reviews.items()
            if category in ("difficult_background", "other_background")
        },
    )


def evaluate_generalization(
    samples,
    predictions,
    threshold,
    *,
    geography_by_scene=None,
    weather_by_scene=None,
    verified_empty_image_ids=(),
    verified_background_image_ids=(),
):
    """Evaluate supplied predictions without reading rasters or changing selection.

    geography_by_scene maps source scene identifiers to country/region labels.
    verified_empty_image_ids comes from a separately documented manual review,
    not from absence of boxes. Samples with ignored boxes cannot be verified empty.
    """
    verified_empty = set(verified_empty_image_ids)
    verified_background = set(verified_background_image_ids)
    if verified_empty & verified_background:
        raise ValueError("verified apron and background categories must be disjoint")
    verified = verified_empty | verified_background
    by_id = {s.image_id: s for s in samples}
    if len(by_id) != len(samples):
        raise ValueError("duplicate sample image IDs")
    if verified - by_id.keys():
        raise ValueError("verified-negative IDs must belong to the evaluated samples")
    if any(by_id[key].boxes or by_id[key].ignored_boxes for key in verified):
        raise ValueError(
            "verified-empty samples must have no positive or ignored boxes"
        )

    result = evaluate_predictions(samples, predictions, threshold, weather_by_scene)
    crowding = defaultdict(list)
    geography = defaultdict(list)
    label_state = defaultdict(list)
    overlap_records = []
    for sample, preds, record in zip(samples, predictions, result["records"]):
        crowding[crowding_bin(len(sample.boxes))].append(record)
        region = (geography_by_scene or {}).get(sample.source.scene_id) or "unknown"
        geography[region].append(record)
        if sample.boxes:
            state = "positive_labeled"
        elif sample.ignored_boxes:
            state = "ignored_only"
        elif sample.image_id in verified_empty:
            state = "verified_empty_apron"
        elif sample.image_id in verified_background:
            state = "verified_background"
        else:
            state = "zero_labeled_unverified"
        label_state[state].append(record)
        record["label_state"] = state
        record["geography"] = region

        chosen = [p for p in preds if p.score >= threshold]
        fp = record["matches"]["false_positives"]
        fn = record["matches"]["false_negatives"]
        pairs = []
        for pi in fp:
            for gi in fn:
                overlap = iou(chosen[pi].xyxy, sample.boxes[gi].xyxy)
                if 0.1 <= overlap < 0.5:
                    pairs.append({"prediction": pi, "truth": gi, "iou": overlap})
        overlap_records.append(
            {
                "image_id": sample.image_id,
                "overlap_pairs": pairs,
                "unmatched_predictions_with_overlap": len(
                    {pair["prediction"] for pair in pairs}
                ),
                "missed_truth_with_overlap": len({pair["truth"] for pair in pairs}),
            }
        )

    result["by_truth_crowding"] = {
        name: summarize(rows) for name, rows in sorted(crowding.items())
    }
    result["by_geography"] = {
        name: summarize(rows) for name, rows in sorted(geography.items())
    }
    result["by_label_state"] = {
        name: summarize(rows) for name, rows in sorted(label_state.items())
    }
    empty = label_state.get("verified_empty_apron", [])
    result["verified_empty_apron"] = {
        **summarize(empty),
        "tiles_with_false_alarm": sum(r["predicted_count"] > 0 for r in empty),
        "false_alarm_tile_fraction": (
            sum(r["predicted_count"] > 0 for r in empty) / len(empty) if empty else None
        ),
        "requires_documented_manual_verification": True,
    }
    background = label_state.get("verified_background", [])
    result["verified_background"] = {
        **summarize(background),
        "tiles_with_false_alarm": sum(r["predicted_count"] > 0 for r in background),
        "false_alarm_tile_fraction": (
            sum(r["predicted_count"] > 0 for r in background) / len(background)
            if background
            else None
        ),
    }
    result["unmatched_overlap_diagnostic"] = {
        "iou_range": [0.1, 0.5],
        "upper_bound_exclusive": True,
        "records": overlap_records,
        "note": (
            "Unmatched prediction/truth overlaps are localization candidates only. "
            "They do not prove aircraft identity and do not change TP/FP/FN."
        ),
    }
    group_count = len({s.source.airfield_group for s in samples})
    result["sampling_limitations"] = {
        "airfield_groups": group_count,
        "images": len(samples),
        "confidence_intervals_computed": False,
        "small_group_sample": group_count < 10,
        "statement": (
            "These are descriptive metrics from a source-correlated sample. "
            "Tiles may overlap within an airfield; counts are per tile and cannot "
            "be summed into an airport inventory. Small geographic and negative "
            "strata do not establish regional or operational reliability."
        ),
    }
    result["strata_notes"].extend(
        [
            "Crowding bins use annotated aircraft per tile: 0, 1, 2-5, and 6+.",
            "Zero labels alone do not verify an empty apron.",
            "Geography labels are supplied source metadata; unknown stays unknown.",
            "Only the supplied fixed operating point is evaluated.",
        ]
    )
    return result
