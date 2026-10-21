"""Offline regressions for fixed-operating-point generalization reports."""

import pytest

from airfield_aircraft_vision.generalization_metrics import (
    crowding_bin,
    evaluate_generalization,
)
from airfield_aircraft_vision.optical_contracts import PixelBox, RasterSource, Sample


def sample(image_id, boxes=(), ignored=(), group="field-a"):
    return Sample(
        image_id=image_id,
        path=image_id + ".tif",
        split="test",
        source=RasterSource(
            scene_id=image_id,
            sensor_id="synthetic",
            native_gsd_m=0.5,
            width=512,
            height=512,
            sha256="a" * 64,
            airfield_group=group,
            capture_group=group,
        ),
        boxes=list(boxes),
        ignored_boxes=list(ignored),
    )


def box(x=0, score=1):
    return PixelBox(xyxy=(x, 0, x + 20, 20), score=score)


@pytest.mark.parametrize(
    "count,label",
    [
        (0, "zero_labeled"),
        (1, "one_labeled"),
        (2, "two_to_five_labeled"),
        (5, "two_to_five_labeled"),
        (6, "six_or_more_labeled"),
    ],
)
def test_crowding_boundaries_are_fixed(count, label):
    assert crowding_bin(count) == label


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_invalid_crowding_count_rejected(bad):
    with pytest.raises(ValueError):
        crowding_bin(bad)


def test_empty_false_alarm_rates_require_explicit_verification():
    samples = [sample("reviewed"), sample("unreviewed")]
    result = evaluate_generalization(
        samples,
        [[box()], [box(), box(40)]],
        0.5,
        verified_empty_image_ids={"reviewed"},
    )
    assert result["metrics"]["fp"] == 3
    assert result["verified_empty_apron"]["images"] == 1
    assert result["verified_empty_apron"]["count_mae"] == 1
    assert result["verified_empty_apron"]["false_alarm_tile_fraction"] == 1
    assert result["by_label_state"]["zero_labeled_unverified"]["fp"] == 2
    assert result["verified_empty_apron"]["recall"] is None


def test_ignored_fragments_cannot_be_declared_verified_empty():
    with pytest.raises(ValueError, match="ignored"):
        evaluate_generalization(
            [sample("fragment", ignored=[box()])],
            [[box()]],
            0.5,
            verified_empty_image_ids={"fragment"},
        )
    result = evaluate_generalization(
        [sample("fragment", ignored=[box()])], [[box()]], 0.5
    )
    assert result["by_label_state"]["ignored_only"]["predicted_count"] == 0
    assert result["verified_empty_apron"]["images"] == 0
    assert result["verified_empty_apron"]["false_alarm_tile_fraction"] is None


def test_verified_empty_positive_or_unknown_id_fails():
    with pytest.raises(ValueError, match="positive"):
        evaluate_generalization(
            [sample("positive", [box()])],
            [[]],
            0.5,
            verified_empty_image_ids={"positive"},
        )
    with pytest.raises(ValueError, match="belong"):
        evaluate_generalization([], [], 0.5, verified_empty_image_ids={"unknown"})


def test_crowded_geography_and_size_strata_preserve_total_matching():
    truth = [box(i * 40) for i in range(6)]
    samples = [sample("crowd", truth), sample("single", [box()], group="field-b")]
    result = evaluate_generalization(
        samples,
        [truth[:2], [box()]],
        0.5,
        geography_by_scene={"crowd": "Example Country"},
    )
    assert result["by_truth_crowding"]["six_or_more_labeled"]["fn"] == 4
    assert result["by_geography"]["Example Country"]["tp"] == 2
    assert result["by_geography"]["unknown"]["tp"] == 1
    assert result["by_bbox_pixel_size"]["small_lt32px"]["matched_truth"] == 3
    assert result["metrics"]["count_mae"] == 2
    assert result["sampling_limitations"]["airfield_groups"] == 2
    assert result["sampling_limitations"]["small_group_sample"] is True


def test_overlap_diagnostic_does_not_reclassify_false_positive_and_miss():
    displaced = PixelBox(xyxy=(10, 0, 30, 20), score=0.8)  # IoU = 1/3
    result = evaluate_generalization(
        [sample("localization", [box()])], [[displaced]], 0.5
    )
    assert result["metrics"]["tp"] == 0
    assert result["metrics"]["fp"] == 1
    assert result["metrics"]["fn"] == 1
    row = result["unmatched_overlap_diagnostic"]["records"][0]
    assert row["unmatched_predictions_with_overlap"] == 1
    assert row["missed_truth_with_overlap"] == 1
    assert row["overlap_pairs"][0]["iou"] == pytest.approx(1 / 3)
    assert result["metrics"]["count_mae"] == 0  # Cancellation is not detection success.


def test_fixed_threshold_respected_without_touching_predictions_or_sample():
    truth = box()
    low_score = box(score=0.2)
    s = sample("fixed", [truth])
    before = s.model_dump()
    result = evaluate_generalization([s], [[low_score]], 0.5)
    assert result["metrics"]["fn"] == 1
    assert result["metrics"]["predicted_count"] == 0
    assert low_score.score == 0.2 and s.model_dump() == before


def test_report_rejects_length_mismatch_and_duplicate_samples():
    with pytest.raises(ValueError, match="length"):
        evaluate_generalization([sample("missing")], [], 0.5)
    s = sample("duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_generalization([s, s], [[], []], 0.5)


def test_reviewed_background_is_not_reported_as_an_empty_apron():
    result = evaluate_generalization(
        [sample("background")],
        [[box()]],
        0.5,
        verified_background_image_ids={"background"},
    )
    assert result["by_label_state"]["verified_background"]["fp"] == 1
    assert result["verified_background"]["false_alarm_tile_fraction"] == 1
    assert result["verified_empty_apron"]["images"] == 0
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_generalization(
            [sample("background")],
            [[]],
            0.5,
            verified_empty_image_ids={"background"},
            verified_background_image_ids={"background"},
        )


def test_review_categories_are_explicit_and_subset_to_current_samples():
    from airfield_aircraft_vision.generalization_metrics import reviewed_negative_ids

    preparation = {
        "verified_negative_image_ids": ["apron", "background", "other_split"],
        "negative_review_details": [
            {"image_id": "apron", "review": {"category": "empty_apron"}},
            {"image_id": "background", "review": {"category": "difficult_background"}},
            {"image_id": "other_split", "review": {"category": "empty_apron"}},
        ],
    }
    assert reviewed_negative_ids(
        preparation, [sample("apron"), sample("background")]
    ) == ({"apron"}, {"background"})


@pytest.mark.parametrize(
    "details",
    [
        [],
        [{"image_id": "negative", "review": {"category": "unknown"}}],
        [{"image_id": "negative", "review": {"category": "empty_apron"}}] * 2,
    ],
)
def test_missing_unknown_or_duplicate_review_categories_are_rejected(details):
    from airfield_aircraft_vision.generalization_metrics import reviewed_negative_ids

    with pytest.raises(ValueError, match="review"):
        reviewed_negative_ids(
            {
                "verified_negative_image_ids": ["negative"],
                "negative_review_details": details,
            },
            [sample("negative")],
        )


def test_other_reviewed_background_category_remains_separate_from_aprons():
    from airfield_aircraft_vision.generalization_metrics import reviewed_negative_ids

    assert reviewed_negative_ids(
        {
            "verified_negative_image_ids": ["other"],
            "negative_review_details": [
                {"image_id": "other", "review": {"category": "other_background"}}
            ],
        },
        [sample("other")],
    ) == (set(), {"other"})
