import math
import pytest
from airfield_aircraft_vision.confidence_calibration import (
    scaled_score,
    reliability,
    choose_temperature,
    candidate_outcomes,
    evaluate_calibration,
)
from airfield_aircraft_vision.optical_contracts import PixelBox, RasterSource, Sample


def sample(name="v", split="validation", group="a"):
    return Sample(
        image_id=name,
        path=name + ".tif",
        split=split,
        source=RasterSource(
            scene_id=name,
            sensor_id="test",
            native_gsd_m=0.5,
            width=512,
            height=512,
            sha256="a" * 64,
            airfield_group=group,
            capture_group=name,
        ),
        boxes=[PixelBox(xyxy=(0, 0, 10, 10))],
    )


def test_temperature_preserves_order_and_bounds():
    values = [scaled_score(p, 2) for p in [0, 0.1, 0.5, 0.9, 1]]
    assert values == sorted(values)
    assert all(0 < p < 1 for p in values)
    assert values[2] == 0.5
    assert scaled_score(0.9, 2) < 0.9


@pytest.mark.parametrize(
    "score,temp",
    [(float("nan"), 1), (-0.1, 1), (1.1, 1), (0.5, 0), (0.5, float("inf"))],
)
def test_invalid_parameters_fail(score, temp):
    with pytest.raises(ValueError):
        scaled_score(score, temp)


def test_reliability_has_meaningful_brier_and_undefined_empty():
    rows = [dict(confidence=0.5, correct=y) for y in [0, 1]]
    result = reliability(rows)
    assert result["ece"] == 0
    assert result["brier"] == 0.25
    assert result["nll"] == pytest.approx(math.log(2))
    assert reliability([])["brier"] is None


@pytest.mark.parametrize("split", ["train", "test"])
def test_calibration_never_selects_on_test_or_train(split):
    with pytest.raises(ValueError, match="validation"):
        candidate_outcomes([sample(split=split)], [[]], split=split)


def test_matching_ignores_regions_and_retains_duplicate_as_wrong():
    s = sample()
    s = s.model_copy(update={"ignored_boxes": [PixelBox(xyxy=(20, 20, 30, 30))]})
    preds = [
        PixelBox(xyxy=(0, 0, 10, 10), score=0.9),
        PixelBox(xyxy=(0, 0, 10, 10), score=0.8),
        PixelBox(xyxy=(20, 20, 30, 30), score=0.7),
    ]
    rows = candidate_outcomes([s], [preds], split="validation")
    assert [r["correct"] for r in rows] == [1, 0]


def test_grouped_calibration_does_not_change_threshold():
    ss = [sample("a", group="a"), sample("b", group="b")]
    pp = [
        [
            PixelBox(xyxy=(0, 0, 10, 10), score=0.9),
            PixelBox(xyxy=(20, 20, 30, 30), score=0.8),
        ]
        for _ in ss
    ]
    report = evaluate_calibration(
        ss, pp, split="validation", checkpoint_sha256="a" * 64
    )
    assert report["acceptance_rule_changed"] is False
    assert report["raw_acceptance_threshold"] == 0.7
    assert len(report["grouped_crossvalidation"]["folds"]) == 2
    assert report["status"] == "experimental_validation_diagnostic"


def test_temperature_fit_requires_both_classes():
    with pytest.raises(ValueError):
        choose_temperature([dict(confidence=0.9, correct=1)])


def test_extreme_temperature_scores_stay_finite_for_nll():
    report = reliability(
        [dict(confidence=1.0, correct=0), dict(confidence=0.0, correct=1)], 0.25
    )
    assert math.isfinite(report["nll"])
    assert math.isfinite(report["brier"])


def test_nonfinite_scores_are_not_silently_filtered_out():
    with pytest.raises(ValueError):
        reliability([dict(confidence=float("nan"), correct=0)])
