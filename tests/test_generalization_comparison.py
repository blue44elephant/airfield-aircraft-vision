"""Offline guards for frozen model selection and one-pass test evaluation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from airfield_aircraft_vision import generalization_comparison as comparison
from airfield_aircraft_vision.optical_contracts import (
    DatasetManifest,
    PixelBox,
    RasterSource,
    Sample,
)
from airfield_aircraft_vision.preprocessing import sha256_file


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    from PIL import Image

    def make_manifest(name, color, group):
        folder = tmp_path / name
        folder.mkdir()
        image = folder / (name + ".png")
        Image.new("RGB", (32, 32), color).save(image)
        sample = Sample(
            image_id=name,
            path=image.name,
            split="test",
            source=RasterSource(
                scene_id=name,
                sensor_id="synthetic",
                native_gsd_m=0.5,
                width=32,
                height=32,
                sha256=sha256_file(image),
                airfield_group=group,
                capture_group=group,
            ),
            boxes=[PixelBox(xyxy=(1, 1, 11, 11))],
        )
        manifest = DatasetManifest(
            samples=[sample],
            preparation={"geography_by_scene": {name: "Synthetic Region"}},
        )
        path = folder / "manifest.json"
        path.write_text(manifest.model_dump_json())
        return path, sample

    manifest, sample = make_manifest("independent", "navy", "unseen-field")
    regression, regression_sample = make_manifest("regression", "gray", "old-field")
    infos = {}
    paths = {}
    for key, f1, mae, threshold in [
        ("candidate_a", 0.7, 1.0, 0.5),
        ("candidate_b", 0.6, 0.5, 0.4),
        ("baseline", 0.5, 2.0, 0.4),
    ]:
        folder = tmp_path / key
        folder.mkdir()
        path = folder / "best.pt"
        path.write_bytes(key.encode())
        paths[key] = path
        infos[str(path)] = {
            "selection_split": "validation",
            "training_manifest_sha256": sha256_file(manifest),
            "selected_threshold": threshold,
            "selected_epoch": 3,
            "selected_validation_metrics": {"f1": f1, "count_mae": mae},
            "checkpoint_sha256": sha256_file(path),
            "architecture": "synthetic",
        }
    events = []

    def detector(path, **kwargs):
        events.append(("load_checkpoint", str(path)))
        return SimpleNamespace(manifest=infos[str(path)], path=Path(path))

    output = tmp_path / "comparison"

    def collect(path, detector, split):
        assert (output / "selection-lock.json").exists()
        assert split == "test"
        events.append(("collect", str(path), detector.path.parent.name))
        data = comparison.load_manifest(path)
        samples = [sample for sample in data.samples if sample.split == "test"]
        # Better test predictions must not overrule validation selection.
        predictions = [
            [] if detector.path.parent.name == "candidate_a" else sample.boxes
            for sample in samples
        ]
        return data, samples, predictions

    monkeypatch.setattr(comparison, "TorchDetector", detector)
    monkeypatch.setattr(comparison, "collect_predictions", collect)
    monkeypatch.setattr(comparison, "_set_cpu_threads", lambda threads: None)
    return SimpleNamespace(
        manifest=manifest,
        sample=sample,
        regression=regression,
        regression_sample=regression_sample,
        infos=infos,
        paths=paths,
        output=output,
        events=events,
    )


def invoke(experiment, *, regression=False):
    return comparison.compare(
        experiment.manifest,
        experiment.paths["baseline"],
        [
            experiment.paths["candidate_a"].parent,
            experiment.paths["candidate_b"].parent,
        ],
        experiment.output,
        regression_manifest=experiment.regression if regression else None,
    )


def test_lock_precedes_test_reads_and_predictions_are_collected_once(
    experiment, monkeypatch
):
    real_load = comparison.load_manifest

    def guarded_load(path):
        assert (experiment.output / "selection-lock.json").exists()
        return real_load(path)

    monkeypatch.setattr(comparison, "load_manifest", guarded_load)
    report = invoke(experiment, regression=True)
    assert report["selection"]["selected"]["checkpoint"] == str(
        experiment.paths["candidate_a"]
    )
    assert report["selection"]["aw01b_baseline"]["threshold"] == 0.4
    calls = [event for event in experiment.events if event[0] == "collect"]
    assert len(calls) == 4
    assert {event[2] for event in calls} == {"candidate_a", "baseline"}
    assert sum(event[1] == str(experiment.manifest) for event in calls) == 2
    independent = report["cohorts"]["independent_test"]
    assert independent["models"]["aw01c_validation_selected"]["metrics"]["recall"] == 0
    assert independent["models"]["aw01b_validation_selected"]["metrics"]["recall"] == 1
    assert (
        report["cohorts"]["aw01b_regression"]["role"] == "previously_exposed_regression"
    )
    assert report["selection_lock_sha256"] == sha256_file(
        experiment.output / "selection-lock.json"
    )


def test_native_overlays_reuse_predictions_and_source_is_unchanged(experiment):
    from PIL import Image

    original = experiment.manifest.parent / experiment.sample.path
    before = original.read_bytes()
    report = invoke(experiment)
    overlay = (
        experiment.output
        / "independent_test"
        / "aw01c"
        / (experiment.sample.image_id + ".overlay.png")
    )
    with Image.open(overlay) as image:
        assert image.size == (32, 32)
    assert original.read_bytes() == before
    assert len([event for event in experiment.events if event[0] == "collect"]) == 2
    raw = json.loads(
        (experiment.output / "independent_test-raw-predictions.json").read_text()
    )
    assert raw["aw01c_predictions"] == [[]]
    assert (
        report["cohorts"]["independent_test"]["models"]["zero_detection_baseline"][
            "metrics"
        ]["count_mae"]
        == 1
    )


@pytest.mark.parametrize(
    "bad_field,bad_value,message",
    [
        ("selection_split", "test", "validation-only"),
        ("training_manifest_sha256", "0" * 64, "manifest checksum"),
        ("selected_threshold", float("nan"), "threshold"),
        ("checkpoint_sha256", "0" * 64, "checkpoint checksum"),
    ],
)
def test_invalid_candidate_is_rejected_before_test_access(
    experiment, bad_field, bad_value, message
):
    experiment.infos[str(experiment.paths["candidate_a"])][bad_field] = bad_value
    with pytest.raises(ValueError, match=message):
        invoke(experiment)
    assert not experiment.output.exists()
    assert not any(event[0] == "collect" for event in experiment.events)


def test_existing_output_blocks_any_model_loading_or_test_repetition(experiment):
    experiment.output.mkdir()
    with pytest.raises(ValueError, match="fresh comparison"):
        invoke(experiment)
    assert experiment.events == []


@pytest.mark.parametrize(
    "mae_a,mae_b,threshold_a,threshold_b,winner",
    [
        (0.5, 1.0, 0.2, 0.9, "candidate_a"),
        (0.5, 0.5, 0.2, 0.9, "candidate_b"),
    ],
)
def test_validation_tie_breaks_use_mae_then_threshold(
    experiment, mae_a, mae_b, threshold_a, threshold_b, winner
):
    for key, mae, threshold in [
        ("candidate_a", mae_a, threshold_a),
        ("candidate_b", mae_b, threshold_b),
    ]:
        info = experiment.infos[str(experiment.paths[key])]
        info["selected_validation_metrics"] = {"f1": 0.7, "count_mae": mae}
        info["selected_threshold"] = threshold
    report = invoke(experiment)
    assert report["selection"]["selected"]["checkpoint"] == str(
        experiment.paths[winner]
    )


def test_changed_locked_checkpoint_blocks_predictions(experiment, monkeypatch):
    def mutate_after_lock(threads):
        experiment.paths["candidate_a"].write_bytes(b"changed")

    monkeypatch.setattr(comparison, "_set_cpu_threads", mutate_after_lock)
    with pytest.raises(ValueError, match="locked checkpoint"):
        invoke(experiment)
    assert (experiment.output / "selection-lock.json").exists()
    assert not any(event[0] == "collect" for event in experiment.events)


def test_same_regression_manifest_cannot_double_evaluate_new_test(experiment):
    with pytest.raises(ValueError, match="different cohort"):
        comparison.compare(
            experiment.manifest,
            experiment.paths["baseline"],
            [experiment.paths["candidate_a"].parent],
            experiment.output,
            regression_manifest=experiment.manifest,
        )
    assert not experiment.output.exists()
