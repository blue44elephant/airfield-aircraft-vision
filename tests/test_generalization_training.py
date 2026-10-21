"""AW-01C deterministic training orchestration regressions; tiny fake model only."""

from collections import Counter
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("torchvision")

from airfield_aircraft_vision import generalization_training as training
from airfield_aircraft_vision.optical_contracts import PixelBox


def sample(name, group, split="train", *, empty=False, ignored=False):
    source = SimpleNamespace(
        sensor_domain="optical",
        native_gsd_m=0.5,
        effective_resolution_m=None,
        panchromatic_gsd_m=None,
        pan_sharpened_gsd_m=None,
        quality="unknown",
        width=32,
        height=32,
        airfield_group=group,
        capture_group="capture-" + name,
        sha256=name.ljust(64, "0"),
    )
    return SimpleNamespace(
        image_id=name,
        split=split,
        source=source,
        boxes=[] if empty else [PixelBox(xyxy=(5, 6, 15, 18))],
        ignored_boxes=[PixelBox(xyxy=(0, 0, 2, 2))] if ignored else [],
    )


def test_round_robin_preserves_every_sample_once_and_balances_airfields():
    samples = [sample(f"a{i}", "a") for i in range(5)]
    samples += [sample(f"b{i}", "b") for i in range(3)]
    samples += [sample("c0", "c")]
    ordered = training.sample_order(samples, 41)
    assert Counter(x.image_id for x in ordered) == Counter(x.image_id for x in samples)
    assert len({id(x) for x in ordered}) == len(samples)
    assert {x.source.airfield_group for x in ordered[:3]} == {"a", "b", "c"}
    assert {x.source.airfield_group for x in ordered[3:5]} == {"a", "b"}
    assert training.sample_order([], 41) == []


def test_round_robin_seed_replay_and_epoch_variation():
    samples = [sample(f"{g}{i}", g) for g in ("a", "b", "c") for i in range(5)]
    before = [s.image_id for s in samples]
    first = [s.image_id for s in training.sample_order(samples, 41)]
    assert first == [s.image_id for s in training.sample_order(samples, 41)]
    assert first != [s.image_id for s in training.sample_order(samples, 42)]
    assert [s.image_id for s in samples] == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("architecture", "unsupported"),
        ("epochs", 0),
        ("epochs", 9),
        ("epochs", 1.5),
        ("epochs", True),
        ("threads", 0),
        ("threads", 9),
        ("threads", 1.5),
        ("threads", True),
        ("lr", 0),
        ("lr", 0.02),
        ("lr", float("nan")),
        ("lr", float("inf")),
        ("max_seconds", 29),
        ("max_seconds", 3601),
        ("max_seconds", float("nan")),
    ],
)
def test_invalid_configuration_fails_before_manifest_or_output(
    monkeypatch, tmp_path, field, value
):
    config = dict(
        architecture=training.RESNET_ARCHITECTURE,
        epochs=1,
        lr=0.0025,
        threads=1,
        max_seconds=30,
    )
    config[field] = value
    monkeypatch.setattr(
        training, "load_manifest", lambda path: pytest.fail("invalid config read data")
    )
    out = tmp_path / "experiment"
    with pytest.raises(ValueError):
        training.train_generalization(
            tmp_path / "missing.json", tmp_path / "missing.pth", out, **config
        )
    assert not out.exists()


@pytest.fixture
def lightweight_training(monkeypatch):
    sources = [
        sample("positive", "train-a"),
        sample("negative", "train-b", empty=True),
        sample("validation", "val", "validation"),
        sample("test-must-not-be-read", "test", "test"),
    ]
    manifest = SimpleNamespace(
        samples=sources, preparation={"verified_negative_image_ids": ["negative"]}
    )
    observed = []
    hooks = []
    validations = []
    initializations = []

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(1, 1)
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.roi_heads = SimpleNamespace(
                score_thresh=0.05,
                fg_bg_sampler=SimpleNamespace(batch_size_per_image=512),
            )
            self.rpn = SimpleNamespace(
                _pre_nms_top_n={"training": 2000, "testing": 1000},
                _post_nms_top_n={"training": 2000, "testing": 1000},
            )

        def forward(self, images, targets):
            assert len(images) == len(targets) == 1
            observed.append(
                {
                    "boxes": targets[0]["boxes"].clone(),
                    "labels": targets[0]["labels"].clone(),
                    "pixels": images[0].detach().clone(),
                }
            )
            for hook in hooks:
                hook(len(observed))
            return {"loss_classifier": self.weight.square().sum()}

    model = TinyModel()

    class Raster:
        def __init__(self, path, source):
            assert "test-must-not-be-read" not in str(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, window, size):
            assert window == (0, 0, 32, 32) and size == 512
            return np.full((3, 512, 512), 128, dtype=np.uint8), np.ones(
                (512, 512), dtype=bool
            )

    def initialize(path, architecture):
        initializations.append((path, architecture))
        return model

    def collect(path, detector, split):
        assert split == "validation"
        validations.append(split)
        assert detector.model is model and not model.training
        return manifest, [sources[2]], [[]]

    def threshold(samples, predictions, *, split):
        assert split == "validation"
        assert all(s.split == "validation" for s in samples)
        return {"selected_threshold": 0.4, "metrics": {"f1": 0.5, "count_mae": 1.0}}

    monkeypatch.setattr(training, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(training, "resolve_image", lambda path, s: s.image_id)
    monkeypatch.setattr(training, "sha256_file", lambda path: "0" * 64)
    monkeypatch.setattr(training, "NativeRaster", Raster)
    monkeypatch.setattr(training, "initialize_architecture_from_coco", initialize)
    monkeypatch.setattr(training, "collect_predictions", collect)
    monkeypatch.setattr(training, "choose_threshold", threshold)
    monkeypatch.setattr(
        training,
        "evaluate_generalization",
        lambda *a, **k: {"metrics": {"f1": 0.5, "count_mae": 1.0, "ap50": 0.2}},
    )
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    yield SimpleNamespace(
        manifest=manifest,
        model=model,
        observed=observed,
        hooks=hooks,
        validations=validations,
        initializations=initializations,
    )
    torch.set_num_threads(previous_threads)
    torch.use_deterministic_algorithms(previous_deterministic)


def run(tmp_path, **kwargs):
    return training.train_generalization(
        tmp_path / "manifest.json",
        tmp_path / "weights.pth",
        tmp_path / "run",
        epochs=kwargs.pop("epochs", 1),
        threads=1,
        max_seconds=30,
        **kwargs,
    )


def test_training_keeps_negative_targets_native_pixels_and_validation_only(
    lightweight_training, tmp_path
):
    state = lightweight_training
    history = run(tmp_path, epochs=2)
    assert [len(x["boxes"]) for x in state.observed].count(0) == 2
    assert [len(x["boxes"]) for x in state.observed].count(1) == 2
    for target in state.observed:
        assert target["boxes"].shape == (len(target["boxes"]), 4)
        assert target["boxes"].dtype == torch.float32
        assert target["labels"].dtype == torch.int64
        assert torch.equal(
            target["labels"], torch.ones(len(target["boxes"]), dtype=torch.int64)
        )
        assert target["pixels"].shape == (3, 512, 512)
        assert torch.all(
            target["pixels"] == torch.tensor(128 / 255, dtype=torch.float32)
        )
    assert state.validations == ["validation", "validation"]
    assert all(not p.requires_grad for p in state.model.backbone.parameters())
    assert state.model.weight.requires_grad
    assert state.model.rpn._post_nms_top_n == {"training": 512, "testing": 1000}
    assert history[1]["learning_rate"] == pytest.approx(
        history[0]["learning_rate"] * 0.3
    )
    checkpoint = torch.load(tmp_path / "run" / "best.pt", weights_only=True)
    assert checkpoint["manifest"]["selection_split"] == "validation"
    assert checkpoint["manifest"]["training_airfields"] == ["train-a", "train-b"]
    assert checkpoint["manifest"]["selected_threshold"] == 0.4
    resources = json.loads((tmp_path / "run" / "resources.json").read_text())
    assert resources["completed_epochs"] == 2
    assert resources["training_tiles"] == 2 and resources["validation_tiles"] == 1


@pytest.mark.parametrize("stop_reason", ["cancellation", "time_limit"])
def test_stop_during_epoch_records_partial_epoch_and_final_validation(
    lightweight_training, monkeypatch, tmp_path, stop_reason
):
    state = lightweight_training
    clock = [0.0]
    monkeypatch.setattr(training.time, "perf_counter", lambda: clock[0])

    def stop_after_one(count):
        if count == 1:
            if stop_reason == "cancellation":
                (tmp_path / "run" / "STOP").write_text("stop requested")
            else:
                clock[0] = 31.0

    state.hooks.append(stop_after_one)
    history = run(tmp_path)
    assert len(state.observed) == 1
    assert state.validations == ["validation"]
    assert history[0]["steps"] == 1 and history[0]["full_epoch"] is False
    resources = json.loads((tmp_path / "run" / "resources.json").read_text())
    assert resources["stop_reason"] == stop_reason
    assert resources["completed_epochs"] == 0
    assert (tmp_path / "run" / "best.pt").is_file()


def test_ignored_aircraft_not_silently_used_as_training_background(
    lightweight_training, tmp_path
):
    state = lightweight_training
    state.manifest.samples[0].ignored_boxes = [PixelBox(xyxy=(0, 0, 2, 2))]
    with pytest.raises(ValueError, match="ignored regions"):
        run(tmp_path)
    assert state.initializations == [] and not (tmp_path / "run").exists()


@pytest.mark.parametrize("split", ["validation", "test"])
@pytest.mark.parametrize(
    "attribute,lineage_key",
    [
        ("airfield_group", "training_airfields"),
        ("capture_group", "training_captures"),
        ("sha256", "training_source_checksums"),
    ],
)
def test_initial_checkpoint_must_preserve_holdout_independence(
    lightweight_training, monkeypatch, tmp_path, split, attribute, lineage_key
):
    state = lightweight_training
    heldout = next(s for s in state.manifest.samples if s.split == split)
    prior = {
        "architecture": training.RESNET_ARCHITECTURE,
        "training_airfields": ["unrelated-prior-training-group"],
        "training_captures": ["unrelated-prior-capture"],
        "training_source_checksums": ["f" * 64],
    }
    prior[lineage_key] = [getattr(heldout.source, attribute)]

    class InitialDetector:
        def __init__(self, *args, **kwargs):
            self.model = state.model
            self.manifest = prior

    monkeypatch.setattr(training, "TorchDetector", InitialDetector)
    with pytest.raises(ValueError, match="leakage"):
        run(tmp_path, initial_checkpoint=tmp_path / "prior-checkpoint.pt")
    assert not state.observed and not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"quality": "poor"},
        {"quality": "unusable"},
        {"effective_resolution_m": 2.0},
        {"panchromatic_gsd_m": 2.0, "pan_sharpened_gsd_m": 0.3},
        {"sensor_domain": "sar"},
        {"sensor_domain": "unknown"},
    ],
)
def test_training_respects_capability_quality_and_physical_resolution_gates(
    lightweight_training, tmp_path, changes
):
    state = lightweight_training
    for field, value in changes.items():
        setattr(state.manifest.samples[0].source, field, value)
    with pytest.raises(ValueError, match="Tier A|TierA|tier_a|high-resolution|optical"):
        run(tmp_path)
    assert state.initializations == [] and not (tmp_path / "run").exists()
