"""AW-01C model compatibility and source-pixel regressions; no downloads."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from airfield_aircraft_vision import architecture_backend as backend
from airfield_aircraft_vision import torch_backend as legacy


def test_resnet_native_grid_fine_features_and_frozen_norm(monkeypatch):
    from torchvision.ops.misc import FrozenBatchNorm2d

    def forbidden(*args, **kwargs):
        raise AssertionError("architecture construction must not download weights")

    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = backend.build_architecture(backend.RESNET_ARCHITECTURE)
        model.eval()
        image = torch.rand(3, 95, 121)
        target = {
            "boxes": torch.tensor([[10.0, 20.0, 35.0, 45.0]]),
            "labels": torch.tensor([1]),
        }
        with torch.inference_mode():
            transformed, targets = model.transform([image], [target])
            assert transformed.image_sizes == [(95, 121)]
            assert transformed.tensors.shape == (1, 3, 96, 128)
            assert torch.equal(targets[0]["boxes"], target["boxes"])
            expected = (
                image - torch.tensor(model.transform.image_mean)[:, None, None]
            ) / torch.tensor(model.transform.image_std)[:, None, None]
            assert torch.equal(transformed.tensors[0, :, :95, :121], expected)
            features = model.backbone(torch.zeros(1, 3, 128, 128))
        assert {k: tuple(v.shape[-2:]) for k, v in features.items()} == {
            "0": (32, 32),
            "1": (16, 16),
            "2": (8, 8),
            "3": (4, 4),
            "pool": (2, 2),
        }
        norms = [m for m in model.modules() if isinstance(m, FrozenBatchNorm2d)]
        assert norms and all(m.eps == 0 for m in norms)
        assert model.rpn.anchor_generator.sizes == (
            (32,),
            (64,),
            (128,),
            (256,),
            (512,),
        )
    finally:
        torch.set_num_threads(previous)


def test_unknown_architecture_and_bad_weight_hash_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        backend.build_architecture("unknown")
    with pytest.raises(ValueError, match="unsupported"):
        backend.initialize_architecture_from_coco(tmp_path / "missing", "unknown")
    fake = tmp_path / "fake.pth"
    fake.write_bytes(b"not official weights")
    with pytest.raises(ValueError, match="official ResNet"):
        backend.initialize_architecture_from_coco(fake, backend.RESNET_ARCHITECTURE)


def test_resnet_pretrained_airplane_head_transfer(monkeypatch):
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    predictor = FastRCNNPredictor(4, 91)
    with torch.no_grad():
        predictor.cls_score.weight.copy_(torch.arange(364).reshape(91, 4))
        predictor.cls_score.bias.copy_(torch.arange(91))
        predictor.bbox_pred.weight.copy_(torch.arange(1456).reshape(364, 4))
        predictor.bbox_pred.bias.copy_(torch.arange(364))
    model = SimpleNamespace(
        roi_heads=SimpleNamespace(box_predictor=predictor),
        load_state_dict=lambda state: None,
    )
    monkeypatch.setattr(backend, "sha256_file", lambda p: "258fb6c6" + "0" * 56)
    monkeypatch.setattr(
        backend, "build_architecture", lambda architecture, classes: model
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    result = backend.initialize_architecture_from_coco(
        "explicit-local.pth", backend.RESNET_ARCHITECTURE
    )
    out = result.roi_heads.box_predictor
    assert torch.equal(out.cls_score.weight, predictor.cls_score.weight[[0, 5]])
    assert torch.equal(out.cls_score.bias, predictor.cls_score.bias[[0, 5]])
    assert torch.equal(
        out.bbox_pred.weight, predictor.bbox_pred.weight[[0, 1, 2, 3, 20, 21, 22, 23]]
    )
    assert torch.equal(
        out.bbox_pred.bias, predictor.bbox_pred.bias[[0, 1, 2, 3, 20, 21, 22, 23]]
    )


@pytest.mark.parametrize("architecture", backend.SUPPORTED_ARCHITECTURES)
def test_checkpoint_loader_dispatch_preserves_mobilenet(monkeypatch, architecture):
    called = []
    fake = SimpleNamespace(
        roi_heads=SimpleNamespace(score_thresh=0.05),
        load_state_dict=lambda state: called.append(("load", state)),
        to=lambda device: fake,
        eval=lambda: fake,
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **kw: {
            "manifest": {
                "architecture": architecture,
                "preprocessing": "native-rgb-v1",
            },
            "state_dict": {"sample": 1},
        },
    )
    monkeypatch.setattr(legacy, "sha256_file", lambda p: "abc")
    monkeypatch.setattr(
        backend,
        "build_architecture",
        lambda selected: called.append(("architecture", selected)) or fake,
    )
    detector = legacy.TorchDetector("checkpoint.pt", candidate_floor=0.005)
    assert detector.manifest["architecture"] == architecture
    assert detector.manifest["checkpoint_sha256"] == "abc"
    assert detector.model.roi_heads.score_thresh == 0.005
    assert called == [("architecture", architecture), ("load", {"sample": 1})]


def test_empty_target_training_is_finite_without_resizing():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = backend.build_architecture(backend.RESNET_ARCHITECTURE)
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        model.train()
        model.backbone.eval()
        losses = model(
            [torch.rand(3, 96, 128)],
            [
                {
                    "boxes": torch.zeros((0, 4)),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
            ],
        )
        assert all(torch.isfinite(loss) for loss in losses.values())
        assert losses["loss_box_reg"].item() == 0
        assert losses["loss_rpn_box_reg"].item() == 0
        sum(losses.values()).backward()
        assert model.roi_heads.box_predictor.cls_score.weight.grad is not None
    finally:
        torch.set_num_threads(previous)
