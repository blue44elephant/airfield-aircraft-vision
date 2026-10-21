"""Optional TorchVision backend. No implicit network access, no pixel resizing."""

from pathlib import Path
from .preprocessing import sha256_file

ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_fpn"
WEIGHTS_URL = (
    "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth"
)


def build_model(classes=2):
    import torch
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.models.detection.transform import GeneralizedRCNNTransform

    class NativeTransform(GeneralizedRCNNTransform):
        def resize(self, image, target=None):
            return image, target

    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=classes,
        box_score_thresh=0.05,
        box_detections_per_img=300,
    )
    model.transform = NativeTransform(
        512, 512, model.transform.image_mean, model.transform.image_std
    )
    return model


def initialize_from_coco(path: Path):
    import torch

    if not sha256_file(path).startswith("fb6a3cc7"):
        raise ValueError(
            "initial weights do not match the official COCO artifact hash prefix"
        )
    model = build_model(91)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    old = model.roi_heads.box_predictor
    new = FastRCNNPredictor(old.cls_score.in_features, 2)
    # COCO category ID 5 is airplane. Copy its learned generic detector.
    with torch.no_grad():
        new.cls_score.weight.copy_(old.cls_score.weight[[0, 5]])
        new.cls_score.bias.copy_(old.cls_score.bias[[0, 5]])
        rows = [0, 1, 2, 3, 20, 21, 22, 23]
        new.bbox_pred.weight.copy_(old.bbox_pred.weight[rows])
        new.bbox_pred.bias.copy_(old.bbox_pred.bias[rows])
    model.roi_heads.box_predictor = new
    return model


class TorchDetector:
    def __init__(self, checkpoint: Path, device="cpu", *, candidate_floor=None):
        import torch

        self.device = torch.device(device)
        checkpoint = Path(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.manifest = payload["manifest"]
        from .architecture_backend import SUPPORTED_ARCHITECTURES, build_architecture

        if (
            self.manifest["architecture"] not in SUPPORTED_ARCHITECTURES
            or self.manifest["preprocessing"] != "native-rgb-v1"
        ):
            raise ValueError("incompatible checkpoint architecture/preprocessing")
        self.manifest = dict(self.manifest, checkpoint_sha256=sha256_file(checkpoint))
        self.model = build_architecture(self.manifest["architecture"])
        self.model.load_state_dict(payload["state_dict"])
        floor = (
            self.manifest.get("candidate_score_floor", 0.05)
            if candidate_floor is None
            else candidate_floor
        )
        if not 0 <= floor <= 1:
            raise ValueError("candidate floor must be in [0,1]")
        self.model.roi_heads.score_thresh = floor
        self.manifest["candidate_score_floor"] = floor
        self.model.to(self.device).eval()

    def predict(self, pixels):
        import torch
        from .optical_contracts import PixelBox

        image = torch.from_numpy(pixels.copy()).float().to(self.device) / 255.0
        with torch.inference_mode():
            out = self.model([image])[0]
        return [
            PixelBox(xyxy=tuple(b), score=s)
            for b, s, label in zip(
                out["boxes"].cpu().tolist(),
                out["scores"].cpu().tolist(),
                out["labels"].cpu().tolist(),
            )
            if label == 1
        ]
