"""Explicit, native-pixel detector architectures for AW-01C experiments.

Constructors never download weights. COCO initialization accepts only an
explicit local artifact with the official SHA-256 prefix. The legacy MobileNet
builder and checkpoint architecture remain supported without changing defaults.
"""

from pathlib import Path

from .preprocessing import sha256_file
from .torch_backend import ARCHITECTURE as MOBILENET_ARCHITECTURE

RESNET_ARCHITECTURE = "fasterrcnn_resnet50_fpn"
RESNET_WEIGHTS_URL = (
    "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
)
SUPPORTED_ARCHITECTURES = (MOBILENET_ARCHITECTURE, RESNET_ARCHITECTURE)


def build_architecture(architecture=MOBILENET_ARCHITECTURE, classes=2):
    """Build a compatible architecture locally, preserving native pixel edges."""
    if architecture == MOBILENET_ARCHITECTURE:
        from .torch_backend import build_model

        return build_model(classes)
    if architecture != RESNET_ARCHITECTURE:
        raise ValueError("unsupported detector architecture")
    if classes < 2:
        raise ValueError("detector requires background and at least one class")

    from functools import partial
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.transform import GeneralizedRCNNTransform
    from torchvision.ops.misc import FrozenBatchNorm2d

    class NativeTransform(GeneralizedRCNNTransform):
        def resize(self, image, target=None):
            return image, target

    # Match the official COCO_V1 graph exactly: FrozenBatchNorm, epsilon 0,
    # C2-C5 FPN features plus pooled P6. No implicit ImageNet weight retrieval.
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,
        norm_layer=partial(FrozenBatchNorm2d, eps=0.0),
        trainable_layers=5,
    )
    model = FasterRCNN(
        backbone,
        num_classes=classes,
        box_score_thresh=0.05,
        box_detections_per_img=300,
    )
    model.transform = NativeTransform(
        512, 512, model.transform.image_mean, model.transform.image_std
    )
    return model


def initialize_architecture_from_coco(path: Path, architecture=MOBILENET_ARCHITECTURE):
    """Copy the pretrained COCO background/airplane rows into a two-class head."""
    if architecture == MOBILENET_ARCHITECTURE:
        from .torch_backend import initialize_from_coco

        return initialize_from_coco(path)
    if architecture != RESNET_ARCHITECTURE:
        raise ValueError("unsupported detector architecture")
    if not sha256_file(Path(path)).startswith("258fb6c6"):
        raise ValueError(
            "initial weights do not match the official ResNet COCO artifact hash prefix"
        )

    import torch
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = build_architecture(architecture, 91)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    old = model.roi_heads.box_predictor
    new = FastRCNNPredictor(old.cls_score.in_features, 2)
    with torch.no_grad():
        new.cls_score.weight.copy_(old.cls_score.weight[[0, 5]])
        new.cls_score.bias.copy_(old.cls_score.bias[[0, 5]])
        rows = [0, 1, 2, 3, 20, 21, 22, 23]
        new.bbox_pred.weight.copy_(old.bbox_pred.weight[rows])
        new.bbox_pred.bias.copy_(old.bbox_pred.bias[rows])
    model.roi_heads.box_predictor = new
    return model
