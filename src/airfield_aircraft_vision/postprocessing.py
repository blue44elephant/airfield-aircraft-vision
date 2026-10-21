"""Deterministic native-grid box filtering and scene-wide duplicate suppression."""

from __future__ import annotations
from .optical_contracts import PixelBox


def iou(a, b):
    ax, ay, bx, by = a
    cx, cy, dx, dy = b
    inter = max(0, min(bx, dx) - max(ax, cx)) * max(0, min(by, dy) - max(ay, cy))
    area = (bx - ax) * (by - ay) + (dx - cx) * (dy - cy) - inter
    return inter / area if area > 0 else 0.0


def suppress(boxes: list[PixelBox], threshold: float = 0.4):
    if not 0 <= threshold <= 1:
        raise ValueError("NMS IoU must be in [0,1]")
    kept = []
    rejected = []
    for box in sorted(boxes, key=lambda b: (-b.score, b.xyxy)):
        if any(iou(box.xyxy, k.xyxy) > threshold for k in kept):
            rejected.append(box)
        else:
            kept.append(box)
    return kept, rejected
