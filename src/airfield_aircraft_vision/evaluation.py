"""One-to-one confidence-ordered IoU metrics, count error and explicit FP/FN."""

from .postprocessing import iou


def match_boxes(predictions, truth, ignored=(), iou_threshold=0.5):
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    matched = set()
    tp = []
    fp = []
    ignored_predictions = []
    for pidx, p in sorted(
        enumerate(predictions), key=lambda item: (-item[1].score, item[0])
    ):
        choices = [
            (iou(p.xyxy, g.xyxy), i) for i, g in enumerate(truth) if i not in matched
        ]
        best = max(choices, default=(0, -1))
        if best[0] >= iou_threshold:
            matched.add(best[1])
            tp.append({"prediction": pidx, "truth": best[1], "iou": best[0]})
        elif any(iou(p.xyxy, g.xyxy) >= iou_threshold for g in ignored):
            ignored_predictions.append(pidx)
        else:
            fp.append(pidx)
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": [i for i in range(len(truth)) if i not in matched],
        "ignored_predictions": ignored_predictions,
    }


def summarize(records):
    tp = sum(len(r["matches"]["true_positives"]) for r in records)
    fp = sum(len(r["matches"]["false_positives"]) for r in records)
    fn = sum(len(r["matches"]["false_negatives"]) for r in records)
    errors = [r["predicted_count"] - r["truth_count"] for r in records]
    return {
        "images": len(records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "count_mae": sum(abs(e) for e in errors) / len(errors) if errors else None,
        "count_bias": sum(errors) / len(errors) if errors else None,
        "count_rmse": (
            (sum(e * e for e in errors) / len(errors)) ** 0.5 if errors else None
        ),
        "count_absolute_error": sum(abs(e) for e in errors),
        "truth_count": sum(r["truth_count"] for r in records),
        "predicted_count": sum(r["predicted_count"] for r in records),
    }
