"""Reproducible AW-01C commands; no implicit dataset or checkpoint downloads."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="AW-01C native-pixel aircraft generalization experiments"
    )
    stages = parser.add_subparsers(dest="stage", required=True)
    prepare = stages.add_parser("prepare")
    prepare.add_argument("--baseline", type=Path, required=True)
    prepare.add_argument("--legacy", type=Path, required=True)
    prepare.add_argument("--root", type=Path, required=True)
    attach = stages.add_parser("attach-negatives")
    attach.add_argument("--manifest", type=Path, required=True)
    attach.add_argument("--reviews", type=Path, required=True)
    attach.add_argument("--output", type=Path, required=True)
    train = stages.add_parser("train")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--weights", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument(
        "--architecture",
        choices=["fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_v3_large_fpn"],
        default="fasterrcnn_resnet50_fpn",
    )
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--lr", type=float, default=0.0025)
    train.add_argument("--max-seconds", type=float, default=1500)
    train.add_argument("--threads", type=int, default=4)
    compare = stages.add_parser("compare")
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--baseline-checkpoint", type=Path, required=True)
    compare.add_argument("--candidates", nargs="+", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--regression-manifest", type=Path)
    compare.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        from .generalization_dataset import prepare_generalization

        prepare_generalization(args.baseline, args.legacy, args.root)
    elif args.stage == "attach-negatives":
        from .negative_crops import attach_reviewed_negatives

        attach_reviewed_negatives(args.manifest, args.reviews, args.output)
    elif args.stage == "train":
        from .generalization_training import train_generalization

        train_generalization(
            args.manifest,
            args.weights,
            args.out,
            architecture=args.architecture,
            epochs=args.epochs,
            lr=args.lr,
            threads=args.threads,
            max_seconds=args.max_seconds,
        )
    else:
        from .generalization_comparison import compare

        compare(
            args.manifest,
            args.baseline_checkpoint,
            args.candidates,
            args.out,
            regression_manifest=args.regression_manifest,
            threads=args.threads,
        )


if __name__ == "__main__":
    main()
